from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
import yfinance as yf
import json
import os
import time
import requests
import xml.etree.ElementTree as ET
from starlette.concurrency import run_in_threadpool
from typing import List, Optional

# Initialize FastAPI App
app = FastAPI(title="Stock Sentiment Predictor API")

# Configure CORS Middleware to allow requests from local browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction Cache (TTL: 5 minutes)
class PredictionCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache = {}

    def get(self, key: str) -> Optional[dict]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, data: dict):
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }

prediction_cache = PredictionCache(ttl_seconds=300)

# Financial Sentiment Lexicon for Google News
BULLISH_WORDS = {
    "gain", "rise", "jump", "surge", "soar", "rally", "bullish", "profit", 
    "upbeat", "upgrade", "outperform", "success", "beat", "positive", 
    "growth", "climb", "high", "strong", "higher", "advance", "buy",
    "record", "momentum", "expanding", "beats", "rises", "climbs", "surges"
}
BEARISH_WORDS = {
    "loss", "fall", "drop", "plunge", "slump", "bearish", "deficit", "warn", 
    "downgrade", "underperform", "decline", "lower", "slip", "sink", "slide", 
    "negative", "weak", "sell", "debt", "shrink", "contracting", "falls", 
    "drops", "slumps", "plummets", "miss", "misses", "warns"
}

def analyze_headline_sentiment(headline: str) -> float:
    words = [w.strip(".,!?\"'()[]").lower() for w in headline.split()]
    bull_count = sum(1 for w in words if w in BULLISH_WORDS)
    bear_count = sum(1 for w in words if w in BEARISH_WORDS)
    
    if bull_count == 0 and bear_count == 0:
        return 0.0
    return (bull_count - bear_count) / (bull_count + bear_count)

def fetch_live_news_sentiment(ticker: str) -> List[dict]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return []
        root = ET.fromstring(response.content)
        headlines = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            
            if title_el is not None and title_el.text:
                title = title_el.text
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0]
                
                score = analyze_headline_sentiment(title)
                sentiment_type = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
                
                headlines.append({
                    "title": title,
                    "link": link_el.text if link_el is not None else "#",
                    "date": pub_el.text if pub_el is not None else "",
                    "score": score,
                    "sentiment": sentiment_type
                })
        return headlines[:10]
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

# Global variables to hold loaded models and data
model = None
feature_names = []
df_sent = None
available_tickers = []

# Map inputs like GOOGL to GOOG transparently for sentiment lookup
TICKER_MAPPING = {
    "GOOGL": "GOOG"
}


@app.on_event("startup")
def startup_event():
    global model, feature_names, df_sent, available_tickers
    
    # 1. Load XGBoost Model
    model_path = "xgb_model.pkl"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file '{model_path}' not found.")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully via pickle.")
    except Exception as e:
        try:
            model = joblib.load(model_path)
            print("Model loaded successfully via joblib.")
        except Exception as e2:
            print(f"Pickle load failed: {e}. Joblib load failed: {e2}")
            raise RuntimeError("Failed to load XGBoost model.")

    # 2. Load Feature Columns
    feature_cols_path = "feature_cols.json"
    if not os.path.exists(feature_cols_path):
        raise RuntimeError(f"Feature columns file '{feature_cols_path}' not found.")
    
    try:
        with open(feature_cols_path, "r") as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature columns: {feature_names}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse feature columns: {e}")

    # 3. Load Latest Sentiment Data
    sentiment_path = "latest_sentiment.csv"
    if not os.path.exists(sentiment_path):
        raise RuntimeError(f"Sentiment data file '{sentiment_path}' not found.")
    
    try:
        df_sent = pd.read_csv(sentiment_path)
        # Normalize stock names
        df_sent['Stock Name'] = df_sent['Stock Name'].str.upper().str.strip()
        available_tickers = sorted(df_sent['Stock Name'].unique().tolist())
        print(f"Loaded sentiment data. Available tickers: {available_tickers}")
    except Exception as e:
        raise RuntimeError(f"Failed to load sentiment data: {e}")

class PredictRequest(BaseModel):
    ticker: str

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all standard technical indicators required by the XGBoost model.
    """
    df = df.copy()
    close = df['Close']
    volume = df['Volume']
    
    # daily_return
    df['daily_return'] = close.pct_change()
    
    # return lags
    df['return_lag1'] = df['daily_return'].shift(1)
    df['return_lag2'] = df['daily_return'].shift(2)
    
    # Moving Averages
    df['ma_5'] = close.rolling(5).mean()
    df['ma_20'] = close.rolling(20).mean()
    df['ma_50'] = close.rolling(50).mean()
    
    # RSI (Cutler's RSI based on rolling averages)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    # MACD & MACD Signal
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands Position
    std_20 = close.rolling(20).std()
    upper_band = df['ma_20'] + 2 * std_20
    lower_band = df['ma_20'] - 2 * std_20
    df['bb_position'] = (close - lower_band) / (upper_band - lower_band + 1e-8)
    
    # Volume Change
    df['volume_change'] = volume.pct_change()
    
    # Volatility
    df['volatility'] = df['daily_return'].rolling(5).std()
    
    return df

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Stock Sentiment Predictor API is running",
        "available_tickers": available_tickers + ["GOOGL"] # Explicitly show GOOGL is supported too
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    ticker_input = request.ticker.upper().strip()
    
    # 1. Check Prediction Cache (TTL cache hit)
    cached_res = prediction_cache.get(ticker_input)
    if cached_res:
        # Return a copy with cached flag marked as true
        res_copy = cached_res.copy()
        res_copy["cached"] = True
        return res_copy

    # Transparently map input ticker to database ticker (e.g. GOOGL -> GOOG)
    sentiment_ticker = TICKER_MAPPING.get(ticker_input, ticker_input)
    
    # Check if we have sentiment data for this ticker
    if sentiment_ticker not in available_tickers:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Sentiment data is not available for ticker '{ticker_input}'.",
                "available_tickers": available_tickers + ["GOOGL"]
            }
        )
    
    # Extract sentiment values from CSV
    sent_row = df_sent[df_sent['Stock Name'] == sentiment_ticker].iloc[0]
    
    # 2. Fetch historical price data via yfinance asynchronously using run_in_threadpool
    try:
        stock = yf.Ticker(ticker_input)
        df_price = await run_in_threadpool(stock.history, period="1y")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch market data from yfinance for '{ticker_input}': {str(e)}"
        )
        
    if df_price.empty or len(df_price) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient historical data available for ticker '{ticker_input}' (need at least 50 trading days)."
        )
        
    # Calculate indicators
    df_features = compute_technical_indicators(df_price)
    
    # Get values from the most recent trading day
    last_row = df_features.iloc[-1]
    
    # 3. Fetch live Google News and analyze sentiment in parallel
    live_news = await run_in_threadpool(fetch_live_news_sentiment, ticker_input)
    
    if live_news:
        scores = [item["score"] for item in live_news]
        live_mean = float(np.mean(scores))
        pos_count = sum(1 for s in scores if s > 0.1)
        neg_count = sum(1 for s in scores if s < -0.1)
        tot = len(scores)
        
        live_pct_pos = pos_count / tot
        live_pct_neg = neg_count / tot
        live_bull_ratio = pos_count / (pos_count + neg_count + 1e-8)
        live_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        
        # Blend: 30% CSV baseline + 70% live sentiment (to adapt to 2026 current market events)
        alpha = 0.7
        blended_sentiment_mean = (1 - alpha) * float(sent_row["sentiment_mean"]) + alpha * live_mean
        blended_pct_positive = (1 - alpha) * float(sent_row["pct_positive"]) + alpha * live_pct_pos
        blended_pct_negative = (1 - alpha) * float(sent_row["pct_negative"]) + alpha * live_pct_neg
        blended_bullish_ratio = (1 - alpha) * float(sent_row["bullish_ratio"]) + alpha * live_bull_ratio
        blended_sentiment_std = (1 - alpha) * float(sent_row["sentiment_std"]) + alpha * live_std
        blended_weighted_sentiment = blended_sentiment_mean * float(sent_row["avg_confidence"])
    else:
        # Fallback to pure CSV data
        blended_sentiment_mean = float(sent_row["sentiment_mean"])
        blended_weighted_sentiment = float(sent_row["weighted_sentiment"])
        blended_pct_positive = float(sent_row["pct_positive"])
        blended_pct_negative = float(sent_row["pct_negative"])
        blended_sentiment_std = float(sent_row["sentiment_std"])
        blended_bullish_ratio = float(sent_row["bullish_ratio"])
        
    # Build complete features dictionary
    features_dict = {
        "daily_return": last_row["daily_return"],
        "return_lag1": last_row["return_lag1"],
        "return_lag2": last_row["return_lag2"],
        "ma_5": last_row["ma_5"],
        "ma_20": last_row["ma_20"],
        "ma_50": last_row["ma_50"],
        "rsi": last_row["rsi"],
        "macd": last_row["macd"],
        "macd_signal": last_row["macd_signal"],
        "bb_position": last_row["bb_position"],
        "volume_change": last_row["volume_change"],
        "volatility": last_row["volatility"],
        
        # Blended sentiment metrics
        "sentiment_mean": blended_sentiment_mean,
        "weighted_sentiment": blended_weighted_sentiment,
        "pct_positive": blended_pct_positive,
        "pct_negative": blended_pct_negative,
        "sentiment_std": blended_sentiment_std,
        "tweet_count": int(sent_row["tweet_count"]),
        "avg_confidence": float(sent_row["avg_confidence"]),
        "bullish_ratio": blended_bullish_ratio
    }
    
    # Convert dict to array matching the feature columns order
    try:
        X = np.array([[features_dict[col] for col in feature_names]])
        # Replace any potential NaNs (e.g. from volume change on first index) with 0
        X = np.nan_to_num(X)
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature alignment mismatch. Missing feature column: {str(e)}"
        )
        
    # Perform model prediction
    try:
        # predict returns class label [0 or 1]
        pred_class = int(model.predict(X)[0])
        # predict_proba returns probability estimates [prob_class_0, prob_class_1]
        probs = model.predict_proba(X)[0]
        
        prob_down = float(probs[0])
        prob_up = float(probs[1])
        
        prediction = "UP" if pred_class == 1 else "DOWN"
        confidence = prob_up if pred_class == 1 else prob_down
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model execution error: {str(e)}"
        )
        
    # Calculate local feature contribution drivers (Explainable AI)
    try:
        # standard dev mapping
        # Calculate standard deviation and mean over the fetched historical price data for technicals
        # Limit to the columns in feature_names that are technical indicators
        tech_cols = [col for col in feature_names if col in last_row.index and col not in df_sent.columns]
        tech_means = df_features[tech_cols].mean()
        tech_stds = df_features[tech_cols].std() + 1e-8
        
        # Calculate standard deviation and mean over the database for sentiment metrics
        sent_cols = [col for col in feature_names if col in df_sent.columns and col != 'Stock Name' and col != 'Date']
        df_sent_filled = df_sent.copy()
        df_sent_filled[sent_cols] = df_sent_filled[sent_cols].fillna(0.0)
        sent_means = df_sent_filled[sent_cols].mean()
        sent_stds = df_sent_filled[sent_cols].std() + 1e-8
        
        importances = model.feature_importances_
        drivers = []
        
        # Reader-friendly name mappings for the UI
        FEATURE_DESCRIPTIONS = {
            "daily_return": "Recent price trend",
            "return_lag1": "Yesterday's price movement",
            "return_lag2": "Two-day lag price movement",
            "ma_5": "Short-term momentum (5-day MA)",
            "ma_20": "Medium-term trend (20-day MA)",
            "ma_50": "Long-term trend (50-day MA)",
            "rsi": "RSI Momentum oscillator",
            "macd": "MACD trend oscillator",
            "macd_signal": "MACD signal line crossover",
            "bb_position": "Bollinger Band volatility boundary",
            "volume_change": "Trading volume volatility change",
            "volatility": "5-day historical price volatility",
            "sentiment_mean": "FinBERT average news sentiment",
            "weighted_sentiment": "Weighted sentiment impact",
            "pct_positive": "Bullish news ratio",
            "pct_negative": "Bearish news ratio",
            "sentiment_std": "Sentiment dispersion/agreement",
            "tweet_count": "Social tweet buzz volume",
            "avg_confidence": "FinBERT classification confidence",
            "bullish_ratio": "Social sentiment bullish ratio"
        }
        
        for i, col in enumerate(feature_names):
            val = features_dict[col]
            if col in tech_cols:
                mean_val = tech_means[col]
                std_val = tech_stds[col]
            else:
                mean_val = sent_means[col]
                std_val = sent_stds[col]
                
            z_score = (val - mean_val) / std_val
            
            # Context-specific direction tweaks
            # positive values indicate bullish contribution, negative is bearish
            direction = 1
            if col in ["pct_negative"]:
                direction = -1
            elif col == "rsi":
                # High RSI is overbought (bearish), low RSI is oversold (bullish)
                z_score = (50 - val) / std_val
            elif col == "bb_position":
                # High BB position is near upper band (resistance - bearish), low is near lower band (bullish)
                z_score = (0.5 - val) / std_val
                
            score = z_score * importances[i] * direction
            drivers.append({
                "feature": col,
                "score": float(score),
                "value": float(val)
            })
            
        # Sort by absolute contribution strength
        drivers_sorted = sorted(drivers, key=lambda x: abs(x["score"]), reverse=True)
        
        formatted_drivers = []
        for d in drivers_sorted:
            feat = d["feature"]
            score = d["score"]
            val = d["value"]
            
            label = FEATURE_DESCRIPTIONS.get(feat, feat)
            
            # Map standardized contribution score to an impact percentage scale (0 to 100)
            impact_val = min(int(abs(score) * 1100), 100)
            if impact_val < 3:
                continue # Skip negligible weights
                
            impact_type = "BULLISH" if score > 0 else "BEARISH"
            
            formatted_drivers.append({
                "feature": feat,
                "label": label,
                "impact": impact_val,
                "type": impact_type,
                "value": val
            })
            
        # Keep top 4 primary drivers
        formatted_drivers = formatted_drivers[:4]
    except Exception as e:
        print(f"XAI calculation failed: {str(e)}")
        formatted_drivers = []
 
    # Extract 30-day candlestick OHLC and moving average (SMA) history
    try:
        df_chart = df_features.tail(30)
        chart_data = []
        for idx, row in df_chart.iterrows():
            timestamp = int(idx.timestamp() * 1000)
            chart_data.append({
                "time": timestamp,
                "ohlc": [
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"])
                ],
                "ma_5": float(row["ma_5"]) if not np.isnan(row["ma_5"]) else float(row["Close"]),
                "ma_20": float(row["ma_20"]) if not np.isnan(row["ma_20"]) else float(row["Close"])
            })
    except Exception as e:
        print(f"Chart formatting failed: {str(e)}")
        chart_data = []
 
    # Store predictions alongside technical metrics, XAI drivers, chart data and headlines in cache
    response_payload = {
        "ticker": ticker_input,
        "prediction": prediction,
        "confidence": confidence,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "current_price": float(last_row["Close"]),
        "rsi": float(last_row["rsi"]) if not np.isnan(last_row["rsi"]) else 50.0,
        "macd": float(last_row["macd"]) if not np.isnan(last_row["macd"]) else 0.0,
        "macd_signal": float(last_row["macd_signal"]) if not np.isnan(last_row["macd_signal"]) else 0.0,
        "volatility": float(last_row["volatility"]) if not np.isnan(last_row["volatility"]) else 0.0,
        "daily_return": float(last_row["daily_return"]) if not np.isnan(last_row["daily_return"]) else 0.0,
        "tweet_count": int(sent_row["tweet_count"]),
        "bullish_ratio": blended_bullish_ratio,
        "sentiment_mean": blended_sentiment_mean,
        "drivers": formatted_drivers,
        "chart_data": chart_data,
        "headlines": live_news,
        "cached": False
    }
    
    prediction_cache.set(ticker_input, response_payload)
    return response_payload


