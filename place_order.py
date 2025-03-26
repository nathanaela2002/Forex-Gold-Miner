import time
import datetime
import numpy as np
import pandas as pd
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.orders import OrderCreate
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
load_dotenv()

OANDA_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ENVIRONMENT = os.getenv("ENVIRONMENT")

# Trading settings: candles lookback, probability cutoffs, risk ratio and loop pause time
LOOKBACK = 10       
THRESHOLD_UP = 0.7  
THRESHOLD_DOWN = 0.3  
RISK_REWARD = 2.0   
SLEEP_INTERVAL = 6000

# Load trained model 
model = load_model("best_model_final.h5")

# Features the model was trained on
FEATURES = [
    "RSI", "MA", "BB_upper", "BB_lower", "pmi_actual",
    "bullish_engulfing", "bearish_engulfing",
    "MACD_line", "MACD_signal", "stoch_%K", "stoch_%D", "ATR"
]

# Get latest candles from OANDA and return them in a DataFrame with date, open, high, low, close, volume.
def fetch_recent_candles(instrument="XAU_USD", count=50, granularity="M30"):
    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=ENVIRONMENT)
    params = {"count": str(count), "granularity": granularity, "price": "M"}
    r = InstrumentsCandles(instrument=instrument, params=params)
    response = client.request(r)
    candles = response.get("candles", [])
    rows = []
    for c in candles:
        dt = pd.to_datetime(c["time"].rstrip("Z")[:26])
        mid = c["mid"]
        rows.append({
            "date": dt,
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": c["volume"],
            "complete": c["complete"]
        })
    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Calculate tech indicators RSI, moving avg, Bollinger bands, MACD, stochastic, ATR and candlestick patterns.
def compute_indicators(df):
    df = df.copy()
    # Calc RSI with a 14 period
    period_rsi = 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period_rsi).mean()
    avg_loss = loss.rolling(window=period_rsi).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    # Calc simple moving average over 20 periods
    period_ma = 20
    df["MA"] = df["close"].rolling(window=period_ma).mean()
    # Calculate Bollinger bands with a 20-period MA and 2 std deviations
    period_boll = 20
    num_std = 2
    df["BB_MA"] = df["close"].rolling(window=period_boll).mean()
    df["BB_STD"] = df["close"].rolling(window=period_boll).std()
    df["BB_upper"] = df["BB_MA"] + num_std * df["BB_STD"]
    df["BB_lower"] = df["BB_MA"] - num_std * df["BB_STD"]
    # Compute MACD line and signal using 12, 26, and 9 periods
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD_line"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD_line"].ewm(span=macd_signal, adjust=False).mean()
    # Get stochastic oscillator values using a 14 period window
    stoch_window = 14
    rolling_low = df["low"].rolling(stoch_window).min()
    rolling_high = df["high"].rolling(stoch_window).max()
    df["stoch_%K"] = 100 * (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    df["stoch_%D"] = df["stoch_%K"].rolling(3).mean()
    # Calculate ATR with a 14 period window
    atr_window = 14
    df["prev_close"] = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["prev_close"]).abs()
    tr3 = (df["low"] - df["prev_close"]).abs()
    df["TR"] = tr1.combine(tr2, max).combine(tr3, max)
    df["ATR"] = df["TR"].rolling(atr_window).mean()
    # Check for engulfing candlestick patterns (bullish and bearish)
    bullish = []
    bearish = []
    for i in range(len(df)):
        if i == 0:
            bullish.append(0)
            bearish.append(0)
            continue
        prev_o = df["open"].iloc[i-1]
        prev_c = df["close"].iloc[i-1]
        curr_o = df["open"].iloc[i]
        curr_c = df["close"].iloc[i]
        if (prev_c < prev_o) and (curr_c > curr_o) and ((curr_c - curr_o) > (prev_o - prev_c)):
            bullish.append(1)
        else:
            bullish.append(0)
        if (prev_c > prev_o) and (curr_c < curr_o) and ((curr_o - curr_c) > (prev_c - prev_o)):
            bearish.append(1)
        else:
            bearish.append(0)
    df["bullish_engulfing"] = bullish
    df["bearish_engulfing"] = bearish
    return df

# Place a market order on OANDA with optional stop loss and take profit prices.
def place_market_order(units, stop_loss=None, take_profit=None, instrument="XAU_USD"):
    order_data = {
        "order": {
            "instrument": instrument,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    if stop_loss is not None:
        order_data["order"]["stopLossOnFill"] = {"price": f"{stop_loss:.3f}"}
    if take_profit is not None:
        order_data["order"]["takeProfitOnFill"] = {"price": f"{take_profit:.3f}"}
    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=ENVIRONMENT)
    r = OrderCreate(OANDA_ACCOUNT_ID, data=order_data)
    response = client.request(r)
    return response

# Main trading loop that fetches data, computes indicators, and places trades.
def trading_loop():
    print("Starting Trading Loop...")
    scaler = StandardScaler()
    first_fit_done = False
    while True:
        try:
            # Grab the newest candle data from OANDA
            df = fetch_recent_candles("XAU_USD", count=50, granularity="M30")
            if len(df) < LOOKBACK + 1:
                print(f"Not enough data. Have {len(df)} candles, need at least {LOOKBACK+1}. Waiting...")
                time.sleep(SLEEP_INTERVAL)
                continue
            # Calculate tech indicators
            df_ind = compute_indicators(df)
            # Set pmi default of 50.0
            if "pmi_actual" not in df_ind.columns:
                df_ind["pmi_actual"] = 50.0
            # Remove rows that are missing needed features
            df_ready = df_ind.dropna(subset=FEATURES).copy()
            if len(df_ready) < LOOKBACK + 1:
                print(f"After dropping NaNs, not enough data. Have {len(df_ready)}. Waiting...")
                time.sleep(SLEEP_INTERVAL)
                continue
            # Scale the feature columns
            if not first_fit_done:
                scaler.fit(df_ready[FEATURES])
                first_fit_done = True
            X_all = scaler.transform(df_ready[FEATURES])
            df_ready.loc[:, "scaled_features"] = list(X_all)
            # Use the last row as current data for prediction
            last_row = df_ready.iloc[-1]
            if not isinstance(last_row["scaled_features"], np.ndarray):
                print("No scaled features for last row. Skipping...")
                time.sleep(SLEEP_INTERVAL)
                continue
            x_pred = last_row["scaled_features"].reshape(1, -1)
            prob_up = model.predict(x_pred, verbose=0)[0][0]
            print(f"[{datetime.datetime.now()}] Probability Up = {prob_up*100:.2f}%")
            # Check if we have a clear bullish or bearish signal
            if prob_up >= THRESHOLD_UP:
                direction = "bullish"
            elif prob_up <= THRESHOLD_DOWN:
                direction = "bearish"
            else:
                direction = None
            if direction is not None:
                entry_price = last_row["close"]
                # Set support and resistance based on the last LOOKBACK candles
                recent = df_ready.iloc[-(LOOKBACK+1) : -1]
                resistance = recent["high"].max()
                support = recent["low"].min()
                if direction == "bullish":
                    potential_profit = resistance - entry_price
                    potential_loss = entry_price - support
                else:
                    potential_profit = entry_price - support
                    potential_loss = resistance - entry_price
                if potential_loss <= 0:
                    print("Potential loss <= 0, skipping trade.")
                else:
                    rr = potential_profit / potential_loss
                    print(f"Direction={direction}, Entry={entry_price:.3f}, Resist={resistance:.3f}, Support={support:.3f}, R/R={rr:.2f}")
                    if rr >= RISK_REWARD:
                        # Prepare to place the trade
                        if direction == "bullish":
                            units = 1
                            stop_loss_price = support
                            take_profit_price = entry_price + 2.0 * (entry_price - support)
                        else:
                            units = -1
                            stop_loss_price = resistance
                            take_profit_price = entry_price - 2.0 * (resistance - entry_price)
                        print(f"Placing {direction.upper()} order with SL={stop_loss_price:.3f}, TP={take_profit_price:.3f}")
                        resp = place_market_order(
                            units=units,
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price
                        )
                        print("Order response:", resp)
                    else:
                        print("Risk/Reward not met. No trade.")
            else:
                print("No strong signal. No trade.")
        except Exception as e:
            print("Error in trading loop:", str(e))
        # Wait before starting the next loop
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    trading_loop()
