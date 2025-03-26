#!/usr/bin/env python3

import time
import datetime
import mysql.connector
import numpy as np
import pandas as pd
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

import os
from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------
# Configuration Constants
# ----------------------------------------
OANDA_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
SQL_HOST = os.getenv("host")
SQL_USER = os.getenv("user")
SQL_PASSWORD = os.getenv("password")
SQL_DATABASE = os.getenv("database")
NUM_ITERATIONS = 100  # Number of optimization cycles

# ----------------------------------------
# Data Pipeline Functions
# ----------------------------------------

def fetch_oanda_data():
    """Fetch OANDA data (6 months of 30-minute candles)"""
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=180)  # ~6 months
    chunk_delta = datetime.timedelta(days=30)
    data_frames = []
    current_start = start

    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment="practice")

    while current_start < end:
        current_end = current_start + chunk_delta
        if current_end > end:
            current_end = end

        from_str = current_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = current_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "from": from_str,
            "to": to_str,
            "granularity": "M30",  # 30-minute candles
            "price": "M"           # mid price
        }
        r = InstrumentsCandles(instrument="XAU_USD", params=params)
        try:
            response = client.request(r)
        except Exception as e:
            print(f"Error fetching candles from {from_str} to {to_str}: {e}")
            current_start = current_end
            continue

        candles = response.get("candles", [])
        rows = []
        for c in candles:
            time_str = c["time"].rstrip("Z")[:26]
            dt = pd.to_datetime(time_str)
            mid = c["mid"]
            row = {
                "date": dt,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": c["volume"],
                "complete": c["complete"]
            }
            rows.append(row)
        if rows:
            df_chunk = pd.DataFrame(rows)
            data_frames.append(df_chunk)
        current_start = current_end
        time.sleep(1)

    if data_frames:
        df = pd.concat(data_frames, ignore_index=True)
        df.sort_values("date", inplace=True)
        return df
    else:
        return pd.DataFrame()

def compute_indicators(df, period_rsi=14, period_ma=20, period_boll=20, num_std=2):
    """Compute RSI, moving average, and Bollinger Bands"""
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period_rsi).mean()
    avg_loss = loss.rolling(window=period_rsi).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average
    df["MA"] = df["close"].rolling(window=period_ma).mean()

    # Bollinger Bands
    df["BB_MA"] = df["close"].rolling(window=period_boll).mean()
    df["BB_STD"] = df["close"].rolling(window=period_boll).std()
    df["BB_upper"] = df["BB_MA"] + num_std * df["BB_STD"]
    df["BB_lower"] = df["BB_MA"] - num_std * df["BB_STD"]
    return df

def compute_extra_indicators(df, macd_fast=12, macd_slow=26, macd_signal=9,
                             stoch_window=14, atr_window=14):
    """Compute extra indicators: MACD, Stochastic, ATR"""
    # --- MACD ---
    # Typical for MACD: fast=12, slow=26, signal=9
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD_line"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD_line"].ewm(span=macd_signal, adjust=False).mean()

    # --- Stochastic Oscillator (fast %K, %D) ---
    # (High/Low within last 'stoch_window', compare with current close)
    rolling_low = df["low"].rolling(stoch_window).min()
    rolling_high = df["high"].rolling(stoch_window).max()
    df["stoch_%K"] = 100 * (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    df["stoch_%D"] = df["stoch_%K"].rolling(3).mean()  # 3-period SMA of %K

    # --- ATR (Average True Range) ---
    # ATR is an average of the True Range over a window
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    df["prev_close"] = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["prev_close"]).abs()
    tr3 = (df["low"] - df["prev_close"]).abs()
    df["TR"] = tr1.combine(tr2, max).combine(tr3, max)
    df["ATR"] = df["TR"].rolling(atr_window).mean()

    return df

def detect_candlestick_patterns(df):
    """Detect bullish and bearish engulfing patterns"""
    patterns = {
        "bullish_engulfing": [],
        "bearish_engulfing": [],
    }
    for i in range(len(df)):
        if i == 0:
            patterns["bullish_engulfing"].append(0)
            patterns["bearish_engulfing"].append(0)
            continue
        prev_open = df["open"].iloc[i-1]
        prev_close = df["close"].iloc[i-1]
        curr_open = df["open"].iloc[i]
        curr_close = df["close"].iloc[i]
        # Bullish engulfing
        if (prev_close < prev_open) and (curr_close > curr_open) and \
           ((curr_close - curr_open) > (prev_open - prev_close)):
            patterns["bullish_engulfing"].append(1)
        else:
            patterns["bullish_engulfing"].append(0)

        # Bearish engulfing
        if (prev_close > prev_open) and (curr_close < curr_open) and \
           ((curr_open - curr_close) > (prev_close - prev_open)):
            patterns["bearish_engulfing"].append(1)
        else:
            patterns["bearish_engulfing"].append(0)

    for pattern_name, values in patterns.items():
        df[pattern_name] = values
    return df

def load_pmi_data():
    """Load PMI data from MySQL"""
    conn = mysql.connector.connect(
        host=SQL_HOST,
        user=SQL_USER,
        password=SQL_PASSWORD,
        database=SQL_DATABASE
    )
    cursor = conn.cursor()
    query = """
        SELECT release_date, actual
        FROM pmi_history
        ORDER BY release_date
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["release_date", "pmi_actual"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    cursor.close()
    conn.close()
    return df

def merge_pmi_with_prices(df_prices, df_pmi):
    """Merge PMI data with price data using as-of merge"""
    df_prices = df_prices.copy()
    df_prices.rename(columns={"date": "timestamp"}, inplace=True)
    df_pmi = df_pmi.copy()
    df_pmi.rename(columns={"release_date": "pmi_date"}, inplace=True)
    df_pmi.sort_values("pmi_date", inplace=True)
    df_prices.sort_values("timestamp", inplace=True)

    merged = pd.merge_asof(
        df_prices, df_pmi,
        left_on="timestamp",
        right_on="pmi_date",
        direction="backward"
    )
    merged["pmi_actual"].fillna(method="ffill", inplace=True)
    merged.rename(columns={"timestamp": "date"}, inplace=True)
    return merged

# ----------------------------------------
# Data Reshaping for LSTM
# ----------------------------------------
def create_timeseries_samples(X, y, seq_len=5):
    """
    Convert (X, y) into 3D sequences for RNN/LSTM:
    X shape => (num_samples, seq_len, num_features)
    y shape => (num_samples,)
    """
    X_list = []
    y_list = []
    for i in range(len(X) - seq_len):
        X_list.append(X[i : i + seq_len])
        y_list.append(y[i + seq_len])
    return np.array(X_list), np.array(y_list)

# ----------------------------------------
# Enhanced Model Architecture
# ----------------------------------------
def create_model(input_dim, hyperparams, seq_len=5):
    """
    Build either an MLP or an LSTM depending on hyperparams['arch'].
    For LSTM, we expect input shape = (seq_len, input_dim).
    For MLP, we expect input shape = (input_dim,).
    """
    # Ensure units are integers
    units1 = int(hyperparams['units1'])
    units2 = int(hyperparams['units2'])
    units3 = int(hyperparams['units3'])
    
    if hyperparams['arch'] == 'LSTM':
        # LSTM
        model = models.Sequential([
            layers.Input(shape=(seq_len, input_dim)),
            layers.LSTM(units1, return_sequences=True),
            layers.BatchNormalization(),
            layers.LSTM(units2, return_sequences=False),
            layers.BatchNormalization(),
            layers.Dense(units3, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1, activation='sigmoid')
        ])
    else:
        # Default to MLP
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(units1, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units3, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1, activation='sigmoid')
        ])
    
    optimizer = optimizers.AdamW(
        learning_rate=hyperparams['lr'],
        weight_decay=hyperparams['wd']
    )
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

# ----------------------------------------
# Hyperparameter Evolution Logic
# ----------------------------------------
def generate_hyperparams(current_best=None):
    """Generate new hyperparameters with evolutionary improvements"""
    arch_choices = ['MLP', 'LSTM']  # we can add or remove
    if current_best:
        # Mutate best parameters with diminishing random noise
        new_params = {
            'lr': np.clip(current_best['lr'] * np.random.normal(1, 0.2), 1e-5, 1e-2),
            'wd': np.clip(current_best['wd'] * np.random.normal(1, 0.2), 1e-6, 1e-3),
            'units1': int(np.clip(current_best['units1'] * np.random.normal(1, 0.1), 16, 1024)),
            'units2': int(np.clip(current_best['units2'] * np.random.normal(1, 0.1), 16, 1024)),
            'units3': int(np.clip(current_best['units3'] * np.random.normal(1, 0.1), 16, 1024)),
            'batch_size': int(np.clip(current_best['batch_size'] + np.random.randint(-16, 17), 16, 256)),
            'arch': current_best['arch']
        }
        # small chance to flip the architecture
        if np.random.rand() < 0.1:
            new_params['arch'] = np.random.choice(arch_choices)
        return new_params
    
    # Initial random parameters
    return {
        'lr': 10**np.random.uniform(-4, -2),   # 1e-4 to 1e-2
        'wd': 10**np.random.uniform(-6, -3),   # 1e-6 to 1e-3
        'units1': np.random.choice([64, 128, 256, 512]),
        'units2': np.random.choice([32, 64, 128, 256]),
        'units3': np.random.choice([16, 32, 64, 128]),
        'batch_size': np.random.choice([32, 64, 128]),
        'arch': np.random.choice(arch_choices)
    }

# ----------------------------------------
# Core Training Loop with Walk-Forward
# ----------------------------------------
def optimize_model(X, y, num_iterations, seq_len=5):
    """
    Run evolutionary optimization of model parameters.
    Uses TimeSeriesSplit for walk-forward validation.
    If arch == LSTM, we create timeseries sequences first.
    """
    best_accuracy = 0.0
    best_params = None
    history = []

    for iteration in range(num_iterations):
        # Generate new parameters based on current best
        hyperparams = generate_hyperparams(best_params)

        # Prepare data depending on architecture
        if hyperparams['arch'] == 'LSTM':
            # For LSTM we reshape data into sequences
            X_seq, y_seq = create_timeseries_samples(X, y, seq_len=seq_len)
            # We scale after forming sequences, or scale before + re-form
            # We'll do: scale each feature across entire dataset before sequence creation
            # but in real usage, you'd do partial scaling per train fold.
            # For simplicity, let's just do one scale pass outside.

            # Because we've already created X_seq, let's just do timeseries split on the length of X_seq
            # TimeSeriesSplit on the first dimension
            data_length = X_seq.shape[0]
            tscv = TimeSeriesSplit(n_splits=5)
            
            val_accuracies = []
            for train_idx, val_idx in tscv.split(range(data_length)):
                X_train = X_seq[train_idx]
                X_val = X_seq[val_idx]
                y_train = y_seq[train_idx]
                y_val = y_seq[val_idx]
                
                model = create_model(X_train.shape[2], hyperparams, seq_len=seq_len)
                
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=min(10, 5 + iteration//20),
                    restore_best_weights=True
                )
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=hyperparams['batch_size'],
                    callbacks=[early_stop],
                    verbose=0
                )
                val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
                val_accuracies.append(val_acc)

            mean_acc = np.mean(val_accuracies)

        else:
            # MLP approach:
            # Scale the entire X first
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            val_accuracies = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_val = X_scaled[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]

                model = create_model(X_train.shape[1], hyperparams, seq_len=seq_len)

                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=min(10, 5 + iteration//20),
                    restore_best_weights=True
                )
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=hyperparams['batch_size'],
                    callbacks=[early_stop],
                    verbose=0
                )
                val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
                val_accuracies.append(val_acc)

            mean_acc = np.mean(val_accuracies)

        history.append(mean_acc)

        # Maintain elite parameters
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_params = hyperparams
            print(f"Iteration {iteration+1}: New best accuracy {best_accuracy:.2%}")

            # Save improved model as final best model
            # Rebuild and train on full data with best_params if you want truly best final model
            # We'll do it quickly now:
            if best_params['arch'] == 'LSTM':
                # Rebuild on full data for final save
                X_seq_full, y_seq_full = create_timeseries_samples(X, y, seq_len=seq_len)
                model_final = create_model(X_seq_full.shape[2], best_params, seq_len=seq_len)
                model_final.fit(
                    X_seq_full, y_seq_full,
                    epochs=30,  # short train
                    batch_size=best_params['batch_size'],
                    verbose=0
                )
                model_final.save("best_model_final.h5")
            else:
                scaler = StandardScaler()
                X_scaled_full = scaler.fit_transform(X)
                model_final = create_model(X_scaled_full.shape[1], best_params, seq_len=seq_len)
                model_final.fit(
                    X_scaled_full, y,
                    epochs=30,  # short train
                    batch_size=best_params['batch_size'],
                    verbose=0
                )
                model_final.save("best_model_final.h5")

        # Adaptive parameter space narrowing
        if iteration > 20 and (max(history[-20:]) - min(history[-20:])) < 0.01:
            print("Convergence detected - refining parameter space")
            hyperparams = generate_hyperparams(best_params)

    return best_params, best_accuracy, history

# ----------------------------------------
# Simulation Function (still MLP-style for demonstration)
# ----------------------------------------
def simulate_trades(df, model, threshold=0.7, lookback=10, risk_reward=2.0):
    """
    For each candle in df (starting from index 'lookback'),
    use the model to predict the probability that the next candle's close will be higher.
    If the probability is above 'threshold' (for bullish) or below (1 - threshold) (for bearish),
    calculate potential profit and loss using resistance/support from the previous 'lookback' candles.
    If the risk/reward ratio meets the requirement, simulate a trade using the next candle's data.
    Returns the win rate (percentage of simulated trades that were correct) and the total number of trades.

    NOTE: This function uses a single-step lookback for the input vector. If you're using LSTM,
    you'd need to sequence the data similarly to your training approach. This is just a demonstration
    of how you might evaluate the MLP approach or quick signals from the last candle.
    """
    trades = 0
    wins = 0

    # For quick demonstration, a subset of features:
    features = [
        "RSI", "MA", "BB_upper", "BB_lower", "pmi_actual",
        "bullish_engulfing", "bearish_engulfing",
        "MACD_line", "MACD_signal", "stoch_%K", "stoch_%D", "ATR"
    ]

    # We need a scaler consistent with MLP usage, so let's do a quick fit:
    # In a real scenario, you'd use the final scaler from training.
    df_clean = df.dropna(subset=features)
    if df_clean.empty:
        return 0, 0

    X_all = df_clean[features].values
    sc = StandardScaler()
    X_all_scaled = sc.fit_transform(X_all)

    # We'll align scaled data with original index
    df_clean.loc[:, 'scaled_features'] = list(X_all_scaled)  # each row is a scaled vector

    for i in range(lookback, len(df_clean) - 1):
        row = df_clean.iloc[i]
        if not isinstance(row['scaled_features'], np.ndarray):
            continue
        x = row['scaled_features'].reshape(1, -1)
        prob = model.predict(x, verbose=0)[0][0]

        if prob >= threshold:
            direction = "bullish"
        elif prob <= (1 - threshold):
            direction = "bearish"
        else:
            continue

        entry = row["close"]

        # find the actual index in the original df
        idx = df_clean.index[i]
        recent_idx_start = i - lookback
        if recent_idx_start < 0:
            continue

        # Because of alignment, let's just do recent in df_clean
        recent = df_clean.iloc[i - lookback : i]
        resistance = recent["high"].max()
        support = recent["low"].min()

        if direction == "bullish":
            potential_profit = resistance - entry
            potential_loss = entry - support
        else:
            potential_profit = entry - support
            potential_loss = resistance - entry

        if potential_loss <= 0:
            continue
        if (potential_profit / potential_loss) < risk_reward:
            continue

        # Next candle
        if i+1 >= len(df_clean):
            continue
        next_candle = df_clean.iloc[i+1]
        next_close = next_candle["close"]
        next_high = next_candle["high"]
        next_low = next_candle["low"]

        if direction == "bullish":
            if next_high >= entry + potential_profit:
                win = True
            elif next_low <= entry - potential_loss:
                win = False
            else:
                win = False
        else:
            if next_low <= entry - potential_profit:
                win = True
            elif next_high >= entry + potential_loss:
                win = False
            else:
                win = False

        trades += 1
        if win:
            wins += 1

    win_rate = wins / trades if trades > 0 else 0
    return win_rate, trades

# ----------------------------------------
# Main Function
# ----------------------------------------
def main():
    # Data collection and preprocessing
    price_df = fetch_oanda_data()
    if price_df.empty:
        print("No data fetched from OANDA.")
        return

    # Compute existing indicators
    price_df = compute_indicators(price_df)
    # Compute extra indicators
    price_df = compute_extra_indicators(price_df)
    # Detect candle patterns
    price_df = detect_candlestick_patterns(price_df)

    # Load PMI data
    pmi_df = load_pmi_data()
    merged_df = merge_pmi_with_prices(price_df, pmi_df)

    # Labeling: shift close by -1, but let's do a threshold-based label
    merged_df["future_close"] = merged_df["close"].shift(-1)
    merged_df.dropna(subset=["future_close"], inplace=True)

    # Alternative labeling: e.g., label=1 if next close is at least +0.1% higher
    # than current close; else 0
    merged_df["label"] = ((merged_df["future_close"] - merged_df["close"]) / merged_df["close"] > 0.001).astype(int)

    # Prepare features. Let's use everything we have:
    all_features = [
        "RSI", "MA", "BB_upper", "BB_lower", "pmi_actual",
        "bullish_engulfing", "bearish_engulfing",
        "MACD_line", "MACD_signal", "stoch_%K", "stoch_%D", "ATR"
    ]
    # Drop any row missing these features
    merged_df.dropna(subset=all_features, inplace=True)

    X = merged_df[all_features].values
    y = merged_df["label"].values

    # Run evolutionary optimization of model hyperparameters
    best_params, best_acc, history = optimize_model(X, y, NUM_ITERATIONS, seq_len=5)

    print(f"\nFinal Best Accuracy: {best_acc:.2%}")
    print("Optimized Parameters:")
    for k, v in best_params.items():
        print(f"{k:>12}: {v}")

    # Load the best model saved during optimization
    best_model = tf.keras.models.load_model("best_model_final.h5")

    # Run trade simulation 100 times
    # (Note: The simulation function is MLP-style, so it won't strictly reflect LSTM logic,
    #  but we'll just demonstrate multiple runs.)
    sim_results = []
    for sim_run in range(100):
        win_rate, num_trades = simulate_trades(merged_df, best_model, threshold=0.7, lookback=10, risk_reward=2.0)
        sim_results.append(win_rate)
        print(f"Simulation {sim_run+1}: Win Rate = {win_rate*100:.2f}% with {num_trades} trades.")

    avg_win_rate = np.mean(sim_results)
    print(f"\nAverage Win Rate over 100 simulations: {avg_win_rate*100:.2f}%")

if __name__ == "__main__":
    main()