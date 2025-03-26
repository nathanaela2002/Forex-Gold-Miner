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

OANDA_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
SQL_HOST = os.getenv("host")
SQL_USER = os.getenv("user")
SQL_PASSWORD = os.getenv("password")
SQL_DATABASE = os.getenv("database")
NUM_ITERATIONS = 100  

# Fetch OANDA data: 6 months of 30-min candles.
def fetch_oanda_data():
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=180)  
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
        params = {"from": from_str, "to": to_str, "granularity": "M30", "price": "M"}  # 30-min candles using mid price.
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
            row = {"date": dt, "open": float(mid["o"]), "high": float(mid["h"]),
                   "low": float(mid["l"]), "close": float(mid["c"]), "volume": c["volume"],
                   "complete": c["complete"]}
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

# Compute RSI, MA, and Bollinger Bands
def compute_indicators(df, period_rsi=14, period_ma=20, period_boll=20, num_std=2):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period_rsi).mean()
    avg_loss = loss.rolling(window=period_rsi).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MA"] = df["close"].rolling(window=period_ma).mean()  # Simple moving average.
    df["BB_MA"] = df["close"].rolling(window=period_boll).mean()  # Bollinger midline.
    df["BB_STD"] = df["close"].rolling(window=period_boll).std()  # Bollinger standard deviation.
    df["BB_upper"] = df["BB_MA"] + num_std * df["BB_STD"]
    df["BB_lower"] = df["BB_MA"] - num_std * df["BB_STD"]
    return df

# Extra indicators: MACD, Stochastic Oscillator, and ATR.
def compute_extra_indicators(df, macd_fast=12, macd_slow=26, macd_signal=9,
                             stoch_window=14, atr_window=14):
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD_line"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD_line"].ewm(span=macd_signal, adjust=False).mean()
    rolling_low = df["low"].rolling(stoch_window).min()
    rolling_high = df["high"].rolling(stoch_window).max()
    df["stoch_%K"] = 100 * (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    df["stoch_%D"] = df["stoch_%K"].rolling(3).mean() 
    df["prev_close"] = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["prev_close"]).abs()
    tr3 = (df["low"] - df["prev_close"]).abs()
    df["TR"] = tr1.combine(tr2, max).combine(tr3, max)
    df["ATR"] = df["TR"].rolling(atr_window).mean()
    return df

# Detect bullish and bearish engulfing patterns.
def detect_candlestick_patterns(df):
    patterns = {"bullish_engulfing": [], "bearish_engulfing": []}
    for i in range(len(df)):
        if i == 0:
            patterns["bullish_engulfing"].append(0)
            patterns["bearish_engulfing"].append(0)
            continue
        prev_open = df["open"].iloc[i-1]
        prev_close = df["close"].iloc[i-1]
        curr_open = df["open"].iloc[i]
        curr_close = df["close"].iloc[i]
        if (prev_close < prev_open) and (curr_close > curr_open) and ((curr_close - curr_open) > (prev_open - prev_close)):
            patterns["bullish_engulfing"].append(1)
        else:
            patterns["bullish_engulfing"].append(0)
        if (prev_close > prev_open) and (curr_close < curr_open) and ((curr_open - curr_close) > (prev_close - prev_open)):
            patterns["bearish_engulfing"].append(1)
        else:
            patterns["bearish_engulfing"].append(0)
    for pattern_name, values in patterns.items():
        df[pattern_name] = values
    return df

# Load PMI data from MySQL
def load_pmi_data():
    conn = mysql.connector.connect(host=SQL_HOST, user=SQL_USER, password=SQL_PASSWORD, database=SQL_DATABASE)
    cursor = conn.cursor()
    query = """SELECT release_date, actual FROM pmi_history ORDER BY release_date"""
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["release_date", "pmi_actual"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    cursor.close()
    conn.close()
    return df

# Merge PMI with price data
def merge_pmi_with_prices(df_prices, df_pmi):
    df_prices = df_prices.copy()
    df_prices.rename(columns={"date": "timestamp"}, inplace=True)
    df_pmi = df_pmi.copy()
    df_pmi.rename(columns={"release_date": "pmi_date"}, inplace=True)
    df_pmi.sort_values("pmi_date", inplace=True)
    df_prices.sort_values("timestamp", inplace=True)
    merged = pd.merge_asof(df_prices, df_pmi, left_on="timestamp", right_on="pmi_date", direction="backward")
    merged["pmi_actual"].fillna(method="ffill", inplace=True)
    merged.rename(columns={"timestamp": "date"}, inplace=True)
    return merged

# Reshape features into 3D sequences for LSTM.
def create_timeseries_samples(X, y, seq_len=5):
    X_list = []
    y_list = []
    for i in range(len(X) - seq_len):
        X_list.append(X[i : i + seq_len])
        y_list.append(y[i + seq_len])
    return np.array(X_list), np.array(y_list)

# Build a model; choose LSTM or MLP based on hyperparams.
def create_model(input_dim, hyperparams, seq_len=5):
    units1 = int(hyperparams['units1'])
    units2 = int(hyperparams['units2'])
    units3 = int(hyperparams['units3'])
    if hyperparams['arch'] == 'LSTM':
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
    optimizer = optimizers.AdamW(learning_rate=hyperparams['lr'], weight_decay=hyperparams['wd'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate new hyperparameters, tweaking the current best if available.
def generate_hyperparams(current_best=None):
    arch_choices = ['MLP', 'LSTM']
    if current_best:
        new_params = {
            'lr': np.clip(current_best['lr'] * np.random.normal(1, 0.2), 1e-5, 1e-2),
            'wd': np.clip(current_best['wd'] * np.random.normal(1, 0.2), 1e-6, 1e-3),
            'units1': int(np.clip(current_best['units1'] * np.random.normal(1, 0.1), 16, 1024)),
            'units2': int(np.clip(current_best['units2'] * np.random.normal(1, 0.1), 16, 1024)),
            'units3': int(np.clip(current_best['units3'] * np.random.normal(1, 0.1), 16, 1024)),
            'batch_size': int(np.clip(current_best['batch_size'] + np.random.randint(-16, 17), 16, 256)),
            'arch': current_best['arch']
        }
        if np.random.rand() < 0.1:
            new_params['arch'] = np.random.choice(arch_choices)
        return new_params
    return {
        'lr': 10**np.random.uniform(-4, -2),
        'wd': 10**np.random.uniform(-6, -3),
        'units1': np.random.choice([64, 128, 256, 512]),
        'units2': np.random.choice([32, 64, 128, 256]),
        'units3': np.random.choice([16, 32, 64, 128]),
        'batch_size': np.random.choice([32, 64, 128]),
        'arch': np.random.choice(arch_choices)
    }

# Evolve and optimize model parameters
def optimize_model(X, y, num_iterations, seq_len=5):
    best_accuracy = 0.0
    best_params = None
    history = []
    for iteration in range(num_iterations):
        hyperparams = generate_hyperparams(best_params)
        if hyperparams['arch'] == 'LSTM':
            X_seq, y_seq = create_timeseries_samples(X, y, seq_len=seq_len)
            data_length = X_seq.shape[0]
            tscv = TimeSeriesSplit(n_splits=5)
            val_accuracies = []
            for train_idx, val_idx in tscv.split(range(data_length)):
                X_train = X_seq[train_idx]
                X_val = X_seq[val_idx]
                y_train = y_seq[train_idx]
                y_val = y_seq[val_idx]
                model = create_model(X_train.shape[2], hyperparams, seq_len=seq_len)
                early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=min(10, 5 + iteration//20), restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=hyperparams['batch_size'], callbacks=[early_stop], verbose=0)
                val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
                val_accuracies.append(val_acc)
            mean_acc = np.mean(val_accuracies)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            tscv = TimeSeriesSplit(n_splits=5)
            val_accuracies = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_val = X_scaled[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]
                model = create_model(X_train.shape[1], hyperparams, seq_len=seq_len)
                early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=min(10, 5 + iteration//20), restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=hyperparams['batch_size'], callbacks=[early_stop], verbose=0)
                val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
                val_accuracies.append(val_acc)
            mean_acc = np.mean(val_accuracies)
        history.append(mean_acc)
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_params = hyperparams
            print(f"Iteration {iteration+1}: New best accuracy {best_accuracy:.2%}")
            if best_params['arch'] == 'LSTM':
                X_seq_full, y_seq_full = create_timeseries_samples(X, y, seq_len=seq_len)
                model_final = create_model(X_seq_full.shape[2], best_params, seq_len=seq_len)
                model_final.fit(X_seq_full, y_seq_full, epochs=30, batch_size=best_params['batch_size'], verbose=0)
                model_final.save("best_model_final.h5")
            else:
                scaler = StandardScaler()
                X_scaled_full = scaler.fit_transform(X)
                model_final = create_model(X_scaled_full.shape[1], best_params, seq_len=seq_len)
                model_final.fit(X_scaled_full, y, epochs=30, batch_size=best_params['batch_size'], verbose=0)
                model_final.save("best_model_final.h5")
        if iteration > 20 and (max(history[-20:]) - min(history[-20:])) < 0.01:
            print("Convergence detected - refining parameter space")
            hyperparams = generate_hyperparams(best_params)
    return best_params, best_accuracy, history

# Simulate trades using model predictions and past candle data.
def simulate_trades(df, model, threshold=0.7, lookback=10, risk_reward=2.0):
    trades = 0
    wins = 0
    features = ["RSI", "MA", "BB_upper", "BB_lower", "pmi_actual",
                "bullish_engulfing", "bearish_engulfing",
                "MACD_line", "MACD_signal", "stoch_%K", "stoch_%D", "ATR"]
    df_clean = df.dropna(subset=features)
    if df_clean.empty:
        return 0, 0
    X_all = df_clean[features].values
    sc = StandardScaler()
    X_all_scaled = sc.fit_transform(X_all)
    df_clean.loc[:, 'scaled_features'] = list(X_all_scaled)
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
        idx = df_clean.index[i]
        recent_idx_start = i - lookback
        if recent_idx_start < 0:
            continue
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

# Main function: get data, train model, and simulate trades.
def main():
    price_df = fetch_oanda_data()  
    if price_df.empty:
        print("No data fetched from OANDA.")
        return
    price_df = compute_indicators(price_df)  # Add basic indicators.
    price_df = compute_extra_indicators(price_df)  # Add extra indicators.
    price_df = detect_candlestick_patterns(price_df)  # Find candle patterns.
    pmi_df = load_pmi_data()  # Load PMI info.
    merged_df = merge_pmi_with_prices(price_df, pmi_df)
    merged_df["future_close"] = merged_df["close"].shift(-1)
    merged_df.dropna(subset=["future_close"], inplace=True)
    merged_df["label"] = ((merged_df["future_close"] - merged_df["close"]) / merged_df["close"] > 0.001).astype(int)
    all_features = ["RSI", "MA", "BB_upper", "BB_lower", "pmi_actual",
                    "bullish_engulfing", "bearish_engulfing",
                    "MACD_line", "MACD_signal", "stoch_%K", "stoch_%D", "ATR"]
    merged_df.dropna(subset=all_features, inplace=True)
    X = merged_df[all_features].values
    y = merged_df["label"].values
    best_params, best_acc, history = optimize_model(X, y, NUM_ITERATIONS, seq_len=5)
    print(f"\nFinal Best Accuracy: {best_acc:.2%}")
    print("Optimized Parameters:")
    for k, v in best_params.items():
        print(f"{k:>12}: {v}")
    best_model = tf.keras.models.load_model("best_model_final.h5")
    sim_results = []
    for sim_run in range(100):
        win_rate, num_trades = simulate_trades(merged_df, best_model, threshold=0.7, lookback=10, risk_reward=2.0)
        sim_results.append(win_rate)
        print(f"Simulation {sim_run+1}: Win Rate = {win_rate*100:.2f}% with {num_trades} trades.")
    avg_win_rate = np.mean(sim_results)
    print(f"\nAverage Win Rate over 100 simulations: {avg_win_rate*100:.2f}%")

if __name__ == "__main__":
    main()
