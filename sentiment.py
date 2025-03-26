import mysql.connector
import pandas as pd
import numpy as np
import requests
import datetime as dt
import re
import time

import matplotlib.pyplot as plt
import mplfinance as mpf

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# For deep learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles

import os
from dotenv import load_dotenv
load_dotenv()

# Make sure NLTK data is downloaded (stopwords, punkt, wordnet)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

OANDA_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

##############################################
#               MYSQL ARTICLES
##############################################

def get_mysql_articles():
    """
    Connect to MySQL and fetch articles with their published date.
    Assumes a table 'Articles' with columns:
      - content (TEXT)
      - published_date (VARCHAR) in the format 'Published 03/20/2025, 01:05 PM'
    """
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="126125Al1245!",
        database="scraped_data"
    )
    cursor = conn.cursor()

    query = """SELECT content, published_date FROM Articles"""
    cursor.execute(query)
    results = cursor.fetchall()

    df = pd.DataFrame(results, columns=["content", "published_date"])

    cursor.close()
    conn.close()

    return df

def parse_published_date(date_str):
    """
    Given a string in the format 'Published 03/20/2025, 01:05 PM',
    parse and return a datetime object (naive) for consistency with OANDA UTC times.
    """
    pattern = r"Published\s+(.*)"
    match = re.match(pattern, date_str)
    if not match:
        return None

    date_part = match.group(1)  # e.g. "03/20/2025, 01:05 PM"
    pub_dt = dt.datetime.strptime(date_part, "%m/%d/%Y, %I:%M %p")
    return pub_dt

##############################################
#       FETCH GOLD DATA FROM OANDA
##############################################

def fetch_gold_candles():
    """
    Fetch XAU/USD candlestick data from OANDA for the past 3 months (1-hour granularity),
    using a chunked approach with oandapyV20.
    Returns a DataFrame with columns: [Date, Open, High, Low, Close, Volume], indexed by Date.
    """
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(days=90)  # 3 months
    chunk_delta = dt.timedelta(days=30)
    data_frames = []
    current_start = start_time

    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment="practice")

    while current_start < end_time:
        current_end = current_start + chunk_delta
        if current_end > end_time:
            current_end = end_time

        from_str = current_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = current_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "from": from_str,
            "to": to_str,
            "granularity": "H1",  # 1-hour candles
            "price": "M"
        }
        request = InstrumentsCandles(instrument="XAU_USD", params=params)
        try:
            response = client.request(request)
        except Exception as e:
            print(f"Error fetching candles from {from_str} to {to_str}: {e}")
            current_start = current_end
            continue

        candles = response.get("candles", [])
        rows = []
        for c in candles:
            # e.g. time: "2025-03-20T13:00:00.000000000Z"
            time_str = c["time"].rstrip("Z")[:26]  # trim extra decimals
            dt_obj = pd.to_datetime(time_str)
            mid = c["mid"]
            row = {
                "Date": dt_obj,
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
                "Volume": c["volume"]
            }
            rows.append(row)

        if rows:
            df_chunk = pd.DataFrame(rows)
            data_frames.append(df_chunk)

        current_start = current_end
        time.sleep(1)  # avoid rate-limit

    if data_frames:
        df = pd.concat(data_frames, ignore_index=True)
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
        return df
    else:
        return pd.DataFrame()

##############################################
#      MARKET SENTIMENT BASED ON PRICE
##############################################

def compute_market_sentiment(gold_df, publish_dt, hours=5):
    """
    Determine how gold price moved in the 'hours' after 'publish_dt'.
    Return a dict with price_before, price_after, price_change, and final sentiment classification.
    
    Steps:
      1. Find the gold price candle on/after publish_dt (price_before).
      2. Find the gold price candle on/after publish_dt + hours.
      3. Compare to see if price rose (Positive), fell (Negative), or stayed about the same (Neutral).
    """
    if gold_df.empty:
        return {
            "price_before": None,
            "price_after_5h": None,
            "price_change": None,
            "market_sentiment": "Neutral"
        }
    
    # 1. Find the index of the candle on or immediately after publish_dt
    idx_before = gold_df.index.searchsorted(publish_dt, side="left")
    if idx_before >= len(gold_df):
        # No candle found after this time => can't compute
        return {
            "price_before": None,
            "price_after_5h": None,
            "price_change": None,
            "market_sentiment": "Neutral"
        }
    
    # Price at the "before" candle
    dt_before = gold_df.index[idx_before]
    price_before = gold_df.loc[dt_before, "Close"]

    # 2. Find the candle on or immediately after publish_dt + hours
    future_time = publish_dt + dt.timedelta(hours=hours)
    idx_after = gold_df.index.searchsorted(future_time, side="left")
    if idx_after >= len(gold_df):
        # No candle found after future_time => can't compute
        return {
            "price_before": float(price_before),
            "price_after_5h": None,
            "price_change": None,
            "market_sentiment": "Neutral"
        }

    dt_after = gold_df.index[idx_after]
    price_after = gold_df.loc[dt_after, "Close"]

    price_change = price_after - price_before
    
    # Classify
    if price_change > 0:
        market_sentiment = "Positive"
    elif price_change < 0:
        market_sentiment = "Negative"
    else:
        market_sentiment = "Neutral"

    return {
        "price_before": float(price_before),
        "price_after_5h": float(price_after),
        "price_change": float(price_change),
        "market_sentiment": market_sentiment
    }

##############################################
#       STORE MARKET SENTIMENT IN SQL
##############################################

def store_market_sentiment_in_sql(sentiment_data):
    """
    Create/Update a table 'sentiment' with columns:
      - article_index (PRIMARY KEY)
      - published_date (DATETIME)
      - price_before (FLOAT)
      - price_after_5h (FLOAT)
      - price_change (FLOAT)
      - market_sentiment (VARCHAR(50))
      - ml_sentiment (VARCHAR(50)) # We'll add an ML-based sentiment column
    """
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="126125Al1245!",
        database="scraped_data"
    )
    cursor = conn.cursor()

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS sentiment (
      article_index INT PRIMARY KEY,
      published_date DATETIME,
      price_before FLOAT,
      price_after_5h FLOAT,
      price_change FLOAT,
      market_sentiment VARCHAR(50),
      ml_sentiment VARCHAR(50)
    )
    """
    cursor.execute(create_table_sql)

    insert_sql = """
    INSERT INTO sentiment (
      article_index,
      published_date,
      price_before,
      price_after_5h,
      price_change,
      market_sentiment,
      ml_sentiment
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      published_date = VALUES(published_date),
      price_before = VALUES(price_before),
      price_after_5h = VALUES(price_after_5h),
      price_change = VALUES(price_change),
      market_sentiment = VALUES(market_sentiment),
      ml_sentiment = VALUES(ml_sentiment)
    """

    for row in sentiment_data:
        data_tuple = (
            row["article_index"],
            row["published_date"],
            row["price_before"],
            row["price_after_5h"],
            row["price_change"],
            row["market_sentiment"],
            row["ml_sentiment"]
        )
        cursor.execute(insert_sql, data_tuple)

    conn.commit()
    cursor.close()
    conn.close()

##############################################
#         NLP & DEEP LEARNING STEPS
##############################################

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def pre_process(text):
    """
    Clean and standardize text, similar to the approach used in the AAPL project:
      1. Lowercase
      2. Remove special chars, HTML, digits
      3. Remove stopwords
      4. Lemmatize
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove non-alpha
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def train_lstm_model(df_articles):
    """
    Train an LSTM sentiment model using (article_text -> market_sentiment) as labels.
    Returns the trained model, tokenizer, label_encoder.
    """
    # 1. Clean text
    df_articles["clean_text"] = df_articles["content"].apply(pre_process)

    # 2. Market sentiment from price movement is used as label
    #    Convert "Positive", "Negative", "Neutral" to numeric
    label_encoder = LabelEncoder()
    df_articles["label_id"] = label_encoder.fit_transform(df_articles["market_sentiment"])

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_articles["clean_text"],
        df_articles["label_id"],
        test_size=0.3,
        random_state=42,
        stratify=df_articles["label_id"]
    )

    # 4. Tokenizer + Sequencing
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # 5. Pad sequences
    max_len = 100
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    # 6. Build LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # 7. Train
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_test_pad, y_test),
        epochs=5,  # can adjust
        batch_size=64,
        verbose=1
    )

    # 8. Evaluate
    print("\n=== ML Model Evaluation ===")
    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # Detailed metrics
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, tokenizer, label_encoder

def ml_sentiment_prediction(model, tokenizer, label_encoder, text):
    """
    Given a trained LSTM model and a new article text, return ML-based sentiment: "Positive", etc.
    """
    cleaned = pre_process(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad_seq = pad_sequences(seq, maxlen=100)
    pred = model.predict(pad_seq)
    label_id = np.argmax(pred, axis=1)[0]
    return label_encoder.classes_[label_id]

##############################################
#               MAIN LOGIC
##############################################

def main():
    # 1. Fetch articles
    articles_df = get_mysql_articles()

    # 2. Fetch gold data (3 months, 1 hour)
    gold_df = fetch_gold_candles()
    if gold_df.empty:
        print("No gold data fetched from OANDA.")
        return

    # 3. For each article, compute market sentiment based on 5-hour price movement
    results = []
    for idx, row in articles_df.iterrows():
        pub_date_str = row["published_date"]
        pub_date = parse_published_date(pub_date_str)
        
        if not pub_date:
            # skip if we can't parse date
            # still store a "Neutral" market_sentiment to avoid missing data
            results.append({
                "article_index": idx,
                "published_date": None,
                "price_before": None,
                "price_after_5h": None,
                "price_change": None,
                "market_sentiment": "Neutral",
                "content": row["content"]  # keep original text for ML
            })
            continue

        analysis = compute_market_sentiment(gold_df, pub_date, hours=5)
        results.append({
            "article_index": idx,
            "published_date": pub_date,
            "price_before": analysis["price_before"],
            "price_after_5h": analysis["price_after_5h"],
            "price_change": analysis["price_change"],
            "market_sentiment": analysis["market_sentiment"],
            "content": row["content"]
        })

    # Convert to DataFrame for ML training
    sentiment_df = pd.DataFrame(results)
    print("\n==== MARKET SENTIMENT BASED ON GOLD PRICE ====\n")
    print(sentiment_df.head(10))  # preview only

    # 4. Train LSTM model on existing data (article text -> price-based sentiment)
    #    Filter out rows without a valid content or market_sentiment
    valid_data = sentiment_df.dropna(subset=["content", "market_sentiment"])
    if len(valid_data["market_sentiment"].unique()) < 2:
        print("Not enough variety in market sentiment labels for ML training. Exiting ML portion.")
        # Even if we can’t train, we’ll just store the table without ML columns
        # Store data in SQL
        sentiment_df["ml_sentiment"] = "N/A"
        store_market_sentiment_in_sql(sentiment_df.to_dict(orient="records"))
        # Plot candlestick
        print("\nPlotting the last 3 months of XAU/USD candlesticks...")
        mpf.plot(
            gold_df,
            type='candle',
            volume=True,
            style='classic',
            title="XAU/USD - Past 3 Months (H1)",
            figsize=(12, 6)
        )
        return

    print("\nTraining LSTM Model on articles (text -> market_sentiment)...")
    model, tokenizer, label_encoder = train_lstm_model(valid_data)

    # 5. Use trained model to predict ML-based sentiment for each article
    ml_sentiments = []
    for i, row in sentiment_df.iterrows():
        predicted_label = ml_sentiment_prediction(
            model, tokenizer, label_encoder, row["content"]
        )
        ml_sentiments.append(predicted_label)

    sentiment_df["ml_sentiment"] = ml_sentiments

    # 6. Store the combined data into SQL table 'sentiment'
    store_market_sentiment_in_sql(sentiment_df.to_dict(orient="records"))

    # 7. Plot candlesticks (optional)
    print("\nPlotting the last 3 months of XAU/USD candlesticks...")
    mpf.plot(
        gold_df,
        type='candle',
        volume=True,
        style='classic',
        title="XAU/USD - Past 3 Months (H1)",
        figsize=(12, 6)
    )

if __name__ == "__main__":
    main()
