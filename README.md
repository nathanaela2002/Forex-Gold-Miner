# FOREX-GOLD-MINING

Welcome to the **FOREX-GOLD-MINING** repository. This project combines **web scraping**, **natural language processing (NLP)** sentiment analysis, and a **quantitative trading algorithm** (featuring **RNN/LSTM** models) to trade Gold (XAU/USD) using the **OANDA** API and **MySQL** for data storage.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technical Stack](#technical-stack)
4. [Data Flow Summary](#data-flow-summary)
5. [Selenium Web Scraper for News Articles](#selenium-web-scraper-for-news-articles)
6. [NLP Market Sentiment Modeling](#nlp-market-sentiment-modeling)
7. [Quantitative Trading on OANDA](#quantitative-trading-on-oanda)
8. [RNN/LSTM Model for Price Prediction](#rnnlstm-model-for-price-prediction)
9. [Files & Directory Structure](#files--directory-structure)
10. [Setting Up & Running](#setting-up--running)
11. [License](#license)

---

## Project Overview
This system is designed to:
1. **Scrape gold-related news articles** (from Investing.com) via **undetected Selenium Chrome**.
2. **Store and process** the articles in a MySQL database.
3. **Compute textual sentiment** (using NLP techniques) from the scraped articles.
4. **Combine** fundamental data (like PMI values from MySQL), **technical indicators** (RSI, Bollinger Bands, MACD, etc.), and **price-based sentiment** to create a **training dataset**.
5. **Train an RNN/LSTM model** (or optionally an MLP) to predict gold’s price direction over the next time frame (using 30-minute data from the **OANDA API**).
6. **Backtest** the model’s performance over the **last 6 months** and **simulate trades** based on thresholds (risk/reward constraints, bullish/bearish signals).
7. **Deploy** a real-time trading loop that automatically places trades via **OANDA** when high-probability opportunities appear.

---

## Key Features
- **Web Scraping** with [`undetected_chromedriver`](https://pypi.org/project/undetected-chromedriver/) to bypass common bot-detections.
- **MySQL Database** integration to store articles and sentiment.
- **NLP Pipeline** (tokenization, stopwords removal, lemmatization) for raw text analysis.
- **RNN/LSTM and MLP** model frameworks with **early stopping** and **walk-forward / time-series splits**.
- **Market Sentiment** labeling based on the actual gold price movement around publication times.
- **Indicators** used: RSI, Moving Averages, MACD, Bollinger Bands, Stochastic, ATR, Candlestick pattern detection (Bullish/Bearish Engulfing).
- **OANDA** integration for historical & real-time 30-minute candle data and for **live order placement**.

---

## Technical Stack
1. **Python** 
2. **Selenium** with [`undetected_chromedriver`](https://github.com/ultrafunkamsterdam/undetected-chromedriver)
3. **MySQL** for data storage
4. [**OandaPyV20**](https://github.com/hootnot/oanda-api-v20) for OANDA RESTful API calls
5. **TensorFlow/Keras** for deep learning (LSTM/MLP)
6. **Pandas**, **NumPy**, **Scikit-learn** for data manipulation and machine learning utilities
7. **Matplotlib / mplfinance** for charting and candlestick plots

---

## Data Flow Summary
1. **Article Scraper** logs into Investing.com, scrapes gold news headlines & content, stores in **MySQL**.
2. **Market Data Fetcher** pulls 30-minute candle data for XAU/USD from the **OANDA** API.
3. **Indicator Computation** adds RSI, MA, Bollinger Bands, MACD, Stochastics, ATR, and candlestick patterns to each row.
4. **PMI** or other fundamental data is merged on a time basis from a MySQL table.
5. **Market Sentiment** for each article is determined by how gold price moved ~5 hours after publication.
6. The **NLP model** is trained, labeling each article as “Positive”, “Negative”, or “Neutral” based on price action. Text is tokenized & vectorized for an LSTM model.
7. The **Price Prediction** model uses 6 months of 30-minute intervals. An LSTM or MLP is trained to predict bullish/bearish probability for the next candle.
8. A **Trading Loop** uses these probabilities + thresholds + risk/reward criteria to place trades on OANDA in real time.

---

## Selenium Web Scraper for News Articles
- **File**: `login_and_scrape_investing.py` (the snippet shown at the beginning).
- Uses **undetected_chromedriver** to bypass typical bot detection.
- Steps:
  1. **Load** environment variables (like Investing.com credentials).
  2. **Login** to Investing.com using Selenium & handle pop-ups.
  3. **Scrape** gold-related news article titles, timestamps, and content.
  4. **Insert** the scraped articles into a `scraped_data.articles` table in MySQL.

This flow ensures an **automated pipeline** that updates regularly, providing fresh textual data.

---

## NLP Market Sentiment Modeling
- **File**: `nlp_sentiment_market.py` (the large script that merges with gold data).
- The main steps:
  1. **Fetch** the scraped articles from MySQL.
  2. **Pull** corresponding gold price data (1-hour candles) from OANDA.
  3. **Clean & Pre-Process** text (lowercase, remove stopwords, lemmatize, etc.).
  4. **Determine** a market sentiment label for each article by looking at gold price changes (e.g., 5 hours post-publication).
  5. **Train** an LSTM sentiment classification model:
     - `Tokenizer` + `pad_sequences` for the textual data.
     - LSTM layers in Keras to learn classifying article text -> final sentiment label.
  6. **Store** final sentiments (both market-based and ML-predicted) back into MySQL.

This creates a **text-based sentiment classification** system that can either stand alone or be merged with price-based data.

---

## Quantitative Trading on OANDA
- **File**: `real_time_trading_loop.py` (or similarly named).
- Main flow:
  1. Periodically **fetch** the latest 30-minute candles.
  2. **Compute** indicators on these candles (RSI, Bollinger Bands, MACD, etc.).
  3. **Load** your best-trained model (e.g., `best_model_final.h5`).
  4. **Predict** probability that the next candle is bullish.
  5. If probability > threshold (or < some lower threshold for bearish):
     - Check **support/resistance** in the last N candles.
     - Verify a **risk/reward** ratio.
     - **Place** an OANDA market order with **Stop Loss** & **Take Profit**.
  6. **Sleep** for a configured interval and repeat.

This script can be run on a server or local machine for real-time, **automated** trading once your pipeline is stable.

---

## RNN/LSTM Model for Price Prediction
- **File**: `model_training_evolution.py` (an example name reflecting the evolutionary training logic shown).
- Steps:
  1. **Fetch 6 months** of 30-minute candle data from OANDA.
  2. **Compute** advanced indicators (RSI, Bollinger, MACD, Stochastics, ATR, candlestick patterns).
  3. **Optional**: Merge fundamental data like PMI from MySQL.
  4. **Label** each row: `label=1` if next candle’s close is above the current by some threshold (or simply higher).
  5. **Preprocess** (scale or standardize data).
  6. Use a **time-series split** (walk-forward) strategy with `TimeSeriesSplit` from scikit-learn.
  7. **Train** an RNN/LSTM (or MLP) in an **evolutionary** manner by iterating over hyperparameters:
     - learning rate, weight decay, hidden units, architecture type, etc.
  8. Save the **best model** to `best_model_final.h5`.
  9. Includes a basic **simulation** function to test risk/reward-based trading over historical data.
