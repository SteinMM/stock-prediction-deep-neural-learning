import os
import secrets
from datetime import datetime
import pandas as pd

from stock_prediction_deep_learning import train_LSTM_network
from stock_prediction_class import StockPrediction

TICKERS = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]
START_DATE = pd.to_datetime("2010-01-01")
VALIDATION_DATE = pd.to_datetime("2020-01-01")

for ticker in TICKERS:
    today_run = datetime.today().strftime("%Y%m%d")
    token = f"{ticker}_{today_run}_{secrets.token_hex(8)}"
    project_folder = os.path.join(os.getcwd(), token)
    os.makedirs(project_folder, exist_ok=True)

    stock = StockPrediction(
        ticker,
        START_DATE,
        VALIDATION_DATE,
        project_folder,
        github_url=None,
        epochs=150,
        time_steps=60,
        token=token,
        batch_size=32,
        features=["Open", "High", "Low", "Close", "Volume"],
    )

    train_LSTM_network(stock)

