from flask import Flask, jsonify, request
import pandas as pd
import requests
from io import StringIO
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import numpy as np

app = Flask(__name__)

BASE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1MCrucUxUJXQQwpQ-thS2yVbVe8vn2xIVM8s7WatKBXE/gviz/tq?tqx=out:csv&gid="

def load_data(gid):
    try:
        sheet_url = BASE_SHEET_URL + str(gid)
        response = requests.get(sheet_url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        df.columns = ["Date", "Total"]

        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
        df = df.sort_values("Date").set_index("Date")

        # ✅ Fill missing values using rolling mean
        df["Total"] = df["Total"].rolling(3, min_periods=1).mean()

        return df
    except Exception as e:
        print(f"❌ Error loading data for GID {gid}: {e}")
        return pd.DataFrame()

def check_stationarity(df):
    result = adfuller(df["Total"])
    return result[1]  # p-value (< 0.05 means data is stationary)

def forecast_sarima(df, steps=12):
    try:
        df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)

        # ✅ Apply log transformation if needed
        if df["Total"].min() > 0:
            df["Total"] = np.log1p(df["Total"])

        # ✅ Train/Test Split (90% train, 10% test)
        train_size = int(len(df) * 0.9)
        train, test = df.iloc[:train_size], df.iloc[train_size:]

        # ✅ Apply Differencing If Needed
        p_value = check_stationarity(train)
        if p_value > 0.05:
            print("🔄 Data is non-stationary, applying differencing")
            train["Total"] = train["Total"].diff().dropna()

        # ✅ Optimized SARIMA Model (Simpler)
        sarima_model = sm.tsa.statespace.SARIMAX(train["Total"], 
                                                 order=(1,1,1),  
                                                 seasonal_order=(0,1,1,12),  
                                                 enforce_stationarity=False,
                                                 enforce_invertibility=False)

        results = sarima_model.fit(disp=False)

        # ✅ Validate Model Performance
        predictions = results.get_prediction(start=len(train), end=len(df)-1)
        predicted_values = predictions.predicted_mean
        actual_values = test["Total"]

        # ✅ SMAPE (Better than MAPE)
        smape = 100 * np.mean(2 * np.abs(actual_values - predicted_values) / (np.abs(actual_values) + np.abs(predicted_values)))
        r2 = r2_score(actual_values, predicted_values)
        accuracy = max(0, 100 - smape)  # Avoid negative accuracy

        # ✅ Forecast next `steps` months
        forecast = results.get_forecast(steps=steps)
        predicted_future = np.expm1(forecast.predicted_mean).round(2)  # Reverse log transformation
        future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, steps + 1)]

        forecast_data = [{"date": str(date.date()), "forecast": float(forecast)} for date, forecast in zip(future_dates, predicted_future)]

        return {
            "forecast": forecast_data,
            "evaluation": {
                "SMAPE": round(smape, 2),
                "R2 Score": round(r2, 2),
                "Accuracy": round(accuracy, 2),
            }
        }
    except Exception as e:
        print(f"❌ Error in SARIMA model: {e}")
        return {"forecast": [], "error": str(e)}

@app.route("/forecast", methods=["GET"])
def get_forecast():
    try:
        gid = request.args.get("gid")

        if not gid:
            return jsonify({"status": "error", "message": "Missing GID (sub-sheet ID)."})

        print(f"🔍 Fetching data for GID: {gid}")

        df = load_data(gid)

        if df.empty:
            return jsonify({"status": "error", "message": f"No data available for GID '{gid}'."})

        result = forecast_sarima(df, steps=12)
        return jsonify({"status": "success", "gid": gid, "data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
