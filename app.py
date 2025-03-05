from flask import Flask, jsonify
import pandas as pd
import requests
from io import StringIO
import statsmodels.api as sm

app = Flask(__name__)

# ✅ Google Sheets public CSV link
sheet_url = "https://docs.google.com/spreadsheets/d/1MCrucUxUJXQQwpQ-thS2yVbVe8vn2xIVM8s7WatKBXE/gviz/tq?tqx=out:csv&gid=1556326888"

# ✅ Load and preprocess data
def load_data(sheet_url):
    response = requests.get(sheet_url)
    response.raise_for_status()
    
    df = pd.read_csv(StringIO(response.text))
    df.columns = ["Date", "Total"]
    
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values("Date").set_index("Date")
    
    # ✅ Set frequency to 'MS' (monthly start), filling missing months with interpolation
    df = df.asfreq("MS").interpolate()
    
    return df

# ✅ SARIMA Forecasting
def forecast_sarima(df, steps=6):
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)

    # ✅ SARIMA model
    sarima_model = sm.tsa.statespace.SARIMAX(df["Total"], 
                                             order=(1, 1, 1),  
                                             seasonal_order=(0, 1, 1, 3),  
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
    
    results = sarima_model.fit()

    # ✅ Predict next `steps` months
    forecast = results.get_forecast(steps=steps)
    predicted_future = forecast.predicted_mean.round(2)

    future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, steps + 1)]
    
    return [{"date": str(date.date()), "forecast": forecast} for date, forecast in zip(future_dates, predicted_future)]

@app.route("/forecast", methods=["GET"])
def get_forecast():
    try:
        df = load_data(sheet_url)
        forecast_data = forecast_sarima(df, steps=6)
        return jsonify({"status": "success", "forecast": forecast_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
