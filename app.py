from flask import Flask, jsonify, request
import pandas as pd
import requests
from io import StringIO
import statsmodels.api as sm
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os  # Required for Render deployment

app = Flask(__name__)

# ✅ Google Sheets Public URLs
MASTER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1MCrucUxUJXQQwpQ-thS2yVbVe8vn2xIVM8s7WatKBXE/gviz/tq?tqx=out:csv&gid=0"
BASE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1MCrucUxUJXQQwpQ-thS2yVbVe8vn2xIVM8s7WatKBXE/gviz/tq?tqx=out:csv&gid="

# ✅ Fetch product-to-gid mapping from the master sheet
def get_product_gid():
    response = requests.get(MASTER_SHEET_URL)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = ["Product", "GID"]  # Ensure master sheet has "Product" and "GID" columns
    df["Product"] = df["Product"].str.strip().str.lower()  # Normalize product names

    return dict(zip(df["Product"], df["GID"]))

# ✅ Load product data from the corresponding sub-sheet
def load_data(gid):
    sheet_url = BASE_SHEET_URL + str(gid)
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

    # ✅ Train/Test Split
    train_size = int(len(df) * 0.8)  # 80% train, 20% test
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # ✅ SARIMA model
    sarima_model = sm.tsa.statespace.SARIMAX(train["Total"], 
                                             order=(1, 1, 1),  
                                             seasonal_order=(1, 1, 1, 12),  
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
    
    results = sarima_model.fit(disp=False)

    # ✅ Validate Model Performance
    predictions = results.get_prediction(start=len(train), end=len(df)-1)
    predicted_values = predictions.predicted_mean
    actual_values = test["Total"]

    mse = mean_squared_error(actual_values, predicted_values)
    rmse = mse ** 0.5
    mae = mean_absolute_error(actual_values, predicted_values)
    mape = (abs((actual_values - predicted_values) / actual_values).mean()) * 100
    r2 = r2_score(actual_values, predicted_values)
    accuracy = 100 - mape

    # ✅ Predict next `steps` months
    forecast = results.get_forecast(steps=steps)
    predicted_future = forecast.predicted_mean.round(2)
    future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, steps + 1)]
    
    forecast_data = [{"date": str(date.date()), "forecast": float(forecast)} for date, forecast in zip(future_dates, predicted_future)]
    
    return {
        "forecast": forecast_data,
        "evaluation": {
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
            "R2 Score": round(r2, 2),
            "Accuracy": round(accuracy, 2),
        }
    }

@app.route("/forecast", methods=["GET"])
def get_forecast():
    try:
        # ✅ Get product name from query params (e.g., /forecast?product=Milk)
        product = request.args.get("product")
        
        if not product:
            return jsonify({"status": "error", "message": "Missing product name."})

        product = product.strip().lower()  # Normalize product name
        product_mapping = get_product_gid()  # Fetch latest product mapping

        if product not in product_mapping:
            return jsonify({"status": "error", "message": f"Product '{product}' not found."})
        
        # ✅ Get the `gid` (sub-sheet ID) for the requested product
        gid = product_mapping[product]
        df = load_data(gid)
        
        result = forecast_sarima(df, steps=6)
        return jsonify({"status": "success", "product": product, "data": result})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Required for Render deployment
    app.run(host="0.0.0.0", port=port)
