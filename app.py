from flask import Flask, render_template, jsonify
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

dataset_path = "dataset/Sony-SAP-Invent_weekly_forecast_2025WK04_2027WK15.csv"
sony_df = pd.read_csv(dataset_path)

@app.route('/dashboard/<product_id>')
def home(product_id):
    # Filter data for "Product 1"
    # add product string from product_id. like "product " + product_id
    product_name = "Product " + product_id
    
    product_df = sony_df[sony_df["Model"] == product_name]
    # Aggregate Sellout by Dealer Group
    dealer_sales = product_df.groupby("Dealer Group")["Sellout"].sum().reset_index()

    # Convert to dictionary format
    dealer_data = {
        "labels": dealer_sales["Dealer Group"].tolist(),
        "values": dealer_sales["Sellout"].tolist()
    }
    # Group data by Year and Week, summing up Sellout
    product_df_total = product_df.groupby(["Year", "Week"], as_index=False).agg({"Sellout": "sum"})
    # Correctly generate date_stamp based on ISO calendar week
    product_df_total["date_stamp"] = pd.to_datetime(product_df_total["Year"].astype(str) + product_df_total["Week"].astype(str) + "-1", format="%Y%W-%w")    #remove dates after 2024
    
    product_df_total = product_df_total[product_df_total["date_stamp"] < "2025-01-01"]
    pf_df = product_df_total[["date_stamp", "Sellout"]].rename(columns={"date_stamp": "ds", "Sellout": "y"})
    # print(pf_df)

    # Train Prophet Model
    pf_df["cap"] = pf_df["y"].max() * 2  # Set a reasonable upper bound (e.g., 20% above max sales)
    print("Max: ", pf_df["cap"].max())
    pf_df["floor"] = 0  # Set minimum sales to zero
    model = Prophet(growth="logistic")
    model.fit(pf_df)

    print(pf_df.describe())  # Check min/max values
    print(pf_df.dtypes)
    # Create Future DataFrame for 2025 (weekly forecasts)
    future = model.make_future_dataframe(periods=52, freq='W-SUN')  # Predict until end of 2025
    future["cap"] = pf_df["cap"].max()  # Use the same cap as historical data
    future["floor"] = 0  # Keep floor as zero
    forecast = model.predict(future)

    # Keep only 2025 data
    forecast_2025 = forecast[(forecast["ds"] >= "2025-01-01") & (forecast["ds"] <= "2025-12-31")]
    print(forecast_2025)

   # Aggregate forecast by month
    # forecast_2025["month"] = pf_df["ds"].dt.strftime("%Y-%m") + forecast_2025["ds"].dt.strftime("%Y-%m")  # Extract month
    forecast_2025["month"] = forecast_2025["ds"].dt.strftime("%Y-%m")
    pf_df["month"] = pf_df["ds"].dt.strftime("%Y-%m")  # Extract month
    monthly_forecast = forecast_2025.groupby("month")["yhat"].sum().reset_index()
    historical_data = pf_df.groupby("month")["y"].sum().reset_index()
    print(len(historical_data))
    # monthly_forecast = [None] * len(historical_data) +  forecast_2025.groupby("month")["yhat"].sum().reset_index()
    print(len(monthly_forecast))
    # Convert into required dictionary format
    forecast_data = {
        "labels": historical_data["month"].tolist() + monthly_forecast["month"].tolist(),  # Monthly timestamps
        "historical_values": historical_data["y"].astype(int).tolist(),  # Actual sales
        "forecast_values": [None] * len(historical_data) + monthly_forecast["yhat"].astype(int).tolist()  # Predicted sales per month
    }

    # # Sample forecast data
    # forecast_data = {
    #     "labels": ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
    #     "values": [100, 150, 130, 180, 200, 230, 250, 270, 300, 320]
    # }

    # forecast_data = {
    #     "labels": pf_df["ds"].dt.strftime("%Y-%m").tolist(),
    #     "values": pf_df["y"].tolist()
    # }

    # df["date_stamp"] = 

    return render_template("dashboard.html", dealer_data=dealer_data, forecast_data=forecast_data)

    # return render_template("dashboard.html", dealer_data=dealer_data)

@app.route('/data')
def get_data():
    dataset_path = "Sony-SAP-Invent_weekly_forecast_2025WK04_2027WK15.csv"
    sony_df = pd.read_csv(dataset_path)
    return sony_df.to_json()

if __name__ == '__main__':
    app.run(debug=True)