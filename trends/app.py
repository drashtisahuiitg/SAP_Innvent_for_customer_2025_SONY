import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from prophet import Prophet
from flask import Flask, render_template, request, jsonify
import io
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en', tz=330, geo='IN')
import base64
from datetime import datetime
import pandas as pd
import time
 
 
app = Flask(__name__)
 
# Load sales data
#df = pd.read_csv("Sony-SAP-Invent_weekly_forecast_2025WK04_2027WK15.csv")
 
# Convert Year, Month, and Week to a proper Date format
#df['Date'] = df.apply(lambda row: datetime.strptime(f"{row['Year']} {row['Month']} {row['Week']}", "%Y %m %W"), axis=1)

trends_data_dict = []
def get_trends_data(pytrends, kw_list, timeframe):
    product=f"{kw_list[0]}"
    productPlusTime=f"{product}_{timeframe}"
    filename=f"{productPlusTime}.csv"
    filepath=f"dataset/{filename}"
    
    datapath="dataset"
    saved_files = os.listdir(datapath)
    print(f"looking for {productPlusTime} in saved data files: {saved_files}")
    if filename in saved_files:
        print(f"Serving local data from {filepath}")
        data = pd.read_csv(filepath)
        print(data)
        return data

    while True:
        print("[!WARNING]remote data fetch..May take time")
        try:
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='IN', gprop='')
            data = pytrends.interest_by_region(resolution='REGION')
            trends_data_dict.append(productPlusTime)
            data.to_csv(filepath)
            print(type(data))
            print(data)
            return data
        except Exception as e:
            print("Error occurred: ", e)
            print("Sleeping for a minute before retrying...")
            time.sleep(10)  # Sleep for 60 seconds before retrying
 
# Function to predict sales for a given product within a specific time window
def predict_sales_for_product(product_name):
 
    kw_list = [product_name]
    timeframe = '2023-01-01 2023-06-30'
    data = get_trends_data(pytrends, kw_list, timeframe)
    print(data)
    df=data
    #df = df.reset_index()
    print("####################")
    print(df)
    plt.figure(figsize=(8, 5))
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', linestyle='-', color='b', label='Trend Data')
    print("BP1")
    # Formatting
    plt.xlabel('X-axis (Index 0)')
    plt.ylabel('Y-axis (Index 1)')
    plt.title('Trend Data Visualization')
    plt.legend()
    plt.xticks(rotation=90)
    plt.grid(True)
 
    # Show the plot
    plt.show()
 
 
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
 
    save_path = os.path.join(save_dir, "trend_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
 
    # product_data = df[df['Model'] == product_name].copy()  # Filter by product
    # if product_data.empty:
    #     return None  # If no data found
 
    # # Prepare data for Prophet
    # product_data = product_data.groupby("Date")["Sales"].sum().reset_index()
    # product_data.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)
    # today = pd.Timestamp.today().normalize()  # Converts today into Pandas Timestamp
    # product_data = product_data[product_data['ds'] < today]  # Keep only past records
    # # Train Prophet Model
    # model = Prophet()
    # model.fit(product_data)
 
    # # Predict future sales
    # future = model.make_future_dataframe(periods=365)  # Predict for up to 1 year
    # forecast = model.predict(future)
 
    # # Ensure no negative values in predictions
    # forecast['yhat'] = forecast['yhat'].clip(lower=0)
 
    # # Filter by time window (start_month to end_month and start_year to end_year)
    # today = datetime.today().date()
    # forecast = forecast[forecast['ds'] >= pd.Timestamp(today)]  # Only future dates
    # forecast['month'] = forecast['ds'].dt.month
    # forecast['year'] = forecast['ds'].dt.year
 
    # if start_month and end_month and start_year and end_year:
    #     forecast = forecast[
    #         (forecast['month'] >= int(start_month)) & (forecast['month'] <= int(end_month)) &
    #         (forecast['year'] >= int(start_year)) & (forecast['year'] <= int(end_year))
    #     ]
 
    # # Plot the results
    # plt.figure(figsize=(10, 5))
    # plt.plot(product_data["ds"], product_data["y"], label="Historical Sales", marker="o")
    # plt.plot(forecast["ds"], forecast["yhat"], label="Predicted Sales", linestyle="dashed")
    # plt.xlabel("Date")
    # plt.ylabel("Sales")
    # plt.title(f"Sales Prediction for {product_name} ({start_month}-{end_month}, {start_year}-{end_year})")
    # plt.legend()
 
    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    print("BP2")
    return base64.b64encode(img.getvalue()).decode()
 
# Home Page
@app.route('/')
def index():
    return render_template('index.html')
 
# Predict Sales
@app.route('/predict', methods=['POST'])
def predict():
    product_name = request.form.get("product_name")
    
    sales_trend_img = predict_sales_for_product(product_name)
    if sales_trend_img:
        return jsonify({"status": "success", "image": sales_trend_img})
    else:
        return jsonify({"status": "error", "message": "Product not found"})
 
# Chatbot API
@app.route('/chat', methods=['POST'])
def chatbot():
    user_message = request.form.get("message").lower()
 
    # Extract product name if mentioned in the message
    words = user_message.split()
    product_name = None
    for word in words:
        if word in df['Model'].unique():
            product_name = word
            break
 
    # Chatbot logic
    if "why sales increased" in user_message:
        response = "Sales increased due to promotions, seasonal demand, or new product features."
        if product_name:
            recent_sales = df[df['Model'] == product_name].sort_values("Date", ascending=False).head(5)
            response += f"\nFor {product_name}, recent sales: {list(recent_sales['Sales'])}"
 
    elif "why sales decreased" in user_message:
        response = "Sales have decreased due to market competition, seasonal changes, or supply chain issues."
        if product_name:
            recent_sales = df[df['Model'] == product_name].sort_values("Date", ascending=False).head(5)
            response += f"\nFor {product_name}, recent sales: {list(recent_sales['Sales'])}"
 
    elif "predict sales" in user_message:
        response = "Enter a product name along with a date range, and I'll predict its sales."
 
    elif "compare sales" in user_message and "last year" in user_message:
        response = "Comparing sales with last year..."
        if product_name:
            this_year = datetime.today().year
            last_year = this_year - 1
 
            this_year_sales = df[(df['Model'] == product_name) & (df['Year'] == this_year)]['Sales'].sum()
            last_year_sales = df[(df['Model'] == product_name) & (df['Year'] == last_year)]['Sales'].sum()
 
            response = f"Sales comparison for {product_name}: \n- {this_year}: {this_year_sales}\n- {last_year}: {last_year_sales}"
            if this_year_sales > last_year_sales:
                response += "\nSales increased compared to last year!"
            else:
                response += "\nSales decreased compared to last year."
 
    elif "what is the trend" in user_message:
        response = "Sales trend is analyzed based on historical data. Would you like a graphical report?"
 
    else:
        response = "I can help with sales trends, predictions, and comparisons. Try asking about 'sales increase', 'sales decrease', or 'predict sales'."
 
    return jsonify({"response": response})
 
 
if __name__ == '__main__':
    app.run(debug=True)
 
