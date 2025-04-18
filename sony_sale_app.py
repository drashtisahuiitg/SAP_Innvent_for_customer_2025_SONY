import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from prophet import Prophet
from flask import Flask, render_template, request, jsonify
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Load sales data
df = pd.read_csv("dataset/Sony-SAP-Invent_weekly_forecast_2025WK04_2027WK15.csv")

# Convert Year, Month, and Week to a proper Date format
df['Date'] = df.apply(lambda row: datetime.strptime(f"{row['Year']} {row['Month']} {row['Week']}", "%Y %m %W"), axis=1)

# Function to predict sales for a given product within a specific time window
def predict_sales_for_product(product_name, start_month, end_month, start_year, end_year):
    product_data = df[df['Model'] == product_name].copy()  # Filter by product
    if product_data.empty:
        return None, None  # If no data found

    # Prepare data for Prophet
    product_data = product_data.groupby("Date")["Sellout"].sum().reset_index()
    product_data.rename(columns={"Date": "ds", "Sellout": "y"}, inplace=True)
    today = pd.Timestamp.today().normalize()
    product_data = product_data[product_data['ds'] < today]  # Keep only past records

    # Train Prophet Model
    model = Prophet()
    model.fit(product_data)

    # Predict future sales
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Ensure no negative values in predictions
    forecast['yhat'] = forecast['yhat'].clip(lower=0)

    # Extract year and month for aggregation
    forecast['month'] = forecast['ds'].dt.month
    forecast['year'] = forecast['ds'].dt.year

    # Filter forecast data based on user-selected time window
    forecast = forecast[
        (forecast['month'] >= int(start_month)) & (forecast['month'] <= int(end_month)) &
        (forecast['year'] >= int(start_year)) & (forecast['year'] <= int(end_year))
    ]

    # Aggregate sales predictions by Month-Year
    monthly_forecast = forecast.groupby(['year', 'month'])['yhat'].sum().reset_index()

    # Generate tabular HTML output
    table_html = "<table><tr><th>Year</th><th>Month</th><th>Predicted Sales</th></tr>"
    for _, row in monthly_forecast.iterrows():
        table_html += f"<tr><td>{row['year']}</td><td>{row['month']}</td><td>{round(row['yhat'], 2)}</td></tr>"
    table_html += "</table>"

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(product_data["ds"], product_data["y"], label="Historical Sales", marker="o")
    plt.plot(forecast["ds"], forecast["yhat"], label="Predicted Sales", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title(f"Sales Prediction for {product_name} ({start_month}-{end_month}, {start_year}-{end_year})")
    plt.legend()

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode(), table_html

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Predict Sales
@app.route('/predict', methods=['POST'])
def predict():
    product_name = request.form.get("product_name")
    start_month = request.form.get("start_month")
    end_month = request.form.get("end_month")
    start_year = request.form.get("start_year")
    end_year = request.form.get("end_year")

    sales_trend_img, sales_table = predict_sales_for_product(product_name, start_month, end_month, start_year, end_year)

    if sales_trend_img:
        return jsonify({"status": "success", "image": sales_trend_img, "table": sales_table})
    else:
        return jsonify({"status": "error", "message": "Product not found"})

# Chatbot API
@app.route('/chat', methods=['POST'])
def chatbot():
    user_message = request.form.get("message").lower()
    words = user_message.split()
    product_name = None

    for word in words:
        if word in df['Model'].unique():
            product_name = word
            break

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
