from pytrends.request import TrendReq
import pandas as pd
import time
 
# Initialize pytrends request
pytrends = TrendReq(hl='en', tz=330, geo='IN')
 
# Define the keyword and timeframe
kw_list = ["Sony headphones", "Sony playstation", "Sony TV"] 
timeframe = '2023-01-01 2023-06-30'  # Adjust the timeframe as needed
 
# Function to get data with retries and delay
def get_trends_data(pytrends, kw_list, timeframe):
    while True:
        try:
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='IN', gprop='')
            data = pytrends.interest_by_region(resolution='REGION')
            return data
        except Exception as e:
            print("Error occurred: ", e)
            print("Sleeping for a minute before retrying...")
            time.sleep(5)  # Sleep for 60 seconds before retrying
 
# Get the Google Trends data
data = get_trends_data(pytrends, kw_list, timeframe)
 
# Remove the 'isPartial' column if it exists
if 'isPartial' in data.columns:
    data = data.drop(columns=['isPartial'])
 
# Reset the index to make 'region' a column
data.reset_index(inplace=True)
 
# Print the full data for regions (states)
print("Full state-level data:")
print(data)
 
# Filter for a specific state, e.g., Maharashtra
# Replace 'Maharashtra' with the exact state name if it's available
if 'Maharashtra' in data['geoName'].values:
    maharashtra_data = data[data['geoName'] == 'Maharashtra']
    print("\nData for Maharashtra:")
    print(maharashtra_data)
else:
    print("\nMaharashtra data not found. Available states:")
    print(data['geoName'].unique())
