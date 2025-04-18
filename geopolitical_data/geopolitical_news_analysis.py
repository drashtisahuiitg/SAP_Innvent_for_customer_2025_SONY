from gnews import GNews
import pandas as pd
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from prettytable import PrettyTable

# Initialize GNews object with English language and country filters
google_news = GNews(language='en', country='US', period='')

# Define the search query for Sony & geopolitics
query = 'Sony AND (Ukraine OR Russia OR war OR sanctions OR trade war OR export ban OR geopolitical tensions OR trade sanctions)'

# Define date range: Start from Jan 1, 2021, until today
start_date = datetime(2021, 1, 1)
end_date = datetime.today()

# Fetch news articles
articles = google_news.get_news(query)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Store filtered news articles
filtered_articles = []

# Create a PrettyTable object for display
table = PrettyTable()
table.field_names = ["Date", "Title", "Sentiment", "Score"]

# Process articles
for article in articles:
    try:
        # Convert published date to datetime format
        published_date = datetime.strptime(article["published date"], "%a, %d %b %Y %H:%M:%S GMT")
        
        # Filter by date range
        if start_date <= published_date <= end_date:
            # Get sentiment score
            text = article["title"] + " " + article.get("description", "")
            sentiment_score = sia.polarity_scores(text)["compound"]  # Ranges from -1 (negative) to +1 (positive)

            # Classify sentiment
            if sentiment_score > 0.05:
                sentiment = "Positive"
            elif sentiment_score < -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            # Store results
            filtered_articles.append({
                "Title": article["title"],
                "Published Date": published_date.strftime("%Y-%m-%d"),
                "Description": article.get("description", "N/A"),
                "URL": article["url"],
                "Sentiment Score": sentiment_score,
                "Sentiment": sentiment
            })

            # Add row to PrettyTable
            table.add_row([published_date.strftime("%Y-%m-%d"), article["title"][:50], sentiment, round(sentiment_score, 2)])

    except Exception as e:
        print(f"Error processing article: {e}")

# Convert to DataFrame and save as CSV
df = pd.DataFrame(filtered_articles)
df.to_csv("sony_geopolitical_news_with_sentiment.csv", index=False)

# Print the table
print(table)

print("✅ News saved successfully with sentiment scores in 'sony_geopolitical_news_with_sentiment.csv'!")
