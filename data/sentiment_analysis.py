import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser  # To parse publication dates

logging.basicConfig(level=logging.INFO)

# Load the tokenizer and model
# Using a financial sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

def get_news_articles(coin, days=1):
    try:
        # Map coin symbols to full names
        coin_names = {
            'BTCUSDT': 'Bitcoin',
            'ETHUSDT': 'Ethereum',
            # Add other mappings as needed
        }
        coin_name = coin_names.get(coin.upper(), coin)

        query = f"{coin_name} cryptocurrency when:{days}d"
        url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
        response = requests.get(url)

        # Parse the XML content
        soup = BeautifulSoup(response.content, 'xml')
        articles = []
        for item in soup.find_all('item'):
            title = item.title.text
            pub_date = item.pubDate.text
            # Parse publication date
            pub_date_parsed = parser.parse(pub_date)
            articles.append({'title': title, 'pub_date': pub_date_parsed})
        logging.info(f"Fetched {len(articles)} articles for {coin_name}.")
        return articles
    except Exception as e:
        logging.exception(f"Error fetching news articles for {coin}: {e}")
        return []

def analyze_sentiment(articles):
    sentiments = []
    texts = [article['title'] for article in articles if 'title' in article and article['title']]

    if not texts:
        logging.error("No texts available for sentiment analysis.")
        return sentiments

    try:
        # Tokenize the inputs
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        # Extract sentiment scores
        sentiments_raw = torch.softmax(logits, dim=1).detach().cpu().numpy()
        # Assuming the model outputs [negative, neutral, positive]
        # We can assign scores: negative=-1, neutral=0, positive=1
        sentiment_scores = sentiments_raw @ np.array([-1, 0, 1])
        sentiments.extend(sentiment_scores)
    except Exception as e:
        logging.exception("Error analyzing sentiment.")
    return sentiments