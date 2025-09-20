# Lee Xiao Syuen

import json
import time
import re
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_json, struct

class NewsKafkaProducer:
    def __init__(self,
                 api_key="29de796cae54413ea620844589651957",
                 query="Malaysian Communications and Multimedia Commission",
                 kafka_broker="localhost:9092",
                 kafka_topic="news_topic",
                 fetch_interval=60):
        self.NEWSAPI_KEY = api_key
        self.NEWSAPI_URL = "https://newsapi.org/v2/everything"
        self.QUERY = query
        self.KAFKA_BROKER = kafka_broker
        self.KAFKA_TOPIC = kafka_topic
        self.FETCH_INTERVAL = fetch_interval
        self.spark = SparkSession.builder.appName("NewsSparkKafkaProducer").getOrCreate()

    def remove_special_chars(self, text):
        return re.sub(r'[\\>\u2666]', '', text)

    def fetch_full_article(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text.strip()
        except Exception as e:
            print(f"[Error] Cannot fetch full text from {url}: {e}")
            return ""

    def clean_article(self, article):
        title = article.get("title", "").strip()
        url = article.get("url", "").strip()
        publishDate = article.get("publishedAt", "")
        newsCompany = article.get("source", {}).get("name", "")
        content = article.get("content") or article.get("description", "")

        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator=" ").lower()
        text = " ".join(text.split())

        if "[+" in text or "…" in text or len(text) < 200:
            full_text = self.fetch_full_article(url)
            if full_text:
                text = " ".join(full_text.lower().split())

        full_text = self.remove_special_chars(text)
        return {
            "title": title,
            "url": url,
            "publishDate": publishDate,
            "newsCompany": newsCompany,
            "full_text": full_text
        }

    def fetch_news(self):
        params = {
            'q': self.QUERY,
            'apiKey': self.NEWSAPI_KEY,
            'pageSize': 100,
            'language': 'en'
        }
        response = requests.get(self.NEWSAPI_URL, params=params)
        if response.status_code == 200:
            return response.json().get('articles', [])
        else:
            print(f"[Error] {response.status_code}: {response.text}")
            return []

    def send_to_kafka(self, cleaned_articles):
        df = self.spark.createDataFrame(cleaned_articles)
        df.select(to_json(struct(*df.columns)).alias("value")) \
          .write \
          .format("kafka") \
          .option("kafka.bootstrap.servers", self.KAFKA_BROKER) \
          .option("topic", self.KAFKA_TOPIC) \
          .save()
        print(f"📝 Sent batch of {len(cleaned_articles)} articles via Spark")

    def run(self):
        try:
            while True:
                raw_articles = self.fetch_news()
                print(f"📥 Fetched {len(raw_articles)} articles from NewsAPI")

                cleaned_list = []
                for raw in raw_articles:
                    if not raw.get("title") or not raw.get("url"):
                        continue
                    cleaned = self.clean_article(raw)
                    if cleaned.get("full_text"):
                        cleaned_list.append(cleaned)

                if cleaned_list:
                    self.send_to_kafka(cleaned_list)

                time.sleep(self.FETCH_INTERVAL)
        finally:
            self.spark.stop()

if __name__ == "__main__":
    NewsKafkaProducer().run()
