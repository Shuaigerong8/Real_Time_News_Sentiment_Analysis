'''spark-submit   --master local[*]   --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1   Task1.py

hdfs dfs -rm -r -f /user/student/cleaned_parquet
hdfs dfs -rm -r -f /user/student/checkpoint
hdfs dfs -rm -r -f /user/student/dedup_store.json
rm -f dedup_store.json
kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic news_topic
kafka-topics.sh --bootstrap-server localhost:9092 --create   --topic news_topic --partitions 1 --replication-factor 1'''
import os
import json
import time
import hashlib
import logging
import argparse
import re
import requests
import unicodedata
from threading import Thread

from bs4 import BeautifulSoup
from newspaper import Article
from kafka import KafkaConsumer
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_json, struct
from pyspark.sql.types import StructType, StructField, StringType
from dateutil import parser as dateparser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class NewsFetcher:
    def __init__(self, query, api_key, page_size=100, retries=3, delay=5):
        self.url = "https://newsapi.org/v2/everything"
        self.query = query
        self.api_key = api_key
        self.page_size = page_size
        self.retries = retries
        self.delay = delay

    def fetch(self):
        params = {"q": self.query, "apiKey": self.api_key, "pageSize": self.page_size}
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.get(self.url, params=params, timeout=10)
                if r.status_code == 200:
                    articles = r.json().get('articles', [])
                    logging.info(f"Fetched {len(articles)} articles on attempt {attempt}")
                    return articles
                logging.warning(f"Fetch {attempt} failed: HTTP {r.status_code}")
            except requests.RequestException as e:
                logging.warning(f"Fetch {attempt} exception: {e}")
            time.sleep(self.delay)
        logging.error("All fetch attempts failed")
        return []


class ArticleCleaner:
    url_pattern     = re.compile(r'https?://\S+')
    email_pattern = re.compile(r'\S+@\S+\.\S+')

    def clean(self, raw: dict) -> dict:
        title     = (raw.get('title') or '').strip()
        url       = (raw.get('url') or '').strip()
        date_raw  = raw.get('publishedAt', '')
        try:
            publish_date = dateparser.parse(date_raw).isoformat()
        except (ValueError, TypeError):
            publish_date = ''
        source = raw.get('source', {}).get('name', '')
        text = raw.get('content') or raw.get('description') or ''
        # strip HTML, normalize, remove URLs/emails, punctuation → lowercase
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = unicodedata.normalize('NFKC', text)
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.lower().split())

        # if too short, pull the full article
        if len(text) < 200:
            full = self._fetch_full_text(url)
            if full:
                text = unicodedata.normalize('NFKC', full)
                text = self.url_pattern.sub('', text)
                text = ' '.join(re.sub(r'[^\w\s]', ' ', text).lower().split())

        return {
            'title': title,
            'url': url,
            'publishDate': publish_date,
            'newsCompany': source,
            'full_text': text
        }

    def _fetch_full_text(self, url: str) -> str:
        try:
            art = Article(url)
            art.download(); art.parse()
            return art.text
        except Exception:
            return ''


class DedupStore:
    def __init__(self, local_file: str, hdfs_path: str, spark: SparkSession):
        self.local_file = local_file
        self.hdfs_path  = hdfs_path
        self.spark      = spark
        self.hashes     = self._load()

    def _load(self) -> set:
        if os.path.exists(self.local_file):
            with open(self.local_file) as f:
                return set(json.load(f))
        return set()

    def is_new(self, h: str) -> bool:
        return h not in self.hashes

    def update(self, new_hashes: set):
        self.hashes |= new_hashes
        with open(self.local_file, 'w') as f:
            json.dump(list(self.hashes), f)
        # mirror to HDFS
        from pyspark.sql import Row
        df = self.spark.createDataFrame([Row(hash=x) for x in self.hashes])
        df.write.mode('overwrite').json(self.hdfs_path)
        logging.info(f"Dedup store updated: {len(self.hashes)} total entries")


class ParquetWriter:
    def __init__(self, path: str, partitions: int):
        self.path       = path
        self.partitions = partitions

    def write(self, df: DataFrame):
        repart = df.repartition(self.partitions, 'publishDate').cache()
        count = repart.count()
        repart.write.mode('append').partitionBy('publishDate').parquet(self.path)
        repart.unpersist()
        logging.info(f"Wrote {count} records to Parquet at {self.path}")


class KafkaPublisher:
    def __init__(self, spark: SparkSession, broker: str, topic: str):
        self.spark = spark
        self.broker = broker
        self.topic = topic

    def publish(self, df: DataFrame):
        if df.rdd.isEmpty():
            logging.info("No records to publish to Kafka")
            return
        df.select(to_json(struct(*df.columns)).alias('value')) \
            .write.format('kafka') \
            .option('kafka.bootstrap.servers', self.broker) \
            .option('topic', self.topic) \
            .save()
        logging.info(f"Published {df.count()} records to Kafka topic '{self.topic}'")


class SampleViewer:
    """
    Read your cleaned Parquet, drop true duplicates,
    grab a sample of N rows and either:
      - show via ace_tools (Jupyter), or
      - fallback to printing JSON lines
    """
    def __init__(self, spark: SparkSession, parquet_path: str,
                 dedup_cols=None, sample_size: int = 10):
        self.spark        = spark
        self.parquet_path = parquet_path
        self.dedup_cols   = dedup_cols or ['url']
        self.sample_size  = sample_size

    def show_sample(self):
        df = self.spark.read.parquet(self.parquet_path)
        samp = df.dropDuplicates(self.dedup_cols).limit(self.sample_size)

        # Try interactive display
        try:
            from ace_tools import display_dataframe_to_user
            pdf = samp.toPandas()
            display_dataframe_to_user("10‑row Sample of Cleaned Articles", pdf)
        except ImportError:
            # fallback to printing JSON lines
            logging.info("ace_tools not available—falling back to printing JSON lines")
            for row in samp.toJSON().collect():
                print(row)


class Producer:
    def __init__(self, args):
        self.fetcher  = NewsFetcher(args.query, args.api_key, args.page_size)
        self.cleaner  = ArticleCleaner()
        self.spark    = SparkSession.builder \
                            .appName('Producer') \
                            .config('spark.sql.shuffle.partitions', args.partitions) \
                            .getOrCreate()
        self.dedup    = DedupStore(args.dedup_local, args.dedup_hdfs, self.spark)
        self.writer   = ParquetWriter(args.hdfs_parquet, args.partitions)
        self.publisher= KafkaPublisher(self.spark, args.broker, args.topic)
        self.viewer   = SampleViewer(self.spark, args.hdfs_parquet)
        self.interval = args.interval
        self.schema   = StructType([
            StructField('title', StringType(), False),
            StructField('url', StringType(), False),
            StructField('publishDate', StringType(), True),
            StructField('newsCompany', StringType(), True),
            StructField('full_text', StringType(), False),
        ])

    def run(self):
        while True:
            raw = self.fetcher.fetch()
            total = len(raw)
            processed, new_hashes = [], set()
            lengths, sources = [], set()

            for art in raw:
                h = hashlib.sha256(art.get('url','').encode()).hexdigest()
                if self.dedup.is_new(h):
                    rec = self.cleaner.clean(art)
                    if rec['full_text']:
                        processed.append(rec)
                        new_hashes.add(h)
                        sources.add(rec['newsCompany'])
                        lengths.append(len(rec['full_text']))

            saved  = len(processed)
            skipped= total - saved

            if saved:
                df = self.spark.createDataFrame(
                    self.spark.sparkContext.parallelize(processed),
                    self.schema
                )
                self.writer.write(df)
                self.publisher.publish(df)
                self.dedup.update(new_hashes)
                # **NEW**: show a sample after each batch
                self.viewer.show_sample()

            avg_len = (sum(lengths)/len(lengths)) if lengths else 0
            logging.info(
                f"Batch summary: fetched={total}, saved={saved}, skipped={skipped}, "
                f"avg_length={avg_len:.1f}, sources={len(sources)}"
            )

            time.sleep(self.interval)


class Consumer:
    def __init__(self, args):
        self.consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=[args.broker],
            auto_offset_reset='latest',
            group_id='consumer_group',
            value_deserializer=lambda m: json.loads(m.decode())
        )
        self.spark      = SparkSession.builder \
                            .appName('Consumer') \
                            .config('spark.sql.shuffle.partitions', args.partitions) \
                            .getOrCreate()
        self.schema     = StructType([
            StructField('title', StringType(), False),
            StructField('url', StringType(), False),
            StructField('publishDate', StringType(), True),
            StructField('newsCompany', StringType(), True),
            StructField('full_text', StringType(), False),
        ])
        self.hdfs_output  = args.hdfs_consumer
        self.batch_size   = args.batch_size
        self.batch_interval = args.batch_interval

    def run(self):
        batch, start = [], time.time()
        for msg in self.consumer:
            batch.append(msg.value)
            now = time.time()
            if len(batch) >= self.batch_size or now - start >= self.batch_interval:
                df = self.spark.createDataFrame(batch, self.schema)
                for row in df.toJSON().collect():
                    print(row)
                if self.hdfs_output:
                    df.write.mode('append').parquet(self.hdfs_output)
                    logging.info(f"Consumer wrote {df.count()} records to {self.hdfs_output}")
                batch, start = [], now


if __name__ == '__main__':
    config = {
        'query':         'Malaysian Communications and Multimedia Commission',
        'api_key':       '29de796cae54413ea620844589651957',  # Replace with your actual API key
        'broker':        'localhost:9092',
        'topic':         'news_topic',
        'interval':      60,
        'page_size':     100,
        'partitions':    4,
        'hdfs_parquet':  'hdfs://localhost:9000/user/student/cleaned_parquet',
        'dedup_local':   'dedup_store.json',
        'dedup_hdfs':    'hdfs://localhost:9000/user/student/dedup_store.json',
        'hdfs_consumer': 'hdfs://localhost:9000/user/student/consumer_output',
        'batch_size':    50,
        'batch_interval': 30,
    }
    args = argparse.Namespace(**config)

    producer = Producer(args)
    consumer = Consumer(args)

    # run both in background threads
    Thread(target=producer.run, daemon=True).start()
    Thread(target=consumer.run, daemon=True).start()

    # keep alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutting down.")