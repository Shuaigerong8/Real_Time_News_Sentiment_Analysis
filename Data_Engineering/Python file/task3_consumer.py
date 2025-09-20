# Lee Xiao Syuen

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sha2, concat_ws, from_json
from pyspark.sql.types import StructType, StringType
from pyspark.ml import PipelineModel
from threading import Timer
from task2 import TextPreprocessing, DataEnrichment
from task2_components import SentenceSplitter, CleanTextTransformer, VaderSentiment

class Task3StreamingPredictor:
    def __init__(self):
        self.KAFKA_BROKER = "localhost:9092"
        self.KAFKA_TOPIC = "news_topic"
        self.MODEL_PATH = "hdfs://localhost:9000/user/student/sentiment_model"
        self.OUTPUT_PATH = "hdfs://localhost:9000/user/student/sentiment_output"
        self.DEDUP_STORE_PATH = "hdfs://localhost:9000/user/student/dedup_store.json"
        self.CHECKPOINT_PATH = "hdfs://localhost:9000/user/student/checkpoint"
        self.EMPTY_THRESHOLD = 3
        self.consecutive_empty_batches = 0

        self.spark = SparkSession.builder \
            .appName("Task3_Streaming_Prediction_With_Sentence_Dedup") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "6g") \
            .getOrCreate()

        self.model = PipelineModel.load(self.MODEL_PATH)
        for stage in self.model.stages:
            if hasattr(stage, "setLabels") and stage.getOutputCol() == "predictedLabel":
                stage.setLabels(["negative", "neutral", "positive"])
                print("✅ IndexToString label mapping restored")

        self.schema = StructType() \
            .add("title", StringType()) \
            .add("url", StringType()) \
            .add("publishDate", StringType()) \
            .add("newsCompany", StringType()) \
            .add("full_text", StringType())

        kafka_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.KAFKA_BROKER) \
            .option("subscribe", self.KAFKA_TOPIC) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()

        self.parsed_df = kafka_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.schema).alias("data")) \
            .select("data.*") \
            .withColumn("hash", sha2(concat_ws("", "title", "url", "full_text"), 256))

    def enrich_and_predict(self, batch_df, batch_id):
        print(f"🔥 Task 3 received batch {batch_id}, size = {batch_df.count()}")
        if batch_df.rdd.isEmpty():
            self.consecutive_empty_batches += 1
            print(f"🚫 Empty batch {batch_id} (consecutive empty: {self.consecutive_empty_batches})")
            if self.consecutive_empty_batches >= self.EMPTY_THRESHOLD:
                print("🛑 No new data for too long. Shutting down Spark...")
                Timer(5.0, lambda: self.query.stop()).start()
                Timer(7.0, lambda: self.spark.stop()).start()
            return
        else:
            self.consecutive_empty_batches = 0

        try:
            existing_hashes_df = self.spark.read.json(self.DEDUP_STORE_PATH)
            existing_hashes = set(r["hash"] for r in existing_hashes_df.collect())
        except:
            existing_hashes = set()

        filtered_df = batch_df.filter(~col("hash").isin(existing_hashes))
        print(f"🔍 Batch {batch_id}: {batch_df.count()} total → {filtered_df.count()} new")

        if filtered_df.rdd.isEmpty():
            self.consecutive_empty_batches += 1
            print(f"🚫 No new data to write in batch {batch_id} (consecutive empty: {self.consecutive_empty_batches})")
            if self.consecutive_empty_batches >= self.EMPTY_THRESHOLD:
                print("🛑 No new data for too long. Shutting down Spark...")
                Timer(5.0, lambda: self.query.stop()).start()
                Timer(7.0, lambda: self.spark.stop()).start()
            return

        self.consecutive_empty_batches = 0

        preprocessing = TextPreprocessing(self.spark)
        enrichment = DataEnrichment(self.spark)

        clean_df = preprocessing.clean_text(filtered_df, input_col="full_text")
        tokenized_df = preprocessing.tokenize(clean_df)
        filtered_token_df = preprocessing.remove_stopwords(tokenized_df)
        enriched_df = enrichment.run_enrichment(filtered_token_df)

        enriched_df = enriched_df.withColumn("topic_summary", concat_ws(" | ",
            col("topic_with_lda"), col("topic_with_ner"), col("department"), col("category")))

        # Join publishDate and hash back into enriched_df
        enriched_df = enriched_df.join(filtered_df.select("title", "url", "full_text", "publishDate", "hash"),
                                       on=["title", "url", "full_text"], how="left")

        prediction_df = self.model.transform(enriched_df)

        prediction_df.select(
            "title", "url", "publishDate", "newsCompany", "full_text", "sentence",
            "predictedLabel", "topic_summary", "topic_with_lda", "topic_with_ner",
            "department", "category", "person_names", "locations"
        ).write.mode("append").parquet(self.OUTPUT_PATH)


        filtered_df.select("hash").write.mode("append").json(self.DEDUP_STORE_PATH)
        print(f"✅ Written predictions for batch {batch_id}")

    def start(self):
        self.query = self.parsed_df.writeStream \
            .foreachBatch(self.enrich_and_predict) \
            .outputMode("append") \
            .option("checkpointLocation", self.CHECKPOINT_PATH) \
            .start()

        self.query.awaitTermination()


if __name__ == "__main__":
    Task3StreamingPredictor().start()
