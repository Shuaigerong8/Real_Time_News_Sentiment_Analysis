# Author: Tan Rou Ming
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF,
    StringIndexer, IndexToString
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import LDA
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, IntegerType, DoubleType, MapType
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
import re
from pyspark.sql.functions import (
    col, lower, regexp_replace, concat_ws, udf,
    explode, trim, split, lit, sha2
)
from task2_components import SentenceSplitter, CleanTextTransformer, VaderSentiment
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF,
    StringIndexer, IndexToString
)
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import DataFrame
from pyspark.ml import Transformer


nltk.download('vader_lexicon')

def create_spark_session():
    return SparkSession.builder \
        .appName("Task2_Sentiment_Pipeline") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

class SentenceSplitter(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="full_text", outputCol="sentence"):
        super().__init__()
        self.setParams(inputCol, outputCol)

    def setParams(self, inputCol=None, outputCol=None):
        if inputCol: self._set(inputCol=inputCol)
        if outputCol: self._set(outputCol=outputCol)
        return self

    def _transform(self, df: DataFrame):
        return df.withColumn(self.getOutputCol(), explode(split(trim(col(self.getInputCol())), "\\."))) \
                 .filter(col(self.getOutputCol()) != "")

class CleanTextTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="sentence", outputCol="clean_text"):
        super().__init__()
        self.setParams(inputCol, outputCol)

    def setParams(self, inputCol=None, outputCol=None):
        if inputCol: self._set(inputCol=inputCol)
        if outputCol: self._set(outputCol=outputCol)
        return self

    def _transform(self, df: DataFrame):
        return df.withColumn(self.getOutputCol(), lower(regexp_replace(col(self.getInputCol()), "[^a-zA-Z\\s]", " ")))

class VaderSentiment(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="clean_text", outputCol="label"):
        super().__init__()
        self.setParams(inputCol, outputCol)

    def setParams(self, inputCol=None, outputCol=None):
        if inputCol: self._set(inputCol=inputCol)
        if outputCol: self._set(outputCol=outputCol)
        return self

    def _transform(self, df: DataFrame):
        analyzer = SentimentIntensityAnalyzer()
        def get_sentiment(text):
            if not text: return "neutral"
            score = analyzer.polarity_scores(text)["compound"]
            return "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        sentiment_udf = udf(get_sentiment, StringType())
        return df.withColumn(self.getOutputCol(), sentiment_udf(col(self.getInputCol())))
        
# TextPreprocessing class: responsible for cleaning, tokenizing, and removing stopwords
class TextPreprocessing:
    def __init__(self, spark_session):
        self.spark = spark_session

    def clean_text(self, df: DataFrame, input_col="full_text", output_col="clean_text"):
        return df.withColumn(output_col, lower(regexp_replace(col(input_col), "[^a-zA-Z\\s]", " ")))

    def tokenize(self, df: DataFrame, input_col="clean_text", output_col="tokenized_text"):
        tokenizer = RegexTokenizer(inputCol=input_col, outputCol=output_col, pattern="\\W")
        return tokenizer.transform(df)

    def remove_stopwords(self, df: DataFrame, input_col="tokenized_text", output_col="filtered_words"):
        remover = StopWordsRemover(inputCol=input_col, outputCol=output_col)
        return remover.transform(df)

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import DataFrame
import nltk
import re

# Download NLTK vader lexicon
nltk.download('vader_lexicon')

class NamedEntityRecognition:
    def __init__(self):
        # Define honorifics and common countries/states
        self.honorifics = ["mr", "mrs", "ms", "dr", "prof", "rev", "sen", "rep", "gov", "president", "ceo", "chairman"]
        self.location_indicators = ["city", "town", "country", "state", "province", "region", "area", "district",
                                     "village", "suburb", "street", "avenue", "boulevard", "road", "highway",
                                     "park", "square", "north", "south", "east", "west", "county", "territory",
                                     "island", "coast", "capital", "metro", "downtown"]
        self.common_countries = ["united states", "canada", "mexico", "brazil", "argentina", "uk", "france",
                                   "germany", "italy", "spain", "china", "japan", "india", "australia", "russia"]
        self.common_us_states = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado",
                                   "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
                                   "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
                                   "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana",
                                   "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york",
                                   "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
                                   "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah",
                                   "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming"]

    def is_title_case(self, word):
        """Check if word is in title case (first letter capital)"""
        return len(word) > 1 and word[0].isupper() and word[1:].islower()

    def extract_person_names(self, text):
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        words = text.split()
        words_lower = text_lower.split()

        potential_names = []

        # Find name patterns
        for i in range(len(words) - 1):
            # Pattern 1: Honorific + Title Case Word
            if i < len(words) - 1 and words_lower[i] in self.honorifics and self.is_title_case(words[i+1]):
                if i < len(words) - 2 and self.is_title_case(words[i+2]):  # Honorific + First + Last
                    potential_names.append(f"{words[i]} {words[i+1]} {words[i+2]}")
                else:  # Honorific + Last
                    potential_names.append(f"{words[i]} {words[i+1]}")

            # Pattern 2: Two consecutive title case words (First + Last)
            elif i < len(words) - 1 and self.is_title_case(words[i]) and self.is_title_case(words[i+1]):
                if i < len(words) - 2 and self.is_title_case(words[i+2]):  # First + Middle + Last
                    potential_names.append(f"{words[i]} {words[i+1]} {words[i+2]}")
                else:  # First + Last
                    potential_names.append(f"{words[i]} {words[i+1]}")

        # Check for common name indicators
        for i, word in enumerate(words_lower):
            if word in ["says", "said", "according", "told", "reported", "claimed"] and i > 0:
                if self.is_title_case(words[i-1]) and i > 1 and self.is_title_case(words[i-2]):
                    potential_names.append(f"{words[i-2]} {words[i-1]}")

        return list(set(potential_names))

    def extract_locations(self, text):
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        words = text.split()

        potential_locations = []

        # Check for common countries and states
        for country in self.common_countries:
            if country in text_lower:
                potential_locations.append(country.title())

        for state in self.common_us_states:
            if state in text_lower or f"state of {state}" in text_lower:
                potential_locations.append(state.title())

        # Look for prepositions followed by potential locations
        for i in range(len(words) - 1):
            if words[i].lower() in ["in", "at", "near", "from", "to"] and i+1 < len(words) and self.is_title_case(words[i+1]):
                # Check if followed by location indicator
                if i+2 < len(words) and words[i+2].lower() in self.location_indicators:
                    potential_locations.append(f"{words[i+1]} {words[i+2]}")
                elif words[i+1].lower() not in ["the", "a", "an"]:  # Avoid common articles
                    potential_locations.append(words[i+1])

        # Look for capitalized words followed by location indicators
        for i in range(len(words) - 1):
            if self.is_title_case(words[i]) and i+1 < len(words) and words[i+1].lower() in self.location_indicators:
                potential_locations.append(f"{words[i]} {words[i+1]}")

        return list(set(potential_locations))

    # Register UDFs for Spark processing
    def register_udfs(self):
        extract_persons_udf = udf(self.extract_person_names, ArrayType(StringType()))
        extract_locations_udf = udf(self.extract_locations, ArrayType(StringType()))
        return extract_persons_udf, extract_locations_udf

# Define infer_topic_from_ner as a standalone function
def infer_topic_from_ner(persons, locations):
    person_count = len(persons) if persons else 0
    location_count = len(locations) if locations else 0

    # More nuanced classification
    if person_count > 4:
        return "People-Focused"
    elif person_count > 2 and location_count > 2:
        return "People & Places"
    elif person_count > 2:
        return "People-Centered"
    elif location_count > 4:
        return "Geography-Focused"
    elif location_count > 2:
        return "Location-Centered"
    elif person_count > 0 and location_count > 0:
        return "Mixed Entities"
    elif person_count > 0:
        return "Limited People"
    elif location_count > 0:
        return "Limited Places"
    else:
        return "No Named Entities"


class DataEnrichment:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.ner = NamedEntityRecognition()  # Instantiate the NamedEntityRecognition class

    def lda_topic_extraction(self, df: DataFrame, num_topics=10, top_n=5):
        cv = CountVectorizer(inputCol="filtered_words", outputCol="features_for_lda", minDF=2.0)
        cv_model = cv.fit(df)
        vectorized = cv_model.transform(df)

        lda = LDA(k=num_topics, maxIter=20, featuresCol="features_for_lda", optimizer="em")
        lda_model = lda.fit(vectorized)

        vocab = cv_model.vocabulary
        topics_matrix = lda_model.topicsMatrix().toArray()

        # Extract topic descriptions before creating UDF
        topic_keywords = {}
        for topic_id in range(num_topics):
            term_indices = topics_matrix[:, topic_id].argsort()[-top_n:][::-1]
            keywords = [vocab[i] for i in term_indices]
            topic_keywords[topic_id] = keywords

        def get_top_topics(topic_distribution):
            if topic_distribution is None:
                return []
            topic_probabilities = list(enumerate(topic_distribution))
            top_topics_with_probs = sorted(topic_probabilities, key=lambda x: x[1], reverse=True)[:top_n]
            return [(int(topic_index), float(prob)) for topic_index, prob in top_topics_with_probs]

        get_top_topics_udf = udf(get_top_topics, ArrayType(StructType([
            StructField("topic_id", IntegerType(), False),
            StructField("probability", DoubleType(), False)
        ])))

        # Create a closure that doesn't require lda_model
        def get_topic_keywords_for_ids(top_topics):
            keywords_list = []
            for topic_id, prob in top_topics:
                keywords = topic_keywords.get(topic_id, ["unknown"])
                keywords_list.append(f"Topic {topic_id}: {', '.join(keywords)} (Prob: {prob:.2f})")
            return keywords_list

        get_topic_keywords_udf = udf(get_topic_keywords_for_ids, ArrayType(StringType()))

        transformed = lda_model.transform(vectorized)
        transformed = transformed.withColumn("top_topics", get_top_topics_udf(col("topicDistribution")))
        transformed = transformed.withColumn("related_topics", get_topic_keywords_udf(col("top_topics")))

        # Create a separate column with just the primary topic (highest probability)
        def get_primary_topic(top_topics):
            if not top_topics or len(top_topics) == 0:
                return "Unknown"
            top_topic_id = top_topics[0][0]
            return f"Topic {top_topic_id}: {', '.join(topic_keywords.get(top_topic_id, ['unknown']))}"

        get_primary_topic_udf = udf(get_primary_topic, StringType())
        transformed = transformed.withColumn("topic_with_lda", get_primary_topic_udf(col("top_topics")))

        # Keep all necessary columns
        cols_to_keep = ["title", "url", "full_text", "related_topics", "topic_with_lda",
                        "clean_text", "newsCompany", "filtered_words", "topicDistribution"]
        return transformed.select(*cols_to_keep)

    def enhance_ner_with_keywords(self, df: DataFrame):
        """Improved keyword-based topic detection with multi-level classification"""
        # Define multilevel taxonomy with hierarchical categories and keywords
        topic_taxonomy = {
            "Politics": {
                "keywords": ["political", "politics", "election", "vote", "government", "policy"],
                "subcategories": {
                    "US Politics": ["president", "congress", "senate", "white house", "washington", "democrat", "republican"],
                    "International Politics": ["diplomat", "treaty", "summit", "foreign", "minister", "international", "global"],
                    "Elections": ["campaign", "ballot", "poll", "voter", "candidate", "primary", "constituency"],
                    "Legislation": ["bill", "law", "regulation", "legislation", "judiciary", "court", "supreme court"]
                }
            },
            "Business": {
                "keywords": ["business", "company", "corporate", "industry", "market", "firm"],
                "subcategories": {
                    "Economy": ["economy", "economic", "growth", "recession", "inflation", "gdp", "fiscal"],
                    "Finance": ["stock", "investment", "investor", "bank", "financial", "fund", "trading", "market"],
                    "Corporate": ["ceo", "executive", "board", "merger", "acquisition", "startup", "corporation"],
                    "Trade": ["trade", "tariff", "export", "import", "commerce", "supply chain", "logistics"]
                }
            },
            # Define more categories here as needed...
        }

        def classify_text_by_keywords(text):
            if not text or not isinstance(text, str):
                return ("Unknown", "Unknown")

            text_lower = text.lower()

            # First level classification
            category_scores = {}
            for category, data in topic_taxonomy.items():
                # Count main category keywords
                main_keywords = data["keywords"]
                score = sum(1 for word in main_keywords if word in text_lower)

                # Add scores from subcategories
                for subcategory, subcategory_keywords in data["subcategories"].items():
                    subcategory_score = sum(1 for word in subcategory_keywords if word in text_lower)
                    score += subcategory_score * 0.5  # Weight subcategory keywords less

                category_scores[category] = score

            # Get top level category
            if not category_scores or max(category_scores.values()) == 0:
                top_category = "General"
            else:
                top_category = max(category_scores.items(), key=lambda x: x[1])[0]

            # If we have a valid top category, find the top subcategory
            subcategory = "General"
            if top_category in topic_taxonomy:
                subcategory_scores = {}
                for subcategory_name, subcategory_keywords in topic_taxonomy[top_category]["subcategories"].items():
                    score = sum(1 for word in subcategory_keywords if word in text_lower)
                    subcategory_scores[subcategory_name] = score

                if subcategory_scores and max(subcategory_scores.values()) > 0:
                    subcategory = max(subcategory_scores.items(), key=lambda x: x[1])[0]

            return (top_category, subcategory)

        # Register UDF here immediately after defining the function
        classify_text_udf = udf(classify_text_by_keywords, StructType([
            StructField("main_category", StringType(), False),
            StructField("subcategory", StringType(), False)
        ]))

        # Apply UDF and create separate columns for main category and subcategory
        df = df.withColumn("keyword_classification", classify_text_udf(col("full_text")))
        df = df.withColumn("department", col("keyword_classification.main_category"))
        df = df.withColumn("category", col("keyword_classification.subcategory"))

        return df.drop("keyword_classification")

    def run_enrichment(self, df: DataFrame):
        """Run the complete enrichment pipeline"""
        # Extract topics with LDA
        enriched_df = self.lda_topic_extraction(df)

        # Register UDFs from the NamedEntityRecognition class
        extract_persons_udf, extract_locations_udf = self.ner.register_udfs()

        # Add named entity recognition columns
        enriched_df = enriched_df.withColumn("person_names", extract_persons_udf(col("full_text")))
        enriched_df = enriched_df.withColumn("locations", extract_locations_udf(col("full_text")))

        # Add NER-based topic classification
        infer_topic_udf = udf(infer_topic_from_ner, StringType())
        enriched_df = enriched_df.withColumn("topic_with_ner", infer_topic_udf(col("person_names"), col("locations")))

        # Add keyword-based topic classification
        enriched_df = self.enhance_ner_with_keywords(enriched_df)

        return enriched_df


class SentimentAnalysisPipeline:
    def __init__(self):
        self.pipeline = Pipeline(stages=[
            SentenceSplitter(),
            CleanTextTransformer(),
            VaderSentiment(),
            RegexTokenizer(inputCol="clean_text", outputCol="words_class", pattern="\\W"),
            StopWordsRemover(inputCol="words_class", outputCol="filtered_words_class"),
            CountVectorizer(inputCol="filtered_words_class", outputCol="raw_features_class"),
            IDF(inputCol="raw_features_class", outputCol="features"),
            StringIndexer(inputCol="label", outputCol="indexedLabel", handleInvalid="keep"),
            LogisticRegression(featuresCol="features", labelCol="indexedLabel"),
            IndexToString(inputCol="prediction", outputCol="predictedLabel")
        ])
    
    def get_pipeline(self):
        return self.pipeline
    
    def fit(self, df: DataFrame):
        """
        Fit the pipeline on the given dataframe.
        
        Args:
            df (DataFrame): The input dataframe to fit the pipeline.
        
        Returns:
            PipelineModel: The trained pipeline model.
        """
        return self.pipeline.fit(df)
    
    def transform(self, model, df: DataFrame):
        """
        Transform the given dataframe using the trained model.
        
        Args:
            model (PipelineModel): The trained pipeline model.
            df (DataFrame): The input dataframe to transform.
        
        Returns:
            DataFrame: The transformed dataframe with predictions.
        """
        return model.transform(df)


def train_and_evaluate(data_path):
    spark = SparkSession.builder.appName("DataEnrichment") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    preprocessing = TextPreprocessing(spark)
    enrichment = DataEnrichment(spark)

    # Load data
    df = spark.read.option("multiLine", "true").json(data_path)
    df.printSchema()

    # Preprocessing
    clean_df = preprocessing.clean_text(df, input_col="full_text")
    tokenized_df = preprocessing.tokenize(clean_df)
    filtered_df = preprocessing.remove_stopwords(tokenized_df)

    # Run enrichment pipeline
    enriched_df = enrichment.run_enrichment(filtered_df)

    # Create a summary column that combines all topic information
    enriched_df = enriched_df.withColumn("topic_summary",
        concat_ws(" | ",
            col("topic_with_lda"),
            col("topic_with_ner"),
            col("department"),
            col("category")
        )
    )

    # Show results
    print("=== ENRICHED DATA SAMPLE ===")
    enriched_df.select("title", "full_text", "url", "topic_with_lda", "topic_with_ner", "department",
                        "category", "person_names", "locations").show(5, truncate=False)

    print("=== DEPARTMENT DISTRIBUTION ===")
    enriched_df.groupBy("department").count().orderBy("count", ascending=False).show()

    print("=== NER TOPIC DISTRIBUTION ===")
    enriched_df.groupBy("topic_with_ner").count().orderBy("count", ascending=False).show()

    spark.stop()


def train_and_save_model(data_path, model_path):
    """Load data, preprocess, train a sentiment analysis model, and save it"""
    # Create Spark session
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    print("✅ Spark session started.")
    
    # Load data
    df = spark.read.option("multiLine", "true").json(data_path)
    print(f"✅ Loaded input data: {df.count()} rows")

    # Preprocess and enrich data
    preprocessing = TextPreprocessing(spark)
    enrichment = DataEnrichment(spark)

    clean_df = preprocessing.clean_text(df, input_col="full_text")
    tokenized_df = preprocessing.tokenize(clean_df)
    filtered_df = preprocessing.remove_stopwords(tokenized_df)
    enriched_df = enrichment.run_enrichment(filtered_df)

    # Create a summary column that combines all topic information
    enriched_df = enriched_df.withColumn("topic_summary",
        concat_ws(" | ",
            col("topic_with_lda"),
            col("topic_with_ner"),
            col("department"),
            col("category")
        )
    )

    # Build and train sentiment model
    pipeline = SentimentAnalysisPipeline().get_pipeline()
    model = pipeline.fit(enriched_df)
    print("✅ Model trained.")

    # Save the model
    model.write().overwrite().save(model_path)
    print(f"✅ Model saved to: {model_path}")

    spark.stop()


if __name__ == "__main__":
    DATA_PATH = "hdfs://localhost:9000/user/student/data.json"
    MODEL_PATH = "hdfs://localhost:9000/user/student/sentiment_model"

    train_and_evaluate(DATA_PATH)  # Only prints sample and stats
    train_and_save_model(DATA_PATH, MODEL_PATH)  # Only saves model to HDFS