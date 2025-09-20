# Author: Tan Rou Ming
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import col, lower, regexp_replace, explode, trim, split, udf
from pyspark.sql.types import StringType

nltk.download('vader_lexicon')

class SentenceSplitter(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="full_text", outputCol="sentence"):
        super().__init__()
        self.setParams(inputCol, outputCol)

    def setParams(self, inputCol=None, outputCol=None):
        if inputCol: self._set(inputCol=inputCol)
        if outputCol: self._set(outputCol=outputCol)
        return self

    def _transform(self, df):
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

    def _transform(self, df):
        return df.withColumn(self.getOutputCol(), lower(regexp_replace(col(self.getInputCol()), "[^a-zA-Z\\s]", " ")))

class VaderSentiment(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="clean_text", outputCol="label"):
        super().__init__()
        self.setParams(inputCol, outputCol)

    def setParams(self, inputCol=None, outputCol=None):
        if inputCol: self._set(inputCol=inputCol)
        if outputCol: self._set(outputCol=outputCol)
        return self

    def _transform(self, df):
        analyzer = SentimentIntensityAnalyzer()

        def get_sentiment(text):
            if not text:
                return "neutral"
            score = analyzer.polarity_scores(text)["compound"]
            return "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"

        sentiment_udf = udf(get_sentiment, StringType())
        return df.withColumn(self.getOutputCol(), sentiment_udf(col(self.getInputCol())))