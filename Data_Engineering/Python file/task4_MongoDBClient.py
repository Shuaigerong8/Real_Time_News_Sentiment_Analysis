#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Nelson Tay Kai Rong
from pymongo import MongoClient
from bson import ObjectId
import numpy as np

class MongoDBClient:
    def __init__(self, uri, db_name):
        """
        Initialize MongoDB connection with access to all 3 collections.
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.articles = self.db['articles']
        self.enrichment_info = self.db['enrichment_info']
        self.model_outputs = self.db['model_outputs']

    def _clean_ndarrays(self, obj):
        """
        Recursively convert NumPy arrays to plain Python lists.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._clean_ndarrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_ndarrays(v) for v in obj]
        else:
            return obj

    def _split_record(self, record):
        """
        Split a full record into 3 components for each collection.
        """
        record = self._clean_ndarrays(record)
        _id = record.get('_id', ObjectId())

        article_doc = {
            '_id': _id,
            'title': record.get('title'),
            'url': record.get('url'),
            'publishDate': record.get('publishDate'),
            'newsCompany': record.get('newsCompany'),
            'full_text': record.get('full_text'),
            'sentence': record.get('sentence')
        }

        enrichment_doc = {
            '_id': _id,
            'topic_summary': record.get('topic_summary'),
            'topic_with_lda': record.get('topic_with_lda'),
            'topic_with_ner': record.get('topic_with_ner'),
            'department': record.get('department'),
            'category': record.get('category'),
            'locations': record.get('locations')
        }

        model_output_doc = {
            '_id': _id,
            'predictedLabel': record.get('predictedLabel')
        }

        return article_doc, enrichment_doc, model_output_doc

    def insert_split_documents(self, records: list[dict]):
        """
        Split and insert each record into the corresponding collections.
        """
        articles = []
        enrichments = []
        outputs = []

        for record in records:
            article, enrich, output = self._split_record(record)
            articles.append(article)
            enrichments.append(enrich)
            outputs.append(output)

        self.articles.insert_many(articles)
        self.enrichment_info.insert_many(enrichments)
        self.model_outputs.insert_many(outputs)

    def delete_all_documents(self):
        """
        Clear all three collections.
        """
        self.articles.delete_many({})
        self.enrichment_info.delete_many({})
        self.model_outputs.delete_many({})

