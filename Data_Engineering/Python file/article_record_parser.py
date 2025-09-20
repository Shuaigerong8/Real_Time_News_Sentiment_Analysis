# Author: Wong Xiao Yuan

# article_record_parser.py

import re
import hashlib

class ArticleRecordParser:
    @staticmethod
    def parse_row(row):
        # extract LDA topics
        raw = row.topic_with_lda or ""
        body = re.sub(r"^Topic\s*\d+:\s*", "", raw)
        topics = [t.strip() for t in body.split(",") if t.strip()]

        # be 100% explicit about None → ""
        sentence_text = row.sentence if row.sentence is not None else ""
        # deterministic ID
        unique = f"{row.title}|{sentence_text}"
        sentence_id = hashlib.md5(unique.encode("utf-8")).hexdigest()

        # 3) build the dict with exactly your available columns
        return {
            "title":        row.title,
            "url":          row.url,
            "publishDate":  row.publishDate,
            "full_text":    row.full_text,
            "sentence":     sentence_text,
            "sentenceId":   sentence_id,
            "newsCompany":  row.newsCompany,
            "sentiment":    getattr(row, "predictedLabel", None),
            "topics":       topics,
            "category":     getattr(row, "category", None),
            "department":   getattr(row, "department", None)
        }
