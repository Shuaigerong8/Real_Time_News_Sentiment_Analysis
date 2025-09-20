#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Nelson Tay Kai Rong
import pandas as pd

class DataQuery:
    def __init__(self, articles_df: pd.DataFrame, enrichment_df: pd.DataFrame, model_df: pd.DataFrame):
        self.articles_df = articles_df
        self.enrichment_df = enrichment_df
        self.model_df = model_df

        self.df = pd.merge(articles_df, enrichment_df, on="_id", how="inner")
        self.df = pd.merge(self.df, model_df, on="_id", how="inner")

    # === Basic Queries ===
    def by_sentiment(self, sentiment: str) -> pd.DataFrame:
        return self.df[self.df['predictedLabel'] == sentiment]


    def by_company(self, company: str) -> pd.DataFrame:
        return self.df[self.df['newsCompany'] == company]

    def by_company_sentiment(self, company: str, sentiment: str) -> pd.DataFrame:
        return self.df[(self.df['newsCompany'] == company) & (self.df['predictedLabel'] == sentiment)]

    def by_keyword(self, keyword: str) -> pd.DataFrame:
        return self.df[self.df['full_text'].str.contains(keyword, case=False, na=False)]

    def by_keywords_all(self, keywords: list) -> pd.DataFrame:
        mask = pd.Series(True, index=self.df.index)
        for kw in keywords:
            mask &= self.df['full_text'].str.contains(kw, case=False, na=False)
        return self.df[mask]

    def by_topic_lda(self, topic: str) -> pd.DataFrame:
        return self.df[self.df['topic_with_lda'].str.startswith(topic, na=False)]

    def by_topic_ner(self, topic: str) -> pd.DataFrame:
        return self.df[self.df['topic_with_ner'] == topic]

    def by_category(self, category: str) -> pd.DataFrame:
        return self.df[self.df['category'] == category]

    def by_department(self, department: str) -> pd.DataFrame:
        return self.df[self.df['department'] == department]

    def fuzzy_search_in_title(self, keyword: str) -> pd.DataFrame:
        return self.df[self.df['title'].str.contains(keyword, case=False, na=False)]

    # === Advanced Analytical Queries ===
    def keyword_frequency_by_category(self, keyword: str) -> pd.Series:
        subset = self.by_keyword(keyword)
        return subset['category'].value_counts()

    def sentiment_distribution_by_company(self) -> pd.DataFrame:
        return self.df.groupby(['newsCompany', 'predictedLabel']).size().unstack().fillna(0)

    def top_keywords_by_sentiment(self, sentiment: str, top_n: int = 10) -> pd.Series:
        filtered = self.df[self.df['predictedLabel'] == sentiment]
        all_words = filtered['topic_summary'].str.split().explode()
        return all_words.value_counts().head(top_n)

    def company_sentiment_ratio(self, company: str) -> pd.Series:
        subset = self.df[self.df['newsCompany'] == company]
        return subset['predictedLabel'].value_counts(normalize=True)

    def count_articles_by_location(self) -> pd.Series:
        return self.df['locations'].explode().value_counts()

    def keyword_sentiment_summary(self, keyword: str) -> pd.Series:
        subset = self.by_keyword(keyword)
        return subset['predictedLabel'].value_counts()

    def department_wise_sentiment_count(self) -> pd.DataFrame:
        return self.df.groupby(['department', 'predictedLabel']).size().unstack().fillna(0)

    def top_lda_topics(self, top_n: int = 5) -> pd.Series:
        return self.df['topic_with_lda'].value_counts().head(top_n)

    def by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        df_copy = self.df.copy()
        df_copy['publishDate'] = pd.to_datetime(df_copy['publishDate'], errors='coerce')
        return df_copy[(df_copy['publishDate'] >= start_date) & (df_copy['publishDate'] <= end_date)]

    def by_topic_summary_keyword(self, keyword: str) -> pd.DataFrame:
        return self.df[self.df['topic_summary'].str.contains(keyword, case=False, na=False)]
    


# In[ ]:




