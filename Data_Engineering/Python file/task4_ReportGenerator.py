#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Nelson Tay Kai Rong
import pandas as pd
import matplotlib.pyplot as plt

class ReportGenerator:
    def __init__(self, articles_df: pd.DataFrame, enrichment_df: pd.DataFrame, model_df: pd.DataFrame):
        df = pd.merge(articles_df, enrichment_df, on="_id", how="inner")
        df = pd.merge(df, model_df, on="_id", how="inner")
        self.df = df

    def report_sentiment_distribution(self):
        counts = self.df['predictedLabel'].value_counts()
        if counts.empty:
            print("No sentiment data.")
            return
        counts.plot(kind='bar')
        plt.title("Overall Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


    def report_count_by_company(self):
        counts = self.df['newsCompany'].value_counts()
        if counts.empty:
            print("No company data to plot.")
            return
        counts.plot(kind='bar')
        plt.title('Article Count by News Company')
        plt.xlabel('News Company')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def report_topics_lda_distribution(self):
        counts = self.df['topic_with_lda'].value_counts()
        if counts.empty:
            print("No LDA topic data to plot.")
            return
        counts.plot(kind='barh')
        plt.title('LDA Topic Distribution')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.show()

    def report_topics_ner_distribution(self):
        counts = self.df['topic_with_ner'].value_counts()
        if counts.empty:
            print("No NER topic data to plot.")
            return
        counts.plot(kind='barh')
        plt.title('NER Topic Distribution')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.show()

    def report_category_distribution(self):
        counts = self.df['category'].value_counts()
        if counts.empty:
            print("No category data to plot.")
            return
        counts.plot(kind='barh')
        plt.title('Category Distribution')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.show()

    def report_department_distribution(self):
        counts = self.df['department'].value_counts()
        if counts.empty:
            print("No department data to plot.")
            return
        counts.plot(kind='barh')
        plt.title('Department Distribution')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.show()

    def report_monthly_article_count(self):
        df_copy = self.df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['publishDate'], errors='coerce').dt.to_period('M')
        monthly_counts = df_copy.groupby('month').size()
        print("\n🗓 Monthly Article Count:\n")
        print(monthly_counts.to_frame("count"))
        monthly_counts.plot(kind='bar')
        plt.title("Monthly Article Count")
        plt.xlabel("Month")
        plt.ylabel("Articles")
        plt.tight_layout()
        plt.show()

    def report_top_companies_by_sentiment(self, sentiment='positive', top_n=5):
        filtered = self.df[self.df['predictedLabel'] == sentiment]
        top_companies = filtered['newsCompany'].value_counts().head(top_n)
        print(f"\n🏆 Top {top_n} Companies by '{sentiment}' Sentiment:\n")
        print(top_companies.to_frame("count"))
        top_companies.plot(kind='bar')
        plt.title(f"Top {top_n} Companies with {sentiment.capitalize()} Sentiment")
        plt.xlabel("News Company")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def report_category_sentiment_pivot(self):
        pivot = self.df.pivot_table(index='category', columns='predictedLabel', aggfunc='size', fill_value=0)
        print("\n📊 Category vs Sentiment Distribution:\n")
        print(pivot)
        pivot.plot(kind='bar', stacked=True)
        plt.title("Category Sentiment Distribution")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def report_topic_distribution_by_company(self):
        pivot = self.df.pivot_table(index='newsCompany', columns='topic_with_lda', aggfunc='size', fill_value=0)
        print("\n📚 Topic Distribution by Company:\n")
        print(pivot)
        ax = pivot.plot(kind='bar', stacked=True, figsize=(12, 6))
    
        # Move legend outside right
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.title("LDA Topics by News Company")
        plt.xlabel("News Company")
        plt.ylabel("Topic Count")
        plt.tight_layout()
        plt.show()



# In[ ]:




