#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Nelson Tay Kai Rong
# === Design Analytical Queries ===
import pandas as pd
import matplotlib.pyplot as plt

def styled_location_article_count(df: pd.Series):
    if df.empty:
        print("No location data available.")
        return
    df = df.head(10).to_frame("Article Count")
    df.index.name = "Location"
    print("\n📍 Top Locations by Article Count:\n")
    print(df)
    df.plot(kind='barh', legend=False)
    plt.title("Location Article Count")
    plt.xlabel("Count")
    plt.ylabel("Location")
    plt.tight_layout()
    plt.show()

def styled_keyword_sentiment_summary(df: pd.Series, keyword: str):
    if df.empty:
        print(f"No sentiment data found for keyword: {keyword}")
        return
    print(f"\n📝 Sentiment Distribution for Keyword: '{keyword}'\n")
    print(df.to_frame("Count"))
    df.plot(kind='bar', legend=False)
    plt.title(f"Sentiment Summary for '{keyword}'")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def styled_department_sentiment_count(df: pd.DataFrame):
    if df.empty:
        print("No department sentiment data.")
        return
    print("\n🏢 Department-wise Sentiment Count:\n")
    print(df)
    df.plot(kind='bar', stacked=True)
    plt.title("Sentiment by Department")
    plt.xlabel("Department")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def styled_top_lda_topics(df: pd.Series):
    if df.empty:
        print("No LDA topics found.")
        return
    df = df.to_frame("Count")
    df.index.name = "Topic"
    print("\n📚 Top LDA Topics:\n")
    print(df)
    df.plot(kind='barh', legend=False)
    plt.title("Top LDA Topics")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:




