# Author: Wong Xiao Yuan

# neo4j_graph_builder.py

from neo4j import GraphDatabase

class Neo4jGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    def close(self):
        self.driver.close()

    # ─── Nodes ────────────────────────────────────────────
    @staticmethod
    def _create_article_node(tx, art):
        tx.run("""
            MERGE (a:Article {title: $title})
            SET a.url = $url,
                a.full_text = $full_text,
                a.publishDate = $publishDate
        """, art)

    @staticmethod
    def _create_publisher_node(tx, company):
        tx.run("MERGE (:Company {name: $company})", {"company": company})

    @staticmethod
    def _create_sentence_node(tx, art):
        tx.run("""
            MERGE (s:Sentence {id: $sentenceId})
            SET s.text = $sentence
        """, {"sentenceId": art["sentenceId"], "sentence": art["sentence"]})

    @staticmethod
    def _create_sentiment_node(tx, sentiment):
        tx.run("MERGE (:Sentiment {label: $sentiment})", {"sentiment": sentiment})

    @staticmethod
    def _create_topic_node(tx, topic):
        tx.run("MERGE (:Topic {keyword: $topic})", {"topic": topic})

    @staticmethod
    def _create_category_node(tx, category):
        tx.run("MERGE (:Category {name: $category})", {"category": category})

    @staticmethod
    def _create_department_node(tx, department):
        tx.run("MERGE (:Department {name: $department})", {"department": department})

    # ─── Relationships ──────────────────────────────────
    @staticmethod
    def _create_published_by_rel(tx, title, company):
        tx.run("""
            MATCH (a:Article {title: $title}), (c:Company {name: $company})
            MERGE (a)-[:PUBLISHED_BY]->(c)
        """, {"title": title, "company": company})

    @staticmethod
    def _create_has_sentence_rel(tx, title, sentenceId):
        tx.run("""
            MATCH (a:Article {title: $title}), (s:Sentence {id: $sentenceId})
            MERGE (a)-[:HAS_SENTENCE]->(s)
        """, {"title": title, "sentenceId": sentenceId})

    @staticmethod
    def _create_has_sentiment_rel(tx, sentenceId, sentiment):
        tx.run("""
            MATCH (s:Sentence {id: $sentenceId}), (sent:Sentiment {label: $sentiment})
            MERGE (s)-[:HAS_SENTIMENT]->(sent)
        """, {"sentenceId": sentenceId, "sentiment": sentiment})

    @staticmethod
    def _create_has_topic_rel(tx, title, topic):
        tx.run("""
            MATCH (a:Article {title: $title}), (t:Topic {keyword: $topic})
            MERGE (a)-[:HAS_TOPIC]->(t)
        """, {"title": title, "topic": topic})

    @staticmethod
    def _create_belongs_to_category_rel(tx, title, category):
        tx.run("""
            MATCH (a:Article {title: $title}), (c:Category {name: $category})
            MERGE (a)-[:BELONGS_TO_CATEGORY]->(c)
        """, {"title": title, "category": category})

    @staticmethod
    def _create_belongs_to_department_rel(tx, title, department):
        tx.run("""
            MATCH (a:Article {title: $title}), (d:Department {name: $department})
            MERGE (a)-[:BELONGS_TO_DEPARTMENT]->(d)
        """, {"title": title, "department": department})
