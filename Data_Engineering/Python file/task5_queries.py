# Author: Wong Xiao Yuan

#task5_queries.py

from neo4j import GraphDatabase
import pandas as pd
from tabulate import tabulate

class Neo4jAdvancedQueryRunner:
    """
    Encapsulates complex Cypher queries and returns results as pandas DataFrames.
    Also provides methods to print results as formatted tables.
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def _run(self, cypher: str, params: dict = None) -> pd.DataFrame:
        """Execute a Cypher query and return a pandas DataFrame."""
        with self.driver.session() as session:
            results = session.run(cypher, params or {})
            records = [record.data() for record in results]
        return pd.DataFrame(records)

    def sentiment_by_dept_category(self) -> pd.DataFrame:
        """Sentence-level sentiment counts per Department & Category."""
        cypher = """
        MATCH 
          (a:Article)-[:HAS_SENTENCE]->(sen:Sentence)-[:HAS_SENTIMENT]->(s:Sentiment),
          (a)-[:BELONGS_TO_DEPARTMENT]->(d:Department),
          (a)-[:BELONGS_TO_CATEGORY]->(c:Category)
        WITH 
          d.name     AS department, 
          c.name     AS category, 
          s.label    AS sentiment, 
          count(sen) AS sentence_count
        RETURN 
          department, category, sentiment, sentence_count
        ORDER BY 
          department, category, sentiment
        """
        return self._run(cypher)

    def company_sentiment_summary(self) -> pd.DataFrame:
        """
        For each Company, count the number of *sentences* by sentiment.
        """
        cypher = """
        MATCH (c:Company)<-[:PUBLISHED_BY]-(a:Article)
        MATCH (a)-[:HAS_SENTENCE]->(sen:Sentence)-[:HAS_SENTIMENT]->(s:Sentiment)
        WITH c.name AS company, s.label AS sentiment, count(DISTINCT sen) AS cnt
        RETURN company, sentiment, cnt
        ORDER BY company, sentiment
        """
        return self._run(cypher)

    def topic_cooccurrence_jaccard(self, limit: int = 10) -> pd.DataFrame:
        """
        Compute Jaccard similarity for each pair of topics:
        |A ∧ B| / |A ∪ B|, ordered by highest overlap.
        """
        cypher = f"""
        // build sets per topic
        CALL {{
          MATCH (a:Article)-[:HAS_TOPIC]->(t:Topic)
          WITH a, collect(t.keyword) AS topics
          RETURN collect({{article:a, topics:topics}}) AS data
        }}
        UNWIND data AS row
        UNWIND row.topics AS t1
        UNWIND row.topics AS t2
        WITH row.article AS a, t1, t2
        WHERE t1 < t2
        WITH t1 AS topicA, t2 AS topicB, count(DISTINCT a) AS intersect

        // get union size via two MATCH + WHERE filters, carrying both topicA/topicB
        CALL {{
          WITH topicA, topicB
          // collect all article IDs for topicA
          MATCH (aA:Article)-[:HAS_TOPIC]->(tA:Topic)
          WHERE tA.keyword = topicA
          WITH collect(id(aA)) AS A, topicA, topicB

          // collect all article IDs for topicB
          MATCH (bB:Article)-[:HAS_TOPIC]->(tB:Topic)
          WHERE tB.keyword = topicB
          WITH A, collect(id(bB)) AS B, topicA, topicB

          RETURN size(apoc.coll.union(A, B)) AS unionSize
        }}

        WITH topicA, topicB, intersect, unionSize,
             round(toFloat(intersect) / unionSize, 3) AS jaccard
        RETURN topicA, topicB, intersect, unionSize, jaccard
        ORDER BY jaccard DESC
        LIMIT {limit}
        """
        return self._run(cypher)

    def sentiment_for_keyword_detailed(self, keyword: str) -> pd.DataFrame:
        """
        For all sentences containing `keyword`, return per‐sentiment:
           – sentiment
           – sentenceCount   (number of sentences)
           – articleCount    (distinct articles)
           – pct             (percentage of sentences)
        """
        cypher = """
        // 1) get overall totals
        CALL {
          MATCH (a0:Article)-[:HAS_SENTENCE]->(s0:Sentence)-[:HAS_SENTIMENT]->(sent0:Sentiment)
          WHERE toLower(s0.text) CONTAINS toLower($kw)
          RETURN
            count(s0)            AS totalSentences,
            count(DISTINCT a0)   AS totalArticles
        }
        // 2) now group by sentiment
        MATCH (a:Article)-[:HAS_SENTENCE]->(s:Sentence)-[:HAS_SENTIMENT]->(sent:Sentiment)
        WHERE toLower(s.text) CONTAINS toLower($kw)
        WITH
          sent.label      AS sentiment,
          count(s)        AS sentenceCount,
          count(DISTINCT a) AS articleCount,
          totalSentences
        RETURN
          sentiment,
          sentenceCount,
          articleCount,
          round(100.0 * toFloat(sentenceCount) / totalSentences, 1) AS sentence_pct
        ORDER BY sentiment
        """
        return self._run(cypher, {"kw": keyword})

    def department_topic_popularity(self, top_n: int = 5) -> pd.DataFrame:
        """Top N topics for each department."""
        cypher = f"""
        MATCH (a:Article)-[:BELONGS_TO_DEPARTMENT]->(d:Department),
              (a)-[:HAS_TOPIC]->(t:Topic)
        WITH d.name AS department, t.keyword AS topic, count(a) AS cnt
        ORDER BY department, cnt DESC
        WITH department, collect({{topic: topic, count: cnt}})[0..{top_n}] AS topTopics
        RETURN department, topTopics;
        """
        return self._run(cypher)

    def weekly_summary(self) -> pd.DataFrame:
        """
        For each calendar week (starting Monday), show:
          • articleCount  – how many articles published that week
          • negative      – total negative sentences that week
          • neutral       – total neutral sentences that week
          • positive      – total positive sentences that week
        """
        cypher = """
        // 1) convert your ISO‑string into a Date, then truncate to week
        MATCH (a:Article)
        WHERE a.publishDate IS NOT NULL
        WITH date(datetime(a.publishDate)) AS pubDate, a
        WITH date.truncate('week', pubDate) AS weekStart, collect(a) AS arts

        // 2) unwind so we can count sentences+sentiment per article
        UNWIND arts AS art
        OPTIONAL MATCH
          (art)-[:HAS_SENTENCE]->(s:Sentence)-[:HAS_SENTIMENT]->(sent:Sentiment)

        // 3) aggregate by weekStart
        WITH
          weekStart,
          size(arts)                                    AS articleCount,
          sum(CASE sent.label WHEN 'negative' THEN 1 ELSE 0 END) AS negative,
          sum(CASE sent.label WHEN 'neutral'  THEN 1 ELSE 0 END) AS neutral,
          sum(CASE sent.label WHEN 'positive' THEN 1 ELSE 0 END) AS positive

        RETURN
          weekStart,
          articleCount,
          negative,
          neutral,
          positive
        ORDER BY weekStart
        """
        return self._run(cypher)

    def print_sentiment_by_dept_category_grouped(self):
        # 1) fetch & pivot
        df = self.sentiment_by_dept_category()
        df_wide = df.pivot_table(
            index=["department","category"],
            columns="sentiment",
            values="sentence_count",
            fill_value=0
        ).reset_index()

        # 2) cast numeric columns to int
        numeric_cols = [c for c in df_wide.columns if c not in ("department","category")]
        df_wide[numeric_cols] = df_wide[numeric_cols].astype(int)

        # 3) select & order columns
        cols = ["department","category","negative","neutral","positive"]
        cols = [c for c in cols if c in df_wide.columns]
        df_wide = df_wide[cols]

        # 4) prepare data & compute column widths
        data    = [list(r) for r in df_wide.itertuples(index=False,name=None)]
        headers = cols
        widths  = [
            max(len(str(v)) for v in [hdr] + [row[i] for row in data])
            for i, hdr in enumerate(headers)
        ]

        # 5) build separator and header line
        sep = "+" + "+".join("-"*(w+2) for w in widths) + "+"
        hdr = "| " + " | ".join(hdr.center(widths[i]) for i, hdr in enumerate(headers)) + " |"

        # 6) print title, header, initial separator
        print("\n\t       --- Sentiment by Department & Category ---")
        print(sep)
        print(hdr)
        print(sep)

        # 7) print each row, blanking repeated departments, and sep between dept blocks
        last_dept = None
        for dept, cat, neg, neu, pos in data:
            # if new department, no action; if same as last, blank it
            display_dept = dept if dept != last_dept else ""
            # when dept changes (and it's not the very first row), print a sep line
            if last_dept is not None and dept != last_dept:
                print(sep)
            # build the row
            cells = [
                display_dept.ljust(widths[0]),
                cat.ljust(widths[1]),
                str(neg).rjust(widths[2]),
                str(neu).rjust(widths[3]),
                str(pos).rjust(widths[4]),
            ]
            print("| " + " | ".join(cells) + " |")
            last_dept = dept

        # 8) print bottom border once more
        print(sep)

    def print_company_sentiment_summary(self):
        """
        Fetch company_sentiment_summary(), pivot it, compute percentages
        (with two decimal places), and print:
          company | negative | neg_% | neutral | neu_% | positive | pos_%
        """
        df = self.company_sentiment_summary()
        if df.empty:
            print("⚠️  No sentence‐level sentiment data found for any company.")
            return

        # Pivot to wide form
        df_wide = df.pivot_table(
            index="company",
            columns="sentiment",
            values="cnt",
            fill_value=0
        ).reset_index()

        # Keep only the sentiment columns we have
        cols = ["company", "negative", "neutral", "positive"]
        cols = [c for c in cols if c in df_wide.columns]
        df_wide = df_wide[cols]

        # Compute total sentences per company
        sentiment_cols = [c for c in ["negative","neutral","positive"] if c in df_wide.columns]
        df_wide["total"] = df_wide[sentiment_cols].sum(axis=1)

        # Compute percentage columns with 2 decimal places
        for c in sentiment_cols:
            pct_col = c + "_pct"
            df_wide[pct_col] = (100.0 * df_wide[c] / df_wide["total"]).round(2)

        # Reorder for display
        display_cols = []
        for c in ["negative","neutral","positive"]:
            if c in df_wide.columns:
                display_cols += [c, c + "_pct"]
        df_wide = df_wide[["company"] + display_cols]

        # Rename pct 列更友好
        df_wide = df_wide.rename(columns={
            "negative_pct": "neg_%", 
            "neutral_pct":  "neu_%", 
            "positive_pct": "pos_%"
        })

        # 打印
        print("\n\t\t--- Company Sentiment Summary (counts and % by Sentence) ---")
        print(tabulate(df_wide, headers="keys", tablefmt="psql", showindex=False))

    def print_topic_cooccurrence_jaccard(self, limit: int = 10):
        df = self.topic_cooccurrence_jaccard(limit)
        print(f"\n\t  --- Top {limit} Topic Pairs by Jaccard Similarity ---")
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

    def print_sentiment_for_keyword_detailed(self, keyword: str):
        """
        Print detailed sentiment for keyword, with a separator above the Total row.
        """
        df = self.sentiment_for_keyword_detailed(keyword)
        if df.empty:
            print(f"⚠️  No sentences found containing “{keyword}.”")
            return

        # Compute totals
        total_sentences = int(df["sentenceCount"].sum())
        total_articles  = int(df["articleCount"].sum())
        total_pct       = 100.0

        # Build the full list of rows, including Total at the end
        rows = df.to_dict(orient="records")
        rows.append({
            "sentiment":     "Total",
            "sentenceCount": total_sentences,
            "articleCount":  total_articles,
            "sentence_pct":  total_pct
        })

        # Determine column order and compute widths
        headers = ["sentiment", "sentenceCount", "articleCount", "sentence_pct"]
        # all values as strings for width calculation
        cols = [[str(r[h]) for r in rows] for h in headers]
        widths = [max(len(h), max(len(v) for v in col)) for h, col in zip(headers, cols)]

        # helper for drawing a sep line
        sep = "+" + "+".join("-"*(w+2) for w in widths) + "+"

        # header line
        hdr = "| " + " | ".join(headers[i].center(widths[i]) for i in range(len(headers))) + " |"

        # print header
        print(f"\n    --- Detailed sentiment for “{keyword}” (with Totals) ---")
        print(sep)
        print(hdr)
        print(sep)

        # print body rows
        last_idx = len(rows) - 1
        for idx, row in enumerate(rows):
            # before printing the Total row, draw a separator
            if idx == last_idx:
                print(sep)
            # prepare cell values
            cells = [
                str(row["sentiment"]).ljust(widths[0]),
                str(row["sentenceCount"]).rjust(widths[1]),
                str(row["articleCount"]).rjust(widths[2]),
                f"{row['sentence_pct']:.1f}".rjust(widths[3]),
            ]
            print("| " + " | ".join(cells) + " |")

        # final bottom border
        print(sep)
        
    def print_department_topic_popularity(self, top_n: int = 5):
        """
        Top N topics per department, printed in grouped table form:
          department | topic | count
        """
        df = self.department_topic_popularity(top_n)
        # df has columns: department, topTopics (list of dicts)
        # explode to one row per topic
        rows = []
        for rec in df.to_dict(orient="records"):
            dept = rec["department"]
            for t in rec["topTopics"]:
                rows.append((dept, t["topic"], t["count"]))

        if not rows:
            print("⚠️  No department–topic data found.")
            return

        # compute column widths
        headers = ["department", "topic", "count"]
        cols = list(zip(*rows))
        widths = [
            max(len(str(v)) for v in [headers[i]] + list(cols[i]))
            for i in range(3)
        ]

        sep = "+" + "+".join("-"*(w+2) for w in widths) + "+"
        hdr = "| " + " | ".join(headers[i].center(widths[i]) for i in range(3)) + " |"

        print(f"\n--- Top {top_n} Topics per Department ---")
        print(sep)
        print(hdr)
        print(sep)

        last_dept = None
        for dept, topic, cnt in rows:
            # blank repeat departments
            disp_dept = dept if dept != last_dept else ""
            # if new block, print separator
            if last_dept is not None and dept != last_dept:
                print(sep)
            line = "| " + disp_dept.ljust(widths[0]) \
                   + " | " + topic.ljust(widths[1]) \
                   + " | " + str(cnt).rjust(widths[2]) + " |"
            print(line)
            last_dept = dept

        print(sep)

    def print_weekly_summary(self):
        """
        Print the combined weekly table:
          weekStart | articleCount | negative | neutral | positive
        """
        df = self.weekly_summary()
        if df.empty:
            print("⚠️  No publishDate or sentiment data found.")
            return

        # reorder columns
        cols = ["weekStart","articleCount","negative","neutral","positive"]
        df = df[[c for c in cols if c in df.columns]]

        from tabulate import tabulate
        print("\n\t    --- Weekly Articles & Sentiment Summary ---")
        print(tabulate(
            df,
            headers="keys",
            tablefmt="psql",
            showindex=False
        ))

# Example standalone script usage:
if __name__ == "__main__":
    URI = "neo4j+s://97144963.databases.neo4j.io"
    USER = "neo4j"
    PASSWORD = "D3LgAxX8FIfix41we1XQ1iNzN6fGAftdjowFjeAJjbk"
    runner = Neo4jAdvancedQueryRunner(URI, USER, PASSWORD)

    runner.print_sentiment_by_dept_category_grouped()
    runner.print_company_sentiment_summary()
    runner.print_department_topic_popularity(top_n=5)
    runner.print_weekly_summary()
    runner.print_topic_cooccurrence_jaccard(limit=10)
    user_kw = input("\nEnter search keyword: ").strip()
    runner.print_sentiment_for_keyword_detailed(user_kw)
    
    runner.close()
