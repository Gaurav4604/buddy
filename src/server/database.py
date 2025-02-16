import os
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.sql as sql

# Environment variables
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_PORT = os.getenv("port_postgres", "5432")


def get_conn_str(database: str) -> str:
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{database}"


def create_database_if_not_exists(db_name: str):
    """
    Connect to the default database (usually 'postgres') and create the target
    database if it doesn't already exist.
    """
    conn = psycopg2.connect(get_conn_str("postgres"), connect_timeout=1)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if cursor.fetchone() is None:
        # Note: Postgres does not support "IF NOT EXISTS" for CREATE DATABASE,
        # so we manually check before creating.
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"Database {db_name} created.")
    else:
        print(f"Database {db_name} already exists.")
    cursor.close()
    conn.close()


# Ensure the 'globals' database exists
create_database_if_not_exists("globals")


class GlobalsDB:
    def __init__(self):
        self.db_name = "globals"
        self.conn = psycopg2.connect(get_conn_str(self.db_name), connect_timeout=1)
        self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Create the topics table (1st table)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                topic_name TEXT PRIMARY KEY,
                uuid UUID DEFAULT gen_random_uuid()
            );
            """
        )
        # Create the topic content table (2nd table)
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS topic_content (
                topic_name TEXT,
                chapter_num INT,
                summary TEXT,
                tags TEXT,
                PRIMARY KEY (topic_name, chapter_num)
            );
            """
        )

    def fetch_all_unique_topics(self):
        """Returns a list of all topics with their topic_name and uuid."""
        self.cursor.execute("SELECT topic_name, uuid FROM topics;")
        rows = self.cursor.fetchall()
        # Convert each row into a dictionary; convert uuid to string if needed.
        return [{"topic_name": row[0], "uuid": str(row[1])} for row in rows]

    def fetch_topic_chapters(self, topic_name: str):
        """
        Returns an array of dictionaries containing chapter_num, summary, and tags (as a list)
        for the given topic. If no topic content exists, returns an empty list.
        """
        self.cursor.execute(
            "SELECT chapter_num, summary, tags FROM topic_content WHERE topic_name = %s ORDER BY chapter_num;",
            (topic_name,),
        )
        rows = self.cursor.fetchall()
        return [
            {
                "chapter_num": row[0],
                "summary": row[1],
                "tags": [tag.strip() for tag in row[2].split(",")] if row[2] else [],
            }
            for row in rows
        ]

    def add_topic(self, topic_name: str):
        """
        Adds a new topic to the topics table. If the topic already exists,
        nothing happens.
        """
        try:
            self.cursor.execute(
                "INSERT INTO topics (topic_name) VALUES (%s) ON CONFLICT (topic_name) DO NOTHING;",
                (topic_name,),
            )
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print("Error adding topic:", e)
            return False

    def add_topic_content(
        self, topic_name: str, chapter_num: int, summary: str, tags: list[str]
    ):
        """
        Inserts or updates a chapter's summary and tags for a given topic.
        Tags is now expected as a list of strings.
        """
        try:
            # Join the list of tags into a comma-separated string
            joined_tags = ", ".join(tags)
            self.cursor.execute(
                """
                INSERT INTO topic_content (topic_name, chapter_num, summary, tags)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (topic_name, chapter_num)
                DO UPDATE SET summary = EXCLUDED.summary, tags = EXCLUDED.tags;
                """,
                (topic_name, chapter_num, summary, joined_tags),
            )
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print("Error adding topic content:", e)
            return False


# Example usage:
if __name__ == "__main__":
    db = GlobalsDB()
    # Add a topic. When fetching chapters for a topic that has no content,
    # an empty list is returned.
    db.add_topic("automata")
    print("Topics:", db.fetch_all_unique_topics())
    print(
        "Topic chapters (before adding content):", db.fetch_topic_chapters("automata")
    )

    # Add topic content for a chapter
    db.add_topic_content("automata", 1, "Summary for chapter 1", "tag1, tag2")
    print("Topic chapters (after adding content):", db.fetch_topic_chapters("automata"))
