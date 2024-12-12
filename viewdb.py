import sqlite3


def view_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table_name in tables:
        print(f"Table: {table_name[0]}")
        cursor.execute(f"SELECT * FROM '{table_name[0]}'")
        rows = cursor.fetchall()

        # Get column names
        cursor.execute(f"PRAGMA table_info('{table_name[0]}')")
        columns = [col[1] for col in cursor.fetchall()]
        print("Columns:", columns)

        for row in rows:
            print(row)
        print("\n")

    conn.close()


if __name__ == "__main__":
    db_path = "llamantin.db"  # Change this to your database path
    view_db(db_path)
