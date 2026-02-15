import psycopg2

db_host = "database-3.cxxtqrexryzl.us-east-1.rds.amazonaws.com"
db_name = "dbname"
db_user = "postgres1"
db_pass = 'programming'

def drop_old_table():
    """
    Deletes the old face_images table if it exists.
    """
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_pass
    )

    cursor = conn.cursor()

    # Drop old table
    cursor.execute("DROP TABLE IF EXISTS face_images;")

    conn.commit()
    cursor.close()
    conn.close()

    print("✅ Old table deleted")

if __name__ == "__main__":
    drop_old_table()