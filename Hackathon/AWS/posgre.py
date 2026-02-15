import psycopg2

db_host = "database-3.cxxtqrexryzl.us-east-1.rds.amazonaws.com"
db_name = "dbname"
db_user = "postgres1"
db_pass = 'programming'

def create_table():
    """
    Creates a new table linking face images to emails.
    """
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_pass
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_images (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            s3_key TEXT NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT face_images_unique_names UNIQUE (name)
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            otp_hash TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            is_used BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

    cursor.close()
    conn.close()

    print("✅ New email-based table created")

if __name__ == "__main__":
    create_table()