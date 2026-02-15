import psycopg2

def get_connection():
    
    db_host = "database-3.cxxtqrexryzl.us-east-1.rds.amazonaws.com"
    db_name = "dbname"
    db_user = "postgres1"
    db_pass = 'programming'
    """
    Creates a connection to the AWS RDS PostgreSQL database.
    """
    return psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_pass
    )

def insert_face_image(s3_key, email):
    """
    Stores S3 image reference with associated email.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO face_images (s3_key, email)
        VALUES (%s, %s)
        """,
        (s3_key, email)
    )

    conn.commit()
    cursor.close()
    conn.close()