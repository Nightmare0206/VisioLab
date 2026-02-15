from datetime import datetime
from.db import conn


def insert_user(name: str, email: str):
    s3_key = f"faces/{name}/{datetime.utcnow().isoformat()}.jpg"

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_images (name, email, s3_key)
                VALUES (%s, %s, %s)
                ON CONFLICT (name)
                DO UPDATE SET email = EXCLUDED.email
                """,
                (name, email, s3_key)
            )
            conn.commit()
            print(f"✅ Saved user: {name}, S3 key: {s3_key}")

    except Exception as e:
        conn.rollback()
        print("❌ Database insert failed:", e)