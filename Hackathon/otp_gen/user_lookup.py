from.db import conn


def get_email_for_user(name):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT email FROM face_images WHERE name=%s",
            (name,)
        )
        row = cur.fetchone()
        return row[0] if row else None