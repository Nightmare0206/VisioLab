import hashlib
from datetime import datetime
from.db import conn



def verify_otp(email, user_input):
    otp_hash = hashlib.sha256(user_input.encode()).hexdigest()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, expires_at FROM otps
            WHERE email=%s AND otp_hash=%s AND is_used=FALSE
            ORDER BY id DESC LIMIT 1
            """,
            (email, otp_hash)
        )

        row = cur.fetchone()
        if not row:
            return False

        otp_id, expires_at = row
        if datetime.utcnow() > expires_at:
            return False

        cur.execute(
            "UPDATE otps SET is_used=TRUE WHERE id=%s",
            (otp_id,)
        )
        conn.commit()
        return True