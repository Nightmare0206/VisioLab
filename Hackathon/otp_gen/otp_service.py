import pyotp
import hashlib
from.db import conn
from.email_service import send_email

OTP_EXPIRY_MINUTES = 5


def generate_and_send_otp(email):
    print("[DEBUG] generate_and_send_otp called for:", email)

    totp = pyotp.TOTP(
        pyotp.random_base32(),
        interval=OTP_EXPIRY_MINUTES * 60,
        digits=3
    )

    otp = totp.now()
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO otps (email, otp_hash, expires_at)
                VALUES (%s, %s, NOW() + (%s || ' minutes')::INTERVAL)
                """,
                (email, otp_hash, OTP_EXPIRY_MINUTES)
            )
            conn.commit()

    except Exception as e:
        print("❌ OTP DB insert failed:", e)
        conn.rollback()
        return

    print("✅ OTP stored in database")
    send_email(email, otp)
    print("✅ OTP email sent")