import smtplib
from email.message import EmailMessage

SMTP_HOST = "smtp-relay.brevo.com"
SMTP_PORT = 587
EMAIL_USER = "piereluigi7@gmail.com"
EMAIL_PASS = "qlks dtgv iqag prtr"
FROM_EMAIL = EMAIL_USER


def send_email(to_email, otp):
    if not EMAIL_USER or not EMAIL_PASS:
        raise RuntimeError("EMAIL_USER or EMAIL_PASS not set")

    msg = EmailMessage()
    msg.set_content(f"Your OTP is: {otp}")
    msg["Subject"] = "Your Door Access OTP"
    msg["From"] = EMAIL_USER
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            print("✅ EMAIL ACTUALLY SENT")

    except Exception as e:
        print("❌ SMTP ERROR:", e)
        raise