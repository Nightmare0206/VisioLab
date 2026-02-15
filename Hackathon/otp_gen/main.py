from.otp_service import generate_and_send_otp
from.otp_verify import verify_otp

email = input("Enter email: ")  # need to connect to webcam so we can retrieve theri email 
generate_and_send_otp(email)

user_otp = input("Enter OTP sent to email: ")

if verify_otp(email, user_otp):
    print("✅ OTP VERIFIED")
else:
    print("❌ INVALID OR EXPIRED OTP")