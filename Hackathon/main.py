import threading
from face_detection import FacialRecognition
from otp_gen.user_lookup import get_email_for_user
from otp_gen.otp_service import generate_and_send_otp
from otp_api import create_app


def run_otp_api():
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)


def on_face_recognized(name):
    print(f"[MAIN] Recognized: {name}")

    email = get_email_for_user(name)
    if not email:
        print("No email found")
        return

    generate_and_send_otp(email)
    print(f"OTP sent to {email}")


if __name__ == "__main__":
    # ✅ Start OTP API in background thread
    api_thread = threading.Thread(target=run_otp_api, daemon=True)
    api_thread.start()

    print("✅ OTP verification API running on port 5000")

    # ✅ Start face recognition
    app = FacialRecognition(on_recognized=on_face_recognized)
    app.open_camera()