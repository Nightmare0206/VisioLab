from flask import Flask, request, jsonify
from otp_gen.otp_verify import verify_otp


def create_app():
    app = Flask(__name__)

    @app.route("/verify-otp", methods=["POST"])
    def verify_otp_api():
        data = request.json
        email = data.get("email")
        otp = data.get("otp")

        if not email or not otp:
            return jsonify({"success": False}), 400

        if verify_otp(email, otp):
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False}), 401

    return app