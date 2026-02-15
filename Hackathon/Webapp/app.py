from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from db import insert_face_image  # DB function that stores image_path + email

app = Flask(__name__)

# Folder where face images will be stored locally
UPLOAD_FOLDER = "face_database"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload():
    """
    Admin uploads a face image and an email.
    Image is stored locally, and its path + email are saved in PostgreSQL.
    """

    if request.method == "POST":
        file = request.files.get("file")
        email = request.form.get("email")

        # Basic validation
        if not file or file.filename == "" or not email:
            return "Image and email are required ❌"

        # Secure the filename
        filename = secure_filename(file.filename)

        # Full local path where image will be saved
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save image locally
        file.save(image_path)

        # Store image path and email in database
        insert_face_image(image_path, email)

        return "✅ Face image and email stored successfully"

    # Show upload page
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)