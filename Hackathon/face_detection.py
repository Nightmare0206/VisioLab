import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis
from otp_gen.user_insert import insert_user


class FacialRecognition:
    """
    Facial recognition system using:
    - MediaPipe FaceLandmarker for face detection + landmarks (mesh)
    - InsightFace ArcFace for face embeddings and recognition
    """

    def __init__(self, known_faces_dir="known_faces", on_recognized=None):
        self.known_faces_dir = known_faces_dir
        self.on_recognized = on_recognized
        self.last_recognized = None

        self.known_embeddings = []
        self.known_names = []

        # ==============================
        # MediaPipe FaceLandmarker setup
        # ==============================
        base_options = python.BaseOptions(
            model_asset_path="face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=3
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

        # ==============================
        # InsightFace ArcFace setup
        # ==============================
        self.arcface = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.arcface.prepare(ctx_id=0, det_size=(640, 640))

        self.load_known_faces()

    # --------------------------------------------------

    def load_known_faces(self):
        os.makedirs(self.known_faces_dir, exist_ok=True)

        for file in os.listdir(self.known_faces_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.known_faces_dir, file)
                image = cv.imread(path)

                emb = self.get_embedding_clean(image)

                if emb is not None:
                    self.known_embeddings.append(emb)
                    self.known_names.append(os.path.splitext(file)[0])
                    print(f"Loaded: {file}")

        print(f"Known faces: {len(self.known_names)}")

    # --------------------------------------------------

    def get_embedding_clean(self, frame):
        faces = self.arcface.get(frame)
        if not faces:
            return None
        return faces[0].embedding

    # --------------------------------------------------

    def get_landmarks(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    # --------------------------------------------------

    def draw_mesh(self, frame, landmarks):
        h, w, _ = frame.shape

        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv.circle(frame, (x, y), 1, (0, 255, 255), -1)

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        return (
            int(min(xs) * w),
            int(min(ys) * h),
            int(max(xs) * w),
            int(max(ys) * h)
        )

    # --------------------------------------------------

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --------------------------------------------------

    def recognize(self, clean_frame, display_frame):
        emb = self.get_embedding_clean(clean_frame)
        landmarks = self.get_landmarks(display_frame)

        bbox = self.draw_mesh(display_frame, landmarks) if landmarks else None

        name = "Unknown"
        score = 0.0

        if emb is not None and self.known_embeddings:
            scores = [
                self.cosine_similarity(emb, known)
                for known in self.known_embeddings
            ]

            best = np.argmax(scores)
            score = scores[best]

            if score > 0.40:
                name = self.known_names[best]

        return name, score, bbox, emb

    # --------------------------------------------------

    def open_camera(self):
        cam = cv.VideoCapture(0)

        if not cam.isOpened():
            print("ERROR: Cannot open camera")
            return

        print("\n=== CONTROLS ===")
        print("ESC - Quit")
        print("S   - Save face")
        print("================\n")

        recognized_at = None
        exit_delay = 5  # seconds

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            clean_frame = frame.copy()
            display_frame = frame.copy()

            name, score, bbox, emb = self.recognize(clean_frame, display_frame)

            # ✅ Start timer after recognition
            if score > 0.40 and recognized_at is None:
                recognized_at = time.time()
                print(f"✅ Recognized {name}, closing in {exit_delay}s")

                # ✅ CALL BACK INTO main.py
                if self.on_recognized:
                    self.on_recognized(name)

            if bbox:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(
                    display_frame, name, (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )

            cv.putText(display_frame, f"Name: {name}", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.putText(display_frame, f"Confidence: {score:.2f}", (20, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            status = "RECOGNIZED" if score > 0.40 else "NOT RECOGNIZED"
            status_color = (0, 255, 0) if score > 0.40 else (0, 0, 255)
            cv.putText(display_frame, status, (20, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv.imshow("Face Recognition", display_frame)
            key = cv.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key in (ord('s'), ord('S')):
                emb = self.get_embedding_clean(clean_frame)
                if emb is not None:
                    face_name = input("Enter name: ").strip()
                    email = input("Enter email: ").strip()

                    if face_name and email:
                        insert_user(face_name, email)

                    if face_name:
                        path = os.path.join(self.known_faces_dir, f"{face_name}.jpg")
                        cv.imwrite(path, clean_frame)
                        self.known_embeddings.append(emb)
                        self.known_names.append(face_name)

                        recognized_at = time.time()
                        print(f"✅ Face added, closing in {exit_delay}s")
                else:
                    print("No face detected - cannot save")

            # ✅ Exit after 5 seconds
            if recognized_at and (time.time() - recognized_at >= exit_delay):
                print("⏱ Closing camera")
                break

        cam.release()
        cv.destroyAllWindows()


# =====================
# RUN APPLICATION
# =====================
if __name__ == "__main__":
    app = FacialRecognition()
    app.open_camera()