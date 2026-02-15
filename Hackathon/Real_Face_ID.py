import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis


class FacialRecognition:
    """
    MediaPipe FaceLandmarker (mesh + detection)
    + InsightFace ArcFace (recognition)
    """

    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_embeddings = []
        self.known_names = []

        # -------- MediaPipe FaceLandmarker --------
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

        # -------- InsightFace (ArcFace) --------
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
                else:
                    print(f"WARNING: Could not get embedding for {file}")

        print(f"Known faces: {len(self.known_names)}")

    # --------------------------------------------------

    def get_embedding_clean(self, frame):
        """
        Get embedding using InsightFace on FULL frame
        """
        # Use InsightFace on full frame
        faces = self.arcface.get(frame)

        if not faces:
            return None

        return faces[0].embedding

    # --------------------------------------------------

    def get_landmarks(self, frame):
        """
        Get MediaPipe landmarks for mesh drawing
        """
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
        """
        Draw face mesh on frame
        """
        h, w, _ = frame.shape

        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv.circle(frame, (x, y), 1, (0, 255, 255), -1)

        # Return bounding box
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        return (x1, y1, x2, y2)

    # --------------------------------------------------

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --------------------------------------------------

    def recognize(self, clean_frame, display_frame):
        """
        clean_frame = for embedding
        display_frame = for drawing
        """
        # Get embedding from clean frame (full frame)
        emb = self.get_embedding_clean(clean_frame)

        # Get landmarks for mesh
        landmarks = self.get_landmarks(display_frame)

        bbox = None
        if landmarks:
            bbox = self.draw_mesh(display_frame, landmarks)

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

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Keep a CLEAN copy for saving/embedding
            clean_frame = frame.copy()

            # Display frame (will have mesh drawn)
            display_frame = frame.copy()

            name, score, bbox, emb = self.recognize(clean_frame, display_frame)

            # Draw bounding box
            if bbox:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                cv.putText(
                    display_frame, name, (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )

            # Confidence display
            cv.putText(
                display_frame, f"Name: {name}", (20, 40),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            cv.putText(
                display_frame, f"Confidence: {score:.2f}", (20, 80),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            cv.putText(
                display_frame, f"Match: {score * 100:.1f}%", (20, 120),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            # Face detected indicator
            face_status = "Face: DETECTED" if emb is not None else "Face: NONE"
            face_color = (0, 255, 0) if emb is not None else (0, 0, 255)
            cv.putText(
                display_frame, face_status, (20, 160),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2
            )

            if score > 0.40:
                status = "RECOGNIZED"
                status_color = (0, 255, 0)
            else:
                status = "NOT RECOGNIZED"
                status_color = (0, 0, 255)

            cv.putText(
                display_frame, status, (20, 200),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
            )

            cv.imshow("Face Recognition", display_frame)

            key = cv.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key in (ord('s'), ord('S')):
                # Get embedding from clean frame
                emb = self.get_embedding_clean(clean_frame)

                if emb is not None:
                    face_name = input("Enter name: ").strip()
                    if face_name:
                        # Save CLEAN frame
                        path = os.path.join(self.known_faces_dir, f"{face_name}.jpg")
                        cv.imwrite(path, clean_frame)
                        print(f"Saved: {path}")

                        # Add to memory
                        self.known_embeddings.append(emb)
                        self.known_names.append(face_name)
                        print(f"Added {face_name} to recognition")
                else:
                    print("No face detected - cannot save")

        cam.release()
        cv.destroyAllWindows()


# =====================
# RUN APPLICATION
# =====================
if __name__ == "__main__":
    app = FacialRecognition()
    app.open_camera()