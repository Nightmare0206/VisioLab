import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis


class FacialRecognition:
    """
    Facial recognition system using:
    - MediaPipe FaceLandmarker for face detection + landmarks (mesh)
    - InsightFace ArcFace for face embeddings and recognition
    """

        
    def __init__(self, known_faces_dir="known_faces"):
        # Directory where known face images are stored
        self.known_faces_dir = known_faces_dir

        # In-memory storage of known face embeddings and labels (names)
        self.known_embeddings = []
        self.known_names = []

        # --------------------------------------------------
        # Current state (for return methods)
        # --------------------------------------------------
        self.face_detected = False
        self.is_recognized_flag = False
        self.current_name = "Unknown"
        self.current_score = 0.0

        # ==============================
        # MediaPipe FaceLandmarker setup
        # ==============================

        # Load MediaPipe model file
        base_options = python.BaseOptions(
            model_asset_path="face_landmarker.task"
        )

        # Configure face landmark detection
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=3  # max number of faces to detect per frame
        )

        # Create MediaPipe face detector
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # ==============================
        # InsightFace ArcFace setup
        # ==============================

        # Initialize ArcFace model for embeddings
        self.arcface = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        # Prepare model (CPU inference, detection resolution)
        self.arcface.prepare(ctx_id=0, det_size=(640, 640))

        # Load known faces from disk
        self.load_known_faces()

    # --------------------------------------------------
    # RETURN METHODS
    # --------------------------------------------------

    def is_face_detected(self):
        """
        Returns True if a face is currently detected
        """
        return self.face_detected

    def is_recognized(self):
        """
        Returns True if face is detected AND recognized
        """
        return self.is_recognized_flag

    def get_name(self):
        """
        Returns the name of the currently detected person
        """
        return self.current_name

    def get_score(self):
        """
        Returns the confidence score (0.0 - 1.0)
        """
        return self.current_score

    def get_result(self):
        """
        Returns full detection result as dictionary
        """
        return {
            "face_detected": self.face_detected,
            "is_recognized": self.is_recognized_flag,
            "name": self.current_name,
            "score": self.current_score
        }

    def check_frame(self, frame):
        """
        Check a single frame for face recognition.
        Returns True if recognized, False otherwise.
        """
        emb = self.get_embedding_clean(frame)

        self.face_detected = emb is not None
        self.current_name = "Unknown"
        self.current_score = 0.0
        self.is_recognized_flag = False

        if emb is not None and self.known_embeddings:
            scores = [
                self.cosine_similarity(emb, known)
                for known in self.known_embeddings
            ]

            best = np.argmax(scores)
            self.current_score = scores[best]

            if self.current_score > 0.40:
                self.current_name = self.known_names[best]
                self.is_recognized_flag = True
                return True

        return False

    # --------------------------------------------------

    def load_known_faces(self):
        """
        Load all known face images from disk, extract embeddings,
        and store them in memory for later comparison.
        """
        os.makedirs(self.known_faces_dir, exist_ok=True)

        for file in os.listdir(self.known_faces_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.known_faces_dir, file)
                image = cv.imread(path)

                # Generate embedding for the stored image
                emb = self.get_embedding_clean(image)

                if emb is not None:
                    self.known_embeddings.append(emb)
                    # Use filename (without extension) as identity label
                    self.known_names.append(os.path.splitext(file)[0])
                    print(f"Loaded: {file}")
                else:
                    print(f"WARNING: Could not get embedding for {file}")

        print(f"Known faces: {len(self.known_names)}")

    # --------------------------------------------------

    def get_embedding_clean(self, frame):
        """
        Generate a face embedding using InsightFace ArcFace.

        Parameters:
        - frame: BGR image containing a face

        Returns:
        - 512-d embedding vector, or None if no face is detected
        """
        # Run InsightFace face detection + embedding extraction
        faces = self.arcface.get(frame)

        if not faces:
            return None

        # Use the first detected face
        return faces[0].embedding

    # --------------------------------------------------

    def get_landmarks(self, frame):
        """
        Detect facial landmarks using MediaPipe FaceLandmarker.

        Parameters:
        - frame: BGR image

        Returns:
        - List of landmarks for the first detected face, or None
        """
        # Convert OpenCV BGR frame to RGB
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Wrap image for MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Run landmark detection
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    # --------------------------------------------------

    def draw_mesh(self, frame, landmarks):
        """
        Draw facial mesh landmarks and compute a bounding box.

        Parameters:
        - frame: image to draw on
        - landmarks: MediaPipe facial landmarks

        Returns:
        - Bounding box (x1, y1, x2, y2)
        """
        h, w, _ = frame.shape

        # Draw each landmark point
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv.circle(frame, (x, y), 1, (0, 255, 255), -1)

        # Compute bounding box from landmarks
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        return (x1, y1, x2, y2)

    # --------------------------------------------------

    def cosine_similarity(self, a, b):
        """
        Compute cosine similarity between two vectors.
        Used to compare face embeddings.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --------------------------------------------------

    def recognize(self, clean_frame, display_frame):
        """
        Perform face recognition on the current frame.

        Parameters:
        - clean_frame: raw frame used for embedding extraction
        - display_frame: frame used for drawing overlays

        Returns:
        - name: recognized identity or "Unknown"
        - score: similarity score
        - bbox: bounding box of detected face
        - emb: extracted face embedding
        """
        # Extract face embedding from clean frame
        emb = self.get_embedding_clean(clean_frame)

        # Detect landmarks for visualization
        landmarks = self.get_landmarks(display_frame)

        bbox = None
        if landmarks:
            bbox = self.draw_mesh(display_frame, landmarks)

        name = "Unknown"
        score = 0.0

        # Update state
        self.face_detected = emb is not None
        self.current_name = "Unknown"
        self.current_score = 0.0
        self.is_recognized_flag = False

        # Compare embedding to known faces
        if emb is not None and self.known_embeddings:
            scores = [
                self.cosine_similarity(emb, known)
                for known in self.known_embeddings
            ]

            best = np.argmax(scores)
            score = scores[best]

            # Update state
            self.current_score = score

            # Recognition threshold
            if score > 0.40:
                name = self.known_names[best]
                self.current_name = name
                self.is_recognized_flag = True

        return name, score, bbox, emb

    # --------------------------------------------------

    def open_camera(self):
        """
        Open webcam stream and run live face recognition.
        Returns True if a face was recognized during the session.
        """
        cam = cv.VideoCapture(0)

        if not cam.isOpened():
            print("ERROR: Cannot open camera")
            return False

        print("\n=== CONTROLS ===")
        print("ESC - Quit")
        print("S   - Save face")
        print("================\n")

        recognized_during_session = False

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Clean copy for embedding (no drawings)
            clean_frame = frame.copy()

            # Display copy for UI overlays
            display_frame = frame.copy()

            # Run recognition pipeline
            name, score, bbox, emb = self.recognize(clean_frame, display_frame)

            # Track if anyone was recognized
            if self.is_recognized_flag:
                recognized_during_session = True

            # Draw bounding box and name
            if bbox:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(
                    display_frame, name, (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )

            # Status text overlays
            cv.putText(display_frame, f"Name: {name}", (20, 40),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.putText(display_frame, f"Confidence: {score:.2f}", (20, 80),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.putText(display_frame, f"Match: {score * 100:.1f}%", (20, 120),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            face_status = "Face: DETECTED" if emb is not None else "Face: NONE"
            face_color = (0, 255, 0) if emb is not None else (0, 0, 255)
            cv.putText(display_frame, face_status, (20, 160),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2)

            status = "RECOGNIZED" if score > 0.40 else "NOT RECOGNIZED"
            status_color = (0, 255, 0) if score > 0.40 else (0, 0, 255)
            cv.putText(display_frame, status, (20, 200),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv.imshow("Face Recognition", display_frame)

            key = cv.waitKey(1) & 0xFF

            # Exit
            if key == 27:
                break

            # Save a new face to the database
            elif key in (ord('s'), ord('S')):
                emb = self.get_embedding_clean(clean_frame)

                if emb is not None:
                    face_name = input("Enter name: ").strip()
                    if face_name:
                        path = os.path.join(self.known_faces_dir, f"{face_name}.jpg")
                        cv.imwrite(path, clean_frame)

                        self.known_embeddings.append(emb)
                        self.known_names.append(face_name)
                        print(f"Added {face_name} to recognition")
                else:
                    print("No face detected - cannot save")

        cam.release()
        cv.destroyAllWindows()

        return recognized_during_session


# =====================
# RUN APPLICATION
# =====================
if __name__ == "__main__":
    app = FacialRecognition()
    result = app.open_camera()
    
    if result:
        (print("✅ Someone was recognized during session"))
        return (True, current_name)
    else:
        print("❌ No one was recognized")