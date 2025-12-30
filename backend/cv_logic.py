import cv2
import numpy as np
import time

# Robust Import for FER
try:
    from fer import FER
except ImportError:
    try:
        from fer.fer import FER
    except ImportError:
        print("Could not import FER. Make sure 'fer' and 'tensorflow' are installed.")
        FER = None

class CVProcessor:
    def __init__(self):
        # Load Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # ---------------------------------------------------------
        # LIVENESS & IDENTITY STATE
        # ---------------------------------------------------------
        self.blink_count = 0
        self.was_closed = False
        
        # To avoid posters, we require at least 1 blink to "verify" logic
        # is running on a live human before we show emotions.
        self.is_verified_human = False
        
        # If we lose the face for X seconds, we reset verification.
        self.last_face_seen_time = 0
        self.face_timeout = 2.0  # seconds

        # ---------------------------------------------------------
        # REAL EMOTION DETECTION (FER Library)
        # ---------------------------------------------------------
        # We load the model once.
        if FER:
            # mtcnn=False uses lighter OpenCV for detection (unused in our direct predict approach)
            self.emotion_detector = FER(mtcnn=False) 
            # Standard FER2013 labels
            self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        else:
            self.emotion_detector = None
            self.emotion_labels = []

        self.current_emotion = "Neutral"
        
        # Throttle emotion detection to every N frames to keep FPS high
        self.frame_counter = 0
        self.emotion_update_interval = 5

    def process_frame(self, frame):
        self.frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = time.time()
        
        data = {
            "face_detected": False,
            "is_verified_human": self.is_verified_human,
            "blink_count": self.blink_count,
            "emotion": "Waiting for Blink..." if not self.is_verified_human else self.current_emotion,
            "rects": [] 
        }

        if len(faces) == 0:
            # If no face seen for a while, reset verification (person left)
            if current_time - self.last_face_seen_time > self.face_timeout:
                self.is_verified_human = False
                data["is_verified_human"] = False
                data["emotion"] = "Neutral"
            return data

        # If we found a face:
        self.last_face_seen_time = current_time
        data["face_detected"] = True
        
        # Process ONLY the largest face to avoid background noise/posters
        # Logic: verify the largest face, ignore others until then.
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = largest_face
        
        data["rects"].append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "type": "face"})
        
        face_roi = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        
        # ---------------------------------------------------------
        # BLINK DETECTION (Haar Cascade on Eyes)
        # ---------------------------------------------------------
        eyes_roi = face_gray[0:int(h/2), :]
        eyes = self.eye_cascade.detectMultiScale(eyes_roi, 1.1, 4)
        
        # Draw eyes
        for (ex, ey, ew, eh) in eyes:
            data["rects"].append({
                "x": int(x + ex), 
                "y": int(y + ey), 
                "w": int(ew), 
                "h": int(eh), 
                "type": "eye"
            })

        eyes_visible = len(eyes) > 0
        
        # ---------------------------------------------------------
        # STRICT BLINK STATE MACHINE
        # ---------------------------------------------------------
        # States: 0=OPEN, 1=CLOSING, 2=CLOSED
        if not hasattr(self, 'blink_state'):
            self.blink_state = 0 # Open
            self.closed_frames = 0

        # Constants for "Human-like" Blink Timing (assuming ~15-30 FPS)
        MIN_CLOSED_FRAMES = 2   # Fast blink
        MAX_CLOSED_FRAMES = 10  # Long blink (approx 0.5s)
        
        if self.blink_state == 0: # OPEN
            if not eyes_visible:
                # Transition to CLOSING
                self.blink_state = 1
                self.closed_frames = 1
                
        elif self.blink_state == 1: # CLOSING / CLOSED
            if not eyes_visible:
                # Still Closed, increment
                self.closed_frames += 1
                if self.closed_frames > MAX_CLOSED_FRAMES:
                    # Too long! Probably looked away or lost tracking. Reset.
                    self.blink_state = 0 
            else:
                # Eyes Re-opened!
                if MIN_CLOSED_FRAMES <= self.closed_frames <= MAX_CLOSED_FRAMES:
                    # VALID BLINK DETECTED!
                    self.blink_count += 1
                    self.is_verified_human = True
                    data["is_verified_human"] = True
                
                # Reset to Open state
                self.blink_state = 0
                self.closed_frames = 0

        # ---------------------------------------------------------
        # REAL EMOTION DETECTION (Only if verified human)
        # ---------------------------------------------------------
        if self.is_verified_human and self.emotion_detector:
            # Update emotion only every N frames to save CPU
            if self.frame_counter % self.emotion_update_interval == 0:
                try:
                    # FER expects RGB
                    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    
                    # top_emotion returns (emotion_name, score)
                    # It may return (None, None) if no face is detected in the crop
                    emotion, score = self.emotion_detector.top_emotion(rgb_face)
                    
                    # Confidence Threshold: prevents random guesses on ambiguous faces
                    if emotion and score is not None and score > 0.35:
                        self.current_emotion = emotion.capitalize()
                    else:
                        self.current_emotion = "Neutral"
                        
                except Exception as e:
                    print(f"Error in emotion detection: {e}")
                    self.current_emotion = "Neutral"

            data["emotion"] = self.current_emotion

        elif not self.is_verified_human:
             data["emotion"] = "Blink to Verify"

        return data
