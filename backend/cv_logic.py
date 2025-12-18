import cv2
import numpy as np
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Using Mock/Heuristic Logic.")

class CVProcessor:
    def __init__(self):
        # Load Haar Cascades
        # Ensure these are loading from the correct cv2 data path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Mood Simulation State
        self.current_mood = "Neutral"
        self.mood_timer = 0
        self.MOOD_DURATION = 20 # frames to hold a simulated mood

        # Blink Detection State (Restored)
        self.blink_count = 0
        self.eye_state_history = []
        self.was_closed = False

        # Emotion Model
        self.emotion_model = None
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.load_emotion_model()

    def load_emotion_model(self):
        # Resolve path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'models', 'emotion_model.h5')
        
        if TF_AVAILABLE and os.path.exists(model_path):
            try:
                self.emotion_model = tf.keras.models.load_model(model_path)
                print("Emotion model loaded successfully.")
            except Exception as e:
                print(f"Failed to load emotion model: {e}")
        else:
            print("Emotion model not found or TensorFlow missing. Using Mock Logic.")

    def detect_emotion(self, face_roi):
        if self.emotion_model:
            try:
                # Preprocess for typical FER model
                roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                
                prediction = self.emotion_model.predict(roi, verbose=0)
                label = self.emotions[np.argmax(prediction)]
                return label
            except:
                return "Neutral"
        else:
            # Fallback: State-based Simulation for Demo
            try:
                # 1. Priority: Smile Detection (Real Computer Vision)
                # We check this EVERY frame. If distinct smile, we force Happy.
                roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
                
                if len(smiles) > 0:
                    self.current_mood = "Happy"
                    self.mood_timer = 10 # Short buffer for smiles to feel responsive
                    return "Happy"
                
                # 2. If NO smile, we rely on the State Machine to hold the mood
                if self.mood_timer > 0:
                    self.mood_timer -= 1
                    return self.current_mood
                
                # 3. Time's up! Pick a NEW mood (Simulated for Demo)
                # We want Sad/Angry to appear frequently and STABLY.
                import random
                roll = random.random()
                
                # 40% Neutral, 30% Angry, 30% Sad (Reversed per request)
                if roll < 0.4:
                    self.current_mood = "Neutral"
                    self.mood_timer = 40
                elif roll < 0.7:
                    self.current_mood = "Angry"  # Swapped from Sad
                    self.mood_timer = 60
                else:
                    self.current_mood = "Sad"    # Swapped from Angry
                    self.mood_timer = 60
                
                return self.current_mood
                
            except Exception as e:
                return "Neutral"

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        data = {
            "face_detected": False,
            "blink_count": self.blink_count,
            "emotion": "Neutral",
            "rects": [] # [x, y, w, h, type]
        }

        if len(faces) > 0:
            data["face_detected"] = True
            # Process the largest face
            (x, y, w, h) = faces[0]
            data["rects"].append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "type": "face"})
            
            face_roi = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # Detect Eyes (Upper half)
            eyes_roi = face_gray[0:int(h/2), :]
            eyes = self.eye_cascade.detectMultiScale(eyes_roi, 1.1, 4)
            
            for (ex, ey, ew, eh) in eyes:
                # Eye coordinates are relative to face ROI
                data["rects"].append({
                    "x": int(x + ex), 
                    "y": int(y + ey), 
                    "w": int(ew), 
                    "h": int(eh), 
                    "type": "eye"
                })

            # Blink Logic
            eyes_open = len(eyes) > 0
            if not eyes_open:
                self.was_closed = True
            else:
                if self.was_closed:
                    self.blink_count += 1
                    self.was_closed = False
            
            # Detect Emotion/Smile
            data["emotion"] = self.detect_emotion(face_roi)
            
            # If happy, assume we found a smile (approximate rect for lower face)
            if data["emotion"] == "Happy":
                 data["rects"].append({
                    "x": int(x + w*0.2), 
                    "y": int(y + h*0.6), 
                    "w": int(w*0.6), 
                    "h": int(h*0.3), 
                    "type": "mouth"
                })
            
            # Eyebrow Region (Upper Face Grid)
            # Simulating the tracking of brows
            data["rects"].append({
                "x": int(x + w*0.15),
                "y": int(y + h*0.20),
                "w": int(w*0.7),
                "h": int(h*0.15),
                "type": "eyebrow"
            })

        data["blink_count"] = self.blink_count
        return data
