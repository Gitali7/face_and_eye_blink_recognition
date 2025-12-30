# Project Overview: Face & Emotion Recognition System

## 1. Introduction
This project is a real-time **Face and Emotion Recognition System** designed to detect human presence and identify emotions (moods) via a web interface. It captures video from the user's camera, processes it on a Python backend, and provides instant feedback on the user's emotional state.

A key feature of this system is its **Anti-Spoofing (Liveness Detection)** capability, allowing it to distinguish between a real human and a static photo or poster.

## 2. Objectives
*   **Real-time Interaction**: Process video frames with low latency using WebSockets.
*   **Robust Liveness Detection**: Filter out non-living objects (photos, posters, screens) to prevent false detections.
*   **Accurate Emotion Recognition**: Use Deep Learning to correctly classify emotions (Happy, Sad, Neutral, etc.) rather than relying on random heuristics.

## 3. Technologies & Techniques Used

### A. Core Stack
*   **Backend**: Python (FastAPI) - High-performance web framework.
*   **Communication**: WebSockets - For real-time, bi-directional image and data transfer.
*   **Frontend**: HTML5, CSS3, JavaScript - Captures webcam feed and renders results.

### B. Computer Vision (OpenCV)
We use **OpenCV (Open Source Computer Vision Library)** for the initial processing steps:
*   **Face Detection**: Uses `Haar Cascade Classifiers` (`haarcascade_frontalface_default.xml`) to locate faces in the frame.
*   **Eye Detection**: Uses `haarcascade_eye.xml` to track eye regions within the detected face.

### C. Anti-Spoofing: Strict Blink State Machine
To solve the "photo problem" (where a photo is detected as a human), we implemented a **Strict Blink State Machine**:
*   **Challenge**: Static photos or poor lighting can cause "noise" where pixels flicker. Simple detectors mistake this for blinking.
*   **Solution**: We require a specific temporal pattern for a valid blink.
    *   **State 0 (OPEN)**: Eyes are visible.
    *   **State 1 (CLOSING)**: Eyes disappear.
    *   **Verification**: The eyes must remain "closed" (undetected) for a **minimum duration** (e.g., ~0.1s) and a **maximum duration** (e.g., ~0.5s).
    *   **Constraint**: Single-frame flickers (noise) are rejected. Long disappearances (tracking loss) are rejected.
    *   **Result**: Only a deliberate, human-like blink verifies the user.

### D. AI Emotion Recognition (Deep Learning)
We moved from mock logic to a real AI model using the **`fer`** library powered by **TensorFlow**:
*   **Model**: Convolutional Neural Network (CNN) trained on the FER-2013 dataset.
*   **Process**:
    1.  Extract the face region (ROI) from the video frame.
    2.  Preprocess (Resize to 48x48, Grayscale/RGB conversion, Normalization).
    3.  Pass through the Neural Network to get probability scores for 7 emotions: *Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*.
*   **Confidence Threshold**: We verify the model's confidence is **> 35%**. If the model is uncertain (ambiguous expression), it defaults to "Neutral" to avoid erratic jumps.

## 4. System Architecture
1.  **Client (Browser)**: Captures webcam video -> Encodes frame to Base64 -> Sends to Backend via WebSocket.
2.  **Server (FastAPI)**: Receives frame -> Decodes image -> Passes to `CVProcessor`.
3.  **CVProcessor**:
    *   Detects Face & Eyes (OpenCV).
    *   Runs **Liveness Check** (Updates State Machine).
    *   IF Verified Human: Runs **Emotion AI** (TensorFlow).
    *   ELSE: Returns "Waiting for Blink...".
4.  **Response**: Server sends JSON (Bounding boxes, Emotion label, Verification status) -> Client draws UI.

## 5. Conclusion
This system successfully integrates classic Computer Vision (Haar Cascades) for speed with modern Deep Learning (CNNs) for accuracy. The addition of the **Strict Blink State Machine** provides a robust layer of security against photo spoofing, making it a reliable tool for liveness and mood detection.
