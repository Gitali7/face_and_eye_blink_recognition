# Project Documentation: Face & Eye Blink Recognition System

## 1. Introduction
This project aims to develop a real-time system for monitoring user status through a webcam. Key metrics include face presence, eye blink frequency (indicative of fatigue or attention), and emotional state. The system is architected with **Federated Learning (FL)** concepts in mind, ensuring user privacy by processing sensitive video data locally or in ephemeral sessions without permanent storage.

## 2. System Architecture

### 2.1 High-Level Flow
1.  **User Browser**: Captures video feed via `navigator.mediaDevices.getUserMedia`.
2.  **WebSocket Stream**: Frames are sent to the backend at ~10 FPS for near real-time processing.
3.  **FastAPI Backend**: Receives frames and orchestrates the analysis.
4.  **Computer Vision Module**:
    *   **Preprocessing**: Grayscale conversion.
    *   **Detection**: Haar Cascades for Face and Eyes.
    *   **Analysis**: Blink counting logic and Emotion Classification (CNN).
5.  **Feedback Loop**: Results (Blink Count, Emotion Label) are sent back to the browser immediately.

### 2.2 Privacy & Federated Learning
In a full FL deployment:
- The **Client** (Browser/Local App) would own the data.
- The **Global Model** would reside on the server.
- **Parametric Updates**: Instead of sending images, the client would train locally and send *weight* updates.
- **Current Implementation**: We simulate this by strictly processing ephemeral frames in memory (RAM) and discarding them immediately, ensuring no personal data hits the disk.

## 3. Key Algorithms

### 3.1 Blink Detection
We utilize a heuristic based on **Haar Cascade Eye Detection**:
- If a face is detected but eyes are *not* detected in the upper facial region for $N$ consecutive frames, the eyes are considered **CLOSED**.
- When eyes reappear, it counts as a **BLINK**.
- *Note*: This is a lightweight alternative to the Eye Aspect Ratio (EAR) method that requires 68-point facial landmarks.

### 3.2 Emotion Recognition
- **Primary Logic**: Computer Vision based **Smile Detection** (Haar Cascade) to instantly detect "Happy" state.
- **Secondary Logic (Simulation)**: For demonstration purposes (without heavy Deep Learning models), the system simulates "Sad", "Angry", and "Neutral" states using a probabilistic state machine when no smile is detected.
- **Optional**: Supports loading a pre-trained Keras CNN model (`emotion_model.h5`) if provided for full AI-based classification.

## 4. Setup & Deployment
Refer to `README.md` for standard installation instructions.

### 4.1 Requirements
*   Python 3.8+
*   FastAPI
*   OpenCV (`opencv-python`)
*   TensorFlow (for Emotion Model)

## 5. Future Scope
- **Edge Deployment**: Move inference entirely to the browser using TensorFlow.js (Complete FL).
- **Personalized Calibration**: Adjust blink sensitivity per user.
- **Multi-Modal Analysis**: Combine voice tone with facial expression.
