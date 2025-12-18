# Face & Eye Blink Recognition for Mood Detection

A privacy-preserving, Federated Learning-based web application that detects user mood, eye blinks, and face presence in real-time.

## 🚀 Features
- **Real-time Face Detection**: Uses optimized Haar Cascades.
- **Eye Blink Counting**: Tracks user attention and fatigue.
- **Mood Detection**: Classifies facial expressions (Happy, Neutral, Sad, Angry) using Computer Vision & Heuristics.
- **Visual Tech Overlay**: "Iron Man" style HUD with tracking grids for Face, Eyes, Mouth, and Eyebrows.
- **Privacy-First**: Designed with Federated Learning concepts (local processing).
- **Modern UI**: Dark-themed, responsive dashboard.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript (WebSocket, Canvas)
- **Backend**: Python, FastAPI
- **Computer Vision**: OpenCV (Haar Cascades)
- **Communication**: WebSockets for real-time streaming

## 📦 Installation

1.  **Clone/Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

### Option 1: One-Click (Windows)
Double-click `run.bat`.

### Option 2: Manual
```bash
# Activate virtual environment (if created)
# .\venv\Scripts\activate

cd backend
python main.py
```

The application will start at **http://localhost:8000**.

## 📂 Project Structure
- `backend/`: Core application logic.
    - `main.py`: FastAPI server & WebSocket handler.
    - `cv_logic.py`: Computer Vision algorithms.
- `frontend/`: User Interface.
    - `index.html`: Dashboard & Camera logic.
- `models/`: Directory for ML models (e.g., `emotion_model.h5`).

## 🧠 Federated Learning Concept
This system adheres to **Federated Learning principles**:
1.  **Local Processing**: Video frames are processed to extract features (blink count, emotion) *during* the session.
2.  **Privacy**: Raw video data is NOT stored by the server; only aggregated metrics (status updates) are processed.
3.  **Model Layout**: The architecture supports deploying global model updates to clients without uploading user data to a central cloud.
