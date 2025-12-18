from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import json
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Face & Eye Blink Recognition for Mood Detection")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Directory '{FRONTEND_DIR}' does not exist")

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def get():
    index_file = FRONTEND_DIR / "index.html"
    with open(index_file, "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

from cv_logic import CVProcessor

# Initialize Processor
processor = CVProcessor()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    try:
        while True:
            # Receive frame from client (base64 string)
            data = await websocket.receive_text()
            
            try:
                # Decode image
                image_bytes = base64.b64decode(data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Process Frame
                result = processor.process_frame(frame)
                
                # Prepare response
                response = {
                    "status": "processing",
                    "blink_count": result["blink_count"],
                    "emotion": result["emotion"],
                    "face_detected": result["face_detected"],
                    "rects": result.get("rects", [])
                }
                
                await websocket.send_text(json.dumps(response))
            except Exception as e:
                # Log frame processing errors but keep connection alive
                print(f"Frame processing error: {e}")
                pass
                
    except WebSocketDisconnect:
        print("Client disconnected normally")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    # No finally block needed for close() as FastAPI/Starlette handles cleanup on disconnect exceptions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
