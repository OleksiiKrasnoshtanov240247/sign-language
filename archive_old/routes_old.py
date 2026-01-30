"""
FastAPI routes for sign language detection API.
Includes WebSocket endpoint for real-time detection and REST endpoints.
"""
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict

from src.backend.api.schemas import DetectionResponse, SessionInfo, ErrorResponse
from src.backend.core.session_manager import SessionManager
from src.backend.core.config import STATIC_MODEL_PATH
from src.backend.detection import SignDetector

# Initialize FastAPI app
app = FastAPI(title="Sign Language Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
session_manager = SessionManager()
sign_detector = None

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"


@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup."""
    global sign_detector
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing detector on device: {device}")
    
    if STATIC_MODEL_PATH.exists():
        sign_detector = SignDetector(
            static_model_path=str(STATIC_MODEL_PATH),
            device=device
        )
        print(f"Detector initialized with model: {STATIC_MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {STATIC_MODEL_PATH}")
        sign_detector = SignDetector(device=device)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Sign Language Detection API</h1><p>Frontend not found</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "detector_loaded": sign_detector is not None,
        "active_sessions": len(session_manager.sessions)
    }


@app.post("/api/session/new", response_model=SessionInfo)
async def create_session():
    """Create a new learning session."""
    session = session_manager.create_session()
    progress = session.get_progress()
    
    return SessionInfo(
        session_id=session.id,
        current_letter=session.current_letter,
        total_correct=session.total_correct,
        total_attempts=session.total_attempts,
        accuracy=progress["accuracy"],
        completed_letters=session.completed_letters
    )


@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    progress = session.get_progress()
    return SessionInfo(
        session_id=session.id,
        current_letter=session.current_letter,
        total_correct=session.total_correct,
        total_attempts=session.total_attempts,
        accuracy=progress["accuracy"],
        completed_letters=session.completed_letters
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_manager.remove_session(session_id)
    return {"status": "deleted"}


def decode_frame(frame_base64: str) -> np.ndarray:
    """Decode base64 frame to numpy array."""
    # Remove data URL prefix if present
    if "," in frame_base64:
        frame_base64 = frame_base64.split(",")[1]
    
    # Decode base64
    img_bytes = base64.b64decode(frame_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return frame


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sign language detection.
    
    Client sends: {"frame": "base64_image", "session_id": "optional"}
    Server responds: DetectionResponse
    """
    await websocket.accept()
    
    session = None
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Get or create session
            session_id = data.get("session_id")
            if session_id:
                session = session_manager.get_session(session_id)
            
            if not session:
                session = session_manager.create_session()
                session_id = session.id
            
            # Decode frame
            try:
                frame = decode_frame(data["frame"])
            except Exception as e:
                await websocket.send_json({
                    "error": "Failed to decode frame",
                    "detail": str(e)
                })
                continue
            
            # Process frame through detector
            landmarks, status, _, prediction_info = sign_detector.process_frame(
                frame,
                session.current_letter
            )
            
            # Prepare response
            response = {
                "session_id": session_id,
                "hand_detected": landmarks is not None,
                "prediction": None,
                "match": False,
                "success": False,
                "timeout": False,
                "show_hint": False,
                "hint_message": "",
                "tutorial_url": session.get_tutorial_url(),
                "progress": session.get_progress(),
                "current_letter": session.current_letter,
                "consecutive_matches": session.consecutive_matches,
                "matches_needed": session.get_progress()["matches_needed"]
            }
            
            # Process prediction if hand detected
            if prediction_info:
                response["prediction"] = prediction_info
                
                # Update session with prediction
                result = session.process_prediction(
                    prediction_info["predicted_class"],
                    prediction_info["confidence"]
                )
                
                response.update({
                    "match": result["match"],
                    "success": result["success"],
                    "timeout": result["timeout"],
                    "show_hint": result["show_hint"],
                    "hint_message": result["hint_message"]
                })
                
                # Advance to next letter if needed
                if result["should_advance"]:
                    session.advance_to_next_letter()
                    response["current_letter"] = session.current_letter
                    response["tutorial_url"] = session.get_tutorial_url()
                    response["progress"] = session.get_progress()
                    response["consecutive_matches"] = 0
            
            # Send response
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print(f"Client disconnected")
        if session:
            # Keep session for potential reconnection
            pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "error": "Internal server error",
                "detail": str(e)
            })
        except:
            pass


# Mount static files after routes to avoid conflicts
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
