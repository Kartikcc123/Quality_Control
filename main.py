"""
FastAPI Application for Fabric Defect Detection
Integrates the FabricCNN model for backend and frontend integration
"""

import base64
import io
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image

# Model classes
CLASSES = ["good", "hole", "objects", "oil_spot", "thread_error"]

class FabricCNN(torch.nn.Module):
    """Fabric Defect Detection CNN Model"""
    def __init__(self):
        super(FabricCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 8 * 8, 256)
        self.fc2 = torch.nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Global model variable
model = None

def load_model():
    """Load the trained model on startup"""
    global model
    model_path = "fabric_cnn_model.pth"
    
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model file '{model_path}' not found!")
        return

    try:
        model = FabricCNN()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        print("Model loaded successfully and ready for predictions!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model prediction"""
    if image.mode != 'L':
        image = image.convert('L')

    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_tensor = torch.tensor(img_array).float().unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def predict_image(image_tensor: torch.Tensor) -> dict:
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        all_probs = probabilities[0].tolist()

    return {
        "predicted_class": CLASSES[predicted_class.item()],
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            CLASSES[i]: round(prob, 4) for i, prob in enumerate(all_probs)
        }
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    load_model()
    yield

# Create FastAPI app
app = FastAPI(
    title="Fabric Defect Detection API",
    description="API for detecting fabric defects using CNN model",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ImageBase64Request(BaseModel):
    image: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict


@app.get("/", include_in_schema=False)
async def serve_html():
    """Serve the frontend HTML page"""
    from fastapi.responses import HTMLResponse
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify model is loaded"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/classes")
async def get_classes():
    """Get list of available defect classes"""
    return {"classes": CLASSES}

@app.post("/predict/file", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """Predict fabric defect from uploaded image file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check if fabric_cnn_model.pth exists.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        image = Image.open(io.BytesIO(contents))
        if image is None:
            raise HTTPException(status_code=400, detail="Could not open image")
            
        image_tensor = preprocess_image(image)
        result = predict_image(image_tensor)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(request: ImageBase64Request):
    """Predict fabric defect from base64 encoded image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        base64_str = request.image
        # Fix: Remove HTML data prefix if it exists (e.g., "data:image/jpeg;base64,")
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        image_tensor = preprocess_image(image)
        result = predict_image(image_tensor)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict/camera-frame")
async def predict_camera_frame(file: UploadFile = File(...)):
    """Predict from camera frame - optimized for real-time detection"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_tensor = preprocess_image(image)
        result = predict_image(image_tensor) # Reused predict_image to keep response format consistent
        
        # Add class_id for backward compatibility if needed by specific camera scripts
        result["class_id"] = CLASSES.index(result["predicted_class"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing frame: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)