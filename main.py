"""
FastAPI Application for Fabric Defect Detection
Integrates the FabricCNN model for backend and frontend integration
"""

import base64
import io
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    model = FabricCNN()
    model.load_state_dict(torch.load("fabric_cnn_model.pth", map_location="cpu"))
    model.eval()
    print("Model loaded successfully!")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model prediction
    - Convert to grayscale
    - Resize to 64x64
    - Normalize to [0, 1]
    - Add batch and channel dimensions
    """
    # Convert to grayscale if RGB
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to 64x64
    image = image.resize((64, 64))

    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0

    # Convert to tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_array).float().unsqueeze(0).unsqueeze(0)

    return img_tensor


def predict_image(image_tensor: torch.Tensor) -> dict:
    """
    Make prediction on preprocessed image
    Returns class label, confidence score, and all class probabilities
    """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        # Get all class probabilities
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
    # Cleanup if needed


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
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ImageBase64Request(BaseModel):
    """Request model for base64 encoded image"""
    image: str  # Base64 encoded image string


class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    predicted_class: str
    confidence: float
    probabilities: dict


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fabric Defect Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict_file": "/predict/file - POST (upload image file)",
            "predict_base64": "/predict/base64 - POST (send base64 image)",
            "health": "/health - GET (check if model is loaded)",
            "classes": "/classes - GET (list available classes)"
        }
    }


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
    """
    Predict fabric defect from uploaded image file
    Supports: JPEG, PNG, BMP, WEBP
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess and predict
        image_tensor = preprocess_image(image)
        result = predict_image(image_tensor)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(request: ImageBase64Request):
    """
    Predict fabric defect from base64 encoded image
    Request body: {"image": "base64_encoded_string"}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))

        # Preprocess and predict
        image_tensor = preprocess_image(image)
        result = predict_image(image_tensor)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict/camera-frame")
async def predict_camera_frame(file: UploadFile = File(...)):
    """
    Predict from camera frame - optimized for real-time detection
    Similar to file endpoint but optimized for quick processing
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess
        image_tensor = preprocess_image(image)

        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return {
            "predicted_class": CLASSES[predicted_class.item()],
            "confidence": round(confidence.item(), 4),
            "class_id": predicted_class.item()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing frame: {str(e)}")


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
