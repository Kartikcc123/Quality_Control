import torch
import cv2
import numpy as np

# Ensure your model.py file is in the same directory
from model import FabricCNN

# --- 1. CONFIGURATION ---
classes = ["good", "hole", "objects", "oil_spot", "thread_error"]
MODEL_PATH = "fabric_cnn_model.pth"

# 🟢 METHOD 1: DroidCam IP Stream (Highly Recommended)
# Open the DroidCam app on your phone, find the "WiFi IP" and "DroidCam Port"
# Format must be exactly like this: "http://<IP>:<PORT>/video"
CAMERA_SOURCE = "http://192.168.1.15:4747/video"

# 🔵 METHOD 2: USB / Virtual Camera Fallback
# If you must use USB, uncomment the line below and change the index if needed.
# CAMERA_SOURCE = 1


# --- 2. LOAD MODEL ---
print("Loading FabricCNN model...")
model = FabricCNN()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'fabric_cnn_model.pth' is in the current directory.")
    exit()


# --- 3. INITIALIZE CAMERA ---
print(f"Connecting to camera at: {CAMERA_SOURCE}...")

# If using Method 2 (USB), add cv2.CAP_DSHOW as the second argument:
# cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(CAMERA_SOURCE)

# If using Method 2 (USB), uncomment these to force MJPG codec to prevent green screen:
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not connect to the camera stream.")
    print("Check your phone's Wi-Fi connection and ensure DroidCam is open.")
    exit()

print("Camera connected. Press 'ESC' to exit.")


# --- 4. MAIN DETECTION LOOP ---
while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Dropped frame, retrying...")
        cv2.waitKey(100)
        continue

    h, w = frame.shape[:2]

    # Define the Region of Interest (ROI) in the center of the frame
    startX, endX = int(w * 0.3), int(w * 0.7)
    startY, endY = int(h * 0.3), int(h * 0.7)
    
    roi = frame[startY:endY, startX:endX]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Calculate variance to check if the camera is pointing at a solid color (like a green screen or blank wall)
    std_dev = np.std(gray)
    
    if std_dev < 15:
        # NO FABRIC DETECTED: Draw a red warning box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, "Place Khadi fabric in frame", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # FABRIC DETECTED: Prepare image for PyTorch
        img = cv2.resize(gray, (64, 64))
        img = img / 255.0
        # Convert to tensor and add batch/channel dimensions: [1, 1, 64, 64]
        img_tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

        # Run Inference
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        pred_idx = pred.item()
        confidence = conf.item()
        label = classes[pred_idx]

        # Determine box color based on defect status (Green for good, Red for defects)
        box_color = (0, 255, 0) if label == "good" else (0, 0, 255)

        # Draw ROI box and prediction text
        cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
        cv2.putText(frame, f"{label.replace('_', ' ').title()}: {confidence:.2f}", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    # Display the final frame
    cv2.imshow("Khadi Quality Inspection", frame)

    # Press 'ESC' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()