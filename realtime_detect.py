import torch
import cv2
import numpy as np

from model import FabricCNN

classes = ["good","hole","objects","oil_spot","thread_error"]

# load model
model = FabricCNN()
model.load_state_dict(torch.load("fabric_cnn_model.pth", map_location="cpu"))
model.eval()

# For Android IP Webcam, use URL like: http://192.168.1.X:8080/video
# Uncomment the line below and replace with your IP Webcam URL
# cap = cv2.VideoCapture("http://192.168.1.100:8080/video")

# Try different backends to fix green screen issue
backends = [cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_DSHOW]

# open camera - try multiple indices for external webcams
# Uses the LAST camera found (usually external webcam at index 1 or higher)
cap = None
camera_idx = -1
for i in range(5):
    for backend in backends:
        temp_cap = cv2.VideoCapture(i, backend)
        if temp_cap.isOpened():
            # Set camera properties for better compatibility
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            temp_cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera found at index {i} with backend {backend}")
            if cap is not None:
                cap.release()
            cap = temp_cap
            camera_idx = i
            break
    else:
        continue
    break

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera")
    exit()

print(f"Using camera at index {camera_idx}")

# Warm up camera - read more frames to stabilize
print("Warming up camera...")
frame_ready = False
for attempt in range(50):
    ret, frame = cap.read()
    cv2.waitKey(100)
    
    if ret and frame is not None and frame.size > 1000:
        # Check if frame is not green/black (mean pixel value check)
        mean_val = np.mean(frame)
        if mean_val > 20 and mean_val < 240:  # Not too dark or too bright
            print(f"Camera ready, frame shape: {frame.shape}, mean: {mean_val:.1f}")
            frame_ready = True
            break
        else:
            print(f"Frame {attempt}: Invalid frame (mean={mean_val:.1f})")
    else:
        print(f"Frame {attempt}: Not received")

if not frame_ready:
    print("Warning: Camera may not be providing valid frames")

while True:

    ret, frame = cap.read()
    cv2.waitKey(1)  # Allow frame buffer to refresh
    
    # Try reading again if first attempt fails
    if not ret or frame is None:
        ret, frame = cap.read()
        
    if not ret or frame is None or frame.size < 1000:
        print(f"Frame not received (ret={ret}, frame={frame is not None}, size={frame.size if frame is not None else 0})")
        cv2.waitKey(500)  # Wait longer before retry
        continue
    
    # Check for invalid/green/black frames
    mean_val = np.mean(frame)
    if mean_val < 20 or mean_val > 240:
        print(f"Invalid frame (mean={mean_val:.1f}), skipping...")
        cv2.waitKey(100)
        continue

    h, w = frame.shape[:2]

    # center ROI
    roi = frame[int(h*0.3):int(h*0.7),
                int(w*0.3):int(w*0.7)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Check if ROI has meaningful content (not just solid background)
    std_dev = np.std(gray)
    if std_dev < 15:  # Low variance = solid color (green screen or blank)
        cv2.rectangle(frame,
                      (int(w*0.3),int(h*0.3)),
                      (int(w*0.7),int(h*0.7)),
                      (0,0,255),2)
        cv2.putText(frame,"Place fabric in frame",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),2)
        cv2.imshow("Fabric Inspection",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    img = cv2.resize(gray,(64,64))

    img = img / 255.0

    img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

        prob = torch.softmax(output,dim=1)

        conf, pred = torch.max(prob,1)

    pred = pred.item()   # FIX

    label = classes[pred]

    # draw ROI box
    cv2.rectangle(frame,
                  (int(w*0.3),int(h*0.3)),
                  (int(w*0.7),int(h*0.7)),
                  (0,255,0),2)

    # show prediction
    cv2.putText(frame,f"{label} {conf.item():.2f}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Fabric Inspection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
