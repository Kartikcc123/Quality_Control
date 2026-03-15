# Deploy Fabric Defect Detection on Render

## Quick Deploy

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/fabric-defect-detection.git
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repo
   - Settings:
     - Build Command: (leave empty)
     - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Click Create

3. **Wait 5-10 minutes** for first build (PyTorch download)

4. **Visit your app** at `https://your-service-name.onrender.com`

## Files Required in Repository
- `main.py` - FastAPI app (already configured)
- `index.html` - Frontend UI (already exists)
- `fabric_cnn_model.pth` - Trained model (already exists)
- `requirements.txt` - Dependencies (already configured)
- `runtime.txt` - Python version (already configured)
- `Procfile` - Web service command (already configured)

## Features
- Upload fabric images for defect detection
- Supports: good, hole, objects, oil_spot, thread_error
- REST API available at `/predict/file` and `/predict/base64`
