# 🤖 AI Navigation Assistant

Real-time AI-powered navigation assistance with object detection, path guidance, and audio feedback using **pretrained YOLO models**.

> **🚀 New to this project?** Check [QUICK_START.md](QUICK_START.md) for 2-step setup!

## 🚀 Production Setup (Recommended)

### 1. Backend Deployed on AWS
Backend is running on AWS: `http://18.222.141.234:8000`

### 2. Use Live Frontend & Client
- **Frontend**: https://arun664.github.io/VisualAssist/
- **Client**: https://arun664.github.io/VisualAssist/client/

**✨ Zero setup needed!** Both frontend/client (GitHub Pages) and backend (AWS) are fully hosted in the cloud.

## 🛠️ Local Development Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start Backend Locally
```bash
cd backend
python main.py
```

### 3. Start Frontend & Client Locally
```bash
# Frontend (Terminal 1)
cd frontend && python -m http.server 3000

# Client (Terminal 2) 
cd client && python -m http.server 3001
```

- **Frontend**: http://localhost:3000
- **Client**: http://localhost:3001

## 🧠 AI Models (Pretrained)

The system uses **pretrained YOLOv11 models** from ultralytics:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolo11n.pt` | 2.6MB | ⚡ Fastest | Good | **Default** - Real-time navigation |
| `yolo11s.pt` | 9.4MB | Fast | Better | Balanced performance |
| `yolo11m.pt` | 20MB | Medium | High | More accurate detection |
| `yolo11l.pt` | 25MB | Slower | Higher | High accuracy needs |
| `yolo11x.pt` | 56MB | Slowest | Highest | Best possible accuracy |

**Default**: `yolo11n.pt` (nano) for optimal real-time performance.

## 🎯 Object Detection Capabilities

The AI detects **40+ navigation-relevant objects**:

### 🪑 Furniture & Obstacles
- Chairs, tables, couches, beds
- Potted plants, TVs, laptops
- Bags, bottles, sports equipment

### 👥 People & Animals  
- Persons, dogs, cats, horses
- Moving and stationary detection

### 🚗 Vehicles
- Cars, motorcycles, trucks, buses
- Bicycles, skateboards

### 🏠 Household Items
- Refrigerators, microwaves, sinks
- Books, clocks, vases, phones

## 🔧 Configuration

### Change AI Model
Edit `backend/computer_vision.py`:
```python
# For faster processing (default)
vision_processor = VisionProcessor(model_name="yolo11n.pt")

# For better accuracy
vision_processor = VisionProcessor(model_name="yolo11m.pt")
```

### Adjust Detection Sensitivity
```python
vision_processor = VisionProcessor(
    model_name="yolo11n.pt",
    confidence_threshold=0.5,  # Lower = more detections
    stationary_threshold=2.0   # Motion sensitivity
)
```

## 🌐 Production Architecture

```
GitHub Pages (HTTPS)          AWS Instance (HTTP)
┌─────────────────┐         ┌──────────────────┐
│  Frontend       │◄────────┤  Backend         │
│  Client         │  CORS   │  + Pretrained AI │
└─────────────────┘         └──────────────────┘
```

### Benefits:
- ✅ **Free frontend hosting** (GitHub Pages)
- ✅ **Cloud backend hosting** (AWS)
- ✅ **Scalable AI processing** (AWS infrastructure)
- ✅ **No model files to manage** (auto-download)
- ✅ **Simple deployment** (GitHub Pages + AWS)
- ✅ **Zero local setup** (fully cloud-hosted)

## 🧪 Test Backend

```bash
# Check AWS backend health
curl http://18.222.141.234:8000/health
```

## 🔍 How It Works

### 1. **Pretrained Model Loading**
```python
from ultralytics import YOLO

# Automatically downloads if not cached
model = YOLO("yolo11n.pt")
```

### 2. **Object Detection**
- Detects 80 COCO classes
- Filters for navigation-relevant objects
- Ignores ceiling/roof objects (top 30% of frame)
- Focuses on ground-level obstacles

### 3. **Path Calculation**
- Creates 6x8 grid for path analysis
- Marks safe vs blocked areas
- Calculates walking corridors
- Provides visual + audio guidance

### 4. **Real-time Processing**
- 0.5 FPS for optimal performance
- Circuit breaker for error handling
- Automatic model caching

## 🚨 Troubleshooting

### Model Download Issues
```bash
# Test internet connection
ping github.com

# Clear ultralytics cache
rm -rf ~/.cache/ultralytics

# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics
```

### CORS Issues
**Production (AWS)**: CORS is configured for HTTP connections from GitHub Pages.

**If CORS issues occur**: 
1. **Use Chrome with CORS disabled**:
   ```
   chrome.exe --disable-web-security --user-data-dir="C:\temp\chrome-cors"
   ```

2. **Or restart local backend**:
   ```bash
   docker-compose restart backend
   ```

### Performance Issues
- **Switch to faster model**: Use `yolo11n.pt` (default)
- **Reduce frame rate**: Already optimized to 0.5 FPS
- **Check CPU usage**: AI processing is CPU-intensive

## 📱 Usage Instructions

### Client Setup:
1. Open: https://arun664.github.io/VisualAssist/client/
2. Click "Enable Camera" → Allow permissions
3. Click "Enable Microphone" → Allow permissions  
4. Click "Start Streaming" → Connects to AWS backend

### Frontend Monitor:
1. Open: https://arun664.github.io/VisualAssist/
2. See real-time AI processed video
3. Click "Start Navigation" → Audio guidance activates
4. Click "Stop Navigation" → Audio mutes, video continues

### What You'll Hear:
- **"Clear path ahead"** → Safe to move forward
- **"Chair detected"** → Obstacle identified
- **"Path blocked"** → Multiple obstacles
- **"Obstacle on left/right"** → Directional guidance

## 🔒 Privacy & Security

- ✅ **Secure cloud processing** - HTTPS encrypted connections
- ✅ **Pretrained models** - No training data exposure  
- ✅ **HTTPS everywhere** - End-to-end encryption
- ✅ **No data storage** - Real-time processing only
- ✅ **Railway security** - Enterprise-grade infrastructure

## 🚀 Deployment Setup

### GitHub Pages Setup (One-time)
1. Go to repository **Settings** → **Pages**
2. Set **Source** to **GitHub Actions**
3. Go to **Actions** → **General** → **Workflow permissions**
4. Select **Read and write permissions**
5. Push to `main` branch to trigger deployment

### Live URLs
- **Frontend**: `https://arun664.github.io/VisualAssist/`
- **Client**: `https://arun664.github.io/VisualAssist/client/`
- **Backend**: `http://18.222.141.234:8000` (AWS)

### Local Development
```bash
# Start local backend
docker-compose up -d backend

# Check status
docker-compose ps
docker-compose logs backend
```

## 📁 Project Structure

```
ai-navigation-assistant/
├── backend/                 # Docker backend service
│   ├── main.py             # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Docker configuration
├── frontend/               # GitHub Pages frontend
│   ├── index.html         # Main interface
│   ├── app.js            # Frontend logic
│   └── styles.css        # Styling
├── client/                # GitHub Pages client
│   ├── index.html        # Client interface
│   ├── client.js         # WebRTC streaming
│   └── client-styles.css # Client styling
├── .github/workflows/     # GitHub Actions
├── docker-compose.yml     # Local development
├── config.js             # Configuration
└── README.md            # This file
```

## 📄 License

MIT License - Free for personal and commercial use.

---

**Powered by:** Ultralytics YOLOv11, FastAPI, WebRTC, AWS, and GitHub Pages.