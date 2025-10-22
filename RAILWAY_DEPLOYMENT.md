# Railway Deployment Guide

This guide explains how to deploy the AI Navigation Assistant Backend to Railway.

## Files Created for Railway Deployment

### Root-Level Dockerfile
- **Location**: `/Dockerfile`
- **Purpose**: Railway-optimized build that copies from the `backend/` directory
- **Key Features**:
  - Builds from root context (required by Railway)
  - Copies `backend/requirements.txt` first for better caching
  - Copies entire `backend/` directory to `/app`
  - Handles Railway's `PORT` environment variable
  - Non-root user for security
  - Graceful YOLO model download

### Updated railway.json
- **Location**: `/railway.json`
- **Changes**:
  - Updated `dockerfilePath` to use root `Dockerfile`
  - Changed `startCommand` to use `entrypoint.py`
  - Health check configured for `/health` endpoint

### Updated Entrypoint
- **Location**: `/backend/entrypoint.py`
- **Railway Support**:
  - Uses `PORT` environment variable (Railway standard)
  - Falls back to `SERVER_PORT` then default 8000
  - Handles both full and minimal application modes

### Docker Ignore
- **Location**: `/.dockerignore`
- **Purpose**: Optimizes build by excluding unnecessary files

## Deployment Steps

### 1. Connect to Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Link to your project (or create new)
railway link
```

### 2. Deploy
```bash
# Deploy directly from repository
railway up

# Or link to GitHub and enable auto-deploys
```

### 3. Configure Environment Variables (Optional)
Set these in Railway dashboard if needed:
- `ENVIRONMENT=production` (default)
- `YOLO_DEVICE=cpu` (recommended for Railway)
- `CORS_ORIGINS=your-domain.com`

## Expected Behavior

### Successful Deployment
✅ **Full Application Mode**: All services running
- FastAPI server on Railway-assigned port
- YOLOv11 model loaded successfully
- Computer vision processing ready
- WebSocket and WebRTC endpoints available
- Navigation FSM operational
- Safety monitoring active

⚠️ **Speech Recognition**: May not initialize (missing Vosk model)
- This is expected and non-critical
- Core functionality remains fully operational

### Fallback Mode
If full application fails, automatically falls back to minimal mode:
- Basic FastAPI server with health checks
- CORS configured
- Basic endpoints available

## API Endpoints

Once deployed, your Railway app will have:
- `GET /` - Root endpoint
- `GET /health` - Health check (used by Railway)
- `GET /fsm/status` - Navigation state machine status
- `GET /computer_vision/status` - Computer vision system status
- `GET /webrtc/connections` - WebRTC connection status
- `WS /ws` - WebSocket connection for real-time communication

## Troubleshooting

### Build Fails with "requirements.txt not found"
- Ensure you're using the root-level `Dockerfile`
- Verify `railway.json` points to correct dockerfile

### Port Issues
- Railway automatically sets the `PORT` environment variable
- The application automatically uses this port
- No manual port configuration needed

### Model Download Issues
- YOLOv11 model downloads automatically during build
- If build times out, the app will try downloading at runtime
- Consider using Railway Pro for longer build times

### Memory Issues
- YOLOv11 + dependencies require significant memory
- Consider Railway Pro plan for better performance
- CPU inference is used by default (no GPU required)

## Performance Considerations

### Build Time
- Initial build: ~10-15 minutes (downloading ML models)
- Subsequent builds: ~2-3 minutes (cached layers)

### Runtime Memory
- Minimum: 1GB RAM
- Recommended: 2GB+ RAM for optimal performance

### Cold Starts
- First request may take 30-60 seconds
- Subsequent requests: <1 second response time
- Consider Railway Pro to reduce cold starts

## Monitoring

### Health Checks
Railway automatically monitors the `/health` endpoint:
- 30-second intervals
- 100-second timeout
- Automatic restarts on failure

### Logs
View logs via Railway dashboard or CLI:
```bash
railway logs
```

### Metrics
- Check Railway dashboard for CPU/Memory usage
- Monitor response times via built-in analytics

## Environment-Specific Notes

### Production Configuration
The app automatically uses production settings:
- JSON logging enabled
- Enhanced error handling
- Performance monitoring
- Safety protocols active

### CORS Configuration
Update CORS origins for your domain:
```bash
railway variables set CORS_ORIGINS=https://yourdomain.com
```

## Support

If deployment issues persist:
1. Check Railway build logs
2. Verify all required files are committed
3. Test Docker build locally first
4. Check Railway service limits