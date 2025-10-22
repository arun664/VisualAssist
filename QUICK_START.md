# 🚀 Quick Start Guide

## Production Setup (1 Step!)

### Open Live Apps
- **Frontend**: https://arun664.github.io/VisualAssist/
- **Client**: https://arun664.github.io/VisualAssist/client/

That's it! Both frontend/client (GitHub Pages) and backend (AWS) are fully hosted in the cloud.

## What You Get

- ✅ **Free frontend hosting** (GitHub Pages)
- ✅ **Cloud AI backend** (AWS)
- ✅ **Scalable processing** (AWS infrastructure)
- ✅ **Simple deployment** (GitHub Pages + AWS)
- ✅ **Zero setup complexity** (just open URLs!)

## Local Development Commands

```bash
# Start local backend (for development)
docker-compose up -d backend

# Check status
docker-compose ps
docker-compose logs backend

# Stop backend
docker-compose down

# Test AWS backend
curl http://18.222.141.234:8000/health
```

## Troubleshooting

### CORS Issues
Use Chrome with CORS disabled:
```bash
chrome.exe --disable-web-security --user-data-dir="C:\temp\chrome-cors"
```

### Backend Not Starting
```bash
# Check Docker
docker --version
docker-compose --version

# Rebuild backend
docker-compose build backend
docker-compose up -d backend
```

### Frontend Not Loading
- Check GitHub Pages deployment in repository Actions tab
- Ensure backend is running on 18.222.141.234:8000
- Try refreshing the page

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.