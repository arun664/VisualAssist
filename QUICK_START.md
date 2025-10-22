# 🚀 Quick Start Guide

## Production Setup (1 Step!)

### Open Live Apps
- **Frontend**: https://arun664.github.io/VisualAssist/
- **Client**: https://arun664.github.io/VisualAssist/client/

That's it! Both frontend/client (GitHub Pages) and backend (Railway) are fully hosted in the cloud.

## What You Get

- ✅ **Free frontend hosting** (GitHub Pages)
- ✅ **Cloud AI backend** (Railway)
- ✅ **Scalable processing** (Railway containers)
- ✅ **Auto-updating everything** (GitHub Actions + Railway)
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

# Test Railway backend
curl https://your-app-name.up.railway.app/health
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
- Ensure backend is running on localhost:8000
- Try refreshing the page

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.