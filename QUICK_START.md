# 🚀 Quick Start Guide

## Production Setup (1 Step!)

### Open Live Apps
- **Frontend**: https://arun664.github.io/VisualAssist/
- **Client**: https://arun664.github.io/VisualAssist/client/

That's it! Both frontend/client (GitHub Pages) and backend (ngrok HTTPS tunnel) are fully hosted and accessible.

## What You Get

- ✅ **Free frontend hosting** (GitHub Pages)
- ✅ **HTTPS backend tunnel** (ngrok)
- ✅ **Secure connections** (HTTPS/WSS)
- ✅ **Mobile compatible** (no Mixed Content issues)
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

# Test ngrok backend
curl https://flagless-clinographic-janita.ngrok-free.dev/health
```

## Troubleshooting

### Connection Issues
- **HTTPS**: All connections now use secure HTTPS (no Mixed Content issues)
- **Mobile**: Works on mobile browsers (no CORS problems)
- **Desktop**: No need to disable browser security

### Backend Not Starting (Local Development)
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
- Ensure backend is running on https://flagless-clinographic-janita.ngrok-free.dev
- Try refreshing the page

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.