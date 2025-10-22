# Railway Deployment Setup

This document explains how to set up automatic deployment to Railway using GitHub Actions. Railway is the exclusive cloud deployment platform for this project.

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Your code should be pushed to GitHub
3. **Railway CLI**: Install locally for initial setup

## Setup Instructions

### 1. Install Railway CLI

```bash
# Using npm
npm install -g @railway/cli

# Or using curl
curl -fsSL https://railway.app/install.sh | sh
```

### 2. Login and Create Project

```bash
# Login to Railway
railway login

# Create new project or link existing
railway init
```

### 3. Get Railway Token

```bash
# Generate deployment token
railway tokens create

# Copy the token - you'll need it for GitHub secrets
```

### 4. Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `RAILWAY_TOKEN` | Railway API token | `railway tokens create` |
| `RAILWAY_PROJECT_ID` | Project ID (optional) | `railway status` |

### 5. Configure Railway Project

In your Railway dashboard:

1. **Create New Project** or select existing
2. **Connect GitHub Repository** 
3. **Set Environment Variables** (optional):
   - `ENVIRONMENT=production`
   - `YOLO_DEVICE=cpu` 
   - `CORS_ORIGINS=your-domain.com`
4. **Configure Build Settings**:
   - Root Directory: `/` (default)
   - Build Command: Automatic (uses Dockerfile)
   - Start Command: `python entrypoint.py` (from railway.json)

### 6. Manual Deployment Test

Test deployment manually first:

```bash
# In your project directory
railway login --token YOUR_TOKEN
railway link YOUR_PROJECT_ID
railway up
```

### 7. Enable Auto-Deploy

Once manual deployment works:

1. Push code changes to `main` branch
2. GitHub Actions will automatically deploy to Railway
3. Monitor deployment in Railway dashboard

## Deployment Workflow

The GitHub Action (`railway-deploy.yml`) will:

1. ✅ **Build Check**: Verify Docker build works
2. ✅ **Deploy**: Push to Railway using CLI
3. ✅ **Verify**: Test health endpoint
4. ✅ **Report**: Create deployment summary

## Expected Deployment Time

- **First Deploy**: 10-15 minutes (downloading ML models)
- **Updates**: 2-5 minutes (cached dependencies)

## Monitoring Deployment

### Railway Dashboard
- View build logs and deployment status
- Monitor resource usage (CPU, Memory)
- Check application logs

### GitHub Actions
- View deployment progress in Actions tab
- See build summaries and error details
- Manual trigger via workflow dispatch

### Health Checks
Railway automatically monitors the `/health` endpoint:
- **Interval**: 30 seconds
- **Timeout**: 100 seconds  
- **Auto-restart**: On failure

## Troubleshooting

### Common Issues

1. **"railway command not found"**
   - Install Railway CLI: `npm install -g @railway/cli`

2. **"Invalid token"** 
   - Generate new token: `railway tokens create`
   - Update GitHub secret: `RAILWAY_TOKEN`

3. **"Project not found"**
   - Check project ID: `railway status`
   - Update GitHub secret: `RAILWAY_PROJECT_ID`

4. **Build timeouts**
   - Railway Pro plan has longer build times
   - Consider pre-building Docker images

5. **Memory issues**
   - AI models require 2GB+ RAM
   - Upgrade to Railway Pro plan

### Debug Commands

```bash
# Check Railway status
railway status

# View deployment logs  
railway logs

# Check environment variables
railway variables

# Test local build
docker build -t test .
docker run -p 8000:8000 -e PORT=8000 test
```

## Environment Variables

Optional variables you can set in Railway:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `production` | Application environment |
| `YOLO_DEVICE` | `cuda` | YOLO processing device |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_PROCESSING_LATENCY_MS` | `80` | Safety threshold |

## Security Notes

- ✅ Railway tokens are encrypted in GitHub secrets
- ✅ Non-root user in Docker container
- ✅ Environment-specific configurations
- ✅ Health check monitoring

## Support

If you encounter issues:

1. Check Railway dashboard for build logs
2. Review GitHub Actions output
3. Test Docker build locally
4. Verify secrets are correctly set
5. Check Railway service limits

For Railway-specific help: [Railway Documentation](https://docs.railway.app/)

## Files Used in Deployment

- `/Dockerfile` - Railway build configuration
- `/railway.json` - Railway deployment settings  
- `/.github/workflows/railway-deploy.yml` - CI/CD pipeline
- `/backend/entrypoint.py` - Application startup with Railway support