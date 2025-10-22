# 🚀 Deployment Setup Instructions

## GitHub Pages Setup (Required)

To fix the deployment permissions error, you need to enable GitHub Pages in your repository settings:

### Step 1: Enable GitHub Pages
1. Go to your repository on GitHub: `https://github.com/arun664/VisualAssist`
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions**
5. Click **Save**

### Step 2: Set Repository Permissions
1. In the same **Settings** tab
2. Go to **Actions** → **General** in the left sidebar
3. Scroll to **Workflow permissions**
4. Select **Read and write permissions**
5. Check **Allow GitHub Actions to create and approve pull requests**
6. Click **Save**

### Step 3: Trigger Deployment
1. Make any commit to the `main` branch
2. Push to GitHub: `git push origin main`
3. GitHub Actions will automatically deploy to Pages

## 🔧 Manual Deployment Alternative

If you prefer not to use GitHub Actions, you can deploy manually:

```bash
# Build the deployment
mkdir dist
cp -r frontend/* dist/
mkdir dist/client
cp -r client/* dist/client/

# Create main index.html
cat > dist/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=./index.html">
    <title>AI Navigation Assistant</title>
</head>
<body>
    <p>Redirecting to <a href="./index.html">AI Navigation Assistant</a>...</p>
</body>
</html>
EOF

# Deploy to gh-pages branch manually
git checkout --orphan gh-pages
git rm -rf .
cp -r dist/* .
git add .
git commit -m "Deploy to GitHub Pages"
git push origin gh-pages --force
git checkout main
```

## 📱 After Deployment

Once deployed, your live URLs will be:
- **Main Page**: `https://arun664.github.io/VisualAssist/`
- **Frontend**: `https://arun664.github.io/VisualAssist/index.html`
- **Client**: `https://arun664.github.io/VisualAssist/client/`

## 🐍 Local Backend Setup

Remember, the backend still runs locally:

```bash
cd backend
python main.py
```

Backend will be available at: `http://localhost:8000`

## 🔍 Troubleshooting

### Permission Denied Error
- Ensure GitHub Pages is enabled with **GitHub Actions** source
- Check workflow permissions are set to **Read and write**
- Verify you're pushing to the `main` branch

### 404 Error on Deployed Site
- Check that `index.html` exists in the root of the deployed files
- Verify the deployment completed successfully in Actions tab
- Wait a few minutes for GitHub Pages to update

### Backend Connection Issues
- Ensure backend is running locally on port 8000
- Check browser console for CORS errors
- Verify firewall isn't blocking localhost:8000

## 📞 Support

If you continue having issues:
1. Check the **Actions** tab for detailed error logs
2. Ensure all files are committed and pushed to `main`
3. Verify repository settings match the instructions above