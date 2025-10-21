@echo off
echo ========================================
echo AI Navigation Assistant - Deployment Setup
echo ========================================
echo.

echo This script helps you set up the deployment:
echo.
echo 1. GitHub Pages (Frontend + Client) - Live URLs
echo 2. Local Backend - AI Processing
echo.

echo Step 1: Enable GitHub Pages
echo ========================================
echo 1. Go to your GitHub repository settings
echo 2. Scroll to "Pages" section  
echo 3. Set Source to "GitHub Actions"
echo 4. The workflow will deploy automatically on push to main
echo.

echo Step 2: Start Local Backend
echo ========================================
echo Run this command in a separate terminal:
echo.
echo   cd backend
echo   python main.py
echo.
echo Backend will be available at: http://localhost:8000
echo.

echo Step 3: Access Live URLs
echo ========================================
echo After GitHub Pages deployment:
echo.
echo Frontend: https://yourusername.github.io/your-repo-name/frontend.html
echo Client:   https://yourusername.github.io/your-repo-name/client/
echo.
echo Replace 'yourusername' and 'your-repo-name' with your actual values.
echo.

echo Step 4: Test the System
echo ========================================
echo 1. Start backend locally (Step 2)
echo 2. Open Client URL in browser
echo 3. Enable camera and microphone permissions
echo 4. Click "Start Streaming"
echo 5. Open Frontend URL in another tab
echo 6. Click "Start Navigation" for audio guidance
echo.

echo ========================================
echo Deployment Benefits:
echo ========================================
echo ✅ Frontend and Client hosted on GitHub Pages (free HTTPS)
echo ✅ Backend runs locally (your AI processing stays private)
echo ✅ No server costs for frontend hosting
echo ✅ Automatic deployments via GitHub Actions
echo ✅ Works from any device with internet access
echo.

pause