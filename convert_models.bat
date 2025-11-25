@echo off
echo ========================================
echo ONNX Model Conversion Script
echo ========================================
echo.
echo This script will convert PyTorch models to ONNX format
echo for browser-based inference.
echo.
echo Requirements:
echo - Python 3.8+
echo - PyTorch
echo - ONNX
echo - Stable Baselines3
echo.

cd backend

echo Installing dependencies...
pip install torch torchvision onnx onnxruntime stable-baselines3 gymnasium numpy scipy

echo.
echo Running conversion script...
python convert_models.py

echo.
echo ========================================
echo Conversion complete!
echo ========================================
echo.
echo Generated files:
echo - static/models/pose_model.onnx
echo - static/models/rl_policy.onnx
echo.
echo Next steps:
echo 1. Test locally: python -m http.server 8000
echo 2. Open http://localhost:8000 in your browser
echo 3. Deploy to Cloudflare Pages
echo.
pause
