# Isotope Box RL Controller

Web-based demonstration integrating pose prediction with reinforcement learning for object stabilization.

## Project Structure

```
main-prediction-RL/
├── index.html              # Main web interface
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css      # UI styling
│   └── models/
│       └── isotopebox.obj # 3D model
├── backend/                # Python backend
│   ├── server.py          # Flask API server
│   ├── requirements.txt   # Python dependencies
│   ├── pose_model_final.pt        # Trained pose model (44MB)
│   └── isotope_upright_with_xyz_arrows.zip  # Trained RL model
└── config/                 # Training scripts & configs
    ├── simple_train.py    # RL training script
    ├── test_iso.py        # RL testing script
    ├── pose_model.py      # Pose model training
    └── isotope.urdf       # URDF definition
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start Server**:
   ```bash
   python server.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5000`

4. **Run Simulation**:
   - Click "Start Simulation"
   - Watch the crate stabilize using RL control
   - Monitor real-time diagnostics

## How It Works

1. **Capture**: Inference camera renders 256x256 image of rotating crate
2. **Predict**: Pose model (ResNet18) predicts quaternion orientation
3. **Control**: RL agent (PPO) computes corrective torques
4. **Apply**: Torques applied to physics simulation
5. **Repeat**: Loop runs at ~4Hz for real-time control

## Features

- ✅ Real-time pose estimation
- ✅ RL-based stabilization control
- ✅ Live diagnostics (quaternions, errors, actions)
- ✅ 240Hz physics simulation
- ✅ Modern glassmorphism UI
- ✅ Axis visualization (RGB = XYZ)

## API Endpoints

- `POST /predict_pose` - Predict quaternion from image
- `POST /predict_action` - Get RL control action
- `POST /quaternion_error` - Calculate prediction error
- `GET /health` - Server health check

## Controls

- **Start/Stop Simulation**: Control physics and inference
- **Reset Crate**: Randomize position and rotation
- **Mouse**: Orbit (left drag), zoom (scroll), pan (right drag)

## Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Flask
- Modern web browser with WebGL support
