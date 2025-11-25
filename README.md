# Client-Side Isotope Box RL Controller

A fully client-side physics simulation with reinforcement learning, running entirely in the browser using Cannon.js and ONNX Runtime Web. No backend required!

## 🚀 Quick Start

### Step 1: Convert Models to ONNX

Before deploying, you need to convert the PyTorch models to ONNX format:

```bash
cd backend
python convert_models.py
```

This will generate:
- `static/models/pose_model.onnx` - Pose prediction model
- `static/models/rl_policy.onnx` - RL policy model

**Requirements for conversion:**
```bash
pip install torch torchvision onnx onnxruntime stable-baselines3 gymnasium numpy scipy
```

### Step 2: Test Locally

Serve the application with a local HTTP server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server -p 8000

# Using PHP
php -S localhost:8000
```

Then open `http://localhost:8000` in your browser.

### Step 3: Deploy to Cloudflare Pages

1. **Create a new Cloudflare Pages project**
   - Go to [Cloudflare Pages](https://pages.cloudflare.com/)
   - Click "Create a project"
   - Connect your Git repository (or upload files directly)

2. **Build settings**
   - Framework preset: `None`
   - Build command: (leave empty)
   - Build output directory: `/`
   - Root directory: `/`

3. **Deploy**
   - Cloudflare will automatically deploy your site
   - Your app will be available at `https://your-project.pages.dev`

## 📁 Project Structure

```
main-prediction-RL/
├── index.html                  # Main application (client-side only)
├── static/
│   ├── css/
│   │   └── style.css          # Styles with loading overlay
│   ├── js/
│   │   ├── physics.js         # Cannon.js physics simulation
│   │   ├── inference.js       # ONNX model inference
│   │   └── simulation.js      # Main simulation controller
│   ├── models/
│   │   ├── isotopebox.glb     # 3D model
│   │   ├── pose_model.onnx    # Pose prediction (generated)
│   │   └── rl_policy.onnx     # RL policy (generated)
└── backend/
    ├── convert_models.py      # Model conversion script
    ├── server.py              # Simple static file server
    ├── pose_model_final.pt    # Original PyTorch model
    └── isotope_upright_with_xyz_arrows.zip  # Original RL model
```

## 🎮 Features

- **Client-Side Physics**: Cannon.js physics engine (no server needed)
- **Browser ML Inference**: ONNX Runtime Web for pose prediction and RL policy
- **Real-time Simulation**: Physics runs at 60 FPS in the browser
- **Interactive Controls**: Drag, position controls, wall size adjustment
- **RL Integration**: Toggle RL-based rotation control
- **Zero Backend**: Fully static site, deployable anywhere

## 🔧 Technology Stack

- **Physics**: [Cannon.js](https://github.com/pmndrs/cannon-es) - JavaScript physics engine
- **ML Inference**: [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) - Browser-based ML
- **3D Rendering**: [Three.js](https://threejs.org/) - WebGL rendering
- **Model Format**: ONNX (converted from PyTorch)

## 📊 How It Works

1. **Initialization**
   - Load ONNX models (pose prediction + RL policy)
   - Initialize Cannon.js physics world
   - Set up Three.js rendering

2. **Simulation Loop** (every frame)
   - Render inference camera view
   - Run pose prediction (every 15 frames)
   - Predict RL action from observation
   - Apply torque to physics body
   - Step physics simulation (4 substeps)
   - Update Three.js visualization

3. **Physics Sync**
   - Cannon.js handles physics simulation
   - Three.js meshes sync with physics bodies
   - Drag controls update physics positions

## 🎯 Controls

- **Start/Stop**: Control simulation loop
- **Reset**: Reset crate to initial position
- **RL Toggle**: Enable/disable RL-based rotation
- **Wall Size**: Adjust boundary walls (0.5m - 5.0m)
- **Position Controls**: Fine-tune crate position (±0.1m increments)
- **Drag**: Click and drag the crate in 3D space

## 🐛 Troubleshooting

### Models not loading
- Ensure you ran `convert_models.py` successfully
- Check browser console for ONNX loading errors
- Verify `.onnx` files exist in `static/models/`

### Physics behaving differently
- Cannon.js parameters are tuned to match PyBullet
- Check timestep (1/240s) and gravity (-9.81 m/s²)
- Verify mass and damping values

### CORS errors
- Must serve via HTTP server (not `file://`)
- Use `python -m http.server` or similar

## 📝 Notes

- **Model Size**: ONNX models are ~45MB (pose) + ~1MB (RL)
- **Performance**: Runs at 60 FPS on modern browsers
- **Compatibility**: Tested on Chrome, Firefox, Edge
- **Mobile**: Works on mobile but performance may vary

## 🔄 Differences from Python Version

| Feature | Python/PyBullet | JavaScript/Cannon.js |
|---------|----------------|---------------------|
| Physics Engine | PyBullet | Cannon.js |
| ML Framework | PyTorch + SB3 | ONNX Runtime Web |
| Server | Flask (required) | None (static) |
| Deployment | Needs Python server | Cloudflare Pages (free) |
| Latency | Network round-trip | Local (instant) |

## 📄 License

Same as original project.
