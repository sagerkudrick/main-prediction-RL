"""
Flask server for pose prediction and RL control integration.
Provides REST API endpoints for:
- Pose prediction from images
- RL action prediction
- Environment reset
"""
import os
import io
import base64
import zipfile
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from flask import send_from_directory
import os

# Get parent directory for serving static files
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PARENT_DIR)

app = Flask(__name__, static_folder=ROOT_DIR)
CORS(app)

# ============== CONFIG ==============
MODEL_PATH = "pose_model_final.pt"
RL_MODEL_ZIP = "isotope_upright_with_xyz_arrows.zip"
RL_MODEL_EXTRACTED = "isotope_upright_with_xyz_arrows"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== DUMMY ENVIRONMENT ==============
class DummyIsotopeEnv(gym.Env):
    """Minimal environment that matches SimpleIsotopeEnv observation/action spaces"""
    def __init__(self):
        super().__init__()
        # Observation: quaternion (4) + angular velocity (3) + Z-axis (3) + orientation one-hot (6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
        # Action: torque along X, Y, Z axes
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        return np.zeros(16, dtype=np.float32), {}
    
    def step(self, action):
        return np.zeros(16, dtype=np.float32), 0.0, False, False, {}

# ============== POSE MODEL ==============
class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ============== LOAD MODELS ==============
print("Loading pose prediction model...")
pose_model = PoseModel().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
pose_model.load_state_dict(state)
pose_model.eval()
print(f"✓ Pose model loaded on {DEVICE}")

# Extract RL model if needed (for reference, but we'll load from zip)
if not os.path.exists(RL_MODEL_EXTRACTED):
    print("Extracting RL model from zip...")
    with zipfile.ZipFile(RL_MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(RL_MODEL_EXTRACTED)
    print("Extracted RL model")

print("Loading RL model...")
# Create dummy environment for model loading (no rendering, no PyBullet)
dummy_env = DummyIsotopeEnv()
# Load directly from zip file
rl_model = PPO.load(RL_MODEL_ZIP, env=dummy_env)
print("RL model loaded")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ============== HELPER FUNCTIONS ==============
def normalize_quaternion(q):
    """Normalize quaternion to unit length"""
    q = np.array(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return q / norm

def quaternion_to_z_axis(quat):
    """Convert quaternion [x,y,z,w] to z-axis vector"""
    rot = R.from_quat(quat)
    rot_matrix = rot.as_matrix()
    return rot_matrix[:, 2]  # Z-axis column

def z_axis_to_orientation_onehot(z_axis):
    """Convert z-axis to one-hot orientation (matching training)"""
    threshold = 0.7
    orientation = [0, 0, 0, 0, 0, 0]
    
    if z_axis[2] > threshold:
        orientation[0] = 1  # up
    elif z_axis[2] < -threshold:
        orientation[1] = 1  # down
    elif z_axis[0] > threshold:
        orientation[3] = 1  # right
    elif z_axis[0] < -threshold:
        orientation[2] = 1  # left
    elif z_axis[1] > threshold:
        orientation[4] = 1  # forward
    elif z_axis[1] < -threshold:
        orientation[5] = 1  # back
    
    return np.array(orientation, dtype=np.float32)

# ============== API ENDPOINTS ==============
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'pose_model': 'loaded',
        'rl_model': 'loaded',
        'device': str(DEVICE)
    })

@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    """
    Predict quaternion from base64-encoded image.
    Expected input: {"image": "base64_string"}
    Returns: {"quaternion": [x, y, z, w], "euler": [x, y, z]}
    """
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[-1])  # Handle data URL prefix
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess and predict
        inp = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = pose_model(inp)[0].cpu().numpy()
        
        # Normalize quaternion
        pred = normalize_quaternion(pred)
        
        # Convert to Euler angles for display
        euler = R.from_quat(pred).as_euler('xyz', degrees=True)
        
        return jsonify({
            'quaternion': pred.tolist(),
            'euler': euler.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_action', methods=['POST'])
def predict_action():
    """
    Predict RL action from current state.
    Expected input: {
        "quaternion": [x, y, z, w],
        "angular_velocity": [x, y, z]
    }
    Returns: {"action": [torque_x, torque_y, torque_z]}
    """
    try:
        data = request.get_json()
        
        # Extract state
        quat = np.array(data['quaternion'], dtype=np.float32)
        ang_vel = np.array(data['angular_velocity'], dtype=np.float32)
        
        # Normalize quaternion
        quat = normalize_quaternion(quat)
        
        # Compute z-axis from quaternion
        z_axis = quaternion_to_z_axis(quat)
        
        # Compute orientation one-hot
        orientation = z_axis_to_orientation_onehot(z_axis)
        
        # Construct observation (matching SimpleIsotopeEnv)
        # obs = quaternion (4) + angular_velocity (3) + z_axis (3) + orientation (6)
        obs = np.concatenate([quat, ang_vel, z_axis, orientation])
        
        # Predict action
        action, _ = rl_model.predict(obs, deterministic=True)
        
        return jsonify({
            'action': action.tolist(),
            'z_axis': z_axis.tolist(),
            'orientation': orientation.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/quaternion_error', methods=['POST'])
def quaternion_error():
    """
    Calculate error between predicted and actual quaternions.
    Expected input: {
        "predicted": [x, y, z, w],
        "actual": [x, y, z, w]
    }
    Returns: {"error": float, "angle_deg": float}
    """
    try:
        data = request.get_json()
        
        pred = normalize_quaternion(data['predicted'])
        actual = normalize_quaternion(data['actual'])
        
        # Calculate quaternion distance (L2 norm)
        error = np.linalg.norm(pred - actual)
        
        # Calculate rotation angle difference
        r_pred = R.from_quat(pred)
        r_actual = R.from_quat(actual)
        r_diff = r_actual * r_pred.inv()
        angle_deg = r_diff.magnitude() * 180 / np.pi
        
        return jsonify({
            'error': float(error),
            'angle_deg': float(angle_deg)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============== STATIC FILE SERVING ==============
@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory(ROOT_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(ROOT_DIR, path)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Server ready!")
    print("="*50)
    print(f"Pose Model: {MODEL_PATH}")
    print(f"RL Model: {RL_MODEL_EXTRACTED}")
    print(f"Device: {DEVICE}")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
