# Client-Side Reinforcement Learning for 3D Orientation Control

**End-to-end ML pipeline: synthetic data generation → deep learning → reinforcement learning → fully browser-based deployment (WebGPU, ONNX, no backend required)**

---

## What It Does

Trains a neural network to predict 3D object orientation from images, then deploys an RL policy that actively re-orients objects upright in real time, all running client-side in the browser at 60 FPS.

---

## 🏗️ End-to-End ML Pipeline

<p align="center">
  <img src="img/pipeline.gif" width="45%">
</p>

Data Generation → Model Training & RL Training → ONNX Export → Browser Deployment (combining model & RL)<br><br>
Complete ownership of the ML lifecycle: from synthetic data in Blender, through parallel GPU-accelerated training (remote SSH), to optimized inference where the orientation model feeds directly into the RL policy running live in JavaScript.

---

## Synthetic Dataset Generation & Orientation Prediction

<table>
  <tr>
    <td align="center" width="45%">
      <strong>🧪 Synthetic Dataset</strong><br><br>
      <img src="img/synthetic.gif" width="100%"><br><br>
      Blender Python scripting generates 23k+ labeled images with perfect ground-truth quaternions. Randomized SO(3) rotations, unlimited data, full control over augmentation.
    </td>
    <td align="center" width="50%">
      <strong>🧠 Orientation Prediction Model</strong><br><br>
      <img src="img/predictions.gif" width="100%"><br><br>
      PyTorch CNN predicts 3D orientation (quaternions) from rendered images. Left: live object, Right: neural network predictions in real time.
    </td>
  </tr>
</table>

---

## Reinforcement Learning Re-Orientation

<p align="center">
  <img src="img/final-orientation.gif" width="720">
</p>

Trained RL policy actively re-orients the object upright by predicting quaternions and applying rotational torque. Generalizes across randomized initial orientations with noisy observations.

---

## Tech Stack

**ML:** PyTorch, ONNX Runtime, Reinforcement Learning, Quaternion math  
**3D:** Three.js, Cannon.js physics, Blender (data generation)  
**Web:** WebGPU/WebGL, JavaScript  
**Infrastructure:** Remote GPU training (SSH), NGINX, automated model deployment & CI/CD

---

## Skills Demonstrated

- Full ML lifecycle ownership (data → training → optimization → deployment)
- Synthetic data generation for computer vision
- Deep learning & reinforcement learning for continuous control
- Physics-based 3D simulation & real-time rendering
- Model optimization & ONNX conversion
- Browser-based ML inference at scale
- Production ML workflows with automated CI/CD
