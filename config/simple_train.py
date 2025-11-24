"""
Training script for isotope box with real-time XYZ axis visualization.
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from stable_baselines3 import PPO


class SimpleIsotopeEnv(gym.Env):
    """Environment to train isotope box to be upright with real-time XYZ arrows"""

    def __init__(self, render=True):
        super().__init__()
        self.render_mode = "human" if render else None
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1. / 240.)

        # Observation: quaternion (4) + angular velocity (3) + Z-axis (3) + orientation one-hot (6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

        # Action: torque along X, Y, Z axes
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.box = None
        self.plane = None
        self.step_count = 0
        self.axis_lines = []  # store debug line ids

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Load plane
        self.plane = p.loadURDF("plane.urdf")

        # Load isotope box at random orientation
        box_pos = [0, 0, 0.3]
        random_euler = np.random.uniform(-np.pi, np.pi, size=3)
        box_ori = p.getQuaternionFromEuler(random_euler)
        self.box = p.loadURDF("isotope.urdf", box_pos, box_ori)

        self.step_count = 0

        # Initialize debug lines for XYZ axes
        self._init_debug_axes()

        return self._get_obs(), {}

    def step(self, action):
        # Apply torque
        torque = action * 0.1
        p.applyExternalTorque(self.box, -1, torqueObj=torque, flags=p.LINK_FRAME)

        # Step physics
        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(1. / 240.)

        # Update XYZ arrows
        self._update_debug_axes()

        obs = self._get_obs()
        reward = self._compute_reward()

        self.step_count += 1
        pos, _ = p.getBasePositionAndOrientation(self.box)
        terminated = self.step_count >= 500 or pos[2] < 0.05

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        """Quaternion + angular velocity + Z-axis + orientation one-hot"""
        _, ori = p.getBasePositionAndOrientation(self.box)
        _, ang_vel = p.getBaseVelocity(self.box)

        rot_matrix = p.getMatrixFromQuaternion(ori)
        z_axis = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]], dtype=np.float32)

        obs = np.array(list(ori) + list(ang_vel), dtype=np.float32)
        obs = np.concatenate([obs, z_axis])
        obs = np.concatenate([obs, self._z_axis_to_orientation_onehot(z_axis)])

        return obs

    def _z_axis_to_orientation_onehot(self, z_axis):
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

    def _compute_reward(self):
        """
        Reward for being upright, penalize spinning too fast or falling.
        """
        pos, ori = p.getBasePositionAndOrientation(self.box)
        _, ang_vel = p.getBaseVelocity(self.box)

        # Rotation matrix from quaternion
        rot_matrix = p.getMatrixFromQuaternion(ori)
        z_axis = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])

        # Alignment with world up (Z-axis)
        world_up = np.array([0, 0, 1])
        alignment = np.dot(z_axis, world_up)  # 1.0 if perfectly upright

        # Penalty for angular velocity (spinning too fast)
        ang_vel_norm = np.linalg.norm(ang_vel)
        ang_vel_penalty = -0.1 * ang_vel_norm  # stronger penalty for spazzing

        # Penalty if box is too low (falling)
        height_penalty = -1.0 if pos[2] < 0.1 else 0.0

        # Total reward
        reward = alignment + ang_vel_penalty + height_penalty

        return reward

    # ------------------- DEBUG AXES -------------------
    def _init_debug_axes(self):
        # Remove old lines
        for line_id in self.axis_lines:
            p.removeUserDebugItem(line_id)
        self.axis_lines = []

        # Add 3 lines for X, Y, Z axes
        self.axis_lines.append(p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [1, 0, 0], 2))
        self.axis_lines.append(p.addUserDebugLine([0, 0, 0], [0, 0.2, 0], [0, 1, 0], 2))
        self.axis_lines.append(p.addUserDebugLine([0, 0, 0], [0, 0, 0.2], [0, 0, 1], 2))

    def _update_debug_axes(self):
        pos, ori = p.getBasePositionAndOrientation(self.box)
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)

        axis_length = 0.2
        x_end = pos + rot_matrix[:, 0] * axis_length
        y_end = pos + rot_matrix[:, 1] * axis_length
        z_end = pos + rot_matrix[:, 2] * axis_length

        p.addUserDebugLine(pos, x_end, [1, 0, 0], 2, replaceItemUniqueId=self.axis_lines[0])
        p.addUserDebugLine(pos, y_end, [0, 1, 0], 2, replaceItemUniqueId=self.axis_lines[1])
        p.addUserDebugLine(pos, z_end, [0, 0, 1], 2, replaceItemUniqueId=self.axis_lines[2])

    # ------------------- END DEBUG -------------------

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = SimpleIsotopeEnv(render=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )

    model.learn(total_timesteps=250_000)
    model.save("isotope_upright_with_xyz_arrows")
    env.close()
