/**
 * Simulation Controller
 * Coordinates physics, rendering, and inference
 */

import { PhysicsSimulation } from './physics.js';
import { InferenceManager } from './inference.js';

export class SimulationController {
    constructor() {
        this.physics = new PhysicsSimulation();
        this.inference = new InferenceManager();

        this.isRunning = false;
        this.rlEnabled = false;
        this.frameCount = 0;
        this.stepCount = 0;

        // Callbacks for UI updates
        this.onStateUpdate = null;
        this.onInferenceUpdate = null;
    }

    /**
     * Initialize simulation (load models and start physics)
     */
    async initialize() {
        console.log('Initializing simulation...');

        try {
            // Load ONNX models
            await this.inference.loadModels();

            // Start physics
            this.physics.start();

            console.log('✓ Simulation initialized');
            return true;
        } catch (error) {
            console.error('Initialization error:', error);
            throw error;
        }
    }

    /**
     * Start the simulation loop
     */
    start() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.frameCount = 0;
        console.log('Simulation started');
    }

    /**
     * Stop the simulation loop
     */
    stop() {
        this.isRunning = false;
        console.log('Simulation stopped');
    }

    /**
     * Reset the simulation
     */
    reset() {
        const state = this.physics.reset();
        this.frameCount = 0;
        this.stepCount = 0;

        if (this.onStateUpdate) {
            this.onStateUpdate(state);
        }

        return state;
    }

    /**
     * Step the simulation (called every frame)
     * @param {HTMLCanvasElement} inferenceCanvas - Canvas for inference rendering
     * @param {number} inferenceInterval - Run inference every N frames
     */
    async step(inferenceCanvas = null, inferenceInterval = 15) {
        if (!this.isRunning) return null;

        this.frameCount++;

        // Run inference periodically
        if (inferenceCanvas && this.frameCount % inferenceInterval === 0) {
            await this.runInference(inferenceCanvas);
        }

        // Predict RL action if enabled
        if (this.rlEnabled && this.inference.ready()) {
            await this.predictAndApplyAction();
        }

        // Step physics
        const state = this.physics.step(4); // 4 substeps like PyBullet
        this.stepCount = state.step_count;

        // Notify UI
        if (this.onStateUpdate) {
            this.onStateUpdate(state);
        }

        return state;
    }

    /**
     * Run pose inference from canvas
     */
    async runInference(canvas) {
        if (!this.inference.ready()) return;

        try {
            // Predict pose from inference camera
            const result = await this.inference.predictPose(canvas);

            // Store predicted quaternion in physics
            this.physics.setPredictedQuaternion(result.quaternion);

            // Notify UI
            if (this.onInferenceUpdate) {
                this.onInferenceUpdate(result);
            }

            return result;
        } catch (error) {
            console.error('Inference error:', error);
        }
    }

    /**
     * Predict RL action and apply to physics
     */
    async predictAndApplyAction() {
        if (!this.physics.predictedQuaternion) {
            this.physics.setAction([0, 0, 0]);
            return;
        }

        try {
            // Get current angular velocity
            const state = this.physics.getState();
            const angVel = state.angular_velocity;

            // Compute z-axis from predicted quaternion
            const quat = this.physics.predictedQuaternion;
            const cannonQuat = { x: quat[0], y: quat[1], z: quat[2], w: quat[3] };
            const zAxis = this.physics.quaternionToZAxis(cannonQuat);

            // Compute orientation one-hot
            const orientation = this.physics.zAxisToOrientationOneHot(zAxis);

            // Construct observation
            const observation = [
                ...quat,
                ...angVel,
                ...zAxis,
                ...orientation
            ];

            // Predict action
            const action = await this.inference.predictAction(observation);

            // Apply action to physics
            this.physics.setAction(action);

        } catch (error) {
            console.error('RL action prediction error:', error);
            this.physics.setAction([0, 0, 0]);
        }
    }

    /**
     * Enable/disable RL control
     */
    setRLEnabled(enabled) {
        this.rlEnabled = enabled;
        this.physics.setRLEnabled(enabled);
    }

    /**
     * Update wall size
     */
    updateWalls(size) {
        this.physics.updateWalls(size);
    }

    /**
     * Teleport box to position
     */
    teleportBox(position, quaternion = null) {
        this.physics.teleportBox(position, quaternion);
    }

    // ============== DRAG CONTROLS ==============

    grabBox(point) {
        this.physics.grabBox(point);
    }

    moveGrabbed(point) {
        this.physics.moveGrabbed(point);
    }

    releaseBox() {
        this.physics.releaseBox();
    }

    /**
     * Get current physics state
     */
    getState() {
        return this.physics.getState();
    }

    /**
     * Get physics body for Three.js sync
     */
    getBoxBody() {
        return this.physics.boxBody;
    }

    /**
     * Check if models are ready
     */
    isReady() {
        return this.inference.ready();
    }

    /**
     * Check if models are loading
     */
    isLoading() {
        return this.inference.loading();
    }
}
