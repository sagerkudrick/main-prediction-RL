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

                // In constructor
        this.oriHistory = [
            [0, 0, 0, 1],  // quat from 2 frames ago (identity)
            [0, 0, 0, 1]   // quat from 1 frame ago
        ];
        this.actionHistory = [
            [0, 0, 0],     // action from 2 frames ago
            [0, 0, 0]      // action from 1 frame ago
        ];

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
     * @param {THREE.WebGLRenderer} renderer - The renderer that has just rendered the inference scene
     */
    async runInference(renderer) {
        if (!this.inference.ready()) return;

        try {
            // Get the canvas from the renderer (it's already been rendered)
            const canvas = renderer.domElement;

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
    /**
         * Predict RL action and apply to physics
         * Uses PREDICTED quaternion from inference (not actual physics state)
         */
    async predictAndApplyAction() {
        if (!this.physics.predictedQuaternion) {
            this.physics.setAction([0, 0, 0]);
            return;
        }
        try {
            // Canonicalize predicted quaternion for consistent RL observations
            const predQuat = this.physics.normalizeAndCanonicalizeQuaternion(this.physics.predictedQuaternion);
            // Compute z-axis from predicted quaternion
            const cannonQuat = { x: predQuat[0], y: predQuat[1], z: predQuat[2], w: predQuat[3] };
            const zAxis = this.physics.quaternionToZAxis(cannonQuat);

            // Compute orientation one-hot from predicted z-axis
            const prevQuat = this.oriHistory[1];
            const angVelEst = [
                predQuat[0] - prevQuat[0],
                predQuat[1] - prevQuat[1],
                predQuat[2] - prevQuat[2]
            ];

            // Construct observation (length 23)
            const observation = [
                predQuat[0], predQuat[1], predQuat[2], predQuat[3], // 4 - current quat
                zAxis[0], zAxis[1], zAxis[2],                       // 3 - current z-axis
                zAxis[0], zAxis[1], zAxis[2],                       // 3 - z-axis (same)
                ...this.oriHistory[0],                              // 4 - quat from 2 frames ago
                angVelEst[0], angVelEst[1], angVelEst[2],           // 3 - angular velocity estimate (from quat diff)
                ...this.actionHistory[0],                           // 3 - action from 2 frames ago
                ...this.actionHistory[1]                            // 3 - action from 1 frame ago
            ];
            
            if (this.frameCount % 60 === 0) {
                console.log('Observation length:', observation.length); // should be 27
                console.log('Observation:', observation);
            }

            const action = await this.inference.predictAction(observation);
            this.physics.setAction(action);

            // Update history for next frame
            this.oriHistory[0] = this.oriHistory[1];
            this.oriHistory[1] = [...predQuat];
            this.actionHistory[0] = this.actionHistory[1];
            this.actionHistory[1] = [...action];

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
     * Update box size
     */
    setBoxSize(x, y, z) {
        this.physics.setBoxSize(x, y, z);
    }

    /**
     * Set mesh shape
     */
    setMeshShape(vertices, indices) {
        this.physics.setMeshShape(vertices, indices);
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
