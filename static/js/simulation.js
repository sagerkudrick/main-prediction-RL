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
        this.onTorqueUpdate = null;

        // History for RL observations
        this.oriHistory = [
            [0, 0, 0, 1],  // quat from 2 frames ago
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
     */
    async step(inferenceCanvas = null, inferenceInterval = 15) {
        if (!this.isRunning) return null;

        this.frameCount++;

        // Setup torque callback
        this.physics.onTorqueApplied = (torque) => {
            if (this.onTorqueUpdate) {
                this.onTorqueUpdate(torque);
            }
        };

        // Run inference periodically
        if (inferenceCanvas && this.frameCount % inferenceInterval === 0) {
            await this.runInference(inferenceCanvas);
        }

        // Predict RL action if enabled
        if (this.rlEnabled && this.inference.ready()) {
            await this.predictAndApplyAction();
        }

        // Step physics
        const state = this.physics.step(4);
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
    async runInference(renderer) {
        if (!this.inference.ready()) return;

        try {
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
     * Predict RL action from PREDICTED physics state
     */
    async predictAndApplyAction() {
        try {
            // Priority: Use PREDICTED quaternion from vision
            // Fallback: Use ACTUAL physics quaternion (e.g. if vision not ready)
            let rawQuat = this.physics.predictedQuaternion;

            if (!rawQuat) {
                // console.warn("No predicted quaternion yet, using ground truth");
                const state = this.physics.getState();
                rawQuat = state.quaternion; // [x, y, z, w]
            }

            // CRITICAL: Coordinate Transformation for RL Model
            // Client (Three.js/Cannon): Y-Up
            // Training (PyBullet): Z-Up
            // Transformation: (x, y, z) -> (x, z, -y)

            // 1. Swizzle Quaternion
            // Client: [x, y, z, w] -> Training: [x, z, -y, w]
            const trainingQuat = {
                x: rawQuat[0],
                y: rawQuat[2],
                z: -rawQuat[1],
                w: rawQuat[3]
            };

            // 2. Compute Z-Axis (in Training Frame)
            // Since we swizzled the quat to Training Frame, this helper calculates the Training Z-axis
            const zAxis = this.physics.quaternionToZAxis(trainingQuat);

            // 3. Compute Angular Velocity Estimate (in Training Frame)
            const prevTrainingQuat = this.oriHistory[1]; // Already stored as [x, z, -y, w]

            // Note: We need to ensure history is stored in Training Frame!
            // If history was stored in RAW frame, we'd need to swizzle it here.
            // Let's look at where history is updated: Lines 198-199

            // Current approach: Let's calculate AngVel from RAW and then Swizzle
            const prevRawQuat = this._lastRawQuat || rawQuat;
            const rawAngVelEst = [
                rawQuat[0] - prevRawQuat[0],
                rawQuat[1] - prevRawQuat[1],
                rawQuat[2] - prevRawQuat[2]
            ];

            const trainingAngVelEst = [
                rawAngVelEst[0],
                rawAngVelEst[2],      // y -> z
                -rawAngVelEst[1]      // z -> -y
            ];

            // Build 23-feature observation (Training Frame)
            const observation = new Float32Array([
                trainingQuat.x, trainingQuat.y, trainingQuat.z, trainingQuat.w,       // 4: current quat
                zAxis[0], zAxis[1], zAxis[2],                                         // 3: z-axis
                zAxis[0], zAxis[1], zAxis[2],                                         // 3: z-axis (clean)
                ...this.oriHistory[0],                                                // 4: prev quat (must be training frame!)
                trainingAngVelEst[0], trainingAngVelEst[1], trainingAngVelEst[2],     // 3: ang vel estimate
                ...this.actionHistory[0],                                             // 3: action t-1
                ...this.actionHistory[1]                                              // 3: action t-2
            ]);

            // if (this.frameCount % 60 === 0) {
            //    console.log('Obs (Training Frame):', observation.slice(0, 4));
            // }

            const action = await this.inference.predictAction(observation);
            this.physics.setAction(action);

            // Update history for next frame
            // Store EVERYTHING in Training Frame for consistency
            this.oriHistory[0] = this.oriHistory[1];
            this.oriHistory[1] = [trainingQuat.x, trainingQuat.y, trainingQuat.z, trainingQuat.w];

            this.actionHistory[0] = this.actionHistory[1];
            this.actionHistory[1] = [...action];

            this._lastRawQuat = [...rawQuat];

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
     * Teleport box
     */
    teleportBox(position, quaternion = null) {
        this.physics.teleportBox(position, quaternion);
    }

    /**
     * Drag controls
     */
    grabBox(point) {
        this.physics.grabBox(point);
    }

    moveGrabbed(point) {
        this.physics.moveGrabbed(point);
    }

    releaseBox() {
        this.physics.releaseBox();
    }

    getState() {
        return this.physics.getState();
    }

    getBoxBody() {
        return this.physics.boxBody;
    }

    isReady() {
        return this.inference.ready();
    }

    isLoading() {
        return this.inference.loading();
    }
}