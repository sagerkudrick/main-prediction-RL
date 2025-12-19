/**
 * ONNX Inference Module
 * Returns predictions with consistent Three.js quaternions [x,y,z,w]
 */
import * as THREE from 'three';

export class InferenceManager {
    constructor() {
        this.poseSession = null;
        this.rlSession = null;
        this.isLoading = false;
        this.isReady = false;
        this.useMockData = false;

        // Reusable canvas + context
        this._preprocCanvas = document.createElement('canvas');
        this._preprocCtx = this._preprocCanvas.getContext('2d');
        this._targetW = 224;
        this._targetH = 224;
        this._tensorBuffer = new Float32Array(3 * this._targetW * this._targetH);

        // Async mutex for serialized runs
        this._runLock = Promise.resolve();
        this._lastQuaternion = [0, 0, 0, 1]; // initial
        this._smoothingFactor = 0.4; // higher = smoother
        // Timing info
        this._lastPoseMs = 0;
        this._lastRlMs = 0;
    }

    smoothQuaternion(newQ) {
        const lastQ = this._lastQuaternion;

        // Handle sign ambiguity (q and -q are equivalent)
        let dot = lastQ[0] * newQ[0] + lastQ[1] * newQ[1] + lastQ[2] * newQ[2] + lastQ[3] * newQ[3];
        const correctedQ = dot < 0 ? newQ.map(v => -v) : newQ;

        const smoothed = lastQ.map((v, i) => this._smoothingFactor * v + (1 - this._smoothingFactor) * correctedQ[i]);
        const norm = Math.sqrt(smoothed.reduce((a, b) => a + b * b, 0));
        this._lastQuaternion = smoothed.map(v => v / norm);
        return this._lastQuaternion;
    }

    async _withRunLock(fn) {
        const previous = this._runLock;
        let release;
        this._runLock = new Promise((resolve) => (release = resolve));
        await previous;
        try {
            return await fn();
        } finally {
            release();
        }
    }

    async loadModels() {
        if (this.isLoading || this.isReady) return;
        this.isLoading = true;
        console.log('Loading ONNX models...');

        if (this.useMockData) {
            console.log('  ⚠ Using mock inference (ONNX models disabled)');
            this.isReady = true;
            this.isLoading = false;
            return true;
        }

        try {
            if (typeof ort === 'undefined') throw new Error('ONNX Runtime not loaded.');

            const providers = ['webgpu']; // Force WebGPU


            // Load Pose model
            console.log('  Loading pose model...');
            let poseLoaded = false;
            for (const provider of providers) {
                try {
                    console.log(`    Trying ${provider} backend...`);

                    // const response = await fetch('http://69.197.134.3:8081/models/pose_model_final.onnx');
                    //if (!response.ok) throw new Error('Failed to fetch model');

                    //const arrayBuffer = await response.arrayBuffer();
                    //console.log('Model loaded into browser memory!');

                    //this.poseSession = await ort.InferenceSession.create(arrayBuffer, {
                    this.poseSession = await ort.InferenceSession.create('static/models/pose_model_best.onnx', {
                        executionProviders: [provider],
                        graphOptimizationLevel: 'all'
                    });
                    console.log(`  ✓ Pose model loaded with ${provider}`);
                    poseLoaded = true;
                    break;
                } catch (err) {
                    console.warn(`    ${provider} failed:`, err.message || err);
                }
            }

            if (!poseLoaded) {
                console.warn('  ⚠ Failed to load pose model, using mock data');
                this.useMockData = true;
                this.isReady = true;
                this.isLoading = false;
                return true;
            }

            // Load RL model
            console.log('  Loading RL policy...');
            let rlLoaded = false;
            for (const provider of providers) {
                try {
                    console.log(`    Trying ${provider} backend...`);
                    this.rlSession = await ort.InferenceSession.create('static/models/rl-model.onnx', {
                        executionProviders: [provider],
                        graphOptimizationLevel: 'all'
                    });
                    console.log(`  ✓ RL policy loaded with ${provider}`);
                    rlLoaded = true;
                    break;
                } catch (err) {
                    console.warn(`    ${provider} failed:`, err.message || err);
                }
            }

            if (!rlLoaded) {
                console.warn('  ⚠ Failed to load RL policy');
            }

            this.isReady = true;
            this.isLoading = false;
            console.log('✓ Models loaded successfully');
            return true;

        } catch (error) {
            console.error('Error loading ONNX models:', error);
            console.warn('  ⚠ Falling back to mock data');
            this.useMockData = true;
            this.isReady = true;
            this.isLoading = false;
            return true;
        }
    }

    async predictPose(rendererOrCanvas) {
        if (!this.isReady) throw new Error('Models not loaded yet');
        if (this.useMockData) return { quaternion: [0, 0, 0, 1] };

        const sourceCanvas = rendererOrCanvas?.domElement || rendererOrCanvas;
        if (!sourceCanvas) return { quaternion: [0, 0, 0, 1] };

        // Draw into preallocated canvas
        const w = sourceCanvas.width;
        const h = sourceCanvas.height;
        this._preprocCanvas.width = this._targetW;
        this._preprocCanvas.height = this._targetH;
        this._preprocCtx.drawImage(sourceCanvas, 0, 0, w, h, 0, 0, this._targetW, this._targetH);

        const imageData = this._preprocCtx.getImageData(0, 0, this._targetW, this._targetH);
        const pixels = imageData.data;
        const tensor = this._tensorBuffer;

        // ImageNet normalization
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        for (let i = 0; i < this._targetH; i++) {
            for (let j = 0; j < this._targetW; j++) {
                const idx = (i * this._targetW + j) * 4;
                const r = pixels[idx] / 255.0;
                const g = pixels[idx + 1] / 255.0;
                const b = pixels[idx + 2] / 255.0;

                tensor[0 * this._targetH * this._targetW + i * this._targetW + j] = (r - mean[0]) / std[0];
                tensor[1 * this._targetH * this._targetW + i * this._targetW + j] = (g - mean[1]) / std[1];
                tensor[2 * this._targetH * this._targetW + i * this._targetW + j] = (b - mean[2]) / std[2];
            }
        }

        const inputTensor = new ort.Tensor('float32', tensor, [1, 3, this._targetH, this._targetW]);

        // Serialized session.run
        const result = await this._withRunLock(async () => {
            const t0 = performance.now();
            const feeds = { input: inputTensor };
            const r = await this.poseSession.run(feeds);
            const t1 = performance.now();
            this._lastPoseMs = t1 - t0;
            return r;
        });

        // Normalize raw quaternion
        let rawQ = Array.from(result.output.data.slice(0, 4));
        let normQ = this.normalizeQuaternion(rawQ);

        // // Initialize last quaternion if undefined
        // if (!this._lastQuaternion) this._lastQuaternion = [0, 0, 0, 1];

        // // Correct sign ambiguity (q vs -q)
        // let dot = this._lastQuaternion[0]*normQ[0] + this._lastQuaternion[1]*normQ[1] + this._lastQuaternion[2]*normQ[2] + this._lastQuaternion[3]*normQ[3];
        // if (dot < 0) normQ = normQ.map(v => -v);

        // // EMA smoothing
        // const alpha = 0.8; // smoothing factor (higher = smoother)
        // const smoothedQ = this._lastQuaternion.map((v, i) => alpha * v + (1 - alpha) * normQ[i]);

        // // Normalize smoothed quaternion
        // const norm = Math.sqrt(smoothedQ.reduce((acc, v) => acc + v*v, 0));
        // const finalQ = smoothedQ.map(v => v / norm);

        // this._lastQuaternion = finalQ;

        return { quaternion: normQ };
    }

    async predictAction(observation) {
        if (!this.isReady) throw new Error('Models not loaded yet');
        if (this.useMockData) return [0, 0, 0];

        const floatObs = new Float32Array(observation);

        const result = await this._withRunLock(async () => {
            const feeds = { input: new ort.Tensor('float32', floatObs, [1, 23]) };
            return await this.rlSession.run(feeds);
        });

        return Array.from(result.actions.data.slice(0, 3));
    }

    normalizeQuaternion(q) {
        const norm = Math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2);
        return norm < 1e-8 ? [0, 0, 0, 1] : q.map((v) => v / norm);
    }

    ready() { return this.isReady; }
    loading() { return this.isLoading; }
}
