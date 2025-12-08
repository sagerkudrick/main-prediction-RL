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

        // Timing info
        this._lastPoseMs = 0;
        this._lastRlMs = 0;
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

                    const response = await fetch('http://69.197.134.3:8081/models/pose_model_best.onnx');
                    if (!response.ok) throw new Error('Failed to fetch model');

                    const arrayBuffer = await response.arrayBuffer();
                    console.log('Model loaded into browser memory!');

                    this.poseSession = await ort.InferenceSession.create(arrayBuffer, {
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

    // Fill channel-first tensor with normalized values
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

    const output = Array.from(result.output.data.slice(0, 4));
    return { quaternion: this.normalizeQuaternion(output) };
}

    async predictAction(observation) {
        if (!this.isReady) throw new Error('Models not loaded yet');
        if (this.useMockData) return [0, 0, 0];

        const floatObs = new Float32Array(observation);

        const result = await this._withRunLock(async () => {
            const feeds = { input: new ort.Tensor('float32', floatObs, [1, 13]) };
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
