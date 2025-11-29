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
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded.');
            }

            const providers = ['webgpu', 'webgl', 'cpu', 'wasm'];

            // Load pose model
            console.log('  Loading pose model...');
            let poseLoaded = false;
            for (const provider of providers) {
                try {
                    console.log(`    Trying ${provider} backend...`);
                    this.poseSession = await ort.InferenceSession.create('static/models/model.onnx', {
                        executionProviders: [provider],
                        graphOptimizationLevel: 'basic'
                    });
                    console.log(`  ✓ Pose model loaded with ${provider}`);
                    poseLoaded = true;
                    break;
                } catch (err) {
                    console.warn(`    ${provider} failed:`, err.message);
                }
            }

            if (!poseLoaded) {
                console.warn('  ⚠ Failed to load pose model, using mock data');
                this.useMockData = true;
                this.isReady = true;
                this.isLoading = false;
                return true;
            }

            // Load RL policy
            console.log('  Loading RL policy...');
            let rlLoaded = false;
            for (const provider of providers) {
                try {
                    console.log(`    Trying ${provider} backend...`);
                    this.rlSession = await ort.InferenceSession.create('static/models/rl-model.onnx', {
                        executionProviders: [provider],
                        graphOptimizationLevel: 'basic'
                    });
                    console.log(`  ✓ RL policy loaded with ${provider}`);
                    rlLoaded = true;
                    break;
                } catch (err) {
                    console.warn(`    ${provider} failed:`, err.message);
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
            this.isLoading = false;
            console.error('Error loading ONNX models:', error);
            console.warn('  ⚠ Falling back to mock data');
            this.useMockData = true;
            this.isReady = true;
            return true;
        }
    }

    /**
     * Predict pose from image
     * @returns {Object} - {quaternion: [x,y,z,w] in Three.js format}
     */
    async predictPose(canvas) {
        if (!this.isReady) {
            throw new Error('Models not loaded yet');
        }

        if (this.useMockData || !canvas) {
            return { quaternion: [0, 0, 0, 1] };
        }

        try {
            const width = canvas.width;
            const height = canvas.height;

            // Draw WebGL canvas onto 2D canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(canvas, 0, 0, width, height);

            // Debug save
            if (window.saveNextInferenceImage) {
                window.saveNextInferenceImage = false;
                const link = document.createElement('a');
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                link.download = `inference_debug_${timestamp}.png`;
                link.href = tempCanvas.toDataURL('image/png');
                link.click();
                console.log('📸 Saved DEBUG inference image:', link.download);
            }

            const imageData = tempCtx.getImageData(0, 0, width, height);
            const tensor = this.preprocessImage(imageData, 224, 224);
            const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 224, 224]);

            const feeds = { input: inputTensor };
            const results = await this.poseSession.run(feeds);

            const output = results.output.data;
            const modelQuat = Array.from(output.slice(0, 4)); // [x,y,z,w] from Blender training

            // Normalize
            const normalized = this.normalizeQuaternion(modelQuat);

            // Convert Blender → Three.js with 180° Y rotation
            // Convert Blender → Three.js with 180° Y rotation
            const threeQuat = this.blenderToThreeQuat(normalized);
            // Canonicalize to ensure consistent sign
            const canonical = InferenceManager.canonicalizeQuaternion(threeQuat);
            return { quaternion: canonical };

        } catch (error) {
            console.error('Pose prediction error:', error);
            return { quaternion: [0, 0, 0, 1] };
        }
    }

    /**
     * Convert Blender quaternion to Three.js with 180° Y rotation
     * Input: [x, y, z, w] from Blender (Z-up)
     * Output: [x, y, z, w] for Three.js (Y-up) with model facing correct direction
     */
    blenderToThreeQuat(q) {
        const [x, y, z, w] = q;

        // Create Three.js quaternion from Blender data
        const qOrig = new THREE.Quaternion(x, y, z, w);

        // Apply 180° rotation around Y axis to fix model orientation
        const qY180 = new THREE.Quaternion();
        qY180.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);

        qOrig.multiply(qY180);

        return [qOrig.x, qOrig.y, qOrig.z, qOrig.w];
    }

    /**
     * Ensure quaternion represents the same rotation with consistent sign
     * (Handles q and -q ambiguity)
     */
    static canonicalizeQuaternion(q) {
        if (q[3] < 0 || (q[3] === 0 && q[0] < 0)) {
            return [-q[0], -q[1], -q[2], -q[3]];
        }
        return [q[0], q[1], q[2], q[3]];
    }

    async predictAction(observation) {
        if (!this.isReady) {
            throw new Error('Models not loaded yet');
        }

        if (this.useMockData) {
            return [0, 0, 0];
        }

        try {
            const inputTensor = new ort.Tensor('float32', new Float32Array(observation), [1, 13]);
            const feeds = { input: inputTensor };
            const results = await this.rlSession.run(feeds);
            const output = results.actions.data;
            return Array.from(output.slice(0, 3));
        } catch (error) {
            console.error('RL action prediction error:', error);
            throw error;
        }
    }

    preprocessImage(imageData, targetWidth, targetHeight) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = targetWidth;
        tempCanvas.height = targetHeight;
        const tempCtx = tempCanvas.getContext('2d');

        const sourceCanvas = document.createElement('canvas');
        sourceCanvas.width = imageData.width;
        sourceCanvas.height = imageData.height;
        const sourceCtx = sourceCanvas.getContext('2d');
        sourceCtx.putImageData(imageData, 0, 0);

        tempCtx.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight);
        const resizedData = tempCtx.getImageData(0, 0, targetWidth, targetHeight);

        const tensor = new Float32Array(3 * targetHeight * targetWidth);
        const pixels = resizedData.data;

        for (let i = 0; i < targetHeight; i++) {
            for (let j = 0; j < targetWidth; j++) {
                const idx = (i * targetWidth + j) * 4;
                const r = pixels[idx] / 127.5 - 1.0;
                const g = pixels[idx + 1] / 127.5 - 1.0;
                const b = pixels[idx + 2] / 127.5 - 1.0;

                tensor[0 * targetHeight * targetWidth + i * targetWidth + j] = r;
                tensor[1 * targetHeight * targetWidth + i * targetWidth + j] = g;
                tensor[2 * targetHeight * targetWidth + i * targetWidth + j] = b;
            }
        }

        return tensor;
    }

    normalizeQuaternion(q) {
        const norm = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (norm < 1e-8) {
            return [0, 0, 0, 1];
        }
        return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
    }

    ready() {
        return this.isReady;
    }

    loading() {
        return this.isLoading;
    }
}