/**
 * ONNX Inference Module - MOCK VERSION
 * Returns dummy predictions until ONNX models are working
 */

export class InferenceManager {
    constructor() {
        this.poseSession = null;
        this.rlSession = null;
        this.isLoading = false;
        this.isReady = false;
        this.useMockData = false; // Try to load real ONNX models
    }


    /**
     * Load both ONNX models (or skip if using mock data)
     */
    async loadModels() {
        if (this.isLoading || this.isReady) return;

        this.isLoading = true;
        console.log('Loading ONNX models...');

        if (this.useMockData) {
            console.log('  ⚠ Using mock inference (ONNX models disabled)');
            console.log('  Physics simulation will work, but ML inference is disabled');
            this.isReady = true;
            this.isLoading = false;
            return true;
        }

        try {
            // Load ONNX Runtime
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded. Make sure to include onnxruntime-web script.');
            }

            // Try different execution providers
            const providers = ['webgl'];

            // Load pose prediction model
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

            // Load RL policy model
            console.log('  Loading RL policy...');
            let rlLoaded = false;
            for (const provider of providers) {
                try {
                    console.log(`    Trying ${provider} backend...`);
                    this.rlSession = await ort.InferenceSession.create('static/models/rl_policy.onnx', {
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
                console.warn('  ⚠ Failed to load RL policy, using mock data');
                //this.useMockData = true;
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
     * Predict pose (quaternion) from image
     * @param {HTMLCanvasElement} canvas - Canvas containing the rendered image
     * @returns {Object} - {quaternion: [x,y,z,w], euler: [x,y,z]}
     */
    /**
    * Predict pose (quaternion) from image
    * @param {HTMLCanvasElement} canvas - Canvas containing the rendered image
    * @returns {Object} - {quaternion: [x,y,z,w], euler: [x,y,z]}
    */
    async predictPose(canvas) {
        if (!this.isReady) {
            throw new Error('Models not loaded yet');
        }

        // Use mock data if models aren't loaded or canvas is missing
        if (this.useMockData || !canvas) {
            console.warn('Using mock data - useMockData:', this.useMockData, 'canvas:', !!canvas);
            return {
                quaternion: [0, 0, 0, 1],
                euler: [0, 0, 0]
            };
        }

        try {
            const width = canvas.width;
            const height = canvas.height;
            //console.log('Running inference on canvas:', width, 'x', height);

            // Draw WebGL canvas onto 2D canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(canvas, 0, 0, width, height);

            // Debug: Save image if requested
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

            // Preprocess to tensor
            const tensor = this.preprocessImage(imageData, 224, 224);
            const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 224, 224]);

            // Run ONNX model
            const feeds = { input: inputTensor };
            const results = await this.poseSession.run(feeds);

            const output = results.output.data;
            const quaternion = Array.from(output.slice(0, 4));
            //console.log('Raw model output (XYZW):', quaternion);

            const normalized = this.normalizeQuaternion(quaternion);
            //console.log('Normalized quaternion:', normalized);

            // -----------------------------
            // CONVERT BLENDER → THREE.JS
            // -----------------------------
            const threeQuat = this.blenderToThreeJSQuat(normalized);
            //console.log('Converted to Three.js:', threeQuat);

            const euler = this.quaternionToEuler(threeQuat);

            return { quaternion: threeQuat, euler };
        } catch (error) {
            console.error('Pose prediction error:', error);
            return {
                quaternion: [0, 0, 0, 1],
                euler: [0, 0, 0]
            };
        }
    }

    /**
     * Convert Blender quaternion [w, x, y, z] → Three.js [x, y, z, w]
     * and map axes Z-up → Y-up
     */
    /**
         * Convert Blender quaternion [x, y, z, w] → Three.js [x, y, z, w]
         * and map axes Z-up → Y-up with correct handedness
         */
    blenderToThreeJSQuat(q) {
        // q = [x, y, z, w] from Blender
        const [x, y, z, w] = q;

        // Flip Z axis to match Three.js forward (-Z)
        const threeX = -x;
        const threeY = -y;
        const threeZ = z;
        const threeW = w;

        return [threeX, threeY, threeZ, threeW];
    }



    /**
     * Predict RL action from observation
     * @param {Array} observation - 16-element observation array
     * @returns {Array} - 3-element action array [torque_x, torque_y, torque_z]
     */
    async predictAction(observation) {
        if (!this.isReady) {
            throw new Error('Models not loaded yet');
        }

        // Use mock data if models aren't loaded
        if (this.useMockData) {
            // Return zero torque
            return [0, 0, 0];
        }

        try {
            // Create ONNX tensor from observation
            const inputTensor = new ort.Tensor('float32', new Float32Array(observation), [1, 16]);

            // Run inference
            const feeds = { observation: inputTensor };
            const results = await this.rlSession.run(feeds);

            // Get output (action)
            const output = results.action.data;
            const action = Array.from(output.slice(0, 3));

            return action;
        } catch (error) {
            console.error('RL action prediction error:', error);
            throw error;
        }
    }

    // ============== IMAGE PREPROCESSING ==============

    /**
     * Preprocess image data to match PyTorch transforms
     * Resize to targetSize x targetSize, normalize to [-1, 1]
     */
    preprocessImage(imageData, targetWidth, targetHeight) {
        // Create temporary canvas for resizing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = targetWidth;
        tempCanvas.height = targetHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw resized image
        const sourceCanvas = document.createElement('canvas');
        sourceCanvas.width = imageData.width;
        sourceCanvas.height = imageData.height;
        const sourceCtx = sourceCanvas.getContext('2d');
        sourceCtx.putImageData(imageData, 0, 0);

        tempCtx.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight);
        const resizedData = tempCtx.getImageData(0, 0, targetWidth, targetHeight);

        // Convert to tensor format: [1, 3, H, W] with normalization
        // PyTorch transform: Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        // Formula: (pixel / 255 - 0.5) / 0.5 = (pixel / 255 - 0.5) * 2 = pixel / 127.5 - 1
        const tensor = new Float32Array(3 * targetHeight * targetWidth);
        const pixels = resizedData.data;

        for (let i = 0; i < targetHeight; i++) {
            for (let j = 0; j < targetWidth; j++) {
                const idx = (i * targetWidth + j) * 4;
                const r = pixels[idx] / 127.5 - 1.0;
                const g = pixels[idx + 1] / 127.5 - 1.0;
                const b = pixels[idx + 2] / 127.5 - 1.0;

                // CHW format (channels first)
                tensor[0 * targetHeight * targetWidth + i * targetWidth + j] = r;
                tensor[1 * targetHeight * targetWidth + i * targetWidth + j] = g;
                tensor[2 * targetHeight * targetWidth + i * targetWidth + j] = b;
            }
        }

        return tensor;
    }

    // ============== HELPER FUNCTIONS ==============

    /**
     * Normalize quaternion to unit length
     */
    normalizeQuaternion(q) {
        const norm = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (norm < 1e-8) {
            return [0, 0, 0, 1];
        }
        return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
    }

    /**
     * Convert quaternion [x, y, z, w] to Euler angles [x, y, z] in degrees
     */
    quaternionToEuler(q) {
        const [x, y, z, w] = q;

        // Roll (x-axis rotation)
        const sinr_cosp = 2 * (w * x + y * z);
        const cosr_cosp = 1 - 2 * (x * x + y * y);
        const roll = Math.atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        const sinp = 2 * (w * y - z * x);
        let pitch;
        if (Math.abs(sinp) >= 1) {
            pitch = Math.sign(sinp) * Math.PI / 2;
        } else {
            pitch = Math.asin(sinp);
        }

        // Yaw (z-axis rotation)
        const siny_cosp = 2 * (w * z + x * y);
        const cosy_cosp = 1 - 2 * (y * y + z * z);
        const yaw = Math.atan2(siny_cosp, cosy_cosp);

        // Convert to degrees
        return [
            roll * 180 / Math.PI,
            pitch * 180 / Math.PI,
            yaw * 180 / Math.PI
        ];
    }

    /**
     * Check if models are ready
     */
    ready() {
        return this.isReady;
    }

    /**
     * Get loading status
     */
    loading() {
        return this.isLoading;
    }
}
