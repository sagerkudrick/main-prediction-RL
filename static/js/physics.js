/**
 * Physics Simulation using Cannon.js - PYBULLET COMPATIBLE
 * Observation format: quat (4) + ang_vel (3) + z_axis (3) + orientation_onehot (6) = 16 features
 * Training used: quat(4) + z_axis(3) + z_noisy(3) + prev_quat(4) + ang_vel_est(3) + action_t-1(3) + action_t-2(3) = 23
 */
import * as CANNON from 'https://cdn.jsdelivr.net/npm/cannon-es@0.20.0/dist/cannon-es.js';

export class PhysicsSimulation {
    constructor() {
        this.world = null;
        this.boxBody = null;
        this.planeBody = null;
        this.wallBodies = [];
        this.stepCount = 0;
        this.currentAction = new CANNON.Vec3(0, 0, 0);
        this.wallSize = 1.0;
        this.rlEnabled = false;
        this.predictedQuaternion = null;
        this.boxSize = new CANNON.Vec3(0.1, 0.1, 0.1);
        this.meshData = null;

        // History for matching training format
        this.oriHistory = [];
        this.actionHistory = [];
        this.angVelEstimate = [0, 0, 0];

        // Torque ramping (from training)
        this.maxTorque = 0.5;
        this.rampSpeed = 0.5;
        this.targetTorque = [0, 0, 0];
        this.currentTorqueApplied = [0, 0, 0];
    }

    /**
     * Initialize the physics world
     */
    start() {
        this.world = new CANNON.World();
        this.world.gravity.set(0, -9.81, 0);

        this.world.defaultContactMaterial.contactEquationStiffness = 1e6;
        this.world.defaultContactMaterial.contactEquationRelaxation = 3;
        this.world.defaultContactMaterial.friction = 0.3;

        this.world.broadphase = new CANNON.NaiveBroadphase();
        this.world.solver.iterations = 10;

        const groundShape = new CANNON.Plane();
        this.planeBody = new CANNON.Body({ mass: 0 });
        this.planeBody.addShape(groundShape);
        this.planeBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
        this.world.addBody(this.planeBody);

        this.createWalls(this.wallSize);
        this.reset();

        return true;
    }

    /**
     * Create 4 walls around the center
     */
    createWalls(size) {
        this.wallBodies.forEach(wall => this.world.removeBody(wall));
        this.wallBodies = [];

        this.wallSize = size;
        const halfSize = size / 2.0;
        const height = 2.0;
        const thickness = 0.1;

        // North wall (Z-)
        const northShape = new CANNON.Box(new CANNON.Vec3(halfSize + thickness, height, thickness));
        const northBody = new CANNON.Body({ mass: 0 });
        northBody.addShape(northShape);
        northBody.position.set(0, height, -halfSize - thickness);
        this.world.addBody(northBody);
        this.wallBodies.push(northBody);

        // South wall (Z+)
        const southShape = new CANNON.Box(new CANNON.Vec3(halfSize + thickness, height, thickness));
        const southBody = new CANNON.Body({ mass: 0 });
        southBody.addShape(southShape);
        southBody.position.set(0, height, halfSize + thickness);
        this.world.addBody(southBody);
        this.wallBodies.push(southBody);

        // East wall (X+)
        const eastShape = new CANNON.Box(new CANNON.Vec3(thickness, height, halfSize));
        const eastBody = new CANNON.Body({ mass: 0 });
        eastBody.addShape(eastShape);
        eastBody.position.set(halfSize + thickness, height, 0);
        this.world.addBody(eastBody);
        this.wallBodies.push(eastBody);

        // West wall (X-)
        const westShape = new CANNON.Box(new CANNON.Vec3(thickness, height, halfSize));
        const westBody = new CANNON.Body({ mass: 0 });
        westBody.addShape(westShape);
        westBody.position.set(-halfSize - thickness, height, 0);
        this.world.addBody(westBody);
        this.wallBodies.push(westBody);
    }

    /**
     * Reset the box to initial position
     */
    reset() {
        if (this.boxBody && this.world) {
            this.world.removeBody(this.boxBody);
        }

        let shape;
        if (this.meshData) {
            const vertices = this.meshData.vertices;
            const indices = this.meshData.indices;
            shape = new CANNON.Trimesh(vertices, indices);
        } else {
            shape = new CANNON.Box(this.boxSize);
        }

        // Random orientation
        const randomAxis = new CANNON.Vec3(
            Math.random() - 0.5,
            Math.random() - 0.5,
            Math.random() - 0.5
        );
        randomAxis.normalize();
        const randomAngle = Math.random() * Math.PI * 2;
        const randomQuat = new CANNON.Quaternion();
        randomQuat.setFromAxisAngle(randomAxis, randomAngle);

        this.boxBody = new CANNON.Body({
            mass: 0.05, // Matching URDF mass (was 1.0)
            shape: shape,
            position: new CANNON.Vec3(0, 1.5, 0),
            quaternion: randomQuat,
            linearDamping: 0.04,
            angularDamping: 0.04
        });

        // Override inertia to match URDF exactly [0.0001, 0.0001, 0.0001]
        // This is critical because Trimesh auto-inertia is often wrong/infinite
        this.boxBody.inertia.set(0.0001, 0.0001, 0.0001);
        this.boxBody.invInertia.set(10000, 10000, 10000); // 1 / 0.0001

        this.boxBody.allowSleep = false;

        if (this.world) {
            this.world.addBody(this.boxBody);
        }

        this.stepCount = 0;
        this.currentAction.set(0, 0, 0);
        this.currentTorqueApplied = [0, 0, 0];
        this.targetTorque = [0, 0, 0];

        // Initialize history
        this.oriHistory = [
            [randomQuat.x, randomQuat.y, randomQuat.z, randomQuat.w],
            [randomQuat.x, randomQuat.y, randomQuat.z, randomQuat.w]
        ];
        this.actionHistory = [
            [0, 0, 0],
            [0, 0, 0]
        ];
        this.angVelEstimate = [0, 0, 0];

        this.predictedQuaternion = null;
        return this.getState();
    }

    /**
     * Step the physics simulation
     */
    /**
        * Step the physics simulation
        */
    step(numSteps = 4) {
        const timeStep = 1.0 / 240.0;

        for (let i = 0; i < numSteps; i++) {
            // Random disturbance (3% chance from training)
            if (Math.random() < 0.03) {
                const dx = (Math.random() - 0.5) * 0.4;
                const dy = (Math.random() - 0.5) * 0.4;
                const dz = (Math.random() - 0.5) * 0.4;

                const worldDist = new CANNON.Vec3(dx, dy, dz);
                this.boxBody.quaternion.vmult(worldDist, worldDist);
                this.boxBody.torque.vadd(worldDist, this.boxBody.torque);
            }

            // Torque ramping
            this.targetTorque[0] = this.currentAction.x * this.maxTorque;
            this.targetTorque[1] = this.currentAction.y * this.maxTorque;
            this.targetTorque[2] = this.currentAction.z * this.maxTorque;

            // Low-pass filter
            this.currentTorqueApplied[0] = this.currentTorqueApplied[0] * (1.0 - this.rampSpeed) + this.targetTorque[0] * this.rampSpeed;
            this.currentTorqueApplied[1] = this.currentTorqueApplied[1] * (1.0 - this.rampSpeed) + this.targetTorque[1] * this.rampSpeed;
            this.currentTorqueApplied[2] = this.currentTorqueApplied[2] * (1.0 - this.rampSpeed) + this.targetTorque[2] * this.rampSpeed;

            // Clamp
            this.currentTorqueApplied[0] = Math.max(-this.maxTorque, Math.min(this.maxTorque, this.currentTorqueApplied[0]));
            this.currentTorqueApplied[1] = Math.max(-this.maxTorque, Math.min(this.maxTorque, this.currentTorqueApplied[1]));
            this.currentTorqueApplied[2] = Math.max(-this.maxTorque, Math.min(this.maxTorque, this.currentTorqueApplied[2]));

            // Apply torque in local frame
            const torque = new CANNON.Vec3(
                this.currentTorqueApplied[0],
                this.currentTorqueApplied[1],
                this.currentTorqueApplied[2]
            );
            const worldTorque = new CANNON.Vec3();
            this.boxBody.quaternion.vmult(torque, worldTorque);
            this.boxBody.torque.vadd(worldTorque, this.boxBody.torque);

            // Step physics once
            this.world.step(timeStep);
            this.stepCount++;

            // Callback on first substep only
            if (i === 0 && this.onTorqueApplied) {
                this.onTorqueApplied(this.currentTorqueApplied);
            }
        }

        // Update history after all substeps
        this._updateHistory();

        return this.getState();
    }


    /**
     * Update observation history
     */
    _updateHistory() {
        const quat = this.boxBody.quaternion;

        // Shift history
        this.oriHistory[0] = this.oriHistory[1];
        this.oriHistory[1] = [quat.x, quat.y, quat.z, quat.w];

        // Estimate angular velocity from quaternion delta
        const dq = [
            this.oriHistory[1][0] - this.oriHistory[0][0],
            this.oriHistory[1][1] - this.oriHistory[0][1],
            this.oriHistory[1][2] - this.oriHistory[0][2]
        ];
        this.angVelEstimate = [
            Math.max(-1.0, Math.min(1.0, dq[0])),
            Math.max(-1.0, Math.min(1.0, dq[1])),
            Math.max(-1.0, Math.min(1.0, dq[2]))
        ];
    }

    /**
     * Get current physics state
     */
    getState() {
        const pos = this.boxBody.position;
        const quat = this.boxBody.quaternion;
        const angVel = this.boxBody.angularVelocity;

        const canonicalQuat = this.normalizeAndCanonicalizeQuaternion([quat.x, quat.y, quat.z, quat.w]);
        const terminated = this.stepCount >= 500 || pos.y < 0.05;

        return {
            position: [pos.x, pos.y, pos.z],
            quaternion: canonicalQuat,
            angular_velocity: [angVel.x, angVel.y, angVel.z],
            step_count: this.stepCount,
            terminated: terminated,
            action: [this.currentAction.x, this.currentAction.y, this.currentAction.z]
        };
    }

    /**
     * Get observation matching training format (23 features)
     */
    getObservation() {
        if (!this.boxBody) {
            return new Float32Array(23);
        }

        const quat = this.boxBody.quaternion;

        // Compute z-axis
        //const zAxis = this.quaternionToZAxis(quat);
        const zAxis = this.quaternionToZAxis(this.predictedQuaternion);

        // 23 features matching training
        return new Float32Array([
            quat.x, quat.y, quat.z, quat.w,                                      // 4: current quat
            zAxis[0], zAxis[1], zAxis[2],                                         // 3: z-axis
            zAxis[0], zAxis[1], zAxis[2],                                      // 3: noisy z-axis
            this.oriHistory[0][0], this.oriHistory[0][1], this.oriHistory[0][2], this.oriHistory[0][3],  // 4: prev quat
            this.angVelEstimate[0], this.angVelEstimate[1], this.angVelEstimate[2],  // 3: ang vel estimate
            this.actionHistory[0][0], this.actionHistory[0][1], this.actionHistory[0][2],  // 3: action t-1
            this.actionHistory[1][0], this.actionHistory[1][1], this.actionHistory[1][2]   // 3: action t-2
        ]);
    }

    /**
     * Set predicted quaternion from inference
     */
    setPredictedQuaternion(quat) {
        this.predictedQuaternion = this.normalizeQuaternion(quat);
    }

    /**
     * Set RL action to apply
     */
    setAction(action) {
        // Update action history
        let x_axis = action[0];
        let y_axis = -action[2];
        let z_axis = action[1];

        this.actionHistory[0] = this.actionHistory[1];
        this.actionHistory[1] = [x_axis, y_axis, z_axis];

        // FLIP Y and Z axes to match Cannon.js (PyBullet Z-up → Cannon Y-up)
        // PyBullet: X, Y, Z → Cannon: X, Z, Y (with Y flip)
        this.currentAction.set(
            x_axis,      // X stays the same
            y_axis,      // Z becomes Y
            z_axis      // Y becomes -Z (flipped)
        );

    }

    /**
     * Enable/disable RL control
     */
    setRLEnabled(enabled) {
        this.rlEnabled = enabled;
        if (!enabled) {
            this.currentAction.set(0, 0, 0);
        }
    }

    /**
     * Teleport box to new position/orientation
     */
    teleportBox(position, quaternion = null) {
        if (!this.boxBody) return;

        this.boxBody.velocity.set(0, 0, 0);
        this.boxBody.angularVelocity.set(0, 0, 0);
        this.boxBody.position.set(position[0], position[1], position[2]);

        if (quaternion) {
            this.boxBody.quaternion.set(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
        }
    }

    /**
     * Update wall size
     */
    updateWalls(size) {
        this.createWalls(size);
    }

    /**
     * Set box size
     */
    setBoxSize(x, y, z) {
        this.boxSize.set(x, y, z);
        this.reset();
    }

    /**
     * Set mesh shape
     */
    setMeshShape(vertices, indices) {
        this.meshData = { vertices, indices };
        this.reset();
    }

    // ============== MOUSE INTERACTION ==============

    grabBox(point) {
        if (!this.boxBody) return;

        if (!this.mouseBody) {
            this.mouseBody = new CANNON.Body({
                mass: 0,
                type: CANNON.Body.KINEMATIC,
                position: new CANNON.Vec3(point[0], point[1], point[2])
            });
            this.mouseBody.collisionFilterGroup = 0;
            this.world.addBody(this.mouseBody);
        } else {
            this.mouseBody.position.set(point[0], point[1], point[2]);
        }

        const worldPoint = new CANNON.Vec3(point[0], point[1], point[2]);
        const localPoint = new CANNON.Vec3();
        this.boxBody.pointToLocalFrame(worldPoint, localPoint);

        this.mouseConstraint = new CANNON.PointToPointConstraint(
            this.boxBody,
            localPoint,
            this.mouseBody,
            new CANNON.Vec3(0, 0, 0)
        );
        this.world.addConstraint(this.mouseConstraint);
        this.mouseConstraint.collideConnected = false;

        this.boxBody.linearDamping = 0.5;
        this.boxBody.angularDamping = 0.5;
        this.boxBody.wakeUp();
    }

    moveGrabbed(point) {
        if (this.mouseBody) {
            this.mouseBody.position.set(point[0], point[1], point[2]);
            if (this.boxBody) this.boxBody.wakeUp();
        }
    }

    releaseBox() {
        if (this.mouseConstraint) {
            this.world.removeConstraint(this.mouseConstraint);
            this.mouseConstraint = null;
        }
        if (this.boxBody) {
            this.boxBody.linearDamping = 0.04;
            this.boxBody.angularDamping = 0.04;
        }
    }

    // ============== HELPER FUNCTIONS ==============

    normalizeQuaternion(q) {
        const norm = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (norm < 1e-8) {
            return [0, 0, 0, 1];
        }
        return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
    }

    normalizeAndCanonicalizeQuaternion(q) {
        const normalized = this.normalizeQuaternion(q);
        if (normalized[3] < 0 || (normalized[3] === 0 && normalized[0] < 0)) {
            return [-normalized[0], -normalized[1], -normalized[2], -normalized[3]];
        }
        return normalized;
    }

    quaternionToZAxis(quat) {
        const x = quat.x, y = quat.y, z = quat.z, w = quat.w;
        const zx = 2 * (x * z + w * y);
        const zy = 2 * (y * z - w * x);
        const zz = 1 - 2 * (x * x + y * y);
        return [zx, zy, zz];
    }

    zAxisToOrientationOneHot(zAxis) {
        const threshold = 0.7;
        const orientation = [0, 0, 0, 0, 0, 0];

        if (zAxis[2] > threshold) orientation[0] = 1;
        else if (zAxis[2] < -threshold) orientation[1] = 1;
        else if (zAxis[0] > threshold) orientation[3] = 1;
        else if (zAxis[0] < -threshold) orientation[2] = 1;
        else if (zAxis[1] > threshold) orientation[4] = 1;
        else if (zAxis[1] < -threshold) orientation[5] = 1;

        return orientation;
    }

    close() {
        if (this.world) {
            this.world = null;
            this.boxBody = null;
            this.planeBody = null;
            this.wallBodies = [];
        }
    }
}