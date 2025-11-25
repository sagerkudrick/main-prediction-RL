/**
 * Physics Simulation using Cannon.js
 * Replaces PyBullet backend with client-side physics
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
    }

    /**
     * Initialize the physics world
     */
    start() {
        // Create physics world
        this.world = new CANNON.World();
        this.world.gravity.set(0, -9.81, 0); // Match PyBullet gravity

        // Set timestep to match PyBullet (1/240 seconds)
        this.world.defaultContactMaterial.contactEquationStiffness = 1e7;
        this.world.defaultContactMaterial.contactEquationRelaxation = 4;

        // Broadphase for better performance
        this.world.broadphase = new CANNON.NaiveBroadphase();
        this.world.solver.iterations = 10;

        // Create ground plane
        const groundShape = new CANNON.Plane();
        this.planeBody = new CANNON.Body({ mass: 0 }); // mass = 0 makes it static
        this.planeBody.addShape(groundShape);
        this.planeBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0); // Rotate to be horizontal
        this.world.addBody(this.planeBody);

        // Create walls
        this.createWalls(this.wallSize);

        // Reset to spawn the box
        this.reset();

        return true;
    }

    /**
     * Create 4 walls around the center
     */
    createWalls(size) {
        // Remove existing walls
        this.wallBodies.forEach(wall => this.world.removeBody(wall));
        this.wallBodies = [];

        this.wallSize = size;
        const halfSize = size / 1.0;
        const height = 0.5;
        const thickness = 0.05;

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
        // Remove existing box if any
        if (this.boxBody) {
            this.world.removeBody(this.boxBody);
        }

        // Create box shape (0.2 x 0.2 x 0.2 to match the visual mesh)
        const boxShape = new CANNON.Box(new CANNON.Vec3(0.1, 0.1, 0.1));

        // Generate random orientation for ragdoll effect
        const randomAxis = new CANNON.Vec3(
            Math.random() - 0.5,
            Math.random() - 0.5,
            Math.random() - 0.5
        );
        randomAxis.normalize();
        const randomAngle = Math.random() * Math.PI * 2;
        const randomQuat = new CANNON.Quaternion();
        randomQuat.setFromAxisAngle(randomAxis, randomAngle);

        // Create box body with mass
        this.boxBody = new CANNON.Body({
            mass: 1.0, // 1kg mass
            shape: boxShape,
            position: new CANNON.Vec3(0, 0.5, 0), // Start in center, slightly elevated
            quaternion: randomQuat, // Random orientation
            linearDamping: 0.01,
            angularDamping: 0.01
        });

        // Disable sleeping to prevent "freezing"
        this.boxBody.allowSleep = false;
        this.boxBody.sleepState = CANNON.Body.AWAKE;

        // Add to world
        this.world.addBody(this.boxBody);

        // Reset state
        this.stepCount = 0;
        this.currentAction.set(0, 0, 0);
        this.predictedQuaternion = null;

        return this.getState();
    }

    /**
     * Step the physics simulation
     */
    step(numSteps = 4) {
        const timeStep = 1.0 / 240.0; // Match PyBullet timestep

        for (let i = 0; i < numSteps; i++) {
            // Apply torque (scaled like PyBullet: action * 0.1)
            const torque = new CANNON.Vec3(
                this.currentAction.x * 0.1,
                this.currentAction.y * 0.1,
                this.currentAction.z * 0.1
            );

            // Apply torque in local frame (LINK_FRAME equivalent)
            const worldTorque = new CANNON.Vec3();
            this.boxBody.quaternion.vmult(torque, worldTorque);
            this.boxBody.torque.copy(worldTorque);

            // Step physics
            this.world.step(timeStep);
            this.stepCount++;
        }

        return this.getState();
    }

    /**
     * Get current physics state
     */
    getState() {
        const pos = this.boxBody.position;
        const quat = this.boxBody.quaternion;
        const angVel = this.boxBody.angularVelocity;

        // Check termination (matching PyBullet logic)
        const terminated = this.stepCount >= 500 || pos.y < 0.05;

        return {
            position: [pos.x, pos.y, pos.z],
            quaternion: [quat.x, quat.y, quat.z, quat.w],
            angular_velocity: [angVel.x, angVel.y, angVel.z],
            step_count: this.stepCount,
            terminated: terminated,
            action: [this.currentAction.x, this.currentAction.y, this.currentAction.z]
        };
    }

    /**
     * Get observation for RL model (matching PyBullet format)
     */
    getObservation() {
        const quat = this.boxBody.quaternion;
        const angVel = this.boxBody.angularVelocity;

        // Compute z-axis from quaternion
        const zAxis = this.quaternionToZAxis(quat);

        // Compute orientation one-hot
        const orientation = this.zAxisToOrientationOneHot(zAxis);

        // Construct observation: quat (4) + ang_vel (3) + z_axis (3) + orientation (6)
        return [
            quat.x, quat.y, quat.z, quat.w,
            angVel.x, angVel.y, angVel.z,
            zAxis[0], zAxis[1], zAxis[2],
            ...orientation
        ];
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
        this.currentAction.set(action[0], action[1], action[2]);
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

        // Reset velocities
        this.boxBody.velocity.set(0, 0, 0);
        this.boxBody.angularVelocity.set(0, 0, 0);

        // Set position
        this.boxBody.position.set(position[0], position[1], position[2]);

        // Set quaternion if provided
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

    // ============== MOUSE INTERACTION ==============

    /**
     * Grab the box at a specific point
     * @param {Array} point - World point [x, y, z] where the click happened
     */
    grabBox(point) {
        if (!this.boxBody) return;

        // Create a kinematic body for the mouse cursor
        if (!this.mouseBody) {
            this.mouseBody = new CANNON.Body({
                mass: 0, // static/kinematic
                type: CANNON.Body.KINEMATIC,
                position: new CANNON.Vec3(point[0], point[1], point[2])
            });
            this.mouseBody.collisionFilterGroup = 0; // Don't collide with anything
            this.world.addBody(this.mouseBody);
        } else {
            this.mouseBody.position.set(point[0], point[1], point[2]);
        }

        // Calculate local point on the box
        const worldPoint = new CANNON.Vec3(point[0], point[1], point[2]);
        const localPoint = new CANNON.Vec3();
        this.boxBody.pointToLocalFrame(worldPoint, localPoint);

        // Create constraint
        this.mouseConstraint = new CANNON.PointToPointConstraint(
            this.boxBody,
            localPoint,
            this.mouseBody,
            new CANNON.Vec3(0, 0, 0)
        );

        this.world.addConstraint(this.mouseConstraint);

        // Wake up the body
        this.boxBody.wakeUp();
    }

    /**
     * Move the grabbed point
     * @param {Array} point - New world point [x, y, z]
     */
    moveGrabbed(point) {
        if (this.mouseBody) {
            this.mouseBody.position.set(point[0], point[1], point[2]);
            // Wake up body to ensure physics keeps running
            if (this.boxBody) this.boxBody.wakeUp();
        }
    }

    /**
     * Release the box
     */
    releaseBox() {
        if (this.mouseConstraint) {
            this.world.removeConstraint(this.mouseConstraint);
            this.mouseConstraint = null;
        }
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
     * Convert quaternion to z-axis vector
     */
    quaternionToZAxis(quat) {
        // Rotation matrix from quaternion
        const x = quat.x, y = quat.y, z = quat.z, w = quat.w;

        // Z-axis is the third column of rotation matrix
        const zx = 2 * (x * z + w * y);
        const zy = 2 * (y * z - w * x);
        const zz = 1 - 2 * (x * x + y * y);

        return [zx, zy, zz];
    }

    /**
     * Convert z-axis to one-hot orientation (matching training)
     */
    zAxisToOrientationOneHot(zAxis) {
        const threshold = 0.7;
        const orientation = [0, 0, 0, 0, 0, 0];

        if (zAxis[2] > threshold) {
            orientation[0] = 1; // up
        } else if (zAxis[2] < -threshold) {
            orientation[1] = 1; // down
        } else if (zAxis[0] > threshold) {
            orientation[3] = 1; // right
        } else if (zAxis[0] < -threshold) {
            orientation[2] = 1; // left
        } else if (zAxis[1] > threshold) {
            orientation[4] = 1; // forward
        } else if (zAxis[1] < -threshold) {
            orientation[5] = 1; // back
        }

        return orientation;
    }

    /**
     * Cleanup
     */
    close() {
        if (this.world) {
            // Cannon.js doesn't need explicit cleanup, but we can clear references
            this.world = null;
            this.boxBody = null;
            this.planeBody = null;
            this.wallBodies = [];
        }
    }
}
