/**
 * Visual Torque Debugger for Three.js
 * Adds visualization of torque vectors and action axes with text labels
 */

import * as THREE from 'three';

export class VisualTorqueDebugger {
  constructor(scene, crateMesh) {
    this.scene = scene;
    this.crateMesh = crateMesh;

    // Store ArrowHelpers
    this.arrows = {
      x: null,
      y: null,
      z: null
    };

    // Store Text Sprites
    this.labels = {
      x: null,
      y: null,
      z: null
    };

    this.init();
  }

  init() {
    // Create arrows and labels for X, Y, Z axes
    this.arrows.x = this._createArrow(0xff0000); // Red
    this.labels.x = this._createLabel(0xff0000);

    this.arrows.y = this._createArrow(0x00ff00); // Green
    this.labels.y = this._createLabel(0x00ff00);

    this.arrows.z = this._createArrow(0x0000ff); // Blue
    this.labels.z = this._createLabel(0x0000ff);
  }

  _createArrow(color) {
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0), // Placeholder dir
      new THREE.Vector3(0, 0, 0), // Placeholder origin
      0,                          // Length 0 initially
      color
    );
    this.scene.add(arrow);
    return arrow;
  }

  _createLabel(colorHex) {
    // Initial creation with placeholder text
    const sprite = this._makeTextSprite(" ", {
      textColor: new THREE.Color(colorHex)
    });
    sprite.visible = false;
    this.scene.add(sprite);
    return sprite;
  }

  updateTorque(torqueX, torqueY, torqueZ) {
    // Optional: visualize the actual applied torque (smoothed)
  }

  updateActionAxes(quaternion, action) {
    // Debug logging
    console.log("VisTorque Update:", action);
    this._updateArrows(action);
  }

  _updateArrows(vector3Array) {
    if (!this.crateMesh) return;

    const origin = this.crateMesh.position;
    const quat = this.crateMesh.quaternion;

    const scale = 0.5;
    // Removed minVis check to FORCE visibility for debugging

    // X Axis
    this._updateSingleArrow(this.arrows.x, this.labels.x, 0, vector3Array[0], origin, quat, scale, "X");

    // Y Axis
    this._updateSingleArrow(this.arrows.y, this.labels.y, 1, vector3Array[1], origin, quat, scale, "Y");

    // Z Axis
    this._updateSingleArrow(this.arrows.z, this.labels.z, 2, vector3Array[2], origin, quat, scale, "Z");
  }

  _updateSingleArrow(arrow, label, axisIndex, value, origin, quart, scale, axisName) {
    const absVal = Math.abs(value);

    // FORCE VISIBLE even if 0
    arrow.visible = true;
    label.visible = true;

    // Determine local direction
    const localDir = new THREE.Vector3(
      axisIndex === 0 ? 1 : 0,
      axisIndex === 1 ? 1 : 0,
      axisIndex === 2 ? 1 : 0
    );

    const sign = value >= 0 ? "+" : "-";
    if (value < 0) {
      localDir.negate();
    }

    // Rotate local direction to world space
    const worldDir = localDir.clone().applyQuaternion(quart);

    // Update Arrow
    // Ensure at least a tiny bit of length so it can be seen
    const arrowLen = Math.max(1, absVal * scale);
    arrow.position.copy(origin);
    arrow.setDirection(worldDir);
    arrow.setLength(arrowLen, 0.2 * scale, 0.1 * scale);

    // Update Label
    const labelPos = origin.clone().add(worldDir.clone().multiplyScalar(arrowLen + 0.2));
    label.position.copy(labelPos);

    this._updateSpriteText(label, `${sign}${axisName}: ${absVal.toFixed(2)}`);
  }

  _makeTextSprite(message, parameters) {
    if (parameters === undefined) parameters = {};

    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    const context = canvas.getContext('2d');

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;

    // CRITICAL for visibility: transparent: true, depthTest: false
    const spriteMaterial = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false
    });

    const sprite = new THREE.Sprite(spriteMaterial);
    // World size: 1 unit wide
    sprite.scale.set(1.0, 0.5, 1);
    sprite.renderOrder = 999; // Render on top

    sprite.userData = {
      context: context,
      canvas: canvas,
      texture: texture,
      parameters: parameters,
      lastMessage: null
    };

    return sprite;
  }

  _updateSpriteText(sprite, message) {
    if (sprite.userData.lastMessage === message) return;

    sprite.userData.lastMessage = message;
    const ctx = sprite.userData.context;
    const canvas = sprite.userData.canvas;
    const params = sprite.userData.parameters;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const fontsize = 64;
    ctx.font = "900 " + fontsize + "px Arial";

    const col = params.textColor || new THREE.Color(1, 1, 1);
    const r = Math.floor(col.r * 255);
    const g = Math.floor(col.g * 255);
    const b = Math.floor(col.b * 255);

    // Stroke for contrast
    ctx.strokeStyle = "rgba(0,0,0, 1.0)";
    ctx.lineWidth = 8;
    ctx.lineJoin = "round";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.strokeText(message, canvas.width / 2, canvas.height / 2);

    // Fill
    ctx.fillStyle = `rgba(${r},${g},${b}, 1.0)`;
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);

    sprite.material.map.needsUpdate = true;
  }

  clear() {
    this.scene.remove(this.arrows.x);
    this.scene.remove(this.arrows.y);
    this.scene.remove(this.arrows.z);
    this.scene.remove(this.labels.x);
    this.scene.remove(this.labels.y);
    this.scene.remove(this.labels.z);
  }
}