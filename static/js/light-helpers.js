/**
 * Light Helpers
 * Adds visual axis helpers to lights so you can see their position and orientation
 */

import * as THREE from 'three';

/**
 * Add visual helpers to a light
 * @param {THREE.Light} light - The light to add helpers to
 * @param {number} axesSize - Size of the axes helper
 * @param {number} markerColor - Color of the marker sphere
 * @param {number} markerSize - Size of the marker sphere
 */
export function addLightHelper(light, axesSize = 0.5, markerColor = 0xffff00, markerSize = 0.1) {
    // Add axis helper to show orientation
    const axes = new THREE.AxesHelper(axesSize);
    light.add(axes);

    // Add marker sphere to show position
    const marker = new THREE.Mesh(
        new THREE.SphereGeometry(markerSize, 16, 16),
        new THREE.MeshBasicMaterial({ color: markerColor })
    );
    light.add(marker);

    return { axes, marker };
}
