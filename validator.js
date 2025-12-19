import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Code } from 'lucide-react';

export default function InferenceAxisDebugger() {
  const [findings, setFindings] = useState([]);

  const runValidation = () => {
    const results = [];

    // What's happening in InferenceManager
    results.push({
      severity: 'info',
      title: '✓ InferenceManager.js - Pose Prediction',
      details: `
YOUR CODE:
1. Renders 224×224 image from canvas
2. ImageNet normalization (correct)
3. Runs ONNX pose model
4. Returns normalized quaternion [x,y,z,w]

GOOD NEWS:
- Image preprocessing looks correct
- Quaternion normalization looks correct
- Output format is [x,y,z,w] (Three.js format)

QUESTION: What coordinate system does ONNX model output?
- If trained on PyBullet images → outputs Z-up quaternion
- If trained on Cannon images → outputs Y-up quaternion
- Most likely: PyBullet Z-up (since training code uses PyBullet)
      `,
      status: 'ok'
    });

    // The real bug location
    results.push({
      severity: 'critical',
      title: '❌ WHERE THE BUG PROBABLY IS',
      details: `
Your InferenceManager returns:
  { quaternion: [x, y, z, w] }  ← PyBullet Z-up format (probably)

But this needs to be CONVERTED before use in Cannon.js!

The problem is likely in your SimulationController or index.html:

CURRENT (WRONG):
  const predResult = await inference.predictPose(canvas);
  simulation.setPredictedQuaternion(predResult.quaternion);
  // ← This directly uses PyBullet quat in Cannon world!

SHOULD BE:
  const predResult = await inference.predictPose(canvas);
  const convertedQuat = convertPyBulletToCannon(predResult.quaternion);
  simulation.setPredictedQuaternion(convertedQuat);

CONVERSION FUNCTION:
function convertPyBulletToCannon(q_pb) {
  // PyBullet: X-forward, Y-right, Z-up
  // Cannon:   X-forward, Y-up, Z-right
  
  // Swap Y and Z components
  return [
    q_pb[0],      // x stays the same
    q_pb[2],      // z → y (was vertical, still vertical but different axis)
    q_pb[1],      // y → z (was right, now along forward-back)
    q_pb[3]       // w unchanged
  ];
}

OR (if axes are inverted):
  return [
    q_pb[0],
    -q_pb[2],     // negate if Z needs to flip
    -q_pb[1],
    q_pb[3]
  ];
      `,
      status: 'error'
    });

    // Observation pipeline
    results.push({
      severity: 'high',
      title: '📊 RL Observation Pipeline',
      details: `
After converting predicted quaternion, you need to use it in RL observation.

QUESTION: How is predicted_quat currently used?

OPTION A - NOT IN OBSERVATION (WRONG):
  getObservation() returns:
    [quat_actual, z_axis, z_noisy, prev_quat, ang_vel, action_t1, action_t2]
  Predicted quat is ignored!

OPTION B - AS GOAL IN REWARD (CORRECT):
  During training, reward was:
    reward = dot(actual_z_axis, [0,0,1]) * 2.0
  
  Should change to:
    target_z = extract_z_from_predicted_quat(convertedQuat)
    reward = dot(actual_z_axis, target_z) * 2.0
  
  But RL model was NOT trained with this!
  → Domain shift: Model expects [0,0,1] goal always

OPTION C - HYBRID (SAFEST):
  Still use [0,0,1] as goal (original training)
  But also compute error from predicted quat
  Add as secondary loss term (requires retraining)

CHECK YOUR CODE:
  - Look at SimulationController.js or simulation.js
  - Search for setPredictedQuaternion()
  - Is it ONLY setting visual representation?
  - Or is it also affecting observation/reward?
      `,
      status: 'warning'
    });

    // Diagnostic steps
    results.push({
      severity: 'high',
      title: '🔧 Debug Steps',
      details: `
1. ADD LOGGING to see actual values:

In your step loop:
  const predQuat = await inference.predictPose(canvas);
  const converted = convertPyBulletToCannon(predQuat.quaternion);
  const actual = simulation.getLastQuaternion();
  
  console.log('Predicted (converted):', converted);
  console.log('Actual current quat:', actual);
  
  // If box is upright both should be ~ [0, 0, 0, 1]
  // If box is tilted, they might differ
  
2. VISUAL TEST:
  - Stop RL (disable checkbox)
  - Manually place box in known pose
  - Render and capture image
  - Check predicted quaternion
  - If it's opposite sign or wrong axis → conversion needed

3. COMPARISON TEST:
  - Let simulation run WITH RL on for 30 steps
  - Log: [predicted_z_axis, actual_z_axis, error]
  - If error increases over time → axes flipped
  - If error decreases → working correctly

4. ISOLATED TEST:
  // Turn off RL
  simulation.setRLEnabled(false);
  
  // Place box upright
  simulation.reset();
  
  // Predict its orientation
  const pred = await inference.predictPose(canvas);
  
  // Both should return ~[0,0,0,1]
  // Log difference
  `;

    results.push({
      severity: 'critical',
      title: '🎯 Most Likely Issues (ranked)',
      details: `
1. [HIGH PROBABILITY] Axis conversion not applied
   Symptom: Predictions look right, but RL goes wrong direction
   Fix: Add Y↔Z swap before using predicted quat
   
2. [MEDIUM PROBABILITY] W sign is flipped
   Symptom: Box tries to flip upside-down
   Fix: If w < 0, negate entire quaternion
   
3. [MEDIUM PROBABILITY] Observation doesn't use predicted quat
   Symptom: RL ignores prediction, always aims for upright
   Fix: Include predicted target in reward calculation
   
4. [LOW PROBABILITY] Image preprocessing is wrong
   Symptom: Predictions are total garbage
   Fix: Verify ImageNet normalization matches training
   
EASIEST TEST:
1. Disable RL
2. Render upright box
3. Check console: predicted quat should be [0, 0, 0, 1]
4. If it's [0, 0, 0, -1], negate w
5. If it's [0, z, y, 1], swap those axes

If predicted matches actual (after conversion), axes are correct!
      `
    });

    // Code snippet
    results.push({
      severity: 'info',
      title: '💻 Copy-Paste Fix',
      details: `
Add this to your simulation.js or wherever you call predictPose():

// At top of file
function convertPyBulletToCannon(q) {
  // PyBullet (Z-up) → Cannon (Y-up)
  // [x, y, z, w] → [x, z, -y, w]
  const converted = [
    q[0],      // x unchanged
    q[2],      // z becomes y (vertical axis)
    -q[1],     // negative y becomes z
    q[3]       // w unchanged
  ];
  
  // Ensure w > 0
  if (converted[3] < 0) {
    return [-converted[0], -converted[1], -converted[2], -converted[3]];
  }
  return converted;
}

// In your step/animate loop:
const predictedQuat = await inference.predictPose(inferenceRenderer);
const convertedQuat = convertPyBulletToCannon(predictedQuat.quaternion);

// Now use converted quat
simulation.setPredictedQuaternion(convertedQuat);

// Log for debugging
if (Math.abs(simulation._lastQuaternion[3]) > 0.01) {
  console.log(
    'Pred:', convertedQuat.map(x => x.toFixed(2)),
    'Actual:', simulation._lastQuaternion.map(x => x.toFixed(2)),
    'Match:', quaternionDistance(convertedQuat, simulation._lastQuaternion).toFixed(2) + '°'
  );
}

function quaternionDistance(q1, q2) {
  const dot = Math.abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]);
  return Math.acos(Math.min(1, dot)) * 2 * 180 / Math.PI;
}
      `
    });

    setFindings(results);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">InferenceManager Axis Validator</h1>
        <p className="text-slate-400 mb-8">Finding where PyBullet→Cannon conversion is missing</p>

        <button
          onClick={runValidation}
          className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded font-semibold mb-8"
        >
          Analyze
        </button>

        <div className="space-y-4">
          {findings.map((finding, i) => (
            <div
              key={i}
              className={`rounded-lg p-4 border-l-4 ${
                finding.severity === 'critical'
                  ? 'bg-red-900 border-red-500'
                  : finding.severity === 'high'
                  ? 'bg-orange-800 border-orange-500'
                  : finding.severity === 'warning'
                  ? 'bg-yellow-800 border-yellow-500'
                  : 'bg-blue-800 border-blue-500'
              }`}
            >
              <h3 className="text-lg font-bold mb-3">{finding.title}</h3>
              <pre className="text-sm whitespace-pre-wrap font-mono bg-slate-900 bg-opacity-50 p-3 rounded border border-slate-700 overflow-x-auto">
                {finding.details}
              </pre>
            </div>
          ))}
        </div>

        {findings.length === 0 && (
          <div className="text-center py-12 text-slate-400">
            Click "Analyze" to check your code
          </div>
        )}
      </div>
    </div>
  );
}