#!/usr/bin/env python3
"""
validate_onnx.py
Quick inspector + dummy-run for ONNX models using onnxruntime (CPU).
Usage: python validate_onnx.py rl-model.onnx
"""

import sys
import numpy as np
import onnxruntime as ort

def inspect_model(path):
    print("Loading ONNX model:", path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # print input metadata
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print("\nInputs:")
    for i in inputs:
        print(" - name:", i.name)
        print("   shape (proto):", i.shape)
        print("   type:", i.type)
    print("\nOutputs:")
    for o in outputs:
        print(" - name:", o.name)
        print("   shape (proto):", o.shape)
        print("   type:", o.type)
    return sess

def run_dummy(sess):
    # Build dummy input(s) according to input metadata
    inputs = sess.get_inputs()
    feed = {}
    for i in inputs:
        # try to infer shape: if first dim is symbolic, use 1 for batch
        shape = []
        for dim in i.shape:
            if isinstance(dim, str) or dim is None:
                shape.append(1)
            else:
                shape.append(int(dim))
        total = int(np.prod(shape))
        x = np.random.randn(*shape).astype(np.float32)
        feed[i.name] = x
        print(f"\nPrepared dummy for '{i.name}' shape {x.shape}, dtype {x.dtype}")

    print("\nRunning session.run(...) with dummy inputs...")
    out = sess.run(None, feed)
    print("Got", len(out), "outputs")
    for idx, o in enumerate(out):
        print(f" output[{idx}] shape={np.shape(o)} dtype={o.dtype}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_onnx.py path/to/rl-model.onnx")
        sys.exit(1)
    path = sys.argv[1]
    sess = inspect_model(path)
    run_dummy(sess)
