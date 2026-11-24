#!/usr/bin/env python3
"""Generate the demo ONNX classifier used by the service.

Graph: input[float32, batch,1,64,64] -> Flatten -> Gemm(4096 -> 3) -> logits
Weights are deterministic (fixed seed) with a small brightness prior so
different images actually land in different classes.
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper

NUM_CLASSES = 3
FLAT_DIM = 64 * 64


def build(path: str) -> None:
    rng = np.random.default_rng(20261123)
    W = rng.normal(0.0, 0.02, size=(FLAT_DIM, NUM_CLASSES)).astype(np.float32)
    W[:, 0] += 1.0 / FLAT_DIM   # class 0 logit rises with brightness
    W[:, 2] -= 1.0 / FLAT_DIM   # class 2 logit falls with brightness
    b = np.array([0.0, 0.1, 0.0], dtype=np.float32)

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 1, 64, 64]
    )
    out = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["batch", NUM_CLASSES]
    )
    nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "b"], ["logits"],
                         alpha=1.0, beta=1.0, transB=0),
    ]
    graph = helper.make_graph(
        nodes,
        "gray64_classifier",
        [inp],
        [out],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.ravel()),
            helper.make_tensor("b", TensorProto.FLOAT, b.shape, b),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="vision-edge",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8  # stay compatible with older onnxruntime builds
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    onnx.save(model, path)


def smoke_test(path: str) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).integers(
        0, 256, size=(2, 1, 64, 64)
    ).astype(np.float32) / 255.0
    logits = sess.run(["logits"], {"input": x})[0]
    assert logits.shape == (2, NUM_CLASSES), logits.shape
    print(f"smoke test ok, logits[0]={logits[0].round(3).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="app/model/classifier.onnx")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    build(args.out)
    print(f"model written to {args.out}")
    if args.smoke_test:
        smoke_test(args.out)
