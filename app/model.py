"""Local ONNX classifier running on CPU.

Inference is executed in a thread-pool executor so the asyncio event
loop stays responsive: frames can be ingested and results queried while
a batch is being computed.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

IMAGE_SIDE = 64
IMAGE_BYTES = IMAGE_SIDE * IMAGE_SIDE


class OnnxClassifier:
    def __init__(self, model_path: str, input_name: str = "input",
                 output_name: str = "logits", infer_delay_ms: int = 0):
        import onnxruntime as ort  # imported lazily to keep startup honest

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = input_name
        self._output_name = output_name
        # demo knob: simulate a heavier CPU model so queueing is observable
        self._infer_delay_s = max(0, infer_delay_ms) / 1000.0

    async def infer(self, images: list[bytes]) -> list[tuple[int, list[float]]]:
        """Classify one batch of raw 64x64 grayscale frames."""
        if not images:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._infer_sync, images)

    def _infer_sync(self, images: list[bytes]) -> list[tuple[int, list[float]]]:
        if self._infer_delay_s:
            time.sleep(self._infer_delay_s)  # occupies the worker thread only
        batch = np.stack([self._to_tensor(img) for img in images])
        logits = self._session.run(
            [self._output_name], {self._input_name: batch}
        )[0]
        probs = _softmax(logits)
        return [
            (int(np.argmax(row)), [round(float(x), 6) for x in row])
            for row in probs
        ]

    @staticmethod
    def _to_tensor(image: bytes) -> np.ndarray:
        if len(image) != IMAGE_BYTES:
            raise ValueError(
                f"expected {IMAGE_BYTES} bytes, got {len(image)}"
            )
        arr = np.frombuffer(image, dtype=np.uint8).reshape(1, IMAGE_SIDE, IMAGE_SIDE)
        return arr.astype(np.float32) / 255.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
