#!/usr/bin/env python3
"""OCR engine abstraction for sticker reading.

The pipeline only needs two capabilities: recognize a batch of pre-localized
single text lines, and (optionally) run full-frame detection.  Recognition
quality on a ~12 px tall serial line decides whether a distant row can be read
at all, so the newer ``rapidocr`` package (PP-OCRv6 models) is preferred and
``rapidocr_onnxruntime`` (PP-OCRv3) is kept as a fallback.
"""

from __future__ import annotations

import numpy as np

_NEW_PACKAGE_ERROR: Exception | None = None
_LEGACY_PACKAGE_ERROR: Exception | None = None

try:  # PP-OCRv6, materially better on small text
    from rapidocr import RapidOCR as _NewRapidOCR
    from rapidocr.ch_ppocr_rec.typings import TextRecInput as _TextRecInput
except Exception as error:  # pragma: no cover - depends on environment
    _NewRapidOCR = None
    _TextRecInput = None
    _NEW_PACKAGE_ERROR = error

try:  # PP-OCRv3
    from rapidocr_onnxruntime import RapidOCR as _LegacyRapidOCR
except Exception as error:  # pragma: no cover - depends on environment
    _LegacyRapidOCR = None
    _LEGACY_PACKAGE_ERROR = error


class RecognitionEngine:
    """Batched single-line text recognition with a stable interface."""

    def __init__(self, threads: int = 4, prefer: str = "auto") -> None:
        self.backend = ""
        self._engine = None
        if prefer in ("auto", "v6") and _NewRapidOCR is not None:
            self._engine = _NewRapidOCR()
            self.backend = "rapidocr-v6"
        elif _LegacyRapidOCR is not None:
            self._engine = _LegacyRapidOCR(
                intra_op_num_threads=max(1, threads), inter_op_num_threads=1
            )
            self.backend = "rapidocr-onnxruntime-v3"
        else:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "no OCR backend available: "
                f"rapidocr={_NEW_PACKAGE_ERROR!r} "
                f"rapidocr_onnxruntime={_LEGACY_PACKAGE_ERROR!r}"
            )

    def recognize_lines(
        self, views: list[np.ndarray]
    ) -> list[tuple[str, float]]:
        """Recognize each image as one text line, without text detection."""
        if not views:
            return []
        if self.backend == "rapidocr-v6":
            output = self._engine.text_rec(_TextRecInput(img=views))
            return [
                (str(text), float(score))
                for text, score in zip(output.txts, output.scores)
            ]
        results, _ = self._engine.text_recognizer(views)
        return [(str(text), float(score)) for text, score in results]

    def detect_lines(self, image: np.ndarray) -> list[dict]:
        """Full-frame detection plus recognition, used for diagnostics only."""
        if self.backend == "rapidocr-v6":
            output = self._engine(image)
            if output is None or output.boxes is None:
                return []
            return [
                {
                    "box": np.asarray(box, dtype=np.float32),
                    "text": str(text),
                    "score": float(score),
                }
                for box, text, score in zip(
                    output.boxes, output.txts, output.scores
                )
            ]
        result, _ = self._engine(image)
        return [
            {
                "box": np.asarray(box, dtype=np.float32),
                "text": str(text),
                "score": float(score),
            }
            for box, text, score in (result or [])
        ]
