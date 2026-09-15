"""Sample frames from a video and run OCR on each."""

import logging
from pathlib import Path

import cv2

from whisper_subtitle.config import OcrConfig
from whisper_subtitle.exceptions import OcrError
from whisper_subtitle.models import OcrDetection
from whisper_subtitle.ocr.engine import PaddleOcrEngine

log = logging.getLogger(__name__)


def sample_and_detect(
    video_path: Path,
    engine: PaddleOcrEngine,
    cfg: OcrConfig,
) -> tuple[list[OcrDetection], int]:
    """Sample frames from a video and run OCR on each.

    Args:
        video_path: video to process.
        engine: an already-constructed PaddleOcrEngine. Reused across
            calls so model loading is amortized.
        cfg: OCR configuration. Uses cfg.sample_fps.

    Returns:
        (detections, frame_height). frame_height is the pixel height of
        the source video — the filter needs it to compute box-height
        ratios.
    """
    if not video_path.exists():
        raise OcrError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OcrError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise OcrError(f"Invalid FPS reported for {video_path}: {fps}")
        if frame_height <= 0:
            raise OcrError(f"Invalid frame height for {video_path}")

        interval = max(1, round(fps / cfg.sample_fps))
        expected_samples = max(1, total_frames // interval)

        log.info(
            "Sampling %s at %d fps (every %d frames, ~%d samples)",
            video_path.name, cfg.sample_fps, interval, expected_samples,
        )

        detections: list[OcrDetection] = []
        frame_number = 0
        sampled = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_number % interval == 0:
                timestamp = frame_number / fps
                found = engine.process_frame(frame, frame_time=timestamp)
                detections.extend(found)
                sampled += 1

                if sampled % 100 == 0:
                    log.debug(
                        "Sampled %d frames, %d detections so far",
                        sampled, len(detections),
                    )

            frame_number += 1

        log.info(
            "OCR complete: %d detections across %d sampled frames",
            len(detections), sampled,
        )
        return detections, frame_height

    finally:
        cap.release()
