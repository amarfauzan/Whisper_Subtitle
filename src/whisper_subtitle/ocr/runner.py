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

    Frames are cropped according to cfg.crop_*_ratio before OCR runs.
    Each ratio is a fraction of the frame's relevant dimension to
    remove from that edge (0.0 = no crop, 0.2 = remove 20%).

    Args:
        video_path: video to process.
        engine: an already-constructed PaddleOcrEngine. Reused across
            calls so model loading is amortized.
        cfg: OCR configuration.

    Returns:
        (detections, cropped_frame_height). Detection boxes are relative
        to the cropped frame — the filter needs the cropped height to
        compute correct box-height ratios.
    """
    if not video_path.exists():
        raise OcrError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OcrError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if fps <= 0:
            raise OcrError(f"Invalid FPS reported for {video_path}: {fps}")
        if frame_h <= 0 or frame_w <= 0:
            raise OcrError(f"Invalid frame size for {video_path}: {frame_w}x{frame_h}")

        # Precompute the crop region in pixels.
        crop_y1, crop_y2, crop_x1, crop_x2 = _crop_region(
            frame_h, frame_w, cfg
        )

        interval = max(1, round(fps / cfg.sample_fps))
        expected_samples = max(1, total_frames // interval)

        log.info(
            "Sampling %s at %d fps (every %d frames, ~%d samples)",
            video_path.name, cfg.sample_fps, interval, expected_samples,
        )
        if (crop_y1, crop_y2, crop_x1, crop_x2) != (0, frame_h, 0, frame_w):
            log.info(
                "Cropping to y=[%d:%d] x=[%d:%d] of %dx%d",
                crop_y1, crop_y2, crop_x1, crop_x2, frame_w, frame_h,
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
                cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                found = engine.process_frame(cropped, frame_time=timestamp)
                detections.extend(found)
                sampled += 1

                if sampled % 100 == 0:
                    log.debug(
                        "Sampled %d frames, %d detections so far",
                        sampled, len(detections),
                    )

            frame_number += 1

        cropped_h = crop_y2 - crop_y1
        log.info(
            "OCR complete: %d detections across %d sampled frames",
            len(detections), sampled,
        )
        return detections, cropped_h

    finally:
        cap.release()


def _crop_region(
    frame_h: int,
    frame_w: int,
    cfg: OcrConfig,
) -> tuple[int, int, int, int]:
    """Compute (y1, y2, x1, x2) from crop ratios.

    Guards against ratios summing to more than 1.0 (which would produce
    an empty or inverted region).
    """
    y1 = int(frame_h * cfg.crop_top_ratio)
    y2 = frame_h - int(frame_h * cfg.crop_bottom_ratio)
    x1 = int(frame_w * cfg.crop_left_ratio)
    x2 = frame_w - int(frame_w * cfg.crop_right_ratio)

    if y2 <= y1 or x2 <= x1:
        raise OcrError(
            f"Crop ratios leave an empty region: "
            f"top={cfg.crop_top_ratio} bottom={cfg.crop_bottom_ratio} "
            f"left={cfg.crop_left_ratio} right={cfg.crop_right_ratio}"
        )

    return y1, y2, x1, x2