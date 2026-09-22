"""Tests for the per-frame spatial merge."""

import pytest

from whisper_subtitle.models import OcrDetection
from whisper_subtitle.ocr.frame_merge import (
    _find_neighbors,
    _is_adjacent,
    _merge_group,
    merge_frame_detections,
)


def det(t, x1, y1, x2, y2, text, conf=0.95):
    return OcrDetection(
        frame_time=t, box=(x1, y1, x2, y2),
        text=text, angle=0.0, confidence=conf,
    )


# ---------------------------------------------------------------- adjacent


class TestIsAdjacent:
    def test_side_by_side_same_row(self):
        a = det(0, 100, 200, 300, 240, "left")
        b = det(0, 310, 200, 500, 240, "right")
        assert _is_adjacent(a, b)

    def test_stacked_above_below(self):
        a = det(0, 100, 200, 400, 240, "top")
        b = det(0, 100, 260, 400, 300, "bottom")
        assert _is_adjacent(a, b)

    def test_far_apart_returns_false(self):
        a = det(0, 100, 200, 300, 240, "left")
        b = det(0, 800, 200, 950, 240, "far right")
        assert not _is_adjacent(a, b)

    def test_diagonally_far_returns_false(self):
        a = det(0, 100, 200, 300, 240, "a")
        b = det(0, 500, 500, 700, 540, "b")
        assert not _is_adjacent(a, b)


# ---------------------------------------------------------------- neighbors


class TestFindNeighbors:
    def test_no_neighbors(self):
        dets = [
            det(0, 0, 0, 100, 100, "a"),
            det(0, 900, 900, 1000, 1000, "b"),
        ]
        assert _find_neighbors(0, dets) == set()

    def test_direct_neighbor(self):
        dets = [
            det(0, 100, 200, 300, 240, "left"),
            det(0, 310, 200, 500, 240, "right"),
        ]
        assert _find_neighbors(0, dets) == {1}

    def test_transitive_chain(self):
        # a — b — c, but a is not adjacent to c
        dets = [
            det(0, 100, 200, 200, 240, "a"),
            det(0, 210, 200, 320, 240, "b"),
            det(0, 330, 200, 440, 240, "c"),
        ]
        # a and c are 130px apart, both h=40. a.x2=200, c.x1=330, gap=130.
        # 130 > 40*1.0 → not adjacent. b bridges them.
        neighbors = _find_neighbors(0, dets)
        assert 1 in neighbors and 2 in neighbors


# ---------------------------------------------------------------- merge


class TestMergeGroup:
    def test_single(self):
        d = det(0, 100, 200, 300, 240, "only")
        assert _merge_group([d]).text == "only"

    def test_side_by_side(self):
        a = det(0, 100, 200, 300, 240, "left")
        b = det(0, 310, 200, 500, 240, "right")
        assert _merge_group([a, b]).text == "left right"

    def test_stacked(self):
        a = det(0, 100, 200, 400, 240, "top")
        b = det(0, 100, 260, 400, 300, "bottom")
        assert _merge_group([a, b]).text == "top\nbottom"

    def test_bounding_box_union(self):
        a = det(0, 100, 200, 300, 240, "a")
        b = det(0, 310, 200, 500, 240, "b")
        merged = _merge_group([a, b])
        assert merged.box == (100, 200, 500, 240)

    def test_confidence_average(self):
        a = det(0, 100, 200, 300, 240, "a", conf=0.8)
        b = det(0, 310, 200, 500, 240, "b", conf=1.0)
        assert _merge_group([a, b]).confidence == pytest.approx(0.9)


# ---------------------------------------------------------------- pipeline


class TestMergeFrameDetections:
    def test_empty(self):
        assert merge_frame_detections([]) == []

    def test_single_frame_single_detection(self):
        d = det(0.5, 100, 200, 300, 240, "x")
        result = merge_frame_detections([d])
        assert len(result) == 1
        assert result[0].text == "x"

    def test_single_frame_two_boxes_merge(self):
        dets = [
            det(0.5, 100, 200, 300, 240, "left"),
            det(0.5, 310, 200, 500, 240, "right"),
        ]
        result = merge_frame_detections(dets)
        assert len(result) == 1
        assert result[0].text == "left right"

    def test_two_frames_each_merged_independently(self):
        dets = [
            det(0.5, 100, 200, 300, 240, "f1-left"),
            det(0.5, 310, 200, 500, 240, "f1-right"),
            det(1.0, 100, 200, 300, 240, "f2-left"),
            det(1.0, 310, 200, 500, 240, "f2-right"),
        ]
        result = merge_frame_detections(dets)
        assert len(result) == 2
        assert result[0].text == "f1-left f1-right"
        assert result[1].text == "f2-left f2-right"

    def test_anchor_is_longest(self):
        # Long text should be the anchor, short text is a neighbor
        dets = [
            det(0.5, 100, 200, 200, 240, "short"),
            det(0.5, 210, 200, 500, 240, "this is the anchor"),
        ]
        result = merge_frame_detections(dets)
        assert len(result) == 1
        # Sorted by x: shorter one is on the left
        assert result[0].text == "short this is the anchor"

    def test_non_adjacent_gets_dropped(self):
        dets = [
            det(0.5, 100, 200, 300, 240, "subtitle"),
            det(0.5, 900, 500, 1000, 540, "far away"),
        ]
        result = merge_frame_detections(dets)
        assert len(result) == 1
        assert result[0].text == "subtitle"

    def test_output_one_per_frame(self):
        # Three frames, three detections each
        dets = []
        for t in [0.0, 1.0, 2.0]:
            dets.append(det(t, 100, 200, 300, 240, f"t{t}-a"))
            dets.append(det(t, 310, 200, 500, 240, f"t{t}-b"))
            dets.append(det(t, 900, 500, 1000, 540, f"t{t}-far"))
        result = merge_frame_detections(dets)
        assert len(result) == 3