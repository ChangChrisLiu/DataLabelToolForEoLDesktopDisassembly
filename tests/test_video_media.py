"""Actual VFR presentation timestamps, including reverse seeks."""
from fractions import Fraction
import av
import numpy as np
import pytest

from tda.video.media import FrameReader, probe


def make_video(path, times=(0, 40, 100, 140, 220, 280, 340, 400, 520, 600, 760, 880)):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 96, 64, 'yuv420p'
        stream.codec_context.max_b_frames = 0
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for i, pts in enumerate(times):
            arr = np.full((64, 96, 3), (i * 15) % 256, np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return path


@pytest.fixture
def media(tmp_path):
    return make_video(tmp_path / 'vfr.mp4')


def test_timestamp_navigation_uses_variable_frame_intervals(media):
    with FrameReader(media) as reader:
        f = reader.frame_at(.12)
        assert (f.pts, f.end) == pytest.approx((.1, .14))
        assert reader.next(f.pts).pts == pytest.approx(.14)
        assert reader.previous(.14).pts == pytest.approx(.1)
        assert reader.frame_at(.61).pts == pytest.approx(.6)
        assert reader.frame_at(.04).pts == pytest.approx(.04)
        assert reader.frame_at(0).pts == 0
        assert reader.previous(0).pts == 0


def test_probe_and_last_frame(media):
    info = probe(media)
    assert (info['width'], info['height']) == (96, 64)
    assert info['duration'] == pytest.approx(.92, abs=.001)
    with FrameReader(media) as reader:
        f = reader.frame_at(.919)
        assert (f.pts, f.end) == pytest.approx((.88, .92), abs=.001)
        assert f.rgb.shape == (64, 96, 3)
        assert np.mean(f.rgb) == pytest.approx(165, abs=3)
        assert reader.next(.88).pts == pytest.approx(.88)


def test_missing_media_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match='视频'):
        probe(tmp_path / 'missing.mp4')
