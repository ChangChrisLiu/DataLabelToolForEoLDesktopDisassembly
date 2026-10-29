"""Source PTS decoder. Preview may skip frames; boundary stepping never does."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import av
import numpy as np


def probe(path):
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                        else float(container.duration or 0) / av.time_base)
            if duration <= 0:
                raise ValueError('无法读取时长')
            return dict(path=str(Path(path).resolve()), duration=duration,
                        width=stream.width, height=stream.height,
                        fps=float(stream.average_rate or 30),
                        start_time=float((stream.start_time or 0) * stream.time_base),
                        roi=[0, 0, stream.width, stream.height])
    except Exception as exc:
        raise ValueError(f'无法读取视频 {path}：{exc}') from exc


@dataclass
class Frame:
    pts: float
    end: float
    rgb: np.ndarray


class FrameReader:
    def __init__(self, path, max_width=None):
        self.info = probe(path)
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_count = 2
        self.origin = float((self.stream.start_time or 0) * self.stream.time_base)
        self.max_width = max_width
        self.iterator = None
        self.current = self.following = None
        self._image = None

    def _pts(self, frame):
        return float(frame.pts * frame.time_base) - self.origin

    def _read(self):
        for frame in self.iterator:
            if frame.pts is not None:
                return frame
        return None

    def _seek(self, seconds):
        timestamp = max(0, int((seconds + self.origin) / self.stream.time_base))
        self.container.seek(timestamp, stream=self.stream, backward=True, any_frame=False)
        self.iterator = iter(self.container.decode(self.stream))
        self.current = self._read()
        if self.current is None:
            raise ValueError('视频中没有可解码的画面。')
        self.following = self._read()
        self._image = None

    def frame_at(self, seconds):
        seconds = min(max(float(seconds), 0), self.info['duration'] - 1e-7)
        if (self.current is None or seconds < self._pts(self.current) - 1e-9 or
                seconds > self._pts(self.current) + 1.):
            self._seek(seconds)
        while self.following is not None and self._pts(self.following) <= seconds + 1e-9:
            self.current, self.following = self.following, self._read()
            self._image = None
        if self._image is None:
            f = self.current
            if self.max_width and f.width > self.max_width:
                f = f.reformat(width=self.max_width,
                               height=round(f.height * self.max_width / f.width))
            self._image = f.to_ndarray(format='rgb24')
        pts = self._pts(self.current)
        end = self._pts(self.following) if self.following is not None else self.info['duration']
        return Frame(pts, max(pts, end), self._image)

    def previous(self, pts):
        return self.frame_at(max(0, pts - 1e-5))

    def next(self, pts):
        current = self.frame_at(pts)
        return self.frame_at(current.end)

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
