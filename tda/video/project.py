"""Editable source ranges, explicit synchronization and atomic project storage."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
import uuid


def output_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    if os.name == 'nt' and path.drive.casefold() != 'd:':
        raise ValueError('工程、缓存和导出请保存在 D: 盘。')
    if 'phd data backup' in str(path).casefold():
        raise ValueError('原始数据目录只读。')
    return path


def finite(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('时间必须是有限数值。')
    return value


class Project:
    def __init__(self, data: dict):
        self.data = copy.deepcopy(data)

    @property
    def segments(self):
        return self.data['segments']

    @property
    def sources(self):
        return self.data['sources']

    @classmethod
    def from_manifest(cls, path, secondary=None):
        path = Path(path).resolve()
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
        width, height = raw['source_resolution']
        oak = dict(path=raw['source_video'], duration=float(raw['source_duration_s']),
                   width=width, height=height, fps=raw.get('source_fps', 30),
                   roi=raw.get('roi_xywh', [0, 0, width, height]))
        rows = []
        for i, row in enumerate(raw['segments']):
            row = copy.deepcopy(row)
            row.update(uid=uuid.uuid4().hex, enabled=True, confirmed=False,
                       baseline={'start': row['start'], 'end': row['end']})
            row.setdefault('clip_id', i + 1)
            row.setdefault('label_zh', f'动作 {i + 1}')
            row.setdefault('note', '')
            rows.append(row)
        p = cls(dict(version=1, title=path.parent.name, import_manifest=str(path),
                     sources={'oak': oak, 'rs': secondary}, segments=rows,
                     anchors=[], drift_correction=False))
        p.validate()
        return p

    @classmethod
    def load(cls, path):
        p = cls(json.loads(Path(path).read_text(encoding='utf-8-sig')))
        p.validate()
        return p

    def validate_segment(self, segment):
        start, end = finite(segment['start']), finite(segment['end'])
        if not 0 <= start < end <= self.sources['oak']['duration'] + 1e-6:
            raise ValueError('片段起止点必须在 OAK 原片范围内，且开始早于结束。')

    def validate(self):
        if self.data.get('version') != 1:
            raise ValueError('不支持的工程版本。')
        for key, source in self.sources.items():
            if source is None:
                continue
            if finite(source['duration']) <= 0:
                raise ValueError(f'{key} 视频时长无效。')
            x, y, w, h = source['roi']
            if any(int(v) != v for v in [x, y, w, h]) or not (
                x >= 0 and y >= 0 and w >= 2 and h >= 2 and
                x + w <= source['width'] and y + h <= source['height']
            ):
                raise ValueError(f'{key} ROI 超出画面或尺寸太小。')
        seen = set()
        for segment in self.segments:
            self.validate_segment(segment)
            if segment['uid'] in seen:
                raise ValueError('片段标识重复。')
            seen.add(segment['uid'])
        self._check_sync(self.data['anchors'], self.data['drift_correction'])

    def _check_sync(self, anchors, drift):
        if len(anchors) > 2:
            raise ValueError('最多支持两对同步点。')
        if drift and len(anchors) != 2:
            raise ValueError('线性校正需要两对同步点。')
        for pair in anchors:
            if len(pair) != 2 or not self.sources.get('rs'):
                raise ValueError('同步需要两个源视频。')
            for value, camera in zip(pair, ('oak', 'rs')):
                if not 0 <= finite(value) < self.sources[camera]['duration']:
                    raise ValueError('同步点超出原片范围。')
        if len(anchors) == 2 and (anchors[1][0] - anchors[0][0] < 1 or
                                  anchors[1][1] <= anchors[0][1]):
            raise ValueError('第二个同步点必须晚于第一个，OAK 两点至少相隔 1 秒。')

    def set_sync(self, anchors, drift=False):
        self._check_sync(anchors, drift)
        self.data['anchors'] = copy.deepcopy(anchors)
        self.data['drift_correction'] = bool(drift)

    def mapping(self):
        points = self.data['anchors']
        if not points:
            raise ValueError('请先在两段原视频中设置同步点。')
        a = 1.
        if self.data['drift_correction']:
            a = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        return a, points[0][1] - a * points[0][0]

    def drift_error(self):
        p = self.data['anchors']
        return None if len(p) < 2 else (p[1][1] - p[0][1]) - (p[1][0] - p[0][0])

    def mapped_range(self, segment):
        self.validate_segment(segment)
        a, b = self.mapping()
        start, end = a * segment['start'] + b, a * segment['end'] + b
        if start < -1e-6 or end > self.sources['rs']['duration'] + 1e-6:
            raise ValueError('对应区间超出 RealSense 原片范围，请检查同步点或片段边界。')
        return max(0., start), min(end, self.sources['rs']['duration'])

    def edit(self, index, **changes):
        row = copy.deepcopy(self.segments[index])
        changed = any(row.get(k) != v for k, v in changes.items())
        row.update(changes)
        self.validate_segment(row)
        if changed and any(k != 'confirmed' for k in changes):
            row['confirmed'] = False
        self.segments[index] = row

    def reset(self, index):
        self.edit(index, **self.segments[index]['baseline'])

    def split(self, index, at):
        old = self.segments[index]
        if not old['start'] < finite(at) < old['end']:
            raise ValueError('拆分点必须位于片段内部。')
        other = copy.deepcopy(old)
        other.update(uid=uuid.uuid4().hex, start=at, confirmed=False)
        self.edit(index, end=at)
        self.segments.insert(index + 1, other)

    def add(self, start, end):
        row = dict(uid=uuid.uuid4().hex, start=start, end=end, label_zh='新增动作',
                   note='', enabled=True, confirmed=False, step=None,
                   baseline={'start': start, 'end': end})
        self.validate_segment(row)
        self.segments.append(row)
        self.segments.sort(key=lambda s: s['start'])
        return row['uid']

    def save(self, path):
        path = output_path(path)
        if path == Path(self.data['import_manifest']).resolve():
            raise ValueError('请保存为新的工程文件，不覆盖初筛清单。')
        if any(s and path == Path(s['path']).resolve() for s in self.sources.values()):
            raise ValueError('不能覆盖源视频。')
        self.validate()
        payload = json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp')
        with temp.open('w', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            backup = path.with_name(path.name + '.bak')
            btemp = path.with_name(path.name + '.bak.tmp')
            shutil.copy2(path, btemp)
            os.replace(btemp, backup)
        os.replace(temp, path)


class History:
    def __init__(self):
        self.past, self.future = [], []

    def push(self, data):
        self.past.append(copy.deepcopy(data))
        self.past = self.past[-100:]
        self.future.clear()

    def undo(self, current):
        if not self.past:
            return current
        self.future.append(copy.deepcopy(current))
        return self.past.pop()

    def redo(self, current):
        if not self.future:
            return current
        self.past.append(copy.deepcopy(current))
        return self.future.pop()
