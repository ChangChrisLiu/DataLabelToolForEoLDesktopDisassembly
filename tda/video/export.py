"""Cancelable paired exports with auditable original-video timestamps."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
from datetime import datetime, timezone

from .media import FrameReader, probe
from .project import Project, output_path


class ExportCancelled(Exception):
    pass


def _json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def export_project(project: Project, output_dir, ids=None, paired=True, cancel=None, progress=None):
    project = Project(project.data)
    project.validate()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('找不到 FFmpeg，请安装并加入 PATH。')
    destination = output_path(output_dir)
    if destination.exists():
        raise FileExistsError('导出目录已经存在，请使用新目录。')
    selected = [s for s in project.segments if s['enabled'] and (ids is None or s['uid'] in ids)]
    selected.sort(key=lambda s: s['start'])
    if not selected:
        raise ValueError('没有可导出的片段。')
    a, b = project.mapping() if paired else (1., 0.)
    cameras = ('oak', 'rs') if paired else ('oak',)
    for camera in cameras:
        source = project.sources[camera]
        if not source or not Path(source['path']).is_file():
            raise ValueError(f'{camera} 原视频不可用，请重新定位文件。')
        actual = probe(source['path'])
        if (abs(actual['duration'] - source['duration']) > .1 or
                (actual['width'], actual['height']) != (source['width'], source['height'])):
            raise ValueError(f'{camera} 视频与工程记录不符，请重新导入或定位正确的原片。')
    for s in selected:
        project.validate_segment(s)
        if paired:
            project.mapped_range(s)
    cancel = cancel or threading.Event()
    progress = progress or (lambda *args: None)
    destination.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'completed': 0, 'total': len(selected) * len(cameras)}
    report = dict(status='running', created_at=datetime.now(timezone.utc).isoformat(),
                  time_basis='Original source video seconds; half-open [start,end)',
                  mapping={'a': a, 'b': b, 'anchors': project.data['anchors'],
                           'drift_correction': project.data['drift_correction']},
                  fps=30, audio=False, secondary_retimed_to_master=bool(paired and a != 1),
                  boundary_policy='Source frame covering the mapped instant; output duration rounded up to 1/30 s.',
                  sources=project.sources, clips=[])
    _json(destination / 'status.json', status)
    _json(destination / 'project_snapshot.json', project.data)
    readers = {}
    try:
        for number, segment in enumerate(selected, 1):
            for camera in cameras:
                if cancel.is_set():
                    raise ExportCancelled('导出已取消，已完成的文件保留在本次目录。')
                source = project.sources[camera]
                start, end = ((segment['start'], segment['end']) if camera == 'oak'
                              else project.mapped_range(segment))
                scale = a if camera == 'rs' else 1.
                duration = segment['end'] - segment['start']
                if camera not in readers:
                    readers[camera] = FrameReader(source['path'], max_width=96)
                reader = readers[camera]
                first = reader.frame_at(start).pts
                last_end = reader.frame_at(max(start, end - 1e-7)).end
                x, y, width, height = source['roi']
                # H.264 yuv420p requires even dimensions; preserve the selected area
                # by padding its right/bottom edge, never by silently shrinking it.
                filters = (f'trim=duration={last_end-first:.9f},'
                           f'setpts=(PTS+({first-start:.9f})/TB)/{scale:.12f},'
                           f'crop={width}:{height}:{x}:{y}:exact=1,'
                           'pad=ceil(iw/2)*2:ceil(ih/2)*2,'
                           'fps=30:start_time=0,tpad=stop_mode=clone:stop_duration=0.1')
                file = f'{number:03d}_{camera}.mp4'
                command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                           '-ss', f'{first:.9f}', '-threads', '2', '-i', source['path'],
                           '-map', '0:v:0', '-an', '-vf', filters,
                           '-frames:v', str(max(1, math.ceil(duration * 30 - 1e-8))),
                           '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '2',
                           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination / file)]
                progress(status['completed'], status['total'], f'{number:03d} · {camera} · {segment["label_zh"]}')
                with (destination / 'ffmpeg.log').open('ab') as log:
                    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    try:
                        while True:
                            if cancel.is_set():
                                process.terminate()
                                raise ExportCancelled('导出已取消。')
                            try:
                                code = process.wait(timeout=.2)
                                break
                            except subprocess.TimeoutExpired:
                                pass
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                if code:
                    raise RuntimeError(f'FFmpeg 导出失败，详情见 {destination / "ffmpeg.log"}')
                info = probe(destination / file)
                if abs(info['duration'] - duration) > 1/30 + .005:
                    raise RuntimeError(f'{file} 导出时长异常，请检查原视频完整性。')
                report['clips'].append(dict(number=number, uid=segment['uid'],
                    camera=camera, label=segment['label_zh'], step=segment.get('step'),
                    confirmed=segment['confirmed'], note=segment.get('note', ''),
                    master_start=segment['start'], master_end=segment['end'],
                    source_start=start, source_end=end, first_decoded_pts=first,
                    roi=source['roi'], actual_duration=info['duration'], file=file))
                status['completed'] += 1
                _json(destination / 'status.json', status)
        status['status'] = report['status'] = 'complete'
        with (destination / 'segments.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(report['clips'][0]))
            writer.writeheader()
            writer.writerows(report['clips'])
        _json(destination / 'manifest.json', report)
        _json(destination / 'status.json', status)
        progress(status['total'], status['total'], '导出完成')
        return destination / 'manifest.json'
    except Exception as exc:
        status.update(status='canceled' if isinstance(exc, ExportCancelled) else 'failed', error=str(exc))
        _json(destination / 'status.json', status)
        raise
    finally:
        for reader in readers.values():
            reader.close()
