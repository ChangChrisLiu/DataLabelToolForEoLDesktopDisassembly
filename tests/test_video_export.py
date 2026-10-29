import json
import threading
import pytest

from test_video_media import make_video
from tda.video.project import Project
from tda.video.media import FrameReader, probe
from tda.video.export import export_project, ExportCancelled


@pytest.fixture
def pair(tmp_path):
    source = make_video(tmp_path / 'oak.mp4', tuple(range(0, 3000, 40)))
    rs = make_video(tmp_path / 'rs.mp4', tuple(range(0, 3200, 40)))
    manifest = tmp_path / 'segments.json'
    manifest.write_text(json.dumps(dict(source_video=str(source), source_duration_s=3.,
        source_resolution=[96, 64], source_fps=25, roi_xywh=[4, 2, 80, 60],
        segments=[dict(clip_id=1, start=.4, end=1., label_zh='测试动作')])) ,encoding='utf-8')
    p = Project.from_manifest(manifest, probe(rs))
    p.sources['rs']['roi'] = [8, 4, 80, 60]
    p.set_sync([[.4, .6]], False)
    return p


def test_export_pair_keeps_source_mapping_and_equal_output_duration(pair, tmp_path):
    dest = tmp_path / 'output'
    path = export_project(pair, dest)
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['status'] == 'complete'
    oak, rs = report['clips']
    assert (oak['source_start'], oak['source_end']) == pytest.approx((.4, 1.))
    assert (rs['source_start'], rs['source_end']) == pytest.approx((.6, 1.2))
    for row in [oak, rs]:
        info = probe(dest / row['file'])
        assert (info['width'], info['height']) == (80, 60)
        assert info['duration'] == pytest.approx(.6, abs=.001)
        assert info['fps'] == 30
    assert (dest / 'segments.csv').is_file()
    with FrameReader(dest / oak['file']) as video:
        assert video.frame_at(0).rgb.mean() == pytest.approx(150, abs=4)


def test_affine_export_retimes_secondary_to_master(pair, tmp_path):
    pair.set_sync([[.4, .6], [2.4, 2.8]], True)
    report = json.loads(export_project(pair, tmp_path / 'drift').read_text(encoding='utf-8'))
    assert report['mapping']['a'] == pytest.approx(1.1)
    assert report['clips'][1]['source_end'] == pytest.approx(1.26)
    assert report['clips'][1]['actual_duration'] == pytest.approx(.6, abs=.001)


def test_bad_sync_and_out_of_range_do_not_create_export(pair, tmp_path):
    dest = tmp_path / 'bad'
    pair.set_sync([], False)
    with pytest.raises(ValueError, match='同步'):
        export_project(pair, dest)
    assert not dest.exists()
    pair.set_sync([[.4, 2.9]], False)
    with pytest.raises(ValueError, match='范围'):
        export_project(pair, dest)
    assert not dest.exists()


def test_existing_destination_not_overwritten_and_cancel_incomplete(pair, tmp_path):
    dest = tmp_path / 'existing'
    dest.mkdir()
    with pytest.raises(FileExistsError):
        export_project(pair, dest)
    cancel = threading.Event()
    cancel.set()
    dest = tmp_path / 'canceled'
    with pytest.raises(ExportCancelled):
        export_project(pair, dest, cancel=cancel)
    status = json.loads((dest / 'status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'canceled'
    assert not (dest / 'manifest.json').exists()


def test_cancel_as_last_encoder_finishes_does_not_publish_success(pair, tmp_path, monkeypatch):
    import subprocess
    real_popen = subprocess.Popen
    cancel = threading.Event()
    class CancelWhenFinished:
        def __init__(self, *args, **kwargs):
            self.process = real_popen(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.process, name)
        def wait(self, *args, **kwargs):
            result = self.process.wait(*args, **kwargs)
            if result == 0:
                cancel.set()
            return result
    monkeypatch.setattr(subprocess, 'Popen', CancelWhenFinished)
    dest = tmp_path / 'late_cancel'
    with pytest.raises(ExportCancelled):
        export_project(pair, dest, paired=False, cancel=cancel)
    assert json.loads((dest / 'status.json').read_text())['status'] == 'canceled'
    assert not (dest / 'manifest.json').exists()
