"""Protect source-time edits, synchronization, and recoverable project storage."""
import copy
import json

import pytest

from tda.video.project import Project, History


@pytest.fixture
def project(tmp_path):
    manifest = tmp_path / 'segments.json'
    manifest.write_text(json.dumps({
        'source_video': str(tmp_path / 'oak.mp4'), 'source_duration_s': 1200,
        'source_resolution': [1920, 1080], 'source_fps': 60,
        'roi_xywh': [560, 600, 720, 480],
        'segments': [
            {'clip_id': 1, 'step': 2, 'start': 74., 'end': 80., 'label_zh': '螺丝', 'note': '保留'},
            {'clip_id': 6, 'step': 7, 'start': 275.5, 'end': 288.5, 'label_zh': '连接器'},
        ]}), encoding='utf-8')
    return Project.from_manifest(manifest, {
        'path': str(tmp_path / 'rs.mp4'), 'duration': 1210., 'width': 1920, 'height': 1080,
        'roi': [0, 0, 1920, 1080], 'fps': 29.98,
    })


def test_import_keeps_original_source_times_and_metadata(project):
    s = project.segments[1]
    assert (s['start'], s['end']) == (275.5, 288.5)
    assert s['baseline']['end'] == 288.5
    assert project.segments[0]['note'] == '保留'
    assert not project.segments[0]['confirmed']


def test_one_anchor_maps_original_source_clock(project):
    with pytest.raises(ValueError, match='同步'):
        project.mapping()
    project.set_sync([[74, 70]], False)
    assert project.mapping() == (1., -4.)
    assert project.mapped_range({'start': 115, 'end': 121}) == (111., 117.)


def test_second_anchor_is_a_check_until_drift_correction_enabled(project):
    project.set_sync([[74, 70], [1074, 1071]], False)
    assert project.mapping() == (1., -4.)
    assert project.drift_error() == pytest.approx(1.)
    project.set_sync([[74, 70], [1074, 1071]], True)
    a, b = project.mapping()
    assert a == pytest.approx(1.001)
    assert a * 74 + b == pytest.approx(70)
    assert project.mapped_range({'start': 1074, 'end': 1084}) == pytest.approx((1071, 1081.01))


@pytest.mark.parametrize('start,end', [(80, 74), (74, 74), (-1, 8), (1199, 1201), (float('nan'), 80)])
def test_invalid_master_cuts_are_rejected_without_change(project, start, end):
    before = copy.deepcopy(project.segments)
    with pytest.raises(ValueError):
        project.edit(0, start=start, end=end)
    assert project.segments == before


def test_edit_invalidates_confirmation_and_reset_keeps_initial_cut(project):
    project.edit(0, confirmed=True)
    project.edit(0, start=73.5)
    assert not project.segments[0]['confirmed']
    project.reset(0)
    assert project.segments[0]['start'] == 74.


def test_outside_secondary_coverage_is_not_silently_clamped(project):
    project.set_sync([[74, 1]], False)
    with pytest.raises(ValueError, match='范围'):
        project.mapped_range({'start': 0, 'end': 10})


@pytest.mark.parametrize('anchors', [[[1, 1], [1, 2]], [[1, 2], [2, 1]], [[-1, 1]], [[float('nan'), 1]]])
def test_invalid_sync_does_not_replace_valid_sync(project, anchors):
    project.set_sync([[74, 70]], False)
    with pytest.raises(ValueError):
        project.set_sync(anchors, True)
    assert project.mapping() == (1., -4.)


def test_split_and_undo_preserve_action_and_unique_identity(project):
    history = History()
    history.push(project.data)
    project.split(0, 77.)
    assert [(s['start'], s['end']) for s in project.segments[:2]] == [(74., 77.), (77., 80.)]
    assert project.segments[1]['step'] == 2
    assert len({s['uid'] for s in project.segments}) == 3
    project.data = history.undo(project.data)
    assert len(project.segments) == 2
    project.data = history.redo(project.data)
    assert len(project.segments) == 3


def test_atomic_save_backup_and_roundtrip(project, tmp_path):
    path = tmp_path / 'review.json'
    project.save(path)
    project.edit(0, start=73.5, label_zh='调整')
    project.save(path)
    saved = Project.load(path)
    assert saved.segments[0]['start'] == 73.5
    assert saved.segments[0]['label_zh'] == '调整'
    assert Project.load(path.with_suffix('.json.bak')).segments[0]['start'] == 74.
    assert saved.segments[0]['baseline']['start'] == 74.
    with pytest.raises(ValueError):
        project.save(project.data['import_manifest'])
