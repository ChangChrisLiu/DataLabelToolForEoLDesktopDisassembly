import time
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from test_video_export import pair
from tda.video.project import Project
from tda.video.app import VideoRefiner


@pytest.fixture
def window(pair, tmp_path):
    app = QApplication.instance() or QApplication([])
    w = VideoRefiner(pair, tmp_path / 'review.json')
    w.show()
    app.processEvents()
    yield w
    w.close()
    app.processEvents()


def wait_until(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), 'Video UI operation did not finish'


def test_boundary_edit_undo_save_and_reopen(window):
    window.start_box.setValue(.48)
    window.apply_form()
    assert window.project.segments[0]['start'] == .48
    window.confirm_current(advance=False)
    assert window.project.segments[0]['confirmed']
    window.start_box.setValue(.52)
    window.apply_form()
    assert not window.project.segments[0]['confirmed']
    window.undo()
    assert window.start_box.value() == .48
    assert window.project.segments[0]['confirmed']
    window.save()
    assert Project.load(window.project_path).segments[0]['start'] == .48


def test_independent_original_frames_define_sync(window):
    window.link_box.setChecked(False)
    window.seek('oak', .4)
    window.seek('rs', .6)
    wait_until(lambda: all(p.frame is not None and not p.busy for p in window.panes.values()))
    window.capture_anchor(0)
    assert window.project.mapping() == pytest.approx((1., .2))
    window.link_box.setChecked(True)
    window.seek('oak', 1.)
    wait_until(lambda: not window.panes['rs'].busy and window.panes['rs'].frame.pts >= 1.19)
    assert window.panes['rs'].frame.pts == pytest.approx(1.2)


def test_invalid_edit_leaves_cut_and_lock_blocks_second_writer(window):
    window.start_box.setValue(2.)
    assert window.apply_form() is False
    assert window.project.segments[0]['start'] == .4
    with pytest.raises(ValueError, match='打开'):
        VideoRefiner(window.project, window.project_path)


def test_confirm_and_save_do_not_round_frame_exact_boundaries(window):
    exact = 29 / 60
    window.mutation(lambda: window.project.edit(0, start=exact))
    window.confirm_current(advance=False)
    assert window.project.segments[0]['start'] == exact
    window.save()
    assert Project.load(window.project_path).segments[0]['start'] == exact


def test_roi_changes_invalidate_confirmation_and_undo_restores_it(window):
    window.confirm_current(advance=False)
    window.set_roi('oak', [8, 2, 80, 60])
    assert not window.project.segments[0]['confirmed']
    window.undo()
    assert window.project.segments[0]['confirmed']
    assert window.project.sources['oak']['roi'] == [4, 2, 80, 60]


def test_sync_mode_plays_original_video_outside_selected_clip(window):
    window.tabs.setCurrentIndex(1)
    window.seek('oak', 1.4)
    window.toggle_play('oak')
    assert window.play_start == 1.4
    window.pause()


def test_split_exclude_and_canvas_source_coordinate(window):
    window.project.split(0, .7)
    window.refresh_list()
    window.list.setCurrentRow(1)
    window.enabled_box.setChecked(False)
    window.apply_form()
    assert not window.project.segments[1]['enabled']
    wait_until(lambda: window.panes['oak'].frame is not None)
    canvas = window.panes['oak'].canvas
    canvas.full_view = True
    canvas.repaint()
    QApplication.processEvents()
    rect = canvas.image_rect()
    center = canvas.source_point(QPointF(rect.center()))
    assert center.x() == pytest.approx(48, abs=1)
    assert center.y() == pytest.approx(32, abs=1)
