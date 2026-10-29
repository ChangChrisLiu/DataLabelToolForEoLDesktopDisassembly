"""User gestures keep screening continuous and preserve source-time synchronization."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest

from test_video_ui import window, wait_until
from test_video_export import pair
import tda.video.app as app_module


def test_change_speed_rebases_clock_without_pausing_or_time_jump(window, monkeypatch):
    now = [10.]
    monkeypatch.setattr(app_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    window.loop_box.setChecked(False)
    window.seek('oak', .4)
    window.toggle_play('oak')
    now[0] = 10.2
    window.speed.setCurrentText('2×')
    assert window.playing == 'oak'
    assert window.positions['oak'] == pytest.approx(.6)
    now[0] = 10.4
    window.tick()
    assert window.positions['oak'] == pytest.approx(1.)
    assert window.positions['rs'] == pytest.approx(1.2)


@pytest.mark.parametrize('rate,elapsed,want', [('1.5×', .2, .7), ('4×', .2, 1.2), ('8×', .2, 2.)])
def test_screening_rates_advance_original_clock(window, monkeypatch, rate, elapsed, want):
    now = [20.]
    monkeypatch.setattr(app_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    window.loop_box.setChecked(False)
    window.speed.setCurrentText(rate)
    window.seek('oak', .4)
    window.toggle_play('oak')
    now[0] += elapsed
    window.tick()
    assert window.positions['oak'] == pytest.approx(want)


def test_drag_previews_before_release_then_resumes_linked_playback(window):
    window.toggle_play('oak')
    slider = window.panes['oak'].slider
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width()//2, slider.height()//2))
    QTest.mouseMove(slider, QPoint(slider.width()*2//3, slider.height()//2))
    wait_until(lambda: window.panes['oak'].frame is not None and window.panes['oak'].frame.pts > 1.5)
    assert window.playing is None
    assert slider.isSliderDown()
    target = slider.value()/1000
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width()*2//3, slider.height()//2))
    wait_until(lambda: window.playing == 'oak')
    assert window.play_start == pytest.approx(target, abs=.04)
    assert not window.loop_box.isChecked()  # Looking outside the cut must not jump back into it.
    assert window.positions['rs'] - window.positions['oak'] == pytest.approx(.2)


def test_click_progress_while_paused_seeks_both_cameras_without_starting(window):
    slider = window.panes['rs'].slider
    QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width()//2, slider.height()//2))
    wait_until(lambda: window.panes['rs'].frame is not None and window.panes['rs'].frame.pts > 1.4)
    assert window.playing is None
    assert window.positions['rs'] - window.positions['oak'] == pytest.approx(.2)


def test_pause_cancels_pending_scrub_resume(window):
    window.toggle_play('oak')
    window.begin_scrub('oak')
    window.finish_scrub('oak', .8)
    window.pause()
    wait_until(lambda: not window.panes['oak'].busy)
    assert window.playing is None
    assert window.pending_resume is None


def test_drag_other_independent_camera_does_not_pause_active_camera(window):
    window.link_box.setChecked(False)
    window.toggle_play('oak')
    window.begin_scrub('rs')
    window.finish_scrub('rs', 1.2)
    assert window.playing == 'oak'
    wait_until(lambda: window.panes['rs'].frame is not None and window.panes['rs'].frame.pts >= 1.19)
    assert window.playing == 'oak'


def test_independent_seek_still_finishes_when_other_camera_loops(window, monkeypatch):
    now = [30.]
    monkeypatch.setattr(app_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    wait_until(lambda: not window.panes['rs'].busy)
    window.link_box.setChecked(False)
    window.toggle_play('oak')
    window.begin_scrub('rs')
    window.finish_scrub('rs', 1.2)
    now[0] = 31.
    window.tick()  # The OAK cut wraps before the queued RS seek result is delivered.
    wait_until(lambda: not window.panes['rs'].busy)
    assert window.panes['rs'].frame.pts == pytest.approx(1.2)
    assert window.playing == 'oak'


def test_local_timeline_drag_resumes_without_changing_cut_boundaries(window):
    before = [(s['start'], s['end']) for s in window.project.segments]
    window.toggle_play('oak')
    bar = window.range_bar
    QTest.mousePress(bar, Qt.MouseButton.LeftButton, pos=QPoint(round(bar.to_x(1.4)), 36))
    QTest.mouseMove(bar, QPoint(round(bar.to_x(1.8)), 36))
    wait_until(lambda: window.panes['oak'].frame is not None and window.panes['oak'].frame.pts > 1.6)
    QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, pos=QPoint(round(bar.to_x(1.8)), 36))
    wait_until(lambda: window.playing == 'oak')
    assert window.play_start == pytest.approx(1.8, abs=.04)
    assert [(s['start'], s['end']) for s in window.project.segments] == before
