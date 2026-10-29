"""Launch with ``python -m tda.video``; defaults to the Desktop 7 first pass."""
import argparse
import logging
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = ROOT / 'exports/video_refiner/desktop07.review.json'
DEFAULT_MANIFEST = ROOT / 'exports/video_desktop07_oak/segments.json'
DEFAULT_RS = Path('G:/PHD Data Backup/Desktop Dataset/Realsense Capture/Dataset Information/Exp/Desktop 7/Desktop7.mp4')


def main(argv=None):
    parser = argparse.ArgumentParser(description='双机位视频精修器：基于原片时间同步与剪辑')
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--secondary', type=Path, default=DEFAULT_RS)
    args = parser.parse_args(argv)
    temp = ROOT / '.cache/tmp'
    temp.mkdir(parents=True, exist_ok=True)
    os.environ['TMP'] = os.environ['TEMP'] = str(temp)
    logs = ROOT / '.cache/logs'
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logs / 'video_refiner.log', level=logging.INFO,
                        encoding='utf-8', format='%(asctime)s %(levelname)s %(message)s')
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('Video Refiner')
    try:
        from .project import Project
        from .media import probe
        from .app import VideoRefiner
        if args.project.exists():
            project = Project.load(args.project)
        else:
            # Resolve a moved external-drive letter through the existing resolver.
            import yaml
            from tda.core import rawroot
            config = ROOT / 'configs/paths.yaml'
            if config.exists():
                rawroot.configure(yaml.safe_load(config.read_text(encoding='utf-8')))
            rs = Path(rawroot.resolve_raw(str(args.secondary)) or args.secondary)
            project = Project.from_manifest(args.manifest, probe(rs) if rs.is_file() else None)
            oak = project.sources['oak']
            oak['path'] = rawroot.resolve_raw(oak['path']) or oak['path']
        window = VideoRefiner(project, args.project)
        window.save()
        window.show()
        # Fit small/high-DPI screens without changing the user's desktop settings.
        screen = window.screen().availableGeometry()
        if window.width() > screen.width() or window.height() > screen.height():
            window.showMaximized()
        logging.info('Opened project %s', args.project)
        return app.exec()
    except Exception as exc:
        logging.exception('Startup failed')
        QMessageBox.critical(None, '视频精修器无法启动', f'{exc}\n\n日志：{logs / "video_refiner.log"}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
