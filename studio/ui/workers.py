"""Background jobs, so loading files or exporting never freezes the window."""

from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class JobSignals(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    """Run ``fn(progress)`` in the thread pool; ``progress`` takes a message."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn
        self.signals = JobSignals()
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: D102
        try:
            result = self.fn(self.signals.progress.emit)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            traceback.print_exc()
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(result)


def start_job(fn: Callable, on_done: Callable, on_error: Callable, on_progress: Callable | None = None) -> Job:
    job = Job(fn)
    job.signals.finished.connect(on_done)
    job.signals.failed.connect(on_error)
    if on_progress is not None:
        job.signals.progress.connect(on_progress)
    QThreadPool.globalInstance().start(job)
    return job
