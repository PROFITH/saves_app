"""Video player with frame stepping, built on PyAV frames shown in a QLabel."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from studio.video import VideoFrame, VideoReader


def format_seconds(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{int(h):02d}:{int(m):02d}:{s:06.3f}"


class VideoPanel(QWidget):
    frame_changed = Signal(float)   # video seconds of the frame on screen
    mark_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reader: Optional[VideoReader] = None  # noqa: UP045
        self.playing = False
        self.speed = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.screen = QLabel("Sin vídeo.\nArrastra aquí un archivo de vídeo.")
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setMinimumSize(320, 180)
        self.screen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.screen.setStyleSheet("background:#111; color:#aaa; border-radius:4px;")
        layout.addWidget(self.screen, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderReleased.connect(self._slider_released)
        layout.addWidget(self.slider)

        self.time_label = QLabel("Vídeo —")
        self.time_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.time_label)
        self.host_label = QLabel("")
        self.host_label.setStyleSheet("font-family: monospace; color:#555;")
        layout.addWidget(self.host_label)

        controls = QHBoxLayout()
        self.buttons = {}
        for key, text, tip, slot in (
            ("back1s", "⏪ 1 s", "Retroceder 1 s (Mayús+←)", lambda: self.step_seconds(-1.0)),
            ("prev", "◀ 1", "Fotograma anterior (←)", self.previous_frame),
            ("play", "▶", "Reproducir / pausa (espacio)", self.toggle_play),
            ("next", "1 ▶", "Fotograma siguiente (→)", self.next_frame),
            ("fwd1s", "1 s ⏩", "Avanzar 1 s (Mayús+→)", lambda: self.step_seconds(1.0)),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            button.setFocusPolicy(Qt.NoFocus)
            controls.addWidget(button)
            self.buttons[key] = button
        self.speed_box = QComboBox()
        self.speed_box.addItems(["0,1×", "0,25×", "0,5×", "1×", "2×", "4×"])
        self.speed_box.setCurrentText("1×")
        self.speed_box.setFocusPolicy(Qt.NoFocus)
        self.speed_box.currentTextChanged.connect(self._speed_changed)
        controls.addWidget(self.speed_box)
        layout.addLayout(controls)

        self.mark_button = QPushButton("Marcar palmada en el vídeo  [M]")
        self.mark_button.setStyleSheet("QPushButton{padding:8px; font-weight:bold; background:#1565c0; color:white; border-radius:4px;} QPushButton:disabled{background:#90a4ae;}")
        self.mark_button.setFocusPolicy(Qt.NoFocus)
        self.mark_button.clicked.connect(self.mark_requested.emit)
        layout.addWidget(self.mark_button)

        self.hint = QLabel(
            "Elige una palmada en la tabla (N / P), avanza fotograma a fotograma (← →) "
            "hasta el primer fotograma del impacto y pulsa M."
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#555;")
        layout.addWidget(self.hint)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._pixmap: Optional[QPixmap] = None  # noqa: UP045
        self._set_enabled(False)

    # ------------------------------------------------------------------

    def _set_enabled(self, enabled: bool) -> None:
        for button in self.buttons.values():
            button.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.speed_box.setEnabled(enabled)
        self.mark_button.setEnabled(enabled)

    def open(self, path: str) -> None:
        self.close_video()
        self.reader = VideoReader(path)
        self._set_enabled(True)
        self._show(self.reader.frame_at(0.0))

    def close_video(self) -> None:
        self.pause()
        if self.reader is not None:
            self.reader.close()
        self.reader = None
        self._set_enabled(False)
        self.screen.setPixmap(QPixmap())
        self.screen.setText("Sin vídeo.\nArrastra aquí un archivo de vídeo.")

    @property
    def current_time(self) -> Optional[float]:  # noqa: UP045
        if self.reader is None or self.reader.current is None:
            return None
        return self.reader.current.time_s

    def set_host_text(self, text: str) -> None:
        self.host_label.setText(text)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def seek(self, video_s: float) -> None:
        if self.reader is None:
            return
        self._show(self.reader.frame_at(video_s))

    def next_frame(self) -> None:
        if self.reader is not None:
            self._show(self.reader.next_frame())

    def previous_frame(self) -> None:
        if self.reader is not None:
            self.pause()
            self._show(self.reader.previous_frame())

    def step_seconds(self, delta: float) -> None:
        if self.current_time is not None:
            self.seek(self.current_time + delta)

    def toggle_play(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.reader is None:
            return
        self.playing = True
        self.buttons["play"].setText("⏸")
        self.timer.start(max(int(1000.0 / (self.reader.fps * self.speed)), 5))

    def pause(self) -> None:
        self.playing = False
        self.timer.stop()
        self.buttons["play"].setText("▶")

    def _tick(self) -> None:
        frame = self.reader.next_frame() if self.reader else None
        if frame is None:
            self.pause()
            return
        self._show(frame)

    def _speed_changed(self, text: str) -> None:
        self.speed = float(text.replace("×", "").replace(",", "."))
        if self.playing:
            self.play()

    def _slider_released(self) -> None:
        if self.reader is not None and self.reader.duration_s:
            self.seek(self.slider.value() / 1000.0 * self.reader.duration_s)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _show(self, frame: Optional[VideoFrame]) -> None:  # noqa: UP045
        if frame is None:
            return
        h, w, _ = frame.rgb.shape
        image = QImage(frame.rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._rescale()
        fps = self.reader.fps if self.reader else 0
        self.time_label.setText(f"Vídeo {format_seconds(frame.time_s)}  · fotograma ≈{frame.index_hint}  · {fps:.2f} fps")
        if self.reader and self.reader.duration_s and not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(int(1000 * frame.time_s / self.reader.duration_s))
            self.slider.blockSignals(False)
        self.frame_changed.emit(frame.time_s)

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.screen.setPixmap(self._pixmap.scaled(self.screen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()
