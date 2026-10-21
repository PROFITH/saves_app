"""
Main window of SAVES Studio.

The work follows the order of the task: open the session folder (or drop
files on the window), let the claps and jumps be detected in every sensor
(each clock's offset is found first), let the audio of the video place the
video, fix the events flagged in red, count the steps on the video if
needed, and export. A banner always says what the next step is.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from importers.manifest_importer import ManifestImporter
from studio.discovery import FORMAT_LABELS, VIDEO_SUFFIXES, classify_file, discover, sensor_entry_for
from studio.engine import FAIR_RMSE_MS, GOOD_RMSE_MS, VIDEO, SyncEngine, analyze_signal
from studio.export import label_times, read_phases
from studio.project import LOCATIONS, SyncProject, VideoEntry
from studio.signals import load_signal
from studio.ui.plots import SignalStack, sensor_color
from studio.ui.video_panel import VideoPanel
from studio.ui.workers import start_job
from studio.video import find_audio_events, has_audio

QUALITY_COLORS = {
    "reference": "#1565c0",
    "good": "#2e7d32",
    "fair": "#f9a825",
    "poor": "#c62828",
    "none": "#9e9e9e",
}
QUALITY_TEXT = {
    "reference": "Referencia",
    "good": "Bien",
    "fair": "Revisar",
    "poor": "Mal",
    "none": "Sin ajuste",
}
VIDEO_SOURCE_TEXT = {
    "claps": "ajustada con las palmadas del vídeo",
    "one_clap": "ajustada con 1 palmada (marca otra para corregir la deriva)",
    "recording_start": "según la hora de inicio registrada (aproximada)",
    "session_start": "aproximada: no hay hora de inicio del vídeo",
    "none": "",
}


def clock(t: Optional[float]) -> str:  # noqa: UP045
    if t is None:
        return "—"
    return datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


def residual_color(value: Optional[float]) -> QColor:  # noqa: UP045
    if value is None:
        return QColor("#9e9e9e")
    a = abs(value)
    if a <= GOOD_RMSE_MS:
        return QColor("#2e7d32")
    if a <= FAIR_RMSE_MS:
        return QColor("#ef8f00")
    return QColor("#c62828")


def _is_na(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and bool(np.isnan(value)))


def readonly_item(text: str, color: Optional[QColor] = None, bold: bool = False) -> QTableWidgetItem:  # noqa: UP045
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    if color is not None:
        item.setForeground(color)
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    return item


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAVES Studio")
        self.resize(1600, 950)
        self.setAcceptDrops(True)

        self.project: Optional[SyncProject] = None  # noqa: UP045
        self.engine: Optional[SyncEngine] = None  # noqa: UP045
        self.loading: dict[str, str] = {}   # sensor_id -> progress text
        self.errors: dict[str, str] = {}
        self._jobs = set()
        self._plotted: list[str] = []
        self._selected_event: Optional[str] = None  # noqa: UP045
        self._step_history: list[tuple[str, dict]] = []  # ("add" | "remove", step)
        self._phases = None
        self._step_counts: dict[int, tuple[int, int]] = {}  # activity -> (main, all phases)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save)

        self._build_actions()
        self._build_ui()
        self._build_shortcuts()
        self.refresh_all()

    # ==================================================================
    # Construction
    # ==================================================================

    def _build_actions(self) -> None:
        bar = QToolBar("Principal")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)

        self.act_open = QAction("📂 Abrir sesión…", self, shortcut=QKeySequence.Open, triggered=self.choose_folder)
        self.act_add = QAction("➕ Añadir archivos…", self, shortcut="Ctrl+Shift+O", triggered=self.choose_files)
        self.act_audio = QAction("🔊 Detectar palmadas y saltos en el audio", self, triggered=self.detect_audio)
        self.act_export = QAction("💾 Exportar", self, shortcut="Ctrl+E", triggered=self.export)
        self.act_xyz = QAction("Ejes X/Y/Z", self, checkable=True, triggered=self._toggle_xyz)
        self.act_xyz.setToolTip("Ver los tres ejes en lugar del módulo (vector magnitude)")
        for action in (self.act_open, self.act_add):
            bar.addAction(action)
        bar.addSeparator()
        bar.addWidget(QLabel(" Referencia: "))
        self.reference_box = QComboBox()
        self.reference_box.setToolTip(
            "Reloj al que se ajustan los demás. Automática: el Polar que más eventos captó (ya está en la hora "
            "del ordenador, y el otro Polar completa lo que le falte); si no hay, el vídeo con 2 o más eventos "
            "marcados; si tampoco, el sensor con más eventos, llevado a la hora de los avisos del cronómetro."
        )
        self.reference_box.currentIndexChanged.connect(self._reference_changed)
        bar.addWidget(self.reference_box)
        bar.addSeparator()
        bar.addAction(self.act_audio)
        bar.addAction(self.act_xyz)
        bar.addSeparator()
        bar.addAction(self.act_export)

        help_menu = self.menuBar().addMenu("Ayuda")
        help_menu.addAction("Abrir una sesión de demostración", self.open_demo)
        help_menu.addAction("Atajos de teclado", self.show_shortcuts)
        file_menu = self.menuBar().addMenu("Archivo")
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_add)
        file_menu.addAction(self.act_export)
        self.menuBar().insertMenu(help_menu.menuAction(), file_menu)

    def _build_ui(self) -> None:
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_welcome())
        self.stack.addWidget(self._build_workspace())

    def _build_welcome(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        box = QFrame()
        box.setStyleSheet("QFrame{border:2px dashed #90a4ae; border-radius:12px; background:#fafafa;}")
        inner = QVBoxLayout(box)
        inner.setContentsMargins(40, 40, 40, 40)
        title = QLabel("<h2>Sincroniza vídeo y sensores offline</h2>")
        title.setAlignment(Qt.AlignCenter)
        text = QLabel(
            "Arrastra aquí la <b>carpeta de la sesión</b> que guardó lab_protocol_timer "
            "(manifiesto, vídeo y CSV del Polar) o ábrela con el botón.<br>"
            "Después arrastra los archivos de los sensores: Axivity <b>.cwa</b>, "
            "ActiGraph <b>.gt3x</b>, GENEActiv o Parmay Matrix <b>.bin</b> y SENS <b>.csv</b>."
        )
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignCenter)
        buttons = QHBoxLayout()
        open_button = QPushButton("📂 Abrir carpeta de sesión")
        open_button.setStyleSheet("padding:10px 18px; font-size:14px;")
        open_button.clicked.connect(self.choose_folder)
        demo_button = QPushButton("Probar con una sesión de demostración")
        demo_button.setStyleSheet("padding:10px 18px;")
        demo_button.clicked.connect(self.open_demo)
        buttons.addStretch()
        buttons.addWidget(open_button)
        buttons.addWidget(demo_button)
        buttons.addStretch()
        for widget in (title, text):
            widget.setStyleSheet("border:none;")
            inner.addWidget(widget)
        inner.addLayout(buttons)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(box, 3)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()
        return page

    def _build_workspace(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(6, 6, 6, 6)

        # Banner with the next step.
        self.banner = QFrame()
        self.banner.setStyleSheet("QFrame{background:#e3f2fd; border-radius:6px;}")
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        self.banner_button = QPushButton()
        self.banner_button.setFocusPolicy(Qt.NoFocus)
        self.banner_button.clicked.connect(self._banner_clicked)
        banner_layout.addWidget(self.banner_label, 1)
        banner_layout.addWidget(self.banner_button)
        outer.addWidget(self.banner)
        self._banner_action = None

        main = QSplitter(Qt.Horizontal)
        outer.addWidget(main, 1)

        # Left: files and quality.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        files_box = QGroupBox("Archivos de la sesión")
        files_layout = QVBoxLayout(files_box)
        self.session_label = QLabel()
        self.session_label.setWordWrap(True)
        files_layout.addWidget(self.session_label)
        self.sensors_table = QTableWidget(0, 5)
        self.sensors_table.setHorizontalHeaderLabels(["", "Sensor", "Ubicación", "Hz", "Estado"])
        self.sensors_table.verticalHeader().hide()
        self.sensors_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.sensors_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.sensors_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sensors_table.setFocusPolicy(Qt.NoFocus)
        self.sensors_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sensors_table.customContextMenuRequested.connect(self._sensor_menu)
        files_layout.addWidget(self.sensors_table)
        add_button = QPushButton("➕ Añadir sensores o vídeo…")
        add_button.setFocusPolicy(Qt.NoFocus)
        add_button.clicked.connect(self.choose_files)
        files_layout.addWidget(add_button)
        left_layout.addWidget(files_box, 3)

        quality_box = QGroupBox("Calidad de la sincronización")
        quality_layout = QVBoxLayout(quality_box)
        self.quality_table = QTableWidget(0, 6)
        self.quality_table.setHorizontalHeaderLabels(["Sensor", "Estado", "Eventos", "Desfase", "Deriva", "Error"])
        self.quality_table.verticalHeader().hide()
        self.quality_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.quality_table.setFocusPolicy(Qt.NoFocus)
        self.quality_table.setToolTip(
            "Eventos: palmadas o saltos encontrados / esperados.\n"
            "Desfase: diferencia de reloj con la referencia en el primer evento.\n"
            "Deriva: lo que se adelanta o atrasa el reloj (ppm = µs por segundo).\n"
            "Error: RMSE de los eventos tras el ajuste (verde ≤ 25 ms, ámbar ≤ 80 ms)."
        )
        self.quality_table.cellDoubleClicked.connect(self._quality_double_clicked)
        quality_layout.addWidget(self.quality_table)
        self.video_quality = QLabel()
        self.video_quality.setWordWrap(True)
        quality_layout.addWidget(self.video_quality)
        left_layout.addWidget(quality_box, 2)
        main.addWidget(left)

        # Centre: plots over the claps table.
        centre = QSplitter(Qt.Vertical)
        self.plots = SignalStack()
        self.plots.clicked.connect(self._plot_clicked)
        self.plots.anchor_moved.connect(self._anchor_moved)
        self.plots.anchor_action.connect(self._anchor_action)
        self.plots.add_anchor.connect(self._anchor_moved)
        centre.addWidget(self.plots)
        self.tabs = QTabWidget()
        claps_box = QWidget()
        claps_layout = QVBoxLayout(claps_box)
        claps_layout.setContentsMargins(4, 4, 4, 4)
        claps_hint = QLabel("Doble clic: ir al evento · clic derecho: opciones")
        claps_hint.setStyleSheet("color:#555;")
        claps_layout.addWidget(claps_hint)
        self.claps_table = QTableWidget(0, 0)
        self.claps_table.verticalHeader().hide()
        self.claps_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.claps_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.claps_table.setFocusPolicy(Qt.NoFocus)
        self.claps_table.itemSelectionChanged.connect(self._clap_selected)
        self.claps_table.cellDoubleClicked.connect(lambda r, _c: self.focus_event(self._event_at(r)))
        self.claps_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.claps_table.customContextMenuRequested.connect(self._clap_menu)
        claps_layout.addWidget(self.claps_table)
        self.tabs.addTab(claps_box, "Palmadas y saltos")
        self.tabs.addTab(self._build_steps_tab(), "👣 Pasos")
        centre.addWidget(self.tabs)
        centre.setStretchFactor(0, 3)
        centre.setStretchFactor(1, 1)
        main.addWidget(centre)

        # Right: video.
        self.video = VideoPanel()
        self.video.frame_changed.connect(self._video_frame)
        self.video.mark_requested.connect(self.mark_video_clap)
        self.video.step_requested.connect(self.mark_step)
        main.addWidget(self.video)

        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 3)
        main.setStretchFactor(2, 1)
        main.setSizes([420, 720, 380])
        return page

    def _build_steps_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        hint = QLabel(
            "Reproduce el vídeo despacio (0,25× o 0,5×) y pulsa <b>S</b> en cada apoyo del pie "
            "(<b>A</b> izquierdo, <b>D</b> derecho si los distingues). <b>Retroceso</b> borra el paso "
            "más cercano y <b>Ctrl+Z</b> deshace. Doble clic en una actividad: ir a su inicio."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        layout.addWidget(hint)
        self.steps_table = QTableWidget(0, 5)
        self.steps_table.setHorizontalHeaderLabels(["Actividad", "Pasos (actividad)", "Pasos (con fases quietas)", "Cadencia", "Duración"])
        self.steps_table.verticalHeader().hide()
        self.steps_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.steps_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.steps_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.steps_table.setFocusPolicy(Qt.NoFocus)
        self.steps_table.cellDoubleClicked.connect(self._steps_row_clicked)
        layout.addWidget(self.steps_table)
        return page

    def _build_shortcuts(self) -> None:
        bindings = {
            "Space": self.video.toggle_play,
            "Right": self.video.next_frame,
            "Left": self.video.previous_frame,
            "Shift+Right": lambda: self.video.step_seconds(1.0),
            "Shift+Left": lambda: self.video.step_seconds(-1.0),
            "Ctrl+Right": lambda: self.video.step_seconds(10.0),
            "Ctrl+Left": lambda: self.video.step_seconds(-10.0),
            "M": self.mark_video_clap,
            "N": lambda: self.select_relative(1),
            "P": lambda: self.select_relative(-1),
            "Delete": self.clear_video_clap,
            "X": lambda: self.act_xyz.trigger(),
            "S": lambda: self.mark_step(""),
            "A": lambda: self.mark_step("L"),
            "D": lambda: self.mark_step("R"),
            "Backspace": self.remove_step,
            "Ctrl+Z": self.undo_step,
        }
        for key, slot in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)

    def show_shortcuts(self) -> None:
        QMessageBox.information(self, "Atajos de teclado", (
            "Espacio: reproducir / pausa\n"
            "← / →: fotograma anterior / siguiente\n"
            "Mayús + ← / →: 1 s atrás / adelante\n"
            "Ctrl + ← / →: 10 s atrás / adelante\n"
            "N / P: evento (palmada o salto) siguiente / anterior\n"
            "M: marcar el evento en el vídeo (fotograma actual)\n"
            "Supr: borrar la marca del vídeo del evento elegido\n"
            "S: paso en el fotograma actual (A: pie izquierdo, D: pie derecho)\n"
            "Retroceso: borrar el paso más cercano · Ctrl+Z: deshacer el último cambio de pasos\n"
            "X: ver ejes X/Y/Z o el módulo\n"
            "Gráficas: rueda para zoom, arrastrar para desplazarse, clic para llevar el vídeo a ese instante,\n"
            "arrastrar una línea de evento para corregirla, clic derecho para más opciones."
        ))

    # ==================================================================
    # Opening sessions and files
    # ==================================================================

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de la sesión")
        if folder:
            self.open_folder(folder)

    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Archivos de sensores o vídeo", self.project.folder if self.project else "",
            "Sesión (*.cwa *.gt3x *.bin *.csv *.json *.mp4 *.mov *.avi *.mkv);;Todos (*.*)",
        )
        if files:
            self.add_paths([Path(f) for f in files])

    def open_demo(self) -> None:
        from studio.demo import build_demo_session

        folder = Path(tempfile.gettempdir()) / "saves_studio_demo"
        if not (folder / "session_manifest.json").exists():
            self.statusBar().showMessage("Creando la sesión de demostración…")
            build_demo_session(folder)
        self.open_folder(folder)

    def open_folder(self, folder: str | Path) -> None:
        folder = Path(folder)
        self._close_session()
        found = discover(folder)
        if SyncProject.exists_in(folder):
            project = SyncProject.load(folder)
            known = {str(Path(s.path).resolve()) for s in project.sensors}
            for entry in found.sensors:
                if str(Path(entry.path).resolve()) not in known:
                    project.add_sensor(entry)
            if not project.videos:
                project.videos = found.videos
            project.manifest_path = project.manifest_path or found.manifest_path
        else:
            project = SyncProject(folder=str(folder), manifest_path=found.manifest_path, videos=found.videos)
            for entry in found.sensors:
                project.add_sensor(entry)
        self.project = project
        self._step_history = []
        self.stack.setCurrentIndex(1)
        self._load_manifest()
        self._open_video()
        for entry in project.sensors:
            self._load_sensor(entry.sensor_id)
        self._schedule_save()
        self.refresh_all()
        # First try on its own: the claps and landings in the video's sound.
        if self.engine is not None and self.project.videos and not self.project.video_events:
            QTimer.singleShot(0, lambda: self.detect_audio(auto=True))

    def add_paths(self, paths: list[Path]) -> None:
        if self.project is None:
            folder = next((p for p in paths if p.is_dir()), None)
            if folder is not None:
                self.open_folder(folder)
                paths = [p for p in paths if p != folder]
            else:
                manifest = next((p for p in paths if p.name == "session_manifest.json"), None)
                self.open_folder((manifest or paths[0]).parent)
        expanded = []
        for p in paths:
            expanded.extend(sorted(x for x in p.iterdir() if x.is_file()) if p.is_dir() else [p])
        added_video = False
        for p in expanded:
            if p.name == "session_manifest.json":
                self.project.manifest_path = str(p)
                self._load_manifest()
            elif p.suffix.lower() in VIDEO_SUFFIXES:
                if all(Path(v.path).resolve() != p.resolve() for v in self.project.videos):
                    self.project.videos.append(VideoEntry(path=str(p)))
                    added_video = True
            else:
                fmt = classify_file(p)
                if fmt is None:
                    continue
                entry = self.project.add_sensor(sensor_entry_for(p, fmt))
                if entry.sensor_id not in self.loading and (self.engine is None or entry.sensor_id not in self.engine.signals):
                    self._load_sensor(entry.sensor_id)
        if added_video and self.video.reader is None:
            self._open_video()
        self._schedule_save()
        self.refresh_all()

    def _close_session(self) -> None:
        self.video.close_video()
        self.project, self.engine = None, None
        self._phases = None
        self.loading, self.errors = {}, {}
        self._plotted = []
        self.plots.set_engine(None)

    def _load_manifest(self) -> None:
        if not self.project or not self.project.manifest_path:
            self.engine = None
            return
        try:
            session = ManifestImporter().load(self.project.manifest_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Manifiesto no válido", f"No se pudo leer el manifiesto:\n{exc}")
            self.engine = None
            return
        old = self.engine
        self.engine = SyncEngine(self.project, session)
        self._phases = None
        if old is not None:
            for sid, signal in old.signals.items():
                self._analyze_in_background(sid, signal)
        participants = ", ".join(p.participant_id for p in session.participants)
        self.setWindowTitle(f"SAVES Studio — {session.session_id}" + (f" ({participants})" if participants else ""))

    def _open_video(self) -> None:
        if not self.project or not self.project.videos:
            return
        try:
            self.video.open(self.project.videos[0].path)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"No se pudo abrir el vídeo: {exc}", 8000)
            return
        if self.engine is not None and self.video.reader is not None:
            self.engine.video_duration_s = self.video.reader.duration_s

    def _load_sensor(self, sensor_id: str) -> None:
        entry = self.project.sensor(sensor_id)
        project_folder = self.project.folder
        timezone = self.project.timezone
        self.loading[sensor_id] = "En cola"
        self.errors.pop(sensor_id, None)

        def work(progress):
            signal = load_signal(entry.path, entry.sensor_format, project_folder, timezone, progress)
            engine = self.engine
            if engine is None:
                return signal, {}, {}, None
            progress("Buscando palmadas y saltos")
            anchors, coverage, search = analyze_signal(engine.session, signal, engine.pad_seconds, engine.min_prominence)
            return signal, anchors, coverage, search

        self._run(work, lambda result: self._sensor_loaded(sensor_id, result),
                  lambda error: self._sensor_failed(sensor_id, error),
                  lambda text: self._sensor_progress(sensor_id, text))

    def _analyze_in_background(self, sensor_id: str, signal) -> None:
        engine = self.engine
        self.loading[sensor_id] = "Buscando palmadas"

        def work(_progress):
            anchors, coverage, search = analyze_signal(engine.session, signal, engine.pad_seconds, engine.min_prominence)
            return signal, anchors, coverage, search

        self._run(work, lambda result: self._sensor_loaded(sensor_id, result),
                  lambda error: self._sensor_failed(sensor_id, error))

    def _run(self, work, on_done, on_error, on_progress=None) -> None:
        holder = {}

        def done(result):
            self._jobs.discard(holder.get("job"))
            on_done(result)

        def failed(error):
            self._jobs.discard(holder.get("job"))
            on_error(error)

        holder["job"] = start_job(work, done, failed, on_progress)
        self._jobs.add(holder["job"])

    def _sensor_progress(self, sensor_id: str, text: str) -> None:
        self.loading[sensor_id] = text
        self.refresh_sensors_table()

    def _sensor_loaded(self, sensor_id: str, result) -> None:
        self.loading.pop(sensor_id, None)
        if self.project is None or self.engine is None:
            return
        try:
            self.project.sensor(sensor_id)
        except KeyError:
            return  # Removed while loading.
        signal, anchors, coverage, search = result
        self.engine.ingest(sensor_id, signal, anchors, coverage, search)
        self.engine.refit()
        self.refresh_all()

    def _sensor_failed(self, sensor_id: str, error: str) -> None:
        self.loading.pop(sensor_id, None)
        self.errors[sensor_id] = error
        self.refresh_all()

    # ==================================================================
    # Drag and drop
    # ==================================================================

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        if self.project is None and len(paths) == 1 and paths[0].is_dir():
            self.open_folder(paths[0])
        else:
            self.add_paths(paths)

    # ==================================================================
    # Refresh
    # ==================================================================

    def refresh_all(self) -> None:
        if self.project is None:
            self.stack.setCurrentIndex(0)
            for action in (self.act_add, self.act_audio, self.act_export, self.act_xyz):
                action.setEnabled(False)
            self.reference_box.setEnabled(False)
            return
        self.stack.setCurrentIndex(1)
        self.act_add.setEnabled(True)
        has_engine = self.engine is not None
        self.act_audio.setEnabled(has_engine and self.video.reader is not None)
        self.act_export.setEnabled(has_engine and bool(self.engine.signals) if has_engine else False)
        self.act_xyz.setEnabled(True)
        self.reference_box.setEnabled(True)
        self.refresh_reference_box()
        self.refresh_session_label()
        self.refresh_sensors_table()
        self.refresh_quality()
        self.refresh_claps_table()
        self.refresh_plots()
        self.refresh_steps()
        self.refresh_video_label()
        self.refresh_banner()

    def refresh_reference_box(self) -> None:
        self.reference_box.blockSignals(True)
        self.reference_box.clear()
        self.reference_box.addItem("Automática", "auto")
        for entry in self.project.sensors:
            self.reference_box.addItem(entry.sensor_id, entry.sensor_id)
        self.reference_box.addItem("Vídeo (palmadas marcadas)", VIDEO)
        index = self.reference_box.findData(self.project.reference)
        self.reference_box.setCurrentIndex(max(index, 0))
        if self.engine is not None and self.project.reference == "auto":
            self.reference_box.setItemText(0, f"Automática: {self.engine.reference_label()}")
        self.reference_box.blockSignals(False)

    def refresh_session_label(self) -> None:
        lines = []
        if self.engine is not None:
            s = self.engine.session
            lines.append(
                f"<b>{s.session_id}</b> · {clock(s.start_time.timestamp())}–{clock(s.end_time.timestamp())[:-4]}"
                f" · {len(s.activities)} actividades · {len(s.sync_windows)} palmadas o saltos"
            )
        else:
            lines.append("<span style='color:#c62828'>Falta session_manifest.json: arrástralo a la ventana.</span>")
        if self.project.videos:
            v = self.project.videos[0]
            extra = "" if v.recording_start_unix_s is not None else " (sin hora de inicio)"
            lines.append(f"🎥 {Path(v.path).name}{extra}")
        else:
            lines.append("🎥 Sin vídeo")
        self.session_label.setText("<br>".join(lines))

    def refresh_sensors_table(self) -> None:
        if self.project is None:
            return
        table = self.sensors_table
        table.blockSignals(True)
        table.setRowCount(len(self.project.sensors))
        all_ids = [s.sensor_id for s in self.project.sensors]
        for row, entry in enumerate(self.project.sensors):
            check = QCheckBox()
            check.setChecked(entry.enabled)
            check.setToolTip("Incluir en la sincronización")
            check.toggled.connect(lambda on, sid=entry.sensor_id: self._toggle_sensor(sid, on))
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(4, 0, 0, 0)
            h.addWidget(check)
            table.setCellWidget(row, 0, holder)
            name = readonly_item(entry.sensor_id, QColor(sensor_color(all_ids.index(entry.sensor_id))), bold=True)
            name.setToolTip(f"{FORMAT_LABELS.get(entry.sensor_format, entry.sensor_format)}\n{entry.path}\n{entry.serial}")
            table.setItem(row, 1, name)
            combo = QComboBox()
            combo.addItems(LOCATIONS)
            if entry.location not in LOCATIONS:
                combo.addItem(entry.location)  # From the manifest (e.g. with side).
            combo.setCurrentText(entry.location)
            combo.setFocusPolicy(Qt.NoFocus)
            combo.currentTextChanged.connect(lambda text, sid=entry.sensor_id: self._location_changed(sid, text))
            table.setCellWidget(row, 2, combo)
            signal = self.engine.signals.get(entry.sensor_id) if self.engine else None
            table.setItem(row, 3, readonly_item(f"{signal.sampling_rate_hz:g}" if signal else ""))
            if entry.sensor_id in self.loading:
                status = readonly_item(f"⏳ {self.loading[entry.sensor_id]}…", QColor("#1565c0"))
            elif entry.sensor_id in self.errors:
                status = readonly_item("✗ Error al leer", QColor("#c62828"))
                status.setToolTip(self.errors[entry.sensor_id])
            elif signal is not None:
                hours = signal.duration_s / 3600
                span = f"{hours:.1f} h" if hours >= 1 else f"{signal.duration_s / 60:.0f} min"
                status = readonly_item(f"✓ {span} · {FORMAT_LABELS.get(entry.sensor_format, '')}")
            else:
                status = readonly_item("—")
            table.setItem(row, 4, status)
        table.blockSignals(False)

    def refresh_quality(self) -> None:
        table = self.quality_table
        if self.engine is None:
            table.setRowCount(0)
            self.video_quality.setText("")
            return
        ids = self.engine.active_sensors()
        table.setRowCount(len(ids))
        all_ids = [s.sensor_id for s in self.project.sensors]
        for row, sid in enumerate(ids):
            st = self.engine.status(sid)
            table.setItem(row, 0, readonly_item(sid, QColor(sensor_color(all_ids.index(sid))), bold=True))
            state = readonly_item(f"● {QUALITY_TEXT[st.quality]}", QColor(QUALITY_COLORS[st.quality]), bold=True)
            state.setToolTip(st.note)
            table.setItem(row, 1, state)
            table.setItem(row, 2, readonly_item(f"{st.n_anchors}/{st.n_windows}"))
            table.setItem(row, 3, readonly_item("" if st.offset_ms is None else f"{st.offset_ms / 1000:+.3f} s"))
            table.setItem(row, 4, readonly_item("" if st.drift_ppm is None else f"{st.drift_ppm:+.0f} ppm"))
            table.setItem(row, 5, readonly_item("" if st.rmse_ms is None else f"{st.rmse_ms:.1f} ms",
                                                residual_color(st.rmse_ms) if st.rmse_ms is not None else None, bold=True))
        if self.project.videos:
            n = len(self.project.video_events)
            residuals = self.engine.video_residuals_ms()
            text = f"🎥 Vídeo: {n} palmada(s) marcada(s); hora {VIDEO_SOURCE_TEXT[self.engine.video_fit_source]}"
            if residuals:
                rmse = (sum(r * r for r in residuals.values()) / len(residuals)) ** 0.5
                text += f". Error {rmse:.0f} ms (1 fotograma = {1000 / self.video.reader.fps:.0f} ms)" if self.video.reader else ""
            self.video_quality.setText(text)
        else:
            self.video_quality.setText("")

    def refresh_claps_table(self) -> None:
        table = self.claps_table
        if self.engine is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        ids = self.engine.active_sensors()
        has_video = bool(self.project.videos)
        headers = ["Evento", "Tipo", "Actividad", "Fase"] + (["Vídeo"] if has_video else []) + ids
        table.blockSignals(True)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        windows = self.engine.session.sync_windows
        table.setRowCount(len(windows))
        names = {a.activity_index: a.activity_name for a in self.engine.session.activities}
        video_res = self.engine.video_residuals_ms()
        phase_text = {"static_prep": "antes", "static_post": "después", "transition_lap": "transición", "activity_main": "durante"}
        for row, window in enumerate(windows):
            eid = window.event_id
            table.setItem(row, 0, readonly_item(eid, bold=True))
            table.setItem(row, 1, readonly_item("salto" if window.sync_kind == "jump" else "palmada"))
            table.setItem(row, 2, readonly_item(names.get(window.activity_index) or f"Actividad {window.activity_index}"))
            table.setItem(row, 3, readonly_item(phase_text.get(window.phase, window.phase)))
            col = 4
            if has_video:
                if eid in self.project.video_events:
                    r = video_res.get(eid)
                    text = "✓" + (f" {r:+.0f} ms" if r is not None else "")
                    table.setItem(row, col, readonly_item(text, residual_color(r) if r is not None else QColor("#1565c0")))
                else:
                    table.setItem(row, col, readonly_item("marcar", QColor("#9e9e9e")))
                col += 1
            for sid in ids:
                state = self.engine.anchor_state(sid, eid)
                r = self.engine.residual_ms(sid, eid)
                if state == "missing":
                    item = readonly_item("✗ no hallada", QColor("#c62828"))
                elif state == "excluded":
                    item = readonly_item("⊘ excluida", QColor("#9e9e9e"))
                else:
                    mark = "✋" if state == "manual" else "✓"
                    if sid == self.engine.resolved_reference():
                        item = readonly_item(f"{mark} ref.", QColor("#1565c0"))
                    else:
                        item = readonly_item(f"{mark} {r:+.0f} ms" if r is not None else mark, residual_color(r))
                table.setItem(row, col, item)
                col += 1
        table.resizeColumnsToContents()
        table.blockSignals(False)
        if self._selected_event:
            self._select_row(self._selected_event, focus=False)

    def refresh_plots(self) -> None:
        if self.engine is None:
            self.plots.set_engine(None)
            self._plotted = []
            return
        active = self.engine.active_sensors()
        if active != self._plotted or self.plots.engine is not self.engine:
            view = self.plots.view_range() if self._plotted else None
            self.plots.set_engine(self.engine)
            if view:
                self.plots.focus(*view)
            self._plotted = list(active)
        else:
            self.plots.refresh_overview()
            self.plots.refresh_markers()
            self.plots.refresh_data()
        self._video_frame(self.video.current_time)

    def refresh_video_label(self) -> None:
        self._video_frame(self.video.current_time)

    # ------------------------------------------------------------------
    # Next-step banner
    # ------------------------------------------------------------------

    def refresh_banner(self) -> None:
        text, button, action, tone = self._next_step()
        colors = {"info": "#e3f2fd", "warn": "#fff8e1", "ok": "#e8f5e9", "error": "#ffebee"}
        self.banner.setStyleSheet(f"QFrame{{background:{colors[tone]}; border-radius:6px;}}")
        self.banner_label.setText(text)
        self.banner_button.setVisible(bool(button))
        self.banner_button.setText(button or "")
        self._banner_action = action

    def _banner_clicked(self) -> None:
        if self._banner_action:
            self._banner_action()

    def _next_step(self):
        if self.engine is None:
            return ("<b>Falta el manifiesto de la sesión</b> (session_manifest.json). Arrástralo a la ventana.",
                    "Añadir archivos…", self.choose_files, "error")
        if not self.project.sensors:
            return ("<b>Añade los sensores:</b> arrastra los archivos .cwa (Axivity), .gt3x (ActiGraph), .bin (GENEActiv o Parmay Matrix) o .csv (SENS).",
                    "Añadir archivos…", self.choose_files, "info")
        if self.loading:
            return (f"⏳ Cargando {len(self.loading)} archivo(s) y buscando palmadas y saltos. La primera vez tarda más; "
                    "luego se abre desde la caché.", None, None, "info")
        ref = self.engine.resolved_reference()
        if ref is None:
            return ("<b>No hay referencia.</b> Elige un sensor como referencia arriba o marca al menos 2 palmadas o saltos en el vídeo.",
                    None, None, "warn")
        for sid in self.engine.active_sensors():
            st = self.engine.status(sid)
            if st.quality == "none":
                missing = next((w.event_id for w in self.engine.session.sync_windows
                                if self.engine.anchor_state(sid, w.event_id) == "missing"), None)
                return (f"<b>{sid}</b> no comparte 2 palmadas o saltos con la referencia. Revisa sus ventanas: "
                        "arrastra la línea o haz clic derecho en la señal para poner el evento a mano.",
                        "Ir a la primera" if missing else None,
                        (lambda e=missing: self.focus_event(e)) if missing else None, "warn")
        for sid in self.engine.active_sensors():
            st = self.engine.status(sid)
            if st.quality == "poor" or (st.quality == "fair" and st.n_pairs >= 3):
                worst = self.engine.worst_event(sid)
                r = self.engine.residual_ms(sid, worst)
                return (f"<b>Revisa {sid}:</b> el evento {worst} se desvía {r:+.0f} ms. Corrígelo arrastrando su línea "
                        "o exclúyelo (clic derecho).", "Ir al evento", lambda e=worst: self.focus_event(e), "warn")
        if self.project.videos and ref != VIDEO and self.engine.video_fit_source != "claps":
            pending = next((w.event_id for w in self.engine.session.sync_windows
                            if w.event_id not in self.project.video_events), None)
            audio = " o pulsa «Detectar palmadas y saltos en el audio»" if self.video.reader else ""
            return (f"<b>Marca las palmadas o saltos en el vídeo</b> (al menos 2){audio}. Al marcar uno, se pasa solo al siguiente.",
                    "Empezar", lambda e=pending: self.focus_event(e), "info")
        steps = f" Llevas {len(self.project.steps)} pasos contados." if self.project.steps else ""
        return ("✅ <b>Todo sincronizado.</b> Revisa la tabla de calidad y exporta los resultados."
                f"{steps} Para contar pasos, abre la pestaña «Pasos».",
                "💾 Exportar", self.export, "ok")

    # ==================================================================
    # Claps: selection, video marks and corrections
    # ==================================================================

    def _event_at(self, row: int) -> Optional[str]:  # noqa: UP045
        item = self.claps_table.item(row, 0)
        return item.text() if item else None

    def _select_row(self, event_id: str, focus: bool = True) -> None:
        for row in range(self.claps_table.rowCount()):
            if self._event_at(row) == event_id:
                self.claps_table.blockSignals(True)
                self.claps_table.selectRow(row)
                self.claps_table.blockSignals(False)
                self.claps_table.scrollToItem(self.claps_table.item(row, 0))
                break
        self._selected_event = event_id
        if focus:
            self.focus_event(event_id)

    def _clap_selected(self) -> None:
        rows = self.claps_table.selectionModel().selectedRows()
        if rows:
            event_id = self._event_at(rows[0].row())
            if event_id != self._selected_event:
                self._selected_event = event_id
                self.focus_event(event_id)

    def select_relative(self, step: int) -> None:
        if self.engine is None or not self.engine.session.sync_windows:
            return
        ids = [w.event_id for w in self.engine.session.sync_windows]
        if self._selected_event in ids:
            index = (ids.index(self._selected_event) + step) % len(ids)
        else:
            index = 0 if step > 0 else len(ids) - 1
        self._select_row(ids[index])

    def focus_event(self, event_id: Optional[str]) -> None:  # noqa: UP045
        """Zoom the plots on a sync window and take the video just before its clap."""
        if self.engine is None or event_id is None:
            return
        if self._selected_event != event_id:
            self._select_row(event_id, focus=False)
        t0, t1 = self.engine.window_display_range(event_id)
        self.plots.focus(t0 - 2.0, t1 + 2.0)
        if self.video.reader is not None:
            if event_id in self.project.video_events:
                self.video.seek(self.project.video_events[event_id])
            else:
                clap = self.engine.display_time_of_anchor(event_id)
                target = (clap - 0.7) if clap is not None else t0
                self.video.seek(float(self.engine.display_to_video(target)))

    def mark_video_clap(self) -> None:
        if self.engine is None or self.video.current_time is None:
            return
        event_id = self._selected_event
        if event_id is None:
            display = float(self.engine.video_to_display(self.video.current_time))
            event_id = self.plots._nearest_window(display).event_id
        self.project.video_events[event_id] = float(self.video.current_time)
        self._changed()
        self.statusBar().showMessage(f"Palmada {event_id} marcada en el vídeo a {self.video.current_time:.3f} s", 4000)
        # Go straight to the next clap still unmarked in the video.
        pending = [w.event_id for w in self.engine.session.sync_windows if w.event_id not in self.project.video_events]
        if pending:
            ids = [w.event_id for w in self.engine.session.sync_windows]
            after = [e for e in pending if ids.index(e) > ids.index(event_id)]
            self._select_row((after or pending)[0])

    def clear_video_clap(self) -> None:
        if self.project and self._selected_event in self.project.video_events:
            del self.project.video_events[self._selected_event]
            self._changed()

    def _anchor_moved(self, sensor_id: str, event_id: str, display_s: float) -> None:
        if self.engine is None:
            return
        native = float(self.engine.display_to_sensor(sensor_id, display_s))
        self.project.set_manual_anchor(sensor_id, event_id, native)
        self.statusBar().showMessage(f"Palmada {event_id} de {sensor_id} corregida a mano", 4000)
        self._changed()

    def _anchor_action(self, sensor_id: str, event_id: str, action: str) -> None:
        if action == "exclude":
            self.project.set_excluded(sensor_id, event_id, True)
        elif action == "include":
            self.project.set_excluded(sensor_id, event_id, False)
        elif action == "auto":
            self.project.clear_manual_anchor(sensor_id, event_id)
        elif action == "focus":
            self.focus_event(event_id)
            return
        self._changed()

    def _clap_menu(self, pos) -> None:
        index = self.claps_table.indexAt(pos)
        if not index.isValid() or self.engine is None:
            return
        event_id = self._event_at(index.row())
        header = self.claps_table.horizontalHeaderItem(index.column()).text()
        menu = QMenu(self)
        menu.addAction("Ir a este evento", lambda: self.focus_event(event_id))
        if header == "Vídeo":
            if self.video.current_time is not None:
                menu.addAction("Marcar en el fotograma actual del vídeo", lambda: (self._select_row(event_id, focus=False), self.mark_video_clap()))
            if event_id in self.project.video_events:
                menu.addAction("Borrar la marca del vídeo", lambda: (self.project.video_events.pop(event_id, None), self._changed()))
        elif header in self.engine.signals:
            state = self.engine.anchor_state(header, event_id)
            if state in ("auto", "manual"):
                menu.addAction(f"Excluir del ajuste de {header}", lambda: self._anchor_action(header, event_id, "exclude"))
            if state == "excluded":
                menu.addAction("Volver a incluir", lambda: self._anchor_action(header, event_id, "include"))
            if state == "manual":
                menu.addAction("Volver a la detección automática", lambda: self._anchor_action(header, event_id, "auto"))
            if self.video.current_time is not None:
                menu.addAction(
                    f"Poner el evento de {header} en el instante del vídeo",
                    lambda: self._anchor_moved(header, event_id, float(self.engine.video_to_display(self.video.current_time))),
                )
        menu.exec(self.claps_table.viewport().mapToGlobal(pos))

    def _quality_double_clicked(self, row: int, _col: int) -> None:
        sid = self.quality_table.item(row, 0).text()
        worst = self.engine.worst_event(sid) if self.engine else None
        if worst:
            self.focus_event(worst)

    # ==================================================================
    # Video and plots interaction
    # ==================================================================

    def _video_frame(self, video_s: Optional[float]) -> None:  # noqa: UP045
        if self.engine is None or video_s is None:
            self.plots.set_cursor(None)
            self.video.set_host_text("")
            return
        display = float(self.engine.video_to_display(video_s))
        self.plots.set_cursor(display, follow=self.video.playing)
        self.video.set_host_text(f"Hora de sesión {clock(display)}  ({VIDEO_SOURCE_TEXT[self.engine.video_fit_source]})")
        self.video.set_steps_text(self._steps_text(display))

    def _plot_clicked(self, display_s: float) -> None:
        if self.engine is not None and self.video.reader is not None:
            self.video.pause()
            self.video.seek(float(self.engine.display_to_video(display_s)))

    def _toggle_xyz(self, enabled: bool) -> None:
        self.plots.set_xyz(enabled)

    # ==================================================================
    # Sensors list edits
    # ==================================================================

    def _toggle_sensor(self, sensor_id: str, enabled: bool) -> None:
        self.project.sensor(sensor_id).enabled = enabled
        self._changed()

    def _location_changed(self, sensor_id: str, location: str) -> None:
        self.project.sensor(sensor_id).location = location
        self._plotted = []  # Titles show the location.
        self._changed()

    def _sensor_menu(self, pos) -> None:
        row = self.sensors_table.indexAt(pos).row()
        if row < 0:
            return
        sid = self.project.sensors[row].sensor_id
        menu = QMenu(self)
        menu.addAction("Usar como referencia", lambda: self._set_reference(sid))
        menu.addAction("Volver a leer el archivo", lambda: self._load_sensor(sid))
        menu.addAction("Abrir la carpeta del archivo", lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(self.project.sensor(sid).path).parent))))
        menu.addSeparator()
        menu.addAction("Quitar de la sesión", lambda: self._remove_sensor(sid))
        menu.exec(self.sensors_table.viewport().mapToGlobal(pos))

    def _remove_sensor(self, sensor_id: str) -> None:
        self.project.remove_sensor(sensor_id)
        if self.engine is not None:
            self.engine.drop_signal(sensor_id)
        self.loading.pop(sensor_id, None)
        self._changed()

    def _set_reference(self, value: str) -> None:
        self.project.reference = value
        self._changed()

    def _reference_changed(self, _index: int) -> None:
        value = self.reference_box.currentData()
        if self.project is not None and value is not None and value != self.project.reference:
            self._set_reference(value)

    # ==================================================================
    # Steps counted on the video
    # ==================================================================

    def _session_phases(self):
        if self._phases is None and self.engine is not None:
            self._phases = read_phases(self.project.folder, self.engine.session)
        return self._phases

    def _activity_names(self) -> dict[int, str]:
        if self.engine is None:
            return {}
        return {a.activity_index: a.activity_name or f"Actividad {a.activity_index}" for a in self.engine.session.activities}

    def mark_step(self, foot: str = "") -> None:
        """Step at the frame on screen (keys S, A and D)."""
        if self.project is None or self.video.current_time is None:
            return
        step = self.project.add_step(self.video.current_time, foot)
        self._step_history.append(("add", step))
        self._steps_changed(f"Paso {len(self.project.steps)} a {step['video_s']:.2f} s" + (f" (pie {'izquierdo' if foot == 'L' else 'derecho'})" if foot else ""))

    def remove_step(self) -> None:
        if self.project is None or self.video.current_time is None:
            return
        removed = self.project.remove_step_near(self.video.current_time)
        if removed is None:
            self.statusBar().showMessage("No hay ningún paso a menos de 0,5 s del fotograma actual", 4000)
            return
        self._step_history.append(("remove", removed))
        self._steps_changed(f"Paso de {removed['video_s']:.2f} s borrado")

    def undo_step(self) -> None:
        if self.project is None or not self._step_history:
            return
        action, step = self._step_history.pop()
        if action == "add":
            self.project.steps = [s for s in self.project.steps if s is not step and s != step]
        else:
            self.project.add_step(step["video_s"], step.get("foot", ""))
        self._steps_changed("Deshecho")

    def _steps_changed(self, message: str) -> None:
        self.statusBar().showMessage(message, 3000)
        self._schedule_save()
        self.refresh_steps()

    def refresh_steps(self) -> None:
        if self.project is None:
            return
        steps = self.project.step_times()
        if self.engine is None:
            self.plots.set_steps([])
            return
        display = [float(t) for t in np.atleast_1d(self.engine.video_to_display(steps))] if steps else []
        self.plots.set_steps(display)
        phases = self._session_phases()
        names = self._activity_names()
        self._step_counts = {}
        if display:
            labels = label_times(np.asarray(display, dtype=float), phases, names)
            for act, phase in zip(labels["activity_index"], labels["phase"]):
                if _is_na(act):
                    continue
                main, total = self._step_counts.get(int(act), (0, 0))
                self._step_counts[int(act)] = (main + (phase == "activity_main"), total + 1)
        table = self.steps_table
        activities = self.engine.session.activities
        table.setRowCount(len(activities))
        for row, activity in enumerate(activities):
            index = activity.activity_index
            main_phases = phases[(phases["activity_index"] == index) & (phases["phase"] == "activity_main")] if phases is not None else []
            main_s = float((main_phases["end_s"] - main_phases["start_s"]).sum()) if len(main_phases) else activity.duration_seconds
            n_main, n_all = self._step_counts.get(index, (0, 0))
            cadence = f"{60.0 * n_main / main_s:.0f} pasos/min" if main_s > 0 and n_main else ""
            item = readonly_item(names.get(index, ""), bold=bool(n_all))
            item.setData(Qt.UserRole, index)
            table.setItem(row, 0, item)
            table.setItem(row, 1, readonly_item(str(n_main) if n_all else "", QColor("#4a148c"), bold=True))
            table.setItem(row, 2, readonly_item(str(n_all) if n_all else ""))
            table.setItem(row, 3, readonly_item(cadence))
            table.setItem(row, 4, readonly_item(f"{main_s:.0f} s"))
        self.tabs.setTabText(1, f"👣 Pasos ({len(steps)})" if steps else "👣 Pasos")
        self._video_frame(self.video.current_time)

    def _steps_row_clicked(self, row: int, _col: int) -> None:
        item = self.steps_table.item(row, 0)
        if item is None or self.engine is None:
            return
        index = item.data(Qt.UserRole)
        activity = next((a for a in self.engine.session.activities if a.activity_index == index), None)
        if activity is None:
            return
        start, end = activity.start_time.timestamp(), activity.end_time.timestamp()
        self.plots.focus(start - 5.0, min(end, start + 60.0))
        if self.video.reader is not None:
            self.video.pause()
            self.video.seek(float(self.engine.display_to_video(start)))

    def _steps_text(self, display: float) -> str:
        """Steps so far and those of the activity on screen."""
        if self.project is None or not self.project.steps or self.engine is None:
            return "Pasos: ninguno todavía (S en cada apoyo del pie)"
        phases = self._session_phases()
        if phases is None or not len(phases):
            return f"Pasos: {len(self.project.steps)}"
        here = label_times(np.asarray([display], dtype=float), phases, self._activity_names())
        index = here["activity_index"][0]
        if _is_na(index):
            return f"Pasos: {len(self.project.steps)} en total"
        main, total = self._step_counts.get(int(index), (0, 0))
        return (f"Pasos: {len(self.project.steps)} en total · «{here['activity'][0]}»: "
                f"{main} en la actividad ({total} con las fases quietas)")

    # ==================================================================
    # Audio, export, saving
    # ==================================================================

    def detect_audio(self, auto: bool = False) -> None:
        """Find the claps and landings in the sound of the video."""
        if self.engine is None or not self.project.videos:
            return
        path = self.project.videos[0].path
        try:
            sound = has_audio(path)
        except Exception:  # noqa: BLE001 - unreadable video
            sound = False
        if not sound:
            if not auto:
                QMessageBox.information(self, "El vídeo no tiene sonido", (
                    "Este vídeo se grabó sin audio, así que las palmadas y los saltos hay que marcarlos a mano.\n\n"
                    "Para las próximas sesiones, graba con audio en lab_protocol_timer "
                    "y se detectarán solos."
                ))
            return
        if self.engine.video_fit is None:
            self.engine.refit()  # the video is placed on the session clock
        # How far the video may be from where it is placed: little when the
        # timer logged its start or claps are marked, the whole video when not.
        if self.engine.video_fit_source in ("claps", "one_clap"):
            search_s = 3.0
        elif self.engine.video_fit_source == "recording_start":
            search_s = 10.0
        else:
            search_s = max(float(self.engine.video_duration_s or 0.0), 3600.0)
        margin = 3.0
        windows, expected = {}, {}
        for w in self.engine.session.sync_windows:
            if w.expected_time is not None:
                # Around the cue: from just before the go to 3.5 s after it.
                cue = w.cue_time.timestamp()
                windows[w.event_id] = (
                    float(self.engine.display_to_video(cue - 1.0)),
                    float(self.engine.display_to_video(cue + 3.5)),
                )
                expected[w.event_id] = float(self.engine.display_to_video(w.expected_time.timestamp()))
            else:
                windows[w.event_id] = (
                    float(self.engine.display_to_video(w.window_start.timestamp() - margin)),
                    float(self.engine.display_to_video(w.window_end.timestamp() + margin)),
                )
        self.statusBar().showMessage("Buscando palmadas y saltos en el audio del vídeo…")

        def done(match):
            if self.project is None:
                return
            found = match.found
            new = {e: t for e, t in found.items() if e not in self.project.video_events}
            self.project.video_events.update(new)
            self._changed()
            kept = len(found) - len(new)
            msg = f"Audio: {len(found)} de {len(windows)} palmadas o saltos encontrados, {len(new)} añadidos"
            if kept:
                msg += f" ({kept} ya estaban marcados a mano y no se tocan)"
            if match.searched and abs(match.shift_s) >= 1.0:
                msg += f"; el vídeo estaba {abs(match.shift_s):.1f} s {'después' if match.shift_s > 0 else 'antes'} de lo previsto"
            if len(found) < len(windows):
                msg += ". Revisa los que faltan en la columna «Vídeo»."
            self.statusBar().showMessage(msg, 15000)

        def failed(error):
            if auto:
                self.statusBar().showMessage(f"No se pudo analizar el audio: {error}", 8000)
            else:
                QMessageBox.warning(self, "Audio", f"No se pudo analizar el audio:\n{error}")

        self._run(lambda _p: find_audio_events(path, windows, expected_video_s=expected, search_s=search_s),
                  done, failed)

    def export(self) -> None:
        if self.engine is None or not self.engine.signals:
            return
        default = str(Path(self.project.folder) / "saves_sync")
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de resultados", default) or None
        if folder is None:
            Path(default).mkdir(exist_ok=True)
            box = QMessageBox(self)
            box.setWindowTitle("Exportar")
            box.setText(f"¿Exportar en {default}?")
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            aligned = QCheckBox("Incluir las señales alineadas (archivos grandes)")
            box.setCheckBox(aligned)
            if box.exec() != QMessageBox.Ok:
                return
            folder, export_aligned = default, aligned.isChecked()
        else:
            export_aligned = QMessageBox.question(
                self, "Exportar", "¿Incluir también las señales alineadas? (archivos grandes)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) == QMessageBox.Yes
        engine = self.engine
        self.statusBar().showMessage("Exportando…")

        def done(paths):
            self.statusBar().showMessage(f"Exportados {len(paths)} archivos en {folder}", 8000)
            box = QMessageBox(self)
            box.setWindowTitle("Exportación terminada")
            box.setText(f"Se han guardado {len(paths)} archivos en:\n{folder}")
            open_button = box.addButton("Abrir carpeta", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Close)
            box.exec()
            if box.clickedButton() is open_button:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

        self._run(lambda _p: engine.export(folder, export_aligned), done,
                  lambda e: QMessageBox.warning(self, "Exportar", f"No se pudo exportar:\n{e}"))

    def _changed(self) -> None:
        if self.engine is not None:
            self.engine.refit()
        self._schedule_save()
        self.refresh_all()

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def _save(self) -> None:
        if self.project is not None:
            try:
                self.project.save()
            except OSError as exc:
                self.statusBar().showMessage(f"No se pudo guardar el proyecto: {exc}", 8000)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save()
        self.video.close_video()
        super().closeEvent(event)
