"""
Stacked signal plots sharing one time axis (the host clock).

Each sensor is drawn in the reference timeline using its current fit, so
after synchronizing, the same clap lines up vertically across sensors. Only
the points needed for the visible range are drawn (min/max pyramid), which
keeps zooming and panning fluid with week-long recordings. The technician
can hide the plots that do not help the task at hand (they still take part
in the synchronization).
"""

from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QMenu, QVBoxLayout, QWidget

from studio.engine import SyncEngine

SENSOR_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
AXIS_COLORS = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
ANCHOR_PENS = {
    "auto": pg.mkPen("#2e7d32", width=2),
    "manual": pg.mkPen("#1565c0", width=2),
    "excluded": pg.mkPen("#9e9e9e", width=1, style=Qt.DashLine),
}
ACTIVITY_BRUSHES = [QColor(255, 193, 7, 28), QColor(0, 150, 136, 28), QColor(156, 39, 176, 24), QColor(3, 169, 244, 26)]
EVENT_COLORS = {"steps": "#6a1b9a", "sit_to_stand": "#00838f"}
OTHER_EVENT_COLOR = "#ad1457"

pg.setConfigOptions(antialias=False, background="w", foreground="#333")


def sensor_color(index: int) -> str:
    return SENSOR_COLORS[index % len(SENSOR_COLORS)]


def event_color(key: str) -> str:
    return EVENT_COLORS.get(key, OTHER_EVENT_COLOR)


class SignalStack(QWidget):
    clicked = Signal(float)                 # display time (s)
    anchor_moved = Signal(str, str, float)  # sensor_id, event_id, display time
    anchor_action = Signal(str, str, str)   # sensor_id, event_id, action
    add_anchor = Signal(str, str, float)    # sensor_id, event_id, display time
    range_changed = Signal(float, float)
    plot_visibility = Signal(str, str)      # sensor_id, "hide" | "only"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine: Optional[SyncEngine] = None  # noqa: UP045
        self.show_xyz = False
        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[str, list[pg.PlotDataItem]] = {}
        self.markers: dict[str, list] = {}
        self.cursors: dict[str, pg.InfiniteLine] = {}
        self.static_items: dict[str, list] = {}
        self.event_ticks: dict[str, pg.VTickGroup] = {}
        self._events: list[float] = []
        self._event_color = event_color("steps")
        self._updating_region = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.overview = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.overview.setMaximumHeight(90)
        self.overview.setMinimumHeight(70)
        self.overview.hideAxis("left")
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.setMenuEnabled(False)
        self.overview_curve = self.overview.plot(pen=pg.mkPen("#607d8b"))
        self.region = pg.LinearRegionItem(brush=QColor(33, 150, 243, 40))
        self.region.setZValue(10)
        self.overview.addItem(self.region)
        self.region.sigRegionChanged.connect(self._region_moved)
        layout.addWidget(self.overview)

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.ci.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        self.canvas.scene().sigMouseClicked.connect(self._scene_clicked)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(30)
        self._redraw_timer.timeout.connect(self.refresh_data)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def set_engine(self, engine: Optional[SyncEngine]) -> None:  # noqa: UP045
        self.engine = engine
        self.rebuild()

    def sensor_ids(self) -> list[str]:
        """Sensors with a plot: those in the fit whose plot is not hidden."""
        if self.engine is None:
            return []
        hidden = set(self.engine.project.hidden_plots)
        return [sid for sid in self.engine.active_sensors() if sid not in hidden]

    def rebuild(self) -> None:
        self.canvas.clear()
        self.plots, self.curves, self.markers, self.cursors, self.static_items = {}, {}, {}, {}, {}
        self.event_ticks = {}
        if self.engine is None:
            return
        if self.engine.active_sensors() and not self.sensor_ids():
            self.canvas.addLabel(
                "Todos los gráficos están ocultos. Vuelve a mostrarlos en «👁 Gráficos» (barra de arriba).",
                row=0, col=0, color="#555",
            )
        first = None
        all_ids = [s.sensor_id for s in self.engine.project.sensors]
        for row, sid in enumerate(self.sensor_ids()):
            entry = self.engine.project.sensor(sid)
            color = sensor_color(all_ids.index(sid))
            axis = pg.DateAxisItem(orientation="bottom")
            plot = self.canvas.addPlot(row=row, col=0, axisItems={"bottom": axis})
            plot.setMenuEnabled(False)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getViewBox().setMouseEnabled(x=True, y=False)
            plot.getViewBox().enableAutoRange(axis="y", enable=True)
            plot.getViewBox().setAutoVisible(y=True)
            plot.setLabel("left", f"<b>{sid}</b>", color=color)
            title = f"<span style='color:{color}'><b>{sid}</b></span> · {entry.location}"
            if entry.serial:
                title += f" · {entry.serial}"
            plot.setTitle(title, size="9pt", justify="left")
            plot.setDownsampling(auto=False)
            plot.setClipToView(True)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            if row < len(self.sensor_ids()) - 1:
                plot.getAxis("bottom").setStyle(showValues=False)
            plot.getViewBox().sigXRangeChanged.connect(self._range_moved)
            self.plots[sid] = plot
            cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f9a825", width=2))
            cursor.setZValue(20)
            plot.addItem(cursor, ignoreBounds=True)
            self.cursors[sid] = cursor
            ticks = pg.VTickGroup(self._events, yrange=[0.86, 1.0], pen=pg.mkPen(self._event_color, width=2))
            ticks.setZValue(18)
            plot.addItem(ticks, ignoreBounds=True)
            self.event_ticks[sid] = ticks
            if self.show_xyz:
                self.curves[sid] = [plot.plot(pen=pg.mkPen(AXIS_COLORS[c], width=1)) for c in ("x", "y", "z")]
            else:
                self.curves[sid] = [plot.plot(pen=pg.mkPen(color, width=1))]
        self.refresh_static()
        self.refresh_overview()
        self.refresh_markers()
        if first is not None:
            start = self.engine.session.start_time.timestamp()
            end = self.engine.session.end_time.timestamp()
            first.setXRange(start, end, padding=0.01)
        self.refresh_data()

    def refresh_static(self) -> None:
        """Activity bands and sync windows (host clock, no fit needed)."""
        if self.engine is None:
            return
        for sid, plot in self.plots.items():
            for item in self.static_items.get(sid, []):
                plot.removeItem(item)
            items = []
            for i, activity in enumerate(self.engine.session.activities):
                band = pg.LinearRegionItem(
                    (activity.start_time.timestamp(), activity.end_time.timestamp()),
                    movable=False, brush=ACTIVITY_BRUSHES[i % len(ACTIVITY_BRUSHES)],
                    pen=pg.mkPen(None),
                )
                band.setZValue(-20)
                plot.addItem(band, ignoreBounds=True)
                items.append(band)
                if sid == next(iter(self.plots)):
                    name = activity.activity_name or f"Actividad {activity.activity_index}"
                    label = pg.TextItem(name, color="#555", anchor=(0, 0))
                    label.setPos(activity.start_time.timestamp(), 0)
                    label.setZValue(-5)
                    plot.addItem(label, ignoreBounds=True)
                    items.append(label)
            for window in self.engine.session.sync_windows:
                shade = pg.LinearRegionItem(
                    (window.window_start.timestamp(), window.window_end.timestamp()),
                    movable=False, brush=QColor(120, 120, 120, 30), pen=pg.mkPen(None),
                )
                shade.setZValue(-15)
                plot.addItem(shade, ignoreBounds=True)
                items.append(shade)
            self.static_items[sid] = items
        self._place_labels()

    def _place_labels(self) -> None:
        if not self.plots:
            return
        plot = next(iter(self.plots.values()))
        top = plot.getViewBox().viewRange()[1][1]
        for item in self.static_items.get(next(iter(self.plots)), []):
            if isinstance(item, pg.TextItem):
                item.setPos(item.pos().x(), top)

    def refresh_overview(self) -> None:
        if self.engine is None or not self.plots:
            self.overview_curve.setData([], [])
            return
        ref = self.engine.resolved_reference()
        sid = ref if ref in self.plots else next(iter(self.plots))
        signal = self.engine.signals[sid]
        start = self.engine.session.start_time.timestamp() - 30
        end = self.engine.session.end_time.timestamp() + 30
        n0 = float(self.engine.display_to_sensor(sid, start))
        n1 = float(self.engine.display_to_sensor(sid, end))
        x, y = signal.envelope("vm").query(n0, n1, max_points=3000)
        self.overview_curve.setData(self.engine.sensor_to_display(sid, x), y)
        self.overview.setXRange(start, end, padding=0)

    # ------------------------------------------------------------------
    # Data and markers
    # ------------------------------------------------------------------

    def view_range(self) -> tuple[float, float]:
        if not self.plots:
            return (0.0, 1.0)
        x0, x1 = next(iter(self.plots.values())).getViewBox().viewRange()[0]
        return float(x0), float(x1)

    def refresh_data(self) -> None:
        if self.engine is None:
            return
        x0, x1 = self.view_range()
        pad = (x1 - x0) * 0.1
        for sid, curves in self.curves.items():
            signal = self.engine.signals.get(sid)
            if signal is None:
                continue
            width = max(int(self.plots[sid].getViewBox().width()), 400)
            n0 = float(self.engine.display_to_sensor(sid, x0 - pad))
            n1 = float(self.engine.display_to_sensor(sid, x1 + pad))
            channels = ("x", "y", "z") if self.show_xyz else ("vm",)
            for curve, channel in zip(curves, channels):
                x, y = signal.envelope(channel).query(n0, n1, max_points=2 * width)
                curve.setData(self.engine.sensor_to_display(sid, x), y)
        self._place_labels()

    def refresh_markers(self) -> None:
        if self.engine is None:
            return
        for sid, plot in self.plots.items():
            for item in self.markers.get(sid, []):
                plot.removeItem(item)
            items = []
            for window in self.engine.session.sync_windows:
                state = self.engine.anchor_state(sid, window.event_id)
                anchor = self.engine.anchor(sid, window.event_id)
                if anchor is None:
                    continue
                pos = float(self.engine.sensor_to_display(sid, anchor.sensor_timestamp_unix_ms))
                # Only the bottom plot names the claps, below the activity names.
                labelled = sid == list(self.plots)[-1]
                line = pg.InfiniteLine(
                    pos=pos, angle=90, movable=True, pen=ANCHOR_PENS[state],
                    hoverPen=pg.mkPen("#ff6f00", width=3),
                    label=window.event_id if labelled else None,
                    labelOpts={"position": 0.08, "color": "#333", "fill": (255, 255, 255, 200)},
                )
                line.setZValue(15)
                line.sensor_id, line.event_id = sid, window.event_id
                line.sigPositionChangeFinished.connect(self._marker_moved)
                plot.addItem(line, ignoreBounds=True)
                items.append(line)
            self.markers[sid] = items

    def set_events(self, display_times: list[float], color: str = EVENT_COLORS["steps"]) -> None:
        """Events counted on the video (one kind), as short ticks at the top."""
        self._events = list(display_times)
        if color != self._event_color:
            self._event_color = color
            for ticks in self.event_ticks.values():
                ticks.setPen(pg.mkPen(color, width=2))
        for ticks in self.event_ticks.values():
            ticks.setXVals(self._events)

    def set_xyz(self, enabled: bool) -> None:
        self.show_xyz = enabled
        view = self.view_range()
        self.rebuild()
        if self.plots:
            next(iter(self.plots.values())).setXRange(*view, padding=0)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def set_cursor(self, display_s: Optional[float], follow: bool = False) -> None:  # noqa: UP045
        for cursor in self.cursors.values():
            cursor.setVisible(display_s is not None)
            if display_s is not None:
                cursor.setPos(display_s)
        if follow and display_s is not None and self.plots:
            x0, x1 = self.view_range()
            if not (x0 <= display_s <= x1):
                width = x1 - x0
                self.focus(display_s - 0.1 * width, display_s + 0.9 * width)

    def focus(self, t0: float, t1: float) -> None:
        if self.plots:
            next(iter(self.plots.values())).setXRange(t0, t1, padding=0)

    def _range_moved(self, *_args) -> None:
        x0, x1 = self.view_range()
        self._updating_region = True
        self.region.setRegion((x0, x1))
        self._updating_region = False
        self.range_changed.emit(x0, x1)
        self._redraw_timer.start()

    def _region_moved(self) -> None:
        if self._updating_region:
            return
        x0, x1 = self.region.getRegion()
        self.focus(x0, x1)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _plot_at(self, scene_pos: QPointF) -> Optional[tuple[str, float]]:  # noqa: UP045
        for sid, plot in self.plots.items():
            if plot.sceneBoundingRect().contains(scene_pos):
                point = plot.getViewBox().mapSceneToView(scene_pos)
                return sid, float(point.x())
        return None

    def _scene_clicked(self, event) -> None:
        hit = self._plot_at(event.scenePos())
        if hit is None:
            return
        sid, x = hit
        if event.button() == Qt.LeftButton and not event.double():
            self.clicked.emit(x)
        elif event.button() == Qt.RightButton:
            event.accept()
            self._context_menu(sid, x, event.screenPos())

    def _nearest_window(self, x: float):
        if self.engine is None or not self.engine.session.sync_windows:
            return None
        return min(
            self.engine.session.sync_windows,
            key=lambda w: 0 if w.window_start.timestamp() <= x <= w.window_end.timestamp()
            else min(abs(x - w.window_start.timestamp()), abs(x - w.window_end.timestamp())),
        )

    def _context_menu(self, sid: str, x: float, screen_pos) -> None:
        menu = QMenu(self)
        window = self._nearest_window(x)
        if window is not None:
            eid = window.event_id
            state = self.engine.anchor_state(sid, eid)
            kind = "el salto" if window.sync_kind == "jump" else "la palmada"
            menu.addAction(f"Poner aquí {kind} {eid} de {sid}", lambda: self.add_anchor.emit(sid, eid, x))
            if state in ("auto", "manual"):
                menu.addAction(f"Excluir {eid} del ajuste de {sid}", lambda: self.anchor_action.emit(sid, eid, "exclude"))
            if state == "excluded":
                menu.addAction(f"Volver a incluir {eid}", lambda: self.anchor_action.emit(sid, eid, "include"))
            if state == "manual":
                menu.addAction(f"Volver a la detección automática de {eid}", lambda: self.anchor_action.emit(sid, eid, "auto"))
            menu.addSeparator()
            menu.addAction(f"Ir a la ventana {eid}", lambda: self.anchor_action.emit(sid, eid, "focus"))
            menu.addSeparator()
        menu.addAction(f"Ocultar el gráfico de {sid}", lambda: self.plot_visibility.emit(sid, "hide"))
        if len(self.plots) > 1:
            menu.addAction(f"Ver solo el gráfico de {sid}", lambda: self.plot_visibility.emit(sid, "only"))
        menu.exec(screen_pos.toPoint())

    def _marker_moved(self, line) -> None:
        self.anchor_moved.emit(line.sensor_id, line.event_id, float(line.value()))
