"""The REDCap connection of Studio: the API address and the technician's token."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from studio import redcap

TOKEN_PATTERN = re.compile(r"^[0-9A-Fa-f]{32}$")
URL_KEY = "redcap_url"
TOKEN_KEY = "redcap_token"


@contextmanager
def busy():
    """Wait cursor while REDCap answers."""
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def saved_token(settings: QSettings) -> str:
    return str(settings.value(TOKEN_KEY, "") or "").strip()


def saved_url(settings: QSettings) -> str:
    return redcap.api_url(str(settings.value(URL_KEY, "") or ""))


class RedcapSettingsDialog(QDialog):
    """
    «Conexión con REDCap»: the API address and the token, kept for this
    Windows user. REDCap is optional: without a token nothing is sent and the
    counts stay in the exported files and in redcap_conteos.csv.
    """

    def __init__(self, settings: QSettings, csv_path: Optional[str | Path] = None, parent=None):  # noqa: UP045
        super().__init__(parent)
        self.settings = settings
        self.removed = False
        self.setWindowTitle("Conexión con REDCap")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        where = (f"en <b>{Path(csv_path).name}</b>, en tu carpeta de exportación," if csv_path is not None
                 else "en redcap_conteos.csv, en tu carpeta de exportación,")
        intro = QLabel(
            "Opcional. Con tu token de la API de REDCap (en el proyecto, <i>API → Request API token</i>, con "
            "permiso de importación), «Enviar a REDCap» manda los conteos a ese proyecto. Sin token no se envía "
            f"nada: los conteos quedan {where} y se pueden subir con <i>Data Import Tool</i>."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.current = QLabel(self._current_text())
        self.current.setWordWrap(True)
        self.current.setStyleSheet("color:#555;")
        layout.addWidget(self.current)

        form = QFormLayout()
        self.url_edit = QLineEdit(saved_url(settings))
        self.url_edit.setPlaceholderText(redcap.DEFAULT_API_URL)
        self.url_edit.textChanged.connect(self._update)
        form.addRow("Dirección de la API", self.url_edit)
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText(
            "Vacío: se mantiene el actual" if self._token_in_use() else "32 letras y números")
        self.token_edit.textChanged.connect(self._update)
        form.addRow("Token de la API", self.token_edit)
        layout.addLayout(form)
        self.show_check = QCheckBox("Mostrar el token")
        self.show_check.toggled.connect(
            lambda on: self.token_edit.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        layout.addWidget(self.show_check)
        self.remember_check = QCheckBox("Recordarlo en este ordenador (solo para tu usuario de Windows)")
        self.remember_check.setChecked(True)
        layout.addWidget(self.remember_check)

        row = QHBoxLayout()
        self.check_button = QPushButton("Probar conexión")
        self.check_button.clicked.connect(self.check)
        row.addWidget(self.check_button)
        row.addStretch()
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Guardar")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Cancelar")
        self.forget_button = None
        if saved_token(settings):
            self.forget_button = self.buttons.addButton("Quitar token", QDialogButtonBox.DestructiveRole)
            self.forget_button.clicked.connect(self.forget)
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update()

    # ------------------------------------------------------------------

    def _token_in_use(self) -> str:
        return saved_token(self.settings) or redcap.env_token()

    def _current_text(self) -> str:
        saved = saved_token(self.settings)
        if saved:
            return f"Token guardado para tu usuario de Windows: {redcap.masked(saved)}."
        env = redcap.env_token()
        if env:
            return f"Token de la variable REDCAP_TOKEN: {redcap.masked(env)}. Uno escrito aquí lo sustituye."
        return "Ahora no hay token: no se envía nada a REDCap."

    def typed_token(self) -> str:
        return self.token_edit.text().strip()

    def token(self) -> str:
        """The token the fields stand for: the one typed, or the one in use."""
        return self.typed_token() or self._token_in_use()

    def url(self) -> Optional[str]:  # noqa: UP045
        return redcap.normalize_url(self.url_edit.text())

    def _problem(self) -> str:
        if self.url() is None:
            return "La dirección tiene que ser una dirección web (https://…/api/)."
        typed = self.typed_token()
        if typed and not TOKEN_PATTERN.match(typed):
            return "Un token de REDCap tiene 32 letras (A-F) y números."
        return ""

    def _update(self, *_args) -> None:
        problem = self._problem()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not problem)
        self.check_button.setEnabled(not problem and bool(self.token()))
        self._show(problem, ok=False)

    def _show(self, text: str, ok: bool = True) -> None:
        self.status.setStyleSheet(f"color:{'#2e7d32' if ok else '#c62828'};")
        self.status.setText(text)

    def check(self) -> Optional[redcap.ProjectInfo]:  # noqa: UP045
        """Asks REDCap for the project of the token and says what it answered."""
        try:
            with busy():
                info = redcap.project_info(self.token(), self.url())
        except redcap.RedcapError as error:
            self._show(str(error), ok=False)
            return None
        title = f"«{info.title}»" if info.title else "el proyecto"
        kind = "con eventos" if info.longitudinal else "sin eventos"
        self._show(f"Conectado con {title} ({kind}).")
        return info

    def save(self) -> None:
        if self._problem():
            return
        self.settings.setValue(URL_KEY, self.url())
        if self.typed_token() and self.remember_check.isChecked():
            self.settings.setValue(TOKEN_KEY, self.typed_token())
        self.accept()

    def forget(self) -> None:
        self.settings.remove(TOKEN_KEY)
        self.removed = True
        self.accept()
