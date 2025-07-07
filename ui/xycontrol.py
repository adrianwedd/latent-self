from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtWidgets import QWidget

class XYControl(QWidget):
    """Simple 2D control emitting normalized coordinates."""

    moved = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.setMouseTracking(True)
        self._pos = QPointF(self.width() / 2, self.height() / 2)

    def _handle(self, event) -> None:
        self._pos = event.position()
        x = (self._pos.x() / self.width()) * 2 - 1
        y = ((self.height() - self._pos.y()) / self.height()) * 2 - 1
        self.moved.emit(float(x), float(y))
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: D401 - Qt override
        self._handle(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: D401 - Qt override
        self._handle(event)

    def paintEvent(self, event) -> None:  # noqa: D401 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.lightGray)
        painter.setPen(QPen(Qt.GlobalColor.black))
        painter.drawLine(self.width() / 2, 0, self.width() / 2, self.height())
        painter.drawLine(0, self.height() / 2, self.width(), self.height() / 2)
        painter.setBrush(QColor(200, 0, 0))
        painter.drawEllipse(self._pos, 5, 5)
        painter.end()
