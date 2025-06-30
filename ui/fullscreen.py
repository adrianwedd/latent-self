"""ui/fullscreen.py - Fullscreen Qt window for the mirror display."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QMainWindow

from .admin import AdminDialog
from directions import Direction

class MirrorWindow(QMainWindow):
    """A fullscreen, borderless window with a QLabel to display video frames."""

    def __init__(self, app) -> None:
        """Create the window bound to ``app``."""
        super().__init__()
        self.app = app
        self.setWindowTitle("Latent Self")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self.video_widget = QLabel(self)
        self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.video_widget)

    def update_frame(self, q_image: QImage) -> None:
        """Update the video widget with a new QImage."""
        pixmap = QPixmap.fromImage(q_image)
        self.video_widget.setPixmap(pixmap)

    def keyPressEvent(self, event) -> None:
        """Handle key presses for closing the application or opening the admin panel."""
        if event.key() == Qt.Key.Key_F12:
            dialog = AdminDialog(self)
            dialog.exec()
        elif event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif event.text():
            direction = Direction.from_key(event.text())
            if direction:
                self.app.video.enqueue_direction(direction)

    def show_fullscreen(self) -> None:
        """Show the window in fullscreen mode and hide the cursor."""
        self.showFullScreen()
        self.setCursor(Qt.CursorShape.BlankCursor)
