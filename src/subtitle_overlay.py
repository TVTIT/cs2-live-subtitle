"""Cửa sổ phụ đề nổi PyQt6: tự cuộn không tràn mép dưới, always on top, ẩn bằng nút taskbar."""

from __future__ import annotations

import sys
from collections import deque
from typing import Callable

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QResizeEvent, QTextOption
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizeGrip,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _enforce_windows_topmost(widget: QWidget) -> None:
    """Ép cửa sổ luôn nằm trên cùng (HWND_TOPMOST) trên Windows mà không cướp focus."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        ctypes.windll.user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
    except Exception:
        pass


class SubtitleOverlay(QWidget):
    """Cửa sổ phụ đề nổi trong suốt, tự cuộn xuống dòng mới nhất, luôn on-top."""

    subtitle_signal = pyqtSignal(str, bool)
    status_signal = pyqtSignal(str)

    def __init__(
        self,
        font_size: int = 22,
        on_close_callback: Callable[[], None] | None = None,
        window_title: str = "CS2 Live Subtitles",
        initial_text: str = "Sẵn sàng nhận phụ đề tiếng Anh...",
        bottom_offset: int = 48,
    ) -> None:
        super().__init__()
        self._font_size = max(14, min(48, font_size))
        self._on_close_callback = on_close_callback
        self._initial_text = initial_text
        self._bottom_offset = bottom_offset
        self._drag_pos: QPoint | None = None
        self._history: deque[str] = deque(maxlen=6)
        self._partial = ""

        # Cửa sổ không viền, luôn nổi trên cùng, vẫn hiện trên taskbar để ẩn/hiện chủ động
        self.setWindowTitle(window_title)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(920, 140)
        self.setMinimumSize(420, 100)

        self._build_ui()
        self._position_bottom_center()

        self.subtitle_signal.connect(self._update_subtitle)
        self.status_signal.connect(self._update_status)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)

        # Khung nền đen xám, opacity 60% (alpha = 153 / 255)
        self.container = QFrame(self)
        self.container.setObjectName("subtitleContainer")
        self.container.setStyleSheet(
            """
            QFrame#subtitleContainer {
                background-color: rgba(32, 33, 38, 153);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 12px;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 22);
                color: #e2e8f0;
                border: none;
                border-radius: 5px;
                padding: 2px 8px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 55);
            }
            QPushButton#closeBtn:hover {
                background-color: rgba(220, 38, 38, 210);
                color: #ffffff;
            }
            """
        )

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(14, 6, 10, 6)
        container_layout.setSpacing(4)

        # Thanh tiêu đề gọn: trạng thái + tăng/giảm cỡ chữ + nút ẩn xuống taskbar + đóng
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        self.status_label = QLabel("Đang khởi tạo...")
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 11.5px;")
        top_bar.addWidget(self.status_label, stretch=1)

        btn_smaller = QPushButton("A-")
        btn_smaller.setToolTip("Giảm cỡ chữ")
        btn_smaller.clicked.connect(lambda: self._change_font_size(-2))
        top_bar.addWidget(btn_smaller)

        btn_larger = QPushButton("A+")
        btn_larger.setToolTip("Tăng cỡ chữ")
        btn_larger.clicked.connect(lambda: self._change_font_size(2))
        top_bar.addWidget(btn_larger)

        btn_min = QPushButton("_")
        btn_min.setToolTip("Ẩn xuống taskbar")
        btn_min.clicked.connect(self.showMinimized)
        top_bar.addWidget(btn_min)

        btn_close = QPushButton("×")
        btn_close.setObjectName("closeBtn")
        btn_close.setToolTip("Thoát")
        btn_close.clicked.connect(self.close)
        top_bar.addWidget(btn_close)

        container_layout.addLayout(top_bar)

        # Vùng hiển thị phụ đề dùng QTextEdit tự cuộn xuống đáy (không bao giờ tràn khuất mép dưới)
        self.subtitle_view = QTextEdit(self.container)
        self.subtitle_view.setReadOnly(True)
        self.subtitle_view.setFrameShape(QFrame.Shape.NoFrame)
        self.subtitle_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.subtitle_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.subtitle_view.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.subtitle_view.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        # Xuyên chuột để bấm giữ bất kỳ chỗ nào trên chữ vẫn kéo di chuyển cửa sổ được
        self.subtitle_view.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.subtitle_view.viewport().setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.subtitle_view.document().setDocumentMargin(2)
        self._apply_subtitle_font()

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(6)
        shadow.setColor(QColor(0, 0, 0, 220))
        shadow.setOffset(1, 1)
        self.subtitle_view.setGraphicsEffect(shadow)

        self._set_display_text(self._initial_text)
        container_layout.addWidget(self.subtitle_view, stretch=1)

        # Góc kéo đổi kích thước cửa sổ
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.addStretch(1)
        grip = QSizeGrip(self.container)
        grip.setFixedSize(14, 14)
        bottom_row.addWidget(grip, alignment=Qt.AlignmentFlag.AlignRight)
        container_layout.addLayout(bottom_row)

        root_layout.addWidget(self.container)

    def _apply_subtitle_font(self) -> None:
        font = QFont("Sans Serif", self._font_size, QFont.Weight.DemiBold)
        self.subtitle_view.setFont(font)
        self.subtitle_view.setStyleSheet(
            "QTextEdit { color: #f8fafc; background: transparent; border: none; }"
        )
        self._scroll_to_bottom()

    def _change_font_size(self, delta: int) -> None:
        self._font_size = max(14, min(48, self._font_size + delta))
        self._apply_subtitle_font()

    def _set_display_text(self, text: str) -> None:
        self.subtitle_view.setPlainText(text)
        cursor = self.subtitle_view.textCursor()
        cursor.select(cursor.SelectionType.Document)
        fmt = cursor.blockFormat()
        fmt.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        cursor.setBlockFormat(fmt)
        cursor.clearSelection()
        self.subtitle_view.setTextCursor(cursor)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        sb = self.subtitle_view.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _position_bottom_center(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = max(geo.y() + 20, geo.y() + geo.height() - self.height() - self._bottom_offset)
        self.move(x, y)

    def _update_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _update_subtitle(self, text: str, is_final: bool) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if is_final:
            self._history.append(cleaned)
            self._partial = ""
        else:
            self._partial = cleaned

        lines = list(self._history)
        if self._partial:
            lines.append(self._partial)
        self._set_display_text("\n".join(lines))

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        QTimer.singleShot(0, lambda: _enforce_windows_topmost(self))

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        # Khi khôi phục từ taskbar, đảm bảo giữ lại trạng thái always-on-top
        if event.type() == QEvent.Type.WindowStateChange and not self.isMinimized():
            QTimer.singleShot(0, lambda: _enforce_windows_topmost(self))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._scroll_to_bottom()

    # Kéo di chuyển cửa sổ mượt trên cả Linux Wayland lẫn Windows
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._on_close_callback is not None:
            self._on_close_callback()
        super().closeEvent(event)
