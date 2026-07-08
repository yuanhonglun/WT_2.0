"""Progress panel: progress bar + log output."""
from __future__ import annotations

from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QProgressBar, QLabel, QTextEdit,
)
from PyQt5.QtCore import Qt


class ProgressPanel(QWidget):
    """Progress bar and scrolling log output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # Progress bar row
        bar_layout = QHBoxLayout()
        self.stage_label = QLabel("Ready")
        self.stage_label.setFixedWidth(300)
        bar_layout.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        bar_layout.addWidget(self.progress_bar)
        layout.addLayout(bar_layout)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setFontFamily("Consolas")
        self.log_text.setFontPointSize(9)
        layout.addWidget(self.log_text)

        # Stage progress weights (approximate time proportion)
        self._stage_weights = {
            "stage0": (0, 15), "stage1": (15, 40), "stage2": (40, 50),
            "stage2b": (50, 55), "stage3": (55, 60),
            "stage4": (60, 70), "stage5": (70, 80),
            "stage6": (80, 88), "stage7": (88, 95), "stage8": (95, 100),
        }

    def update_progress(self, stage: str, current: int, total: int, msg: str):
        """Update progress bar and stage label."""
        # Map stage + within-stage progress to overall 0-100%
        if stage in self._stage_weights:
            s_start, s_end = self._stage_weights[stage]
            if total > 0:
                within = current / total
            else:
                within = 0.5
            pct = int(s_start + (s_end - s_start) * within)
            self.progress_bar.setValue(min(pct, 100))
        elif total > 0:
            pct = int(current / total * 100)
            self.progress_bar.setValue(pct)
        self.stage_label.setText(f"{stage}: {msg}")
        self.log(f"[{stage}] {current}/{total} - {msg}")

    def log(self, message: str):
        """Append a timestamped message to the log."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"{ts} {message}")
        # Auto-scroll
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def reset(self):
        """Reset progress bar and log."""
        self.progress_bar.setValue(0)
        self.stage_label.setText("Running...")
        self.log_text.clear()

    def set_complete(self, n_features: int):
        """Mark pipeline as complete."""
        self.progress_bar.setValue(100)
        self.stage_label.setText(f"Complete: {n_features} features")

    def set_error(self, error_msg: str):
        """Show error state."""
        self.stage_label.setText(f"Error: {error_msg[:80]}")
        self.log(f"ERROR: {error_msg}")
