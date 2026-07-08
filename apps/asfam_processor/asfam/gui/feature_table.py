"""Feature table view with sorting and filtering."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTableView, QLineEdit, QLabel, QHeaderView, QAbstractItemView,
)
from PyQt5.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, pyqtSignal,
)
from PyQt5.QtGui import QColor

from asfam.models import Feature


COLUMNS = [
    ("ID", "feature_id"),
    ("Status", "_feedback_status"),
    ("m/z", "precursor_mz"),
    ("RT (min)", "rt"),
    ("Type", "signal_type"),
    ("MS1 m/z Source", "mz_source"),
    ("Detection Mode", "detection_source"),
    ("Height", "mean_height"),
    ("Fragments", "n_fragments"),
    ("CV", "cv"),
    ("S/N", "sn_ratio"),
    ("Formula", "formula"),
    ("Adduct", "adduct"),
    ("Name", "name"),
    # Plan F-followup-5 mirror: surface the selected match's score so
    # users can see how confident the annotation is. Always shown
    # regardless of threshold; the "Annotated" row filter is the gate.
    ("Score", "_score"),
    ("WDP", "_wdp"),
    ("RDP", "_rdp"),
]

HIGH_COLOR = QColor(255, 255, 255)       # white (unused, kept for reference)
LOW_COLOR = QColor(232, 240, 248)         # light blue (theme)
LOW_TEXT_COLOR = QColor(45, 106, 159)     # theme blue for low-response text


class FeatureTableModel(QAbstractTableModel):
    """Model for the feature table."""

    # Index of the "Status" column (delegate-rendered; model returns "").
    STATUS_COLUMN_INDEX: int = next(
        i for i, (_, attr) in enumerate(COLUMNS) if attr == "_feedback_status"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._features: list[Feature] = []

    def set_features(self, features: list[Feature]):
        self.beginResetModel()
        self._features = features
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._features)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        feat = self._features[index.row()]
        col_name, col_attr = COLUMNS[index.column()]

        if role == Qt.DisplayRole:
            if col_attr == "_feedback_status":
                return ""  # rendered entirely by FeatureStatusDelegate
            if col_attr in ("_score", "_wdp", "_rdp"):
                sel = getattr(feat, "selected_annotation", None)
                if sel is None:
                    return ""
                attr = {"_score": "score", "_wdp": "wdp", "_rdp": "rdp"}[col_attr]
                try:
                    return f"{float(getattr(sel, attr, 0.0)):.3f}"
                except (TypeError, ValueError):
                    return ""
            val = getattr(feat, col_attr, None)
            if val is None:
                return ""
            # Show duplicate type in Type column for duplicate features
            if col_attr == "signal_type" and feat.is_duplicate and feat.duplicate_type:
                return feat.duplicate_type
            # Special display for mz_source: append confidence for nl_consensus
            if col_attr == "mz_source" and val == "nl_consensus":
                conf = getattr(feat, "mz_confidence", "")
                return f"nl_consensus ({conf})" if conf else "nl_consensus"
            if isinstance(val, float):
                if col_attr == "precursor_mz":
                    return f"{val:.5f}"
                elif col_attr in ("rt",):
                    return f"{val:.3f}"
                elif col_attr in ("cv",):
                    return f"{val:.3f}"
                elif col_attr in ("mean_height", "sn_ratio"):
                    ion_mz = getattr(feat, "height_ion_mz", None)
                    suffix = f" (ion {ion_mz:.3f})" if ion_mz is not None else ""
                    return f"{val:.1f}{suffix}"
                return f"{val:.4f}"
            return str(val)

        if role == Qt.BackgroundRole:
            # Don't return BackgroundRole — let stylesheet handle selection
            return None

        if role == Qt.ForegroundRole:
            # Use blue text for low-response to distinguish visually
            if feat.signal_type == "ms2_only":
                return LOW_TEXT_COLOR
            return None

        if role == Qt.TextAlignmentRole:
            if col_attr in ("precursor_mz", "rt", "mean_height", "n_fragments",
                            "cv", "sn_ratio", "_score", "_wdp", "_rdp"):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter

        if role == Qt.UserRole:
            # Return sortable value
            if col_attr == "_feedback_status":
                return ""
            if col_attr in ("_score", "_wdp", "_rdp"):
                sel = getattr(feat, "selected_annotation", None)
                if sel is None:
                    return 0.0
                attr = {"_score": "score", "_wdp": "wdp", "_rdp": "rdp"}[col_attr]
                try:
                    return float(getattr(sel, attr, 0.0))
                except (TypeError, ValueError):
                    return 0.0
            val = getattr(feat, col_attr, None)
            if val is None:
                return 0
            return val

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COLUMNS[section][0]
        return None

    def get_feature(self, row: int) -> Optional[Feature]:
        if 0 <= row < len(self._features):
            return self._features[row]
        return None

    def feature_id_at_row(self, row: int) -> "str | None":
        """Return the feature_id at *row* (source index), or None if out of range.

        Used by FeatureStatusDelegate as its feature_id_provider.
        """
        feat = self.get_feature(row)
        return feat.feature_id if feat is not None else None


class FeatureSortProxy(QSortFilterProxyModel):
    """Sort/filter proxy that filters by text search."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_text = ""
        self._show_duplicates = False
        self._annotated_only = False
        self._annotated_threshold = 0.8
        # High-confidence also requires >= this many matched peaks (set from
        # config by main_window). 0 = no count constraint.
        self._annotated_min_matched = 0

    def set_filter_text(self, text: str):
        self._filter_text = text.lower()
        self.invalidateFilter()

    def set_show_duplicates(self, show: bool):
        self._show_duplicates = show
        self.invalidateFilter()

    def set_annotated_only(self, on: bool) -> None:
        self._annotated_only = bool(on)
        self.invalidateFilter()

    def set_annotated_threshold(self, thr: float) -> None:
        self._annotated_threshold = float(thr)
        self.invalidateFilter()

    def set_annotated_min_matched(self, n: int) -> None:
        self._annotated_min_matched = int(n)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        feature = (model._features[source_row]
                   if source_row < len(model._features) else None)
        # Duplicate filter
        if not self._show_duplicates:
            if feature and feature.is_duplicate:
                return False
        # Plan F-followup-5 mirror: row filter for "Annotated only".
        # A feature passes iff its top match's score reaches the
        # configured display threshold.
        if self._annotated_only:
            if feature is None:
                return False
            sel = getattr(feature, "selected_annotation", None)
            if sel is None:
                return False
            try:
                if float(sel.score) < self._annotated_threshold:
                    return False
            except (TypeError, ValueError):
                return False
            if int(getattr(sel, "n_matched", 0) or 0) < self._annotated_min_matched:
                return False
        # Text filter
        if not self._filter_text:
            return True
        for col in range(model.columnCount()):
            idx = model.index(source_row, col, source_parent)
            data = model.data(idx, Qt.DisplayRole)
            if data and self._filter_text in str(data).lower():
                return True
        return False

    def lessThan(self, left, right):
        left_val = self.sourceModel().data(left, Qt.UserRole)
        right_val = self.sourceModel().data(right, Qt.UserRole)
        try:
            return float(left_val) < float(right_val)
        except (TypeError, ValueError):
            return str(left_val) < str(right_val)


class FeatureTableWidget(QWidget):
    """Feature table with search bar."""

    featureSelected = pyqtSignal(int)  # emits row index in source model

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter features...")
        self.search_input.textChanged.connect(self._on_search)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # Table
        self.model = FeatureTableModel()
        self.proxy = FeatureSortProxy()
        self.proxy.setSourceModel(self.model)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.selectionModel().selectionChanged.connect(self._on_selection)
        layout.addWidget(self.table)

    def set_features(self, features: list[Feature]):
        self.model.set_features(features)
        # Auto-resize columns
        for i in range(len(COLUMNS)):
            self.table.resizeColumnToContents(i)

    def get_selected_feature(self) -> Optional[Feature]:
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return None
        proxy_idx = indexes[0]
        source_idx = self.proxy.mapToSource(proxy_idx)
        return self.model.get_feature(source_idx.row())

    def select_feature_by_id(self, feature_id: str):
        """Select a row matching the given feature_id."""
        for row in range(self.model.rowCount()):
            feat = self.model.get_feature(row)
            if feat and feat.feature_id == feature_id:
                source_idx = self.model.index(row, 0)
                proxy_idx = self.proxy.mapFromSource(source_idx)
                self.table.selectRow(proxy_idx.row())
                self.table.scrollTo(proxy_idx)
                return

    def _on_search(self, text):
        self.proxy.set_filter_text(text)

    def _on_selection(self, selected, deselected):
        feat = self.get_selected_feature()
        if feat:
            # Find source row index
            indexes = self.table.selectionModel().selectedRows()
            if indexes:
                source_idx = self.proxy.mapToSource(indexes[0])
                self.featureSelected.emit(source_idx.row())
