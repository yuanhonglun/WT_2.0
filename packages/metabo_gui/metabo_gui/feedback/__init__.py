"""Feedback subsystem: annotation, tracking, and inspection of feature quality."""

from .controller import FeedbackController
from .models import FeedbackEntry, FeedbackStore, FeatureSignature, RunContext
from .status_column import FeatureStatusDelegate
from .widget import NotesDock
from .matcher import MatchResult, match_entries_to_features
from .run_context import build_run_context, feature_signature_from_components
from .exporter import dump_feedback_to_export_dir
from .store import load_alongside, save_alongside, sidecar_path_for
from .tags import ISSUE_TAGS, TAG_LABELS, VERIFIED_GOOD_TAG

__all__ = [
    # Core classes
    "FeedbackController",
    "FeedbackEntry",
    "FeedbackStore",
    "FeatureSignature",
    "RunContext",
    # UI
    "FeatureStatusDelegate",
    "NotesDock",
    # Matching and utility
    "MatchResult",
    "match_entries_to_features",
    # Context and export
    "build_run_context",
    "feature_signature_from_components",
    "dump_feedback_to_export_dir",
    # Storage
    "load_alongside",
    "save_alongside",
    "sidecar_path_for",
    # Constants
    "ISSUE_TAGS",
    "TAG_LABELS",
    "VERIFIED_GOOD_TAG",
]
