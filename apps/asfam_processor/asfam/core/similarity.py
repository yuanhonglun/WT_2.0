"""Compatibility shim: re-export shared similarity helpers from metabo_core."""
from metabo_core.algorithms.similarity import (  # noqa: F401
    ms2_isotope_step_score,
    greedy_match,
    modified_cosine,
    neutral_loss_cosine,
    cosine_similarity,
    weighted_dot_product,
    reverse_dot_product,
    simple_dot_product,
    composite_similarity,
    eic_pearson_correlation,
    eic_pearson_in_range,
)
