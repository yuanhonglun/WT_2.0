import math

from metabo_gui.spectrum_display import (
    normalize_for_display,
    variant_best_matching_query,
)


class _M:  # minimal AnnotationMatch stand-in
    def __init__(self, name, ref_peaks):
        self.name = name
        self.ref_peaks = ref_peaks


def test_scales_base_peak_to_target():
    peaks = [(100.0, 999.0), (150.0, 152.0), (200.0, 5.0)]
    out = normalize_for_display(peaks, target=100.0)
    assert math.isclose(out[0][1], 100.0)
    assert math.isclose(out[1][1], 152.0 / 999.0 * 100.0, rel_tol=1e-9)
    assert math.isclose(out[2][1], 5.0 / 999.0 * 100.0, rel_tol=1e-9)


def test_does_not_mutate_input():
    peaks = [(100.0, 999.0), (150.0, 152.0)]
    snapshot = [tuple(p) for p in peaks]
    normalize_for_display(peaks, target=100.0)
    assert peaks == snapshot  # 入参逐位不变


def test_preserves_mz_and_order():
    peaks = [(100.0, 10.0), (150.0, 999.0)]
    out = normalize_for_display(peaks, target=100.0)
    assert [round(m, 6) for m, _ in out] == [100.0, 150.0]  # m/z 与顺序不变


def test_empty_and_zero_safe():
    assert normalize_for_display([], target=100.0) == []
    assert normalize_for_display([(100.0, 0.0)], target=100.0) == [(100.0, 0.0)]


def test_no_binning():
    peaks = [(100.00, 999.0), (100.01, 500.0)]  # 相邻 m/z 不合并
    out = normalize_for_display(peaks, target=100.0)
    assert len(out) == 2


def test_variant_picks_same_name_best_cosine():
    query = [(91.05, 100.0), (119.05, 60.0)]
    matches = [
        _M("Coniferaldehyde", [(147.04, 100.0), (161.06, 99.0)]),  # top-score, 谱形不贴
        _M("Other", [(91.05, 100.0)]),                              # 同谱形但异名，忽略
        _M("Coniferaldehyde", [(91.05, 100.0), (119.05, 61.0)]),   # 同名且贴合 -> 选它
    ]
    assert variant_best_matching_query(query, matches, "Coniferaldehyde") == 2


def test_variant_no_same_name_returns_zero():
    query = [(91.05, 100.0)]
    matches = [_M("A", [(50.0, 100.0)]), _M("B", [(91.05, 100.0)])]
    assert variant_best_matching_query(query, matches, "A") == 0


def test_variant_empty_matches_returns_zero():
    assert variant_best_matching_query([(91.05, 100.0)], [], "X") == 0
