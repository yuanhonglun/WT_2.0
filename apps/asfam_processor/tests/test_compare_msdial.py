import sys
from pathlib import Path
# 本测试在 apps/asfam_processor/tests/ (depth 3): parents[3] = repo root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))               # scripts/ 不在 pythonpath, 手动加 root
from scripts.compare_asfam_msdial import (
    match_features, report, build_segment,
)


def test_match_by_mz_and_rt():
    msdial = [{"mz": 285.05, "rt": 0.61}, {"mz": 300.02, "rt": 0.62}]
    metra  = [{"mz": 285.052, "rt": 0.60}]  # 命中第 1 个
    matched, msdial_only, metra_only = match_features(
        metra, msdial, mz_tol=0.02, rt_tol=0.1)
    assert len(matched) == 1
    assert len(msdial_only) == 1      # 300.02 未被覆盖
    assert len(metra_only) == 0


def test_match_is_one_to_one_two_metra_one_msdial():
    """贪心一一匹配: 两个 METRA 落在同一个 MS-DIAL 容差内 -> 只有一个命中,
    另一个进 metra_only; MS-DIAL 不被重复占用。"""
    msdial = [{"mz": 100.0, "rt": 1.0}]
    metra = [
        {"mz": 100.001, "rt": 1.0, "height": 10.0},
        {"mz": 100.002, "rt": 1.0, "height": 999.0},  # 强度更高, 先抢到
    ]
    matched, msdial_only, metra_only = match_features(
        metra, msdial, mz_tol=0.01, rt_tol=0.1)
    assert len(matched) == 1
    assert len(msdial_only) == 0
    assert len(metra_only) == 1
    # 强度更高者 (height=999) 抢到匹配。
    assert matched[0][0]["height"] == 999.0


def test_match_nearest_within_tolerance_wins():
    """单个 METRA 在容差内有两个候选 -> 取组合距离最近的那个。"""
    metra = [{"mz": 200.000, "rt": 5.00, "height": 1.0}]
    msdial = [
        {"mz": 200.008, "rt": 5.00},   # dmz 大
        {"mz": 200.001, "rt": 5.00},   # 最近
    ]
    matched, msdial_only, metra_only = match_features(
        metra, msdial, mz_tol=0.01, rt_tol=0.1)
    assert len(matched) == 1
    assert matched[0][1]["mz"] == 200.001
    assert len(msdial_only) == 1


def test_empty_inputs():
    assert match_features([], [], 0.01, 0.1) == ([], [], [])
    matched, msdial_only, metra_only = match_features(
        [], [{"mz": 100.0, "rt": 1.0}], 0.01, 0.1)
    assert matched == [] and len(msdial_only) == 1 and metra_only == []
    matched, msdial_only, metra_only = match_features(
        [{"mz": 100.0, "rt": 1.0}], [], 0.01, 0.1)
    assert matched == [] and msdial_only == [] and len(metra_only) == 1


def test_coverage_recall_excludes_isotope_copies():
    """覆盖召回只数 MS-DIAL 单同位素 (isotope==0); 命中同位素拷贝不计入。"""
    msdial = [
        {"mz": 100.0, "rt": 1.0, "isotope": 0},   # mono
        {"mz": 101.0, "rt": 1.0, "isotope": 1},   # M+1 copy
    ]
    metra = [
        {"mz": 100.0, "rt": 1.0, "height": 5.0, "isotope": 0},
        {"mz": 101.0, "rt": 1.0, "height": 4.0, "isotope": 1},
    ]
    seg = build_segment("seg", metra, msdial, mz_tol=0.01, rt_tol=0.1)
    rep = report(seg)
    assert rep["matched"] == 2            # both pairs matched
    assert rep["msdial_mono"] == 1        # only the mono counts as denominator
    assert rep["matched_msdial_mono"] == 1
    assert rep["coverage_recall"] == 1.0  # 1/1, isotope copy not double-counted


def test_mae_none_when_no_annotations():
    """无注释时 (features 全空 name) MAE 返回 None, 计数为 0, 不崩。"""
    msdial = [{"mz": 100.0, "rt": 1.0, "isotope": 0, "name": "", "annotated": False,
               "total_score": None}]
    metra = [{"mz": 100.0, "rt": 1.0, "isotope": 0, "height": 1.0, "name": "",
              "annotated": False, "total_score": None}]
    seg = build_segment("seg", metra, msdial, mz_tol=0.01, rt_tol=0.1)
    rep = report(seg)
    assert rep["mae"] is None
    assert rep["mae_n"] == 0


def test_mae_same_compound_only():
    """MAE 只在 m/z+RT 命中且两边对同一化合物有可信注释的 pair 上计算。"""
    msdial = [{"mz": 100.0, "rt": 1.0, "isotope": 0, "name": "Glucose",
               "annotated": True, "total_score": 0.90, "inchikey": ""}]
    metra = [{"mz": 100.0, "rt": 1.0, "isotope": 0, "height": 1.0, "name": "glucose",
              "annotated": True, "total_score": 0.80, "inchikey": ""}]
    seg = build_segment("seg", metra, msdial, mz_tol=0.01, rt_tol=0.1)
    rep = report(seg)
    assert rep["mae_n"] == 1
    assert abs(rep["mae"] - 0.10) < 1e-9


def test_net_add_counts_ms2_driven():
    """净增只数 detection_source 为 MS2-驱动 (ms2_driven/both) 的 metra_only。"""
    msdial = []   # nothing on MS-DIAL side -> all metra are metra_only
    metra = [
        {"mz": 100.0, "rt": 1.0, "height": 3.0, "detection_source": "ms2_driven"},
        {"mz": 101.0, "rt": 1.0, "height": 2.0, "detection_source": "both"},
        {"mz": 102.0, "rt": 1.0, "height": 1.0, "detection_source": "ms1_driven"},
    ]
    seg = build_segment("seg", metra, msdial, mz_tol=0.01, rt_tol=0.1)
    rep = report(seg)
    assert rep["net_add"] == 2            # ms2_driven + both
    assert rep["net_add_degraded"] is False


def test_net_add_degrades_without_detection_source():
    """老 CSV 无 detection_source 列 -> 净增退化为数全部 metra_only 并打标。"""
    metra = [
        {"mz": 100.0, "rt": 1.0, "height": 3.0},   # no detection_source key
        {"mz": 101.0, "rt": 1.0, "height": 2.0},
    ]
    seg = build_segment("seg", metra, [], mz_tol=0.01, rt_tol=0.1)
    rep = report(seg)
    assert rep["net_add"] == 2
    assert rep["net_add_degraded"] is True
