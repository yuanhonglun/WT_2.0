"""Internationalization support for ASFAMProcessor GUI."""
from __future__ import annotations

import json
import os

# Persistent settings file
_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".asfam_settings.json")

# Current language (module-level state)
_current_lang = "en"


def _load_saved_lang() -> str:
    """Load saved language preference from disk."""
    try:
        if os.path.exists(_SETTINGS_PATH):
            with open(_SETTINGS_PATH, "r") as f:
                data = json.load(f)
                return data.get("language", "en")
    except Exception:
        pass
    return "en"


def _save_lang(lang: str):
    """Persist language preference to disk."""
    try:
        data = {}
        if os.path.exists(_SETTINGS_PATH):
            with open(_SETTINGS_PATH, "r") as f:
                data = json.load(f)
        data["language"] = lang
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# Initialize from saved preference
_current_lang = _load_saved_lang()

# Translation dictionary: key -> {lang: text}
_TRANSLATIONS = {
    # Toolbar
    "Run Pipeline": {"zh": "运行流水线"},
    "Stop": {"zh": "停止"},
    "Re-annotate": {"zh": "重新定性"},
    "Save Project": {"zh": "保存项目"},
    "Open Project": {"zh": "打开项目"},
    "Export": {"zh": "导出"},
    " View: ": {"zh": " 视图: "},
    "Aligned (all samples)": {"zh": "对齐 (所有样品)"},
    "Aligned (all replicates)": {"zh": "对齐 (所有重复)"},

    # Setup panel
    "mzML Data Files": {"zh": "mzML 数据文件"},
    "Add Files": {"zh": "添加文件"},
    "Remove": {"zh": "移除"},
    "Edit Samples": {"zh": "编辑样品"},
    "Spectral Library": {"zh": "谱库"},
    "Browse": {"zh": "浏览"},
    "Output Directory": {"zh": "输出目录"},
    "General": {"zh": "常规"},
    "Detection": {"zh": "检测"},
    "Dedup": {"zh": "去重"},
    "Alignment": {"zh": "对齐"},
    "Ion Mode:": {"zh": "离子模式:"},
    "Workers:": {"zh": "线程数:"},
    "Peak Height Min:": {"zh": "最低峰高:"},
    "S/N Threshold:": {"zh": "信噪比阈值:"},
    "Peak Width Min:": {"zh": "最小峰宽:"},
    "EIC m/z Tol (Da):": {"zh": "EIC m/z 容差 (Da):"},
    "RT Cluster Tol (min):": {"zh": "RT 聚类容差 (min):"},
    "Min Fragments:": {"zh": "最少碎片:"},
    "Gaussian Sim Thr:": {"zh": "高斯相似度阈值:"},
    "MS1 Height Min:": {"zh": "MS1 最低峰高:"},
    "Infer Min Frags:": {"zh": "推断最少碎片:"},
    "Library Match Thr:": {"zh": "库匹配阈值:"},
    "MS-Buddy:": {"zh": "MS-Buddy:"},
    "Enable MS-Buddy formula prediction": {"zh": "启用 MS-Buddy 分子式预测"},
    "Save Config": {"zh": "保存配置"},
    "Load Config": {"zh": "加载配置"},

    # Feature table
    "Search:": {"zh": "搜索:"},
    "Filter features...": {"zh": "筛选特征..."},

    # Scatter plot
    "Show:": {"zh": "显示:"},
    "All": {"zh": "全部"},
    "MS1 only": {"zh": "仅 MS1"},
    "MS2 only": {"zh": "仅 MS2"},
    "Feature Overview": {"zh": "特征概览"},
    "No features loaded": {"zh": "未加载特征"},

    # EIC plot
    "Smoothed": {"zh": "平滑"},
    "Raw": {"zh": "原始"},
    "No raw data loaded": {"zh": "未加载原始数据"},
    "No MS1 signal": {"zh": "无 MS1 信号"},
    "Select a feature": {"zh": "选择一个特征"},

    # MS2 plot
    "Match:": {"zh": "匹配:"},
    "No library matches": {"zh": "无库匹配"},
    "No MS2 spectrum": {"zh": "无 MS2 谱"},

    # Progress
    "Ready": {"zh": "就绪"},

    # About
    "About": {"zh": "关于"},

    # Dialogs
    "No Files": {"zh": "无文件"},
    "Please add mzML files first.": {"zh": "请先添加 mzML 文件。"},
    "No Output": {"zh": "无输出"},
    "Please select an output directory.": {"zh": "请选择输出目录。"},
    "No Data": {"zh": "无数据"},

    # Sample edit
    "New Sample": {"zh": "新样品"},
    "Delete Sample": {"zh": "删除样品"},
    "Unassigned": {"zh": "未分配"},
}


def set_language(lang: str):
    """Set the current language ('en' or 'zh') and persist."""
    global _current_lang
    _current_lang = lang
    _save_lang(lang)


def get_language() -> str:
    return _current_lang


def tr(text: str) -> str:
    """Translate text to the current language. Returns original if no translation."""
    if _current_lang == "en":
        return text
    entry = _TRANSLATIONS.get(text)
    if entry and _current_lang in entry:
        return entry[_current_lang]
    return text
