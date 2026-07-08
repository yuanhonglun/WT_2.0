"""Friendly stage labels for cross-app progress reporting.

The Qt progress panel in every app shows ``"[stage] current/total - msg"``
in the log. ``stage`` is the internal pipeline name (e.g. ``"load"``,
``"features"``), but ``msg`` is what the user reads. Keeping the ``msg``
phrasing in a single shared table guarantees that the same logical
operation reads the same way across ASFAM / DDA / GC-MS.

Apps wrap each stage call with::

    label = stage_label("dda", stage_name, "start")
    cb(stage_name, 0, total, label)
    ...
    label = stage_label("dda", stage_name, "done", elapsed=t)
    cb(stage_name, total, total, label)

``stage_message`` remains the low-level builder; ``stage_label`` is the
thin convenience wrapper that pulls the correct app-specific overrides
from :func:`get_stage_titles`.
"""
from __future__ import annotations

from typing import Literal, Optional

# Common, cross-app stage labels — sentence-case, active voice. Apps
# that have additional stages (DDA isotope_adduct, GC-MS deconvolve)
# extend this via their own dicts.
COMMON_STAGE_TITLES: dict[str, str] = {
    "load": "Loading mzML files",
    "features": "Detecting MS1 features",
    "annotate": "Library annotation",
    "align": "Cross-replicate alignment",
    "export": "Writing results",
}

# ASFAM 专属阶段文案。ASFAM 流水线在 stage0/1/.../8 上对每个阶段都有
# 独立的内部名，这里将每个 ASFAM 内部 stage key 映射到面向用户的描述。
# 这些文案与 COMMON_STAGE_TITLES 风格保持一致：sentence-case、active voice。
ASFAM_STAGE_TITLES: dict[str, str] = {
    "stage0": "Loading mzML files",
    "stage1": "Detecting MS2-driven features",
    "stage1b": "Detecting MS1-driven features",
    "stage2": "Assigning MS1 signal to features",
    "stage2b": "Inferring features from MS2 spectra",
    "stage3": "Merging segments across acquisitions",
    "stage4": "Deduplicating isotope peaks",
    "stage5": "Deduplicating adduct peaks",
    "stage5b": "Flagging duplicate features",
    "stage6": "Detecting in-source fragments",
    "stage6b": "Library annotation",
    "stage7": "Cross-replicate alignment",
    "stage8": "Writing results",
}

# DDA 专属阶段文案。DDA 流水线（load → features → isotope_adduct →
# ms2_assoc → annotate → align → export）大多数 stage 可以直接借用
# COMMON_STAGE_TITLES；这里只补充 DDA 独有的两步。
DDA_STAGE_TITLES: dict[str, str] = {
    "isotope_adduct": "Grouping isotope and adduct peaks",
    "ms2_assoc": "Associating MS2 spectra to features",
}

# GC-MS 专属阶段文案。GC-MS 流水线（load → features → filter_quality →
# deconvolve → dedup_components → compute_ri → annotate → align →
# export）在 features / annotate / align / export 上可以复用公共文案，
# 这里补全其余几个 GC-MS 特有的阶段。
GCMS_STAGE_TITLES: dict[str, str] = {
    "filter_quality": "Filtering low-quality features",
    "deconvolve": "Deconvolving co-eluting components",
    "dedup_components": "Deduplicating reconstructed components",
    "compute_ri": "Computing retention indices",
}


AppName = Literal["asfam", "dda", "gcms"]

_APP_TITLES: dict[str, dict[str, str]] = {
    "asfam": ASFAM_STAGE_TITLES,
    "dda": DDA_STAGE_TITLES,
    "gcms": GCMS_STAGE_TITLES,
}


def get_stage_titles(app_name: AppName) -> dict[str, str]:
    """返回某个 app 可见的全部 stage 文案。

    将 :data:`COMMON_STAGE_TITLES` 与 ``app_name`` 对应的 app 专属字典
    合并后返回一个新的 ``dict``。app 专属 key 会覆盖同名的公共 key
    （便于个别 app 想自定义公共阶段文案时使用）。

    Parameters
    ----------
    app_name : ``"asfam"`` | ``"dda"`` | ``"gcms"``
        app 标识。未注册的 app 会抛出 ``KeyError``。
    """
    if app_name not in _APP_TITLES:
        raise KeyError(f"unknown app_name: {app_name!r}")
    merged: dict[str, str] = dict(COMMON_STAGE_TITLES)
    merged.update(_APP_TITLES[app_name])
    return merged


def stage_message(
    stage: str,
    phase: str,
    extra_titles: Optional[dict[str, str]] = None,
    elapsed: Optional[float] = None,
    detail: Optional[str] = None,
) -> str:
    """Build a human-readable progress message for a stage.

    Parameters
    ----------
    stage : str
        Internal stage key (e.g. ``"load"``).
    phase : ``"start"`` | ``"done"``
        Whether this is the start or finish marker.
    extra_titles : optional dict
        App-specific overrides / additions to ``COMMON_STAGE_TITLES``.
    elapsed : optional float
        Elapsed wall time in seconds; only shown on ``phase="done"``.
    detail : optional str
        Trailing detail appended after the activity ("done in 1.9s · 1430 features").
    """
    title = stage
    if stage in COMMON_STAGE_TITLES:
        title = COMMON_STAGE_TITLES[stage]
    if extra_titles and stage in extra_titles:
        title = extra_titles[stage]

    if phase == "start":
        msg = f"{title}…"
    else:  # "done"
        msg = f"{title} done"
        if elapsed is not None:
            msg += f" in {elapsed:.1f}s"
    if detail:
        msg += f" · {detail}"
    return msg


def stage_label(
    app_name: AppName,
    stage: str,
    phase: str,
    elapsed: Optional[float] = None,
    detail: Optional[str] = None,
) -> str:
    """生成某个 app 的某个 stage 的进度文案。

    这是 :func:`stage_message` 的便捷封装：内部会用
    :func:`get_stage_titles` 取到该 app 的完整文案字典，再交给
    ``stage_message`` 渲染。orchestrator 只需要写
    ``stage_label("dda", "load", "start")``，无需自己维护 extra_titles。

    Parameters
    ----------
    app_name : ``"asfam"`` | ``"dda"`` | ``"gcms"``
        app 标识。
    stage : str
        内部 stage key（如 ``"stage0"``、``"load"``）。
    phase : ``"start"`` | ``"done"``
        阶段起始 / 结束标记。
    elapsed : optional float
        ``phase="done"`` 时显示的耗时（秒）。
    detail : optional str
        在文案末尾追加的尾部信息。
    """
    return stage_message(
        stage,
        phase,
        extra_titles=get_stage_titles(app_name),
        elapsed=elapsed,
        detail=detail,
    )
