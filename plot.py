#!/usr/bin/env python3
"""Plot the retained fixed-cardinality penalty and theorem figures from unified CSVs.

This module contains both the shared plotting helpers and the single plot
orchestration function used by run_experiments.py.
"""
from __future__ import annotations

import csv
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys

csv.field_size_limit(sys.maxsize)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter, LogFormatter, LogFormatterMathtext

# Use SciencePlots for ALL manuscript panels when installed.  'no-latex'
# avoids requiring a LaTeX executable on headless/GPU worker hosts.  Local
# minimal verification environments without SciencePlots retain a clear
# fallback; the distributed requirements install it for publication runs.
try:
    import scienceplots  # noqa: F401  (registers 'science' Matplotlib styles)
except ImportError:
    plt.style.use('default')
else:
    plt.style.use(['science', 'no-latex'])


THEOREM_PENALTY_ORACLE_ETA_RATIOS = [-.9,-.75,-.5,-.25,-.1,0.,.1,.25,.5,1.,2.,4.,8.,16.,32.]
THEOREM_ORDER_ORACLE_ETA_RATIOS = [-1.,-.75,-.5,-.25,-.1,0.,.1,.25,.5,1.,3.]
# The VRP and legacy optional-parameter plotting constants were removed.

_SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*")

# Visual grammar for theorem-validation reference markers.  The measured
# diagnostic curve uses the normal first series color; vertical references use
# distinct cycle colors so exact boundaries and certificates remain visually
# separable.  Any across-instance IQR uses the same color at low opacity.
_THEOREM_REF_COLORS = ("C0", "C1", "C2", "C3", "C4")


def _coerce(value: str) -> Any:
    if value == "":
        return None
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"none", "null"}:
        return None
    try:
        if any(c in value.lower() for c in (".", "e", "nan", "inf")):
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _finite(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        v = row.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            vals.append(f)
    return np.asarray(vals, dtype=float)


def _mean_std(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float]:
    vals = _finite(rows, key)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


def _mean_std_by(rows, x_key, x_values, y_key):
    means, stds = [], []
    for x in x_values:
        rr = [r for r in rows if r.get(x_key) is not None and np.isclose(float(r[x_key]), float(x))]
        m, s = _mean_std(rr, y_key)
        means.append(m); stds.append(s)
    return means, stds


def _threshold_distribution(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float, float]:
    vals_by_instance = {}
    for r in rows:
        try:
            value = float(r[key]); idx = int(r["instance_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals_by_instance[idx] = value
    vals = np.asarray(list(vals_by_instance.values()), dtype=float)
    if vals.size == 0:
        return (float("nan"),) * 3
    q25, med, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
    return float(med), float(q25), float(q75)


def _add_threshold_marker(
    ax, rows, key, label, *, linestyle="--", color=None, add_label=True
):
    """Draw a colored median threshold line with a matching IQR band."""
    med, q25, q75 = _threshold_distribution(rows, key)
    if not np.isfinite(med):
        return
    if np.isfinite(q25) and np.isfinite(q75) and q75 > q25 + 1e-15:
        ax.axvspan(q25, q75, color=color, alpha=0.07, zorder=0)
    ax.axvline(
        med, color=color, linestyle=linestyle, linewidth=1.15, zorder=1,
        label=(label if add_label else None),
    )


def _fmt_plain_tick(value: float) -> str:
    """Render tick values in decimal notation, never powers-of-ten notation."""
    v = float(value)
    if not np.isfinite(v):
        return ""
    if abs(v) < 1e-12:
        return "0"
    if abs(v - round(v)) < 1e-10:
        return str(int(round(v)))
    # Matplotlib's positional formatter selects the decimal precision needed.
    return np.format_float_positional(v, precision=10, unique=True, trim="-")


def _fmt_plain_num(value: float, _pos=None) -> str:
    return _fmt_plain_tick(value)


def _fmt_power10_tick(value: float, _pos=None) -> str:
    """Mathtext decade labels on the band-violation symlog y-axis only."""
    v = float(value)
    if not np.isfinite(v):
        return ""
    if abs(v) < 1e-30:
        return "0"
    exponent = int(round(math.log10(abs(v))))
    if np.isclose(abs(v), 10.0 ** exponent, rtol=1e-8, atol=0.0):
        sign = "-" if v < 0.0 else ""
        return rf"${sign}10^{{{exponent}}}$"
    return _fmt_plain_tick(v)


def _prepare_numeric_tick_formatters(fig) -> None:
    """Disable scientific notation/offset text without touching categorical labels."""
    for ax in fig.axes:
        if not ax.axison:
            continue
        for axis_name in ('x', 'y'):
            axis = getattr(ax, f'{axis_name}axis')
            formatter = axis.get_major_formatter()
            # Ignore explicitly authored discrete labels (e.g. eta heatmap).
            if isinstance(formatter, (LogFormatter, LogFormatterMathtext)):
                axis.set_major_formatter(FuncFormatter(_fmt_plain_num))
            elif isinstance(formatter, ScalarFormatter):
                formatter.set_scientific(False)
                formatter.set_useOffset(False)
                formatter.set_powerlimits((-100, 100))
                # Custom formatter also avoids scientific-looking 1e-6 offsets.
                axis.set_major_formatter(FuncFormatter(_fmt_plain_num))
        ax.xaxis.offsetText.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)


def _tick_priority(value: float, *, axis: str) -> tuple:
    """Prioritize the origin and multiplier 1 if collision forces us to hide labels."""
    if np.isclose(value, 0, rtol=0, atol=1e-12):
        return (0,)
    if axis == 'x' and np.isclose(value, 1, rtol=0, atol=1e-12):
        return (1,)
    return (2,)


def _prune_overlapping_ticks(fig, *, padding_px: float = 5.) -> None:
    """Use the final rendered label rectangles, not data spacing, to avoid collisions.
    We hide only the *text*, not the ticks or underlying observations. This is
    especially important around negative symlog coordinates (such as -0.75).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.axison:
            continue
        for which in ('x', 'y'):
            axis = getattr(ax, f'{which}axis')
            entries = []
            for tick in axis.get_major_ticks():
                label = tick.label1
                if not label.get_visible() or not label.get_text():
                    continue
                bb = label.get_window_extent(renderer=renderer)
                if bb.width < 0.1 or bb.height < 0.1:
                    continue
                value = tick.get_loc()
                if not np.isfinite(value):
                    continue
                lo = bb.x0 if which == 'x' else bb.y0
                hi = bb.x1 if which == 'x' else bb.y1
                entries.append((lo, hi, float(value), label))
            # Greedy keep: first preferred reference marks, then distribute
            # remaining labels left-to-right (bottom-to-top on y axes).
            kept = []
            for low, high, value, label in sorted(
                    entries, key=lambda ent: (_tick_priority(ent[2], axis=which), ent[0])):
                if any(low < hi + padding_px and high > lo - padding_px
                       for lo, hi in kept):
                    label.set_visible(False)
                else:
                    kept.append((low, high))
    fig.canvas.draw()


def _savefig(fig, path: Path, *, tight_rect=None, clean_ticks=True) -> None:
    """Save a 300-DPI PDF using the same Agg raster renderer as the old PNGs.

    Matplotlib's native PDF backend uses different font metrics from Agg, which
    can subtly change text spacing and layout (most visibly for multi-line
    annotations such as the budget heatmap mean/SEM labels).  The publication
    figures were originally tuned against Agg-rendered PNG output, so first
    render the final figure with Agg at 300 DPI and then embed those exact
    pixels losslessly in a one-page PDF.  This preserves the old PNG appearance
    while delivering the requested PDF files at an effective 300 DPI.
    """
    if fig.get_layout_engine() is None:
        if tight_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_rect)

    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Manuscript figures must be saved as PDF, got: {path}")

    # Render through Agg exactly as the former PNG exporter did, only at 300 DPI.
    # Let unmodified figures (the budget heatmap) keep their original tick styling.
    if clean_ticks:
        _prepare_numeric_tick_formatters(fig)
        _prune_overlapping_ticks(fig)
    raster = BytesIO()
    fig.savefig(raster, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    raster.seek(0)

    image = plt.imread(raster, format="png")
    height_px, width_px = image.shape[:2]

    # Make the PDF page exactly match the raster dimensions at 300 pixels/inch.
    # interpolation='none' tells vector backends to embed the native image
    # rather than resample it, retaining the Agg rendering pixel-for-pixel.
    pdf_fig = plt.figure(
        figsize=(width_px / 300.0, height_px / 300.0),
        dpi=300,
        frameon=False,
    )
    ax = pdf_fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(image, interpolation="none", aspect="auto")
    ax.set_axis_off()
    pdf_fig.savefig(path, format="pdf", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(pdf_fig)


def _series_marker(index: int) -> str:
    return _SERIES_MARKERS[index % len(_SERIES_MARKERS)]


def _legend(ax, **kwargs):
    """Draw a compact legend with duplicate labels removed."""
    handles, labels = ax.get_legend_handles_labels()
    unique_handles, unique_labels = [], []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        defaults = dict(handlelength=2.0, handletextpad=0.55, labelspacing=0.35, borderpad=0.4)
        defaults.update(kwargs)
        return ax.legend(unique_handles, unique_labels, **defaults)
    return None


def _figure_legend(fig, axes, **kwargs):
    """Draw one de-duplicated legend for a group of merged panels."""
    axes_iter = np.asarray(axes, dtype=object).reshape(-1)
    handles, labels = [], []
    seen = set()
    for ax in axes_iter:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if not handles:
        return None
    defaults = dict(
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.35,
        borderpad=0.4,
        frameon=True,
    )
    defaults.update(kwargs)
    return fig.legend(handles, labels, **defaults)


def _panel_labels(axes, labels=None, *, fontsize=10):
    """Add consistent panel labels inside each axes' upper-left corner.

    Labels are anchored in axes coordinates rather than relative to tick-label
    or y-label extents.  This keeps their placement identical when panels have
    different tick widths or when right-hand panels omit y-axis labels.
    """
    axes_flat = np.asarray(axes, dtype=object).reshape(-1)
    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(axes_flat))]
    if len(labels) != len(axes_flat):
        raise ValueError("Panel-label count must match the number of axes")
    for ax, label in zip(axes_flat, labels):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.0),
            zorder=20,
        )


def _format_penalty_oracle_axis(ax) -> None:
    ax.set_xscale("symlog", linthresh=0.5, linscale=1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_plain_num))
    ax.grid(alpha=0.25)


def plot_theorem_local(rows, out: Path):
    if not rows: return
    x = THEOREM_PENALTY_ORACLE_ETA_RATIOS
    g, gs = _mean_std_by(rows, "eta_l_oracle_ratio", x, "g_l")
    gamma, gammas = _mean_std_by(rows, "eta_l_oracle_ratio", x, "Gamma_l")
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    axes[0].errorbar(x, g, yerr=gs, marker="o", capsize=3); axes[0].axhline(0, linestyle=":", linewidth=1); axes[0].set_ylabel(r"$g_{\mathrm{loc}}$")
    axes[1].errorbar(x, gamma, yerr=gammas, marker="o", capsize=3); axes[1].axhline(0, linestyle=":", linewidth=1); axes[1].set_ylabel(r"$\Gamma_{\mathrm{loc}}$")
    axes[1].set_xlabel(r"$\eta_{\mathrm{loc}}$")
    for i, ax in enumerate(axes):
        ax.axvline(0, color=_THEOREM_REF_COLORS[0], linestyle="-", linewidth=1.25, label=(r"$P_{\mathrm{loc}}^\star$" if i == 0 else None))
        _add_threshold_marker(ax, rows, "exact_gain_eta_ratio", r"$\Delta_{\mathrm{1bit}}$", linestyle="--", color=_THEOREM_REF_COLORS[1], add_label=i == 0)
        _add_threshold_marker(ax, rows, "Delta_star_eta_ratio", r"$\Delta_\star$", linestyle="-.", color=_THEOREM_REF_COLORS[2], add_label=i == 0)
        _format_penalty_oracle_axis(ax)
    _panel_labels(axes)
    _legend(axes[0], fontsize=8)
    _savefig(fig, out / "theorem_local.pdf")


def plot_theorem_global(rows, out: Path):
    if not rows: return
    x = THEOREM_PENALTY_ORACLE_ETA_RATIOS
    g, gs = _mean_std_by(rows, "eta_g_oracle_ratio", x, "g_g")
    success = []
    for ratio in x:
        rr = [r for r in rows if r.get("eta_g_oracle_ratio") is not None and np.isclose(float(r["eta_g_oracle_ratio"]), ratio)]
        success.append(float(np.mean([bool(r.get("all_global_minimizers_feasible")) for r in rr])) if rr else float("nan"))
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    axes[0].errorbar(x, g, yerr=gs, marker="o", capsize=3); axes[0].axhline(0, linestyle=":", linewidth=1); axes[0].set_ylabel(r"$g_{\mathrm{glob}}$")
    axes[1].plot(x, success, marker="o"); axes[1].set_ylim(-0.05, 1.05); axes[1].set_ylabel(r"$f_{\mathrm{glob}}$")
    axes[1].set_xlabel(r"$\eta_{\mathrm{glob}}$")
    for i, ax in enumerate(axes):
        ax.axvline(0, color=_THEOREM_REF_COLORS[0], linestyle="-", linewidth=1.25, label=(r"$P_{\mathrm{glob}}^\star$" if i == 0 else None))
        _add_threshold_marker(ax, rows, "G_F_eta_ratio", r"$G_{\mathcal{F}}$", linestyle="--", color=_THEOREM_REF_COLORS[1], add_label=i == 0)
        _add_threshold_marker(ax, rows, "Ghat_F_tight_eta_ratio", r"$\widehat G_{\mathcal{F}}^{\mathrm{tight}}$", linestyle=":", color=_THEOREM_REF_COLORS[2], add_label=i == 0)
        _add_threshold_marker(ax, rows, "Ghat_F_eta_ratio", r"$\widehat G_{\mathcal{F}}$", linestyle="-.", color=_THEOREM_REF_COLORS[3], add_label=i == 0)
        _format_penalty_oracle_axis(ax)
    _panel_labels(axes)
    _legend(axes[0], fontsize=7.5)
    _savefig(fig, out / "theorem_global.pdf")


def plot_theorem_band(rows, out: Path):
    if not rows: return
    x = THEOREM_PENALTY_ORACLE_ETA_RATIOS
    g, gs = _mean_std_by(rows, "eta_u_oracle_ratio", x, "g_u")
    gamma, gammas = _mean_std_by(rows, "eta_u_oracle_ratio", x, "Gamma_u")
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    axes[0].errorbar(x, g, yerr=gs, marker="o", capsize=3); axes[0].axhline(0, linestyle=":", linewidth=1); axes[0].set_ylabel(r"$g_{\mathrm{band}}$")
    axes[1].errorbar(x, gamma, yerr=gammas, marker="o", capsize=3); axes[1].axhline(0, linestyle=":", linewidth=1); axes[1].set_yscale("symlog", linthresh=1e-6); axes[1].yaxis.set_major_formatter(FuncFormatter(_fmt_power10_tick)); axes[1].set_ylabel(r"$\Gamma_{\mathrm{band}}$")
    axes[1].set_xlabel(r"$\eta_{\mathrm{band}}$")
    for i, ax in enumerate(axes):
        ax.axvline(0, color=_THEOREM_REF_COLORS[0], linestyle="-", linewidth=1.25, label=(r"$P_{\mathrm{band}}^\star$" if i == 0 else None))
        _add_threshold_marker(ax, rows, "Delta_base_max_eta_ratio", r"$\Delta_{\max}^{\mathrm{base}}$", linestyle="--", color=_THEOREM_REF_COLORS[1], add_label=i == 0)
        _add_threshold_marker(ax, rows, "B_base_eta_ratio", r"$B_{\mathrm{base}}$", linestyle="-.", color=_THEOREM_REF_COLORS[2], add_label=i == 0)
        _format_penalty_oracle_axis(ax)
    _panel_labels(axes)
    _legend(axes[0], fontsize=8)
    _savefig(fig, out / "theorem_band.pdf")


def plot_theorem_ordering(rows, out: Path):
    if not rows: return
    x = THEOREM_ORDER_ORACLE_ETA_RATIOS
    metrics = [("g_ord", r"$g_{\mathrm{ord}}$"), ("Gamma_ord", r"$\Gamma_{\mathrm{ord}}$"), ("kendall_tau", r"Kendall $\tau$"), ("pearson_r", r"Pearson $r$")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharex=True)
    for idx, (ax, (key, ylabel)) in enumerate(zip(axes.flat, metrics)):
        means, stds = _mean_std_by(rows, "eta_lambda_oracle_ratio", x, key)
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=3)
        ax.axvline(0, color=_THEOREM_REF_COLORS[0], linestyle="-", linewidth=1.25,
                   label=(r"$\lambda_{\mathrm{ord}}^\star$" if idx == 0 else None))
        _add_threshold_marker(ax, rows, "lambda_cert_eta_ratio", r"$\lambda_{\mathrm{ord,cert}}$", linestyle="--", color=_THEOREM_REF_COLORS[1], add_label=idx == 0)
        _add_threshold_marker(ax, rows, "lambda_practical_eta_ratio", r"$\lambda_{\mathrm{ord,coef}}$", linestyle="-.", color=_THEOREM_REF_COLORS[2], add_label=idx == 0)
        if key in {"g_ord", "Gamma_ord"}: ax.axhline(0, linestyle=":", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\eta_{\mathrm{ord}}$")
    _panel_labels(axes)
    _legend(axes.flat[0], fontsize=7.5)
    _savefig(fig, out / "theorem_ordering.pdf")


def _target_instance_values(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    """One value per independent instance, averaging repeated runs first."""
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            f = float(value)
            idx = int(row.get("instance_index", -1))
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            grouped.setdefault(idx, []).append(f)
    vals = [float(np.mean(v)) for v in grouped.values() if v]
    return np.asarray(vals, dtype=float)


def _target_mean_sem(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float, int]:
    """Mean and standard error across independent problem instances.

    Multiple QAOA restarts for the same instance are averaged before computing
    between-instance uncertainty, so restarts are not treated as independent
    problem instances.
    """
    vals = _target_instance_values(rows, key)
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


def _target_marker_summary(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float, float]:
    """Median and IQR of a recommendation coordinate, one value per instance."""
    by_instance: Dict[int, float] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            f = float(value)
            idx = int(row.get("instance_index", -1))
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            by_instance[idx] = f
    vals = np.asarray(list(by_instance.values()), dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
    return float(med), float(q1), float(q3)


def _add_target_vertical_marker(
    ax,
    rows: Sequence[Dict[str, Any]],
    key: str,
    label: str,
    *,
    linestyle: str,
    color: Optional[str] = None,
    linewidth: float = 1.3,
    centered_eta: bool = False,
) -> None:
    """Draw a median recommendation line and shade its across-instance IQR."""
    med, q1, q3 = _target_marker_summary(rows, key)
    if not np.isfinite(med):
        return
    if centered_eta:
        med = _eta_from_ratio(med)
        q1 = _eta_from_ratio(q1) if np.isfinite(q1) else q1
        q3 = _eta_from_ratio(q3) if np.isfinite(q3) else q3
    if np.isfinite(q1) and np.isfinite(q3) and q3 > q1 + 1e-15:
        ax.axvspan(q1, q3, color=color, alpha=0.09, zorder=0)
    ax.axvline(med, linestyle=linestyle, color=color, linewidth=linewidth, label=label)


def _penalty_multiplier_x(row: Dict[str, Any]) -> float:
    """Selected penalty/reference ratio, with legacy CSV fallback."""
    value = row.get("penalty_multiplier", row.get("P_over_P_global"))
    return float(value)


def _eta_from_ratio(ratio: float) -> float:
    """Convert a positive coefficient/reference ratio to the centered eta coordinate."""
    return float(ratio) - 1.0


def _penalty_eta_x(row: Dict[str, Any]) -> float:
    """Global-penalty centered coordinate eta_glob = P/P_glob^star - 1."""
    return _eta_from_ratio(_penalty_multiplier_x(row))


def _fmt_eta_tick(value: float) -> str:
    """Compact numeric tick labels for centered coefficient coordinates."""
    if abs(value) < 5e-13:
        return "0"
    av = abs(value)
    if av >= 100:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if av >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _set_eta_axis(ax, values: Sequence[float], label: str, *, show_label: bool = True) -> None:
    """Use a centered eta axis while retaining readable spacing over wide sweeps."""
    finite = sorted({float(v) for v in values if np.isfinite(v) and float(v) > -1.0})
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.0, base=2)
    if finite:
        # Extremely wide optional sweeps need fewer labels near eta=-1.
        if finite[-1] > 100.0:
            neg = [v for v in finite if v < -1e-12]
            pos = [v for v in finite if v > 1e-12]
            ticks = [neg[0]] if neg else []
            if any(abs(v) <= 1e-12 for v in finite):
                ticks.append(0.0)
            if pos:
                if len(pos) <= 4:
                    ticks.extend(pos)
                else:
                    idx = [0, len(pos)//3, (2*len(pos))//3, len(pos)-1]
                    ticks.extend(pos[i] for i in idx)
            ticks = sorted(set(ticks))
            ax.set_xticks(ticks)
            ax.set_xticklabels([_fmt_eta_tick(v) for v in ticks])
            ax.set_xlabel(label if show_label else "")
            ax.grid(alpha=0.25)
            return
        # Prefer eta values corresponding to powers-of-two coefficient ratios.
        # This keeps the original sweep structure visible without overcrowding.
        power_ticks = []
        for v in finite:
            ratio = 1.0 + v
            if ratio <= 0.0:
                continue
            q = np.log2(ratio)
            if abs(q - round(q)) < 1e-9:
                power_ticks.append((int(round(q)), v))
        if len(power_ticks) >= 4:
            power_ticks.sort()
            if len(power_ticks) <= 7:
                ticks = [v for _q, v in power_ticks]
            else:
                idx = np.unique(np.rint(np.linspace(0, len(power_ticks)-1, 7)).astype(int))
                ticks = [power_ticks[int(i)][1] for i in idx]
                zero_candidates = [v for q, v in power_ticks if q == 0]
                if zero_candidates and not any(np.isclose(t, 0.0) for t in ticks):
                    # Replace the selected tick nearest the boundary with eta=0.
                    k = int(np.argmin(np.abs(np.asarray(ticks))))
                    ticks[k] = 0.0
                ticks = sorted(set(ticks))
        elif len(finite) <= 7:
            ticks = finite
        else:
            idx = np.unique(np.rint(np.linspace(0, len(finite)-1, 7)).astype(int))
            ticks = [finite[int(i)] for i in idx]
            if 0.0 in finite and not any(np.isclose(t, 0.0) for t in ticks):
                k = int(np.argmin(np.abs(np.asarray(ticks))))
                ticks[k] = 0.0
                ticks = sorted(set(ticks))
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_eta_tick(v) for v in ticks])
    ax.set_xlabel(label if show_label else "")
    ax.grid(alpha=0.25)


def _add_target_ratio_vertical_marker(
    ax,
    rows: Sequence[Dict[str, Any]],
    numerator_key: str,
    label: str,
    *,
    denominator_key: Optional[str] = None,
    linestyle: str,
    color: Optional[str] = None,
    linewidth: float = 1.3,
    centered_eta: bool = False,
) -> None:
    """Draw a median/IQR marker for a per-instance ratio of two stored scales."""
    by_instance: Dict[int, float] = {}
    for row in rows:
        try:
            num = float(row.get(numerator_key))
            if denominator_key is None:
                den_raw = row.get("penalty_multiplier_reference_value", row.get("P_global"))
            else:
                den_raw = row.get(denominator_key)
            den = float(den_raw)
            idx = int(row.get("instance_index", -1))
        except (TypeError, ValueError):
            continue
        if np.isfinite(num) and np.isfinite(den) and den > 0.0:
            by_instance[idx] = num / den
    vals = np.asarray(list(by_instance.values()), dtype=float)
    if vals.size == 0:
        return
    q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
    if centered_eta:
        q1, med, q3 = (_eta_from_ratio(q1), _eta_from_ratio(med), _eta_from_ratio(q3))
    if q3 > q1 + 1e-15:
        ax.axvspan(float(q1), float(q3), color=color, alpha=0.07, zorder=0)
    ax.axvline(float(med), linestyle=linestyle, color=color, linewidth=linewidth, label=label)


def _add_penalty_recommendation_markers(
    ax,
    rows: Sequence[Dict[str, Any]],
    *,
    include_local: bool,
    include_global: bool = True,
    include_band: bool = True,
    centered_eta: bool = True,
) -> None:
    """Draw exact/tight property boundaries and computable surrogate references.

    On small enumerated QAOA instances the rows contain both the brute-force
    exact property boundaries and the computable recommendations.  Both are
    shown so the practical solver plots expose certificate conservativeness.
    Older result files without those fields fall back to the original markers.
    """
    # Visual grammar for vertical recommendation lines:
    #   color     -> bound family (local / global / full-band)
    #   linestyle -> exact (solid) / surrogate (dashed)
    bound_colors = {
        "local": "tab:blue",
        "global": "tab:orange",
        "band": "tab:green",
    }

    has_comparison = any(r.get("P_global_tight_boundary") not in (None, "") for r in rows)
    if has_comparison:
        if include_local:
            _add_target_ratio_vertical_marker(
                ax, rows, "P_local_tight_boundary", r"$P_{\mathrm{loc}}^\star$",
                linestyle="-", color=bound_colors["local"], centered_eta=centered_eta
            )
            _add_target_ratio_vertical_marker(
                ax, rows, "P_local_surrogate", r"$\Delta_\star$",
                linestyle="--", color=bound_colors["local"], centered_eta=centered_eta
            )
        if include_global:
            _add_target_ratio_vertical_marker(
                ax, rows, "P_global_tight_boundary", r"$P_{\mathrm{glob}}^\star$",
                linestyle="-", color=bound_colors["global"], centered_eta=centered_eta
            )
            _add_target_ratio_vertical_marker(
                ax, rows, "P_global_surrogate", r"$\widehat G_{\mathcal{F}}^{\mathrm{tight}}$",
                linestyle="--", color=bound_colors["global"], centered_eta=centered_eta
            )
        if include_band:
            _add_target_ratio_vertical_marker(
                ax, rows, "P_band_tight_boundary", r"$P_{\mathrm{band}}^\star$",
                linestyle="-", color=bound_colors["band"], centered_eta=centered_eta
            )
            _add_target_ratio_vertical_marker(
                ax, rows, "P_band_surrogate", r"$B_{\mathrm{base}}$",
                linestyle="--", color=bound_colors["band"], centered_eta=centered_eta
            )
        return

    empirical = any(
        str(r.get("penalty_scale_source", "")).startswith("extrapolated_")
        for r in rows
    )
    local_label = r"$P_{\mathrm{loc}}^\star$"
    global_label = r"$P_{\mathrm{glob}}^\star$"
    band_label = r"$P_{\mathrm{band}}^\star$"
    if include_local:
        _add_target_vertical_marker(
            ax, rows, "P_local_over_P_global", local_label,
            linestyle="--", color=bound_colors["local"], centered_eta=centered_eta
        )
    if include_global:
        ax.axvline(
            0.0 if centered_eta else 1.0, linestyle="--", color=bound_colors["global"],
            linewidth=1.3, label=global_label
        )
    if include_band:
        _add_target_vertical_marker(
            ax, rows, "P_band_over_P_global", band_label,
            linestyle="--", color=bound_colors["band"], centered_eta=centered_eta
        )


def _format_penalty_eta_axis(ax, rows: Optional[Sequence[Dict[str, Any]]] = None) -> None:
    """Centered global-penalty coordinate eta_glob = P/P_glob^star - 1."""
    rows = list(rows or [])
    values = [_penalty_eta_x(r) for r in rows if np.isfinite(_penalty_eta_x(r))]
    _set_eta_axis(ax, values, r"$\eta_{\mathrm{glob}}$")


def _format_penalty_multiplier_axis(ax, rows: Optional[Sequence[Dict[str, Any]]] = None) -> None:
    """Penalty ratio P/P_glob^star on a base-2 logarithmic axis."""
    rows = list(rows or [])
    values = sorted({_penalty_multiplier_x(r) for r in rows
                     if np.isfinite(_penalty_multiplier_x(r)) and _penalty_multiplier_x(r) > 0.0})
    ax.set_xscale("log", base=2)
    if values:
        qmin = int(np.ceil(np.log2(min(values))))
        qmax = int(np.floor(np.log2(max(values))))
        powers = list(range(qmin, qmax + 1))
        if len(powers) > 7:
            stride = max(1, int(math.ceil((len(powers) - 1) / 6)))
            powers = powers[::stride]
            if qmax not in powers:
                powers.append(qmax)
            if 0 >= qmin and 0 <= qmax and 0 not in powers:
                powers.append(0)
            powers = sorted(set(powers))
        ticks = [2.0 ** q for q in powers]
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_plain_tick(t) for t in ticks])
    ax.set_xlabel(r"$P/P_{\mathrm{glob}}^\star$")
    ax.grid(alpha=0.25)


def plot_spectral_gap(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    """Merge normalized-gap and spectral-mechanism views into one figure."""
    if not rows:
        return
    xvals = sorted({_penalty_eta_x(r) for r in rows
                    if np.isfinite(_penalty_eta_x(r)) and _penalty_eta_x(r) > -1.0})

    normalized, normalized_sem = [], []
    low, low_sem, span, span_sem = [], [], [], []
    for x in xvals:
        rr = [r for r in rows if np.isclose(_penalty_eta_x(r), x)]
        m, se, _n = _target_mean_sem(rr, "normalized_gap")
        normalized.append(m); normalized_sem.append(se)
        m, se, _n = _target_mean_sem(rr, "low_gap")
        low.append(m); low_sem.append(se)
        m, se, _n = _target_mean_sem(rr, "energy_range")
        span.append(m); span_sem.append(se)

    # Two 4x2.5 panels side-by-side preserve the preferred single-panel ratio.
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

    axes[0].errorbar(xvals, normalized, yerr=normalized_sem, marker="o", capsize=3,
                     label=r"$\rho_{\mathrm{spec}}$")
    axes[0].set_ylabel(r"$\rho_{\mathrm{spec}}$")

    axes[1].errorbar(xvals, low, yerr=low_sem, marker="o", capsize=3,
                     label=r"$\Delta_{\mathrm{low}}$")
    axes[1].errorbar(xvals, span, yerr=span_sem, marker="s", capsize=3,
                     label=r"$W_{\mathrm{spec}}$")
    axes[1].set_ylabel(r"$\Delta E$")

    _set_eta_axis(axes[0], xvals, r"$\eta_{\mathrm{glob}}$", show_label=False)
    _add_penalty_recommendation_markers(
        axes[0], rows, include_local=False, include_global=True, include_band=True
    )
    _format_penalty_eta_axis(axes[1], rows)
    _add_penalty_recommendation_markers(
        axes[1], rows, include_local=False, include_global=True, include_band=True
    )

    _panel_labels(axes)
    # The bound legend is identical in both panels; show it once, outside.
    _figure_legend(fig, axes, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                   fontsize=7.5, ncol=3)
    _savefig(fig, out / "spectral.pdf")


def _prepare_qaoa_penalty_series(rows: Sequence[Dict[str, Any]]):
    """Return the data used by the practical penalty-multiplier figure."""
    if not rows:
        return None
    depths = sorted({int(r["qaoa_depth"]) for r in rows if r.get("qaoa_depth") is not None})
    if not depths:
        return None
    depth = int(max(depths))
    selected = [r for r in rows if int(r.get("qaoa_depth", -1)) == depth]
    xvals = sorted({_penalty_multiplier_x(r) for r in selected
                    if np.isfinite(_penalty_multiplier_x(r)) and _penalty_multiplier_x(r) > 0.0})
    p_feas, p_feas_sem, p_best, p_best_sem = [], [], [], []
    for x in xvals:
        rr = [r for r in selected if np.isclose(_penalty_multiplier_x(r), x)]
        m, se, _n = _target_mean_sem(rr, "p_feasible")
        p_feas.append(m); p_feas_sem.append(se)
        m, se, _n = _target_mean_sem(rr, "p_best_feasible")
        p_best.append(m); p_best_sem.append(se)
    return selected, xvals, p_feas, p_feas_sem, p_best, p_best_sem


def _draw_qaoa_penalty_panel(ax, rows: Sequence[Dict[str, Any]]) -> bool:
    """Draw the former qaoa_penalty.pdf contents on an existing axes."""
    prepared = _prepare_qaoa_penalty_series(rows)
    if prepared is None:
        return False
    selected, xvals, p_feas, p_feas_sem, p_best, p_best_sem = prepared
    ax.errorbar(xvals, p_feas, yerr=p_feas_sem, marker="o", capsize=3,
                label=r"$p_{\mathrm{feas}}$")
    ax.errorbar(xvals, p_best, yerr=p_best_sem, marker="s", capsize=3,
                label=r"$p_{\mathrm{opt}}$")
    _add_penalty_recommendation_markers(
        ax, selected, include_local=True, include_global=True, include_band=True,
        centered_eta=False
    )
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)

    ax.set_ylabel(r"$p_{\mathrm{feas}},\ p_{\mathrm{opt}}$")
    ax.set_ylim(0.0, 1.0)
    return True


def plot_practical_penalty(
    rows: Sequence[Dict[str, Any]],
    out: Path,
    *,
    qaoa_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Two-panel practical figure: QAOA sampling (left) and quality (right).

    The left panel is exactly the content of the former standalone
    ``qaoa_penalty.pdf`` and is sourced from ``qaoa_penalty.csv``.  The right
    panel keeps the practical feasible-sample quality curve from
    ``practical_penalty.csv``.  The obsolete standalone QAOA figure is no
    longer emitted.
    """
    qaoa_rows = list(qaoa_rows or [])
    rows = list(rows or [])
    if not qaoa_rows and not rows:
        return

    have_qaoa = bool(_prepare_qaoa_penalty_series(qaoa_rows))
    have_quality = bool(rows)
    n_panels = int(have_qaoa) + int(have_quality)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    # axes = list(axes_raw.flat)
    panel_idx = 0

    if have_qaoa:
        _draw_qaoa_penalty_panel(axes[panel_idx], qaoa_rows)
        panel_idx += 1

    if have_quality:
        ax = axes[panel_idx]
        xvals = sorted({_penalty_multiplier_x(r) for r in rows
                        if np.isfinite(_penalty_multiplier_x(r)) and _penalty_multiplier_x(r) > 0.0})
        qaoa_gap_pct, qaoa_gap_sem_pct = [], []
        for x in xvals:
            rr = [r for r in rows if np.isclose(_penalty_multiplier_x(r), x)]
            mean_gap, gap_se, _n = _target_mean_sem(rr, "relative_primary_gap")
            qaoa_gap_pct.append(100.0 * mean_gap if np.isfinite(mean_gap) else float("nan"))
            qaoa_gap_sem_pct.append(100.0 * gap_se if np.isfinite(gap_se) else float("nan"))

        ax.errorbar(xvals, qaoa_gap_pct, yerr=qaoa_gap_sem_pct, marker="o", capsize=3,
                    label=r"$\mathrm{gap}_{\mathrm{pri}}$")
        ax.axhline(0.0, linestyle=(0, (7, 2, 1, 2)), linewidth=1.3,
                   label=r"$\mathrm{gap}_{\mathrm{pri}}=0$")
        _add_penalty_recommendation_markers(
            ax, rows, include_local=True, include_global=True, include_band=True,
            centered_eta=False
        )
        _format_penalty_multiplier_axis(ax, rows)
        ax.set_ylabel(r"$\mathrm{gap}_{\mathrm{pri}}\;(\%)$")

    _panel_labels(axes[:n_panels])
    # One de-duplicated legend for the whole two-panel figure.  It sits above
    # the axes (outside the panels) while remaining inside the 8x2.5 canvas.
    _figure_legend(fig, axes, loc="upper center", bbox_to_anchor=(0.5, 1.17),
                   fontsize=7.5, ncol=3)
    _savefig(fig, out / "penalty_multiplier.pdf")


def plot_budget_heatmap(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    """B1: feasible sampling probability versus penalty and QAOA optimizer budget."""
    if not rows:
        return
    qrows = [r for r in rows if str(r.get("solver", "")).lower() == "qaoa"]
    if not qrows:
        return
    budgets = sorted({int(float(r.get("optimizer_iterations", r.get("work_budget")))) for r in qrows})
    ratios = sorted({_penalty_eta_x(r) for r in qrows if np.isfinite(_penalty_eta_x(r))})
    z = np.full((len(budgets), len(ratios)), np.nan)
    z_sem = np.full_like(z, np.nan)
    for i, budget in enumerate(budgets):
        for j, ratio in enumerate(ratios):
            rr = [r for r in qrows
                  if int(float(r.get("optimizer_iterations", r.get("work_budget")))) == budget
                  and np.isclose(_penalty_eta_x(r), ratio)]
            if rr:
                z[i, j], z_sem[i, j], _n = _target_mean_sem(rr, "p_feasible")

    fig, ax = plt.subplots(figsize=(4, 2.5))
    im = ax.imshow(z, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(ratios)), [_fmt_eta_tick(x) for x in ratios])
    ax.set_yticks(np.arange(len(budgets)), [str(v) for v in budgets])
    ax.set_xlabel(r"$\eta_{\mathrm{glob}}$")
    ax.set_ylabel("QAOA iterations")
    pvals = sorted({int(r["qaoa_depth"]) for r in qrows if r.get("qaoa_depth") not in (None, "")})
    ptxt = f" (p={pvals[0]})" if len(pvals) == 1 else ""
    for i in range(len(budgets)):
        for j in range(len(ratios)):
            if np.isfinite(z[i, j]):
                text = f"{z[i,j]:.2f}"
                if np.isfinite(z_sem[i,j]) and z_sem[i,j] > 0.0:
                    text += "\n" + rf"$\pm {z_sem[i,j]:.2f}$"
                ax.text(j, i, text, ha="center", va="center", linespacing=0.9)
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.set_label(r"$p_{\mathrm{feas}}$")
    _savefig(fig, out / "budget_heatmap.pdf", clean_ticks=False)


def plot_scaling_target(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    """S1: QAOA scaling, comparing tight and surrogate penalty prescriptions."""
    if not rows:
        return
    sizes = sorted({int(r["N_x"]) for r in rows})
    regimes = [
        "under",
        "local_tight", "local_surrogate",
        "global_tight", "global_surrogate",
        "band_tight", "band_surrogate",
    ]
    labels = {
        "under": "Under-scaled",
        "local_tight": r"$P_{\mathrm{loc}}^\star$",
        "local_surrogate": r"$\Delta_\star$",
        "global_tight": r"$P_{\mathrm{glob}}^\star$",
        "global_surrogate": r"$\widehat G_{\mathcal{F}}^{\mathrm{tight}}$",
        "band_tight": r"$P_{\mathrm{band}}^\star$",
        "band_surrogate": r"$B_{\mathrm{base}}$",
    }
    markers = {
        "under": "o",
        "local_tight": "D", "local_surrogate": "d",
        "global_tight": "s", "global_surrogate": "P",
        "band_tight": "^", "band_surrogate": "v",
    }
    linestyles = {
        "under": "-",
        "local_tight": "-", "local_surrogate": "--",
        "global_tight": "-", "global_surrogate": "--",
        "band_tight": "-", "band_surrogate": "--",
    }

    pvals = sorted({int(r["qaoa_depth"]) for r in rows if r.get("qaoa_depth") not in (None, "")})
    ptxt = f" (p={pvals[0]})" if len(pvals) == 1 else ""
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

    for regime in regimes:
        feas_vals, feas_sems = [], []
        energy_vals, energy_sems = [], []
        for n in sizes:
            rr = [r for r in rows if r.get("penalty_regime") == regime and int(r["N_x"]) == n]
            m, se, _count = _target_mean_sem(rr, "p_feasible")
            feas_vals.append(m); feas_sems.append(se)
            metric = (
                "expected_H_primary_plus_reporting_penalty"
                if any(r.get("expected_H_primary_plus_reporting_penalty") not in (None, "") for r in rr)
                else "expected_H_primary_plus_penalty"
            )
            m, se, _count = _target_mean_sem(rr, metric)
            energy_vals.append(m); energy_sems.append(se)
        axes[0].errorbar(
            sizes, feas_vals, yerr=feas_sems, marker=markers[regime],
            linestyle=linestyles[regime], capsize=3, label=labels[regime]
        )
        axes[1].errorbar(
            sizes, energy_vals, yerr=energy_sems, marker=markers[regime],
            linestyle=linestyles[regime], capsize=3, label=labels[regime]
        )

    axes[0].set_ylabel(r"$p_{\mathrm{feas}}$")
    axes[0].set_ylim(0.0, 1.03)
    axes[0].grid(alpha=0.25)
    # _legend(axes[0], ncol=2, fontsize="small")

    reporting_refs = sorted({
        str(r.get("reporting_penalty_reference"))
        for r in rows if r.get("reporting_penalty_reference") not in (None, "")
    })
    reporting_ref = reporting_refs[0] if len(reporting_refs) == 1 else "common"
    axes[1].set_xlabel(r"$n$")
    axes[1].set_ylabel(r"$\langle\mathcal{H}_{\mathrm{rep}}\rangle$")
    axes[1].grid(alpha=0.25)

    _panel_labels(axes)
    # fig.tight_layout()
    _figure_legend(fig, axes, loc="upper center", bbox_to_anchor=(0.5, 1.1),
                   fontsize=7.5, ncol=3)
    _savefig(fig, out / "penalty_nx.pdf")


def _conservativeness_series(
    rows: Sequence[Dict[str, Any]],
    sizes: Sequence[int],
    key: str,
) -> Tuple[List[float], List[float], List[int]]:
    """Mean ± between-instance SEM for one coefficient scale at each N_x."""
    means: List[float] = []
    sems: List[float] = []
    counts: List[int] = []
    for size in sizes:
        rr = [
            r for r in rows
            if r.get("N_x") is not None and int(r["N_x"]) == int(size)
        ]
        mean, sem, n = _target_mean_sem(rr, key)
        means.append(mean)
        sems.append(sem)
        counts.append(n)
    return means, sems, counts


def plot_conservativeness(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    """Plot exact property boundaries against theorem/computable surrogates.

    The five panels deliberately keep the raw coefficient scale on the y-axis
    rather than normalizing by the oracle boundary.  This makes the absolute
    conservativeness inflation visible as the QUBO grows.  Error bars are SEM
    across independent generated instances at each N_x.
    """
    if not rows:
        return
    sizes = sorted({int(r["N_x"]) for r in rows if r.get("N_x") is not None})
    if not sizes:
        return

    panels = [
        (
            "Penalty: local robustness",
            [
                ("P_local_property_boundary", r"$P_{\mathrm{loc}}^\star$"),
                ("Delta_local_exact_gain", r"$\Delta_{\mathrm{1bit}}$"),
                ("Delta_star", r"$\Delta_\star$"),
            ],
        ),
        (
            "Penalty: global-minimizer feasibility",
            [
                ("P_global_property_boundary", r"$P_{\mathrm{glob}}^\star$"),
                ("G_F_exact", r"$G_{\mathcal{F}}$"),
                ("Ghat_F_tight", r"$\widehat G_{\mathcal{F}}^{\mathrm{tight}}$"),
                ("Ghat_F", r"$\widehat G_{\mathcal{F}}$"),
            ],
        ),
        (
            "Penalty: complete band separation",
            [
                ("P_band_property_boundary", r"$P_{\mathrm{band}}^\star$"),
                ("Delta_base_max_exact", r"$\Delta_{\max}^{\mathrm{base}}$"),
                ("B_base", r"$B_{\mathrm{base}}$"),
            ],
        ),
        (
            "Optional: full ordering",
            [
                ("lambda_oracle", r"$\lambda_{\mathrm{ord}}^\star$"),
                ("lambda_cert", r"$\lambda_{\mathrm{ord,cert}}$"),
                ("lambda_practical", r"$\lambda_{\mathrm{ord,coef}}$"),
            ],
        ),
        (
            "Optional: optimum preservation",
            [
                ("lambda_opt_property_boundary", r"$\lambda_{\mathrm{disp}}^\star$"),
                ("lambda_opt_cert", r"$\lambda_{\mathrm{disp,gap}}$"),
                ("lambda_opt_struct", r"$\lambda_{\mathrm{disp,struct}}$"),
            ],
        ),
    ]

    # Five vertically stacked panels make the guarantee hierarchy read in the
    # same top-to-bottom order as the manuscript.  The 4-inch width matches the
    # other single-column manuscript figures; the taller canvas preserves enough
    # vertical room for five readable panels after one-column LaTeX scaling.
    fig, axes = plt.subplots(5, 1, figsize=(4, 9), sharex=True)
    for ax, (title, series) in zip(axes, panels):
        for sidx, (key, label) in enumerate(series):
            means, sems, counts = _conservativeness_series(rows, sizes, key)
            yy = np.asarray(means, dtype=float)
            ee = np.asarray(sems, dtype=float)
            finite = np.isfinite(yy)
            if not finite.any():
                continue
            ax.errorbar(
                np.asarray(sizes, dtype=float)[finite],
                yy[finite],
                yerr=ee[finite],
                marker=_series_marker(sidx),
                capsize=3,
                linewidth=1.2,
                label=label,
            )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        _legend(ax, fontsize=7.2)
        ax.set_ylabel("Coefficient value")
        ax.set_xticks(sizes)
    axes[-1].set_xlabel(r"$n$")
    _panel_labels(axes)
    _savefig(fig, out / "conservativeness.pdf")

# ---------------------------------------------------------------------------
# Optional-term and optimum-preservation figures
# ---------------------------------------------------------------------------

# The penalty size plots use a separate default-cycle color per prescription,
# solid for tight and dashed for surrogate.  Follow the same visual grammar.
ARM_STYLES = (
    ('ordering_tight', r'$0.95\lambda_{\mathrm{ord}}^\star$', 'C0', 'o', '-'),
    ('ordering_surrogate', r'$0.95\lambda_{\mathrm{ord,coef}}$', 'C1', 'D', '--'),
    ('optimum_tight', r'$0.95\lambda_{\mathrm{disp}}^\star$', 'C2', 's', '-'),
    ('optimum_surrogate', r'$0.95\lambda_{\mathrm{disp,struct}}$', 'C3', '^', '--'),
    ('beyond_displacement', r'$1.05\lambda_{\mathrm{disp}}^\star$', 'C4', 'v', ':'),
)


def _stats(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return math.nan, math.nan, 0
    vals = np.asarray(values, dtype=float)
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else 0.0
    return mean, sem, len(vals)


def _ratio_marker(ax, rows, key, label, color, linestyle='--', *, show=True):
    """One median threshold line and one translucent interquartile region."""
    per_instance = {}
    for row in rows:
        try:
            value = float(row[key])
            instance = int(row['instance'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            per_instance[instance] = value
    if not per_instance:
        return False
    vals = np.fromiter(per_instance.values(), dtype=float)
    q25, med, q75 = np.quantile(vals, [.25, .5, .75])
    q25, med, q75 = q25 - 1.0, med - 1.0, q75 - 1.0
    if q75 > q25 + 1e-15:
        ax.axvspan(float(q25), float(q75), color=color, alpha=.07, zorder=0)
    ax.axvline(float(med), color=color, ls=linestyle, lw=1.15,
               label=label if show else None)
    return True


def plot_theorem_optima(rows, audits, out):
    """Two stacked 4x4 panels, matching theorem_local/global/band.pdf."""
    if not rows:
        return
    x = sorted({float(r['lambda_over_disp']) - 1.0 for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    metrics = (
        ('worst_global_minimizer_primary_regret',
         r'$R_{\mathrm{disp}}$'),
        ('strictly_better_suboptimal_percent', r'$\Gamma_{\mathrm{disp}}\;(\%)$'),
    )
    ref_rows = [r for r in rows if np.isclose(float(r['lambda_over_disp']), 1.)]
    for idx, (ax, (key, ylabel)) in enumerate(zip(axes, metrics)):
        means, sems = [], []
        for ratio in x:
            subset = [r for r in rows if np.isclose(float(r['lambda_over_disp']) - 1.0, ratio)]
            mean, sem, _ = _stats(subset, key)
            means.append(mean); sems.append(sem)
        ax.errorbar(x, means, yerr=sems, marker='o', capsize=3,
                    color='C0')
        ax.axhline(0, linestyle=':', linewidth=1, color='0.35')
        ax.axvline(0., linestyle='-', linewidth=1.25, color=_THEOREM_REF_COLORS[0],
                   label=r'$\lambda_{\mathrm{disp}}^\star$' if idx == 0 else None)
        _ratio_marker(ax, ref_rows, 'lambda_order_over_disp',
                      r'$\lambda_{\mathrm{ord}}^\star$', _THEOREM_REF_COLORS[1],
                      linestyle='--', show=idx == 0)
        _ratio_marker(ax, ref_rows, 'lambda_opt_cert_over_disp',
                      r'$\lambda_{\mathrm{disp,gap}}$', _THEOREM_REF_COLORS[2],
                      linestyle='-.', show=idx == 0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel(r'$\eta_{\mathrm{disp}}$')
    # Keep zero-valued markers fully visible instead of clipping them against
    # the lower axis spine. Preserve the autoscaled upper limit and add only a
    # small amount of headroom below zero.
    _, optima_bottom_panel_top = axes[-1].get_ylim()
    axes[-1].set_ylim(bottom=-0.05 * max(1.0, optima_bottom_panel_top))
    _panel_labels(axes)
    _legend(axes[0], fontsize=7.5)
    _savefig(fig, Path(out)/'theorem_optima.pdf')


def _constructions(rows):
    """Support old CSVs (random only) and new paired construction CSVs."""
    found = {str(r.get('construction') or 'random') for r in rows
             if str(r.get('arm')) != 'baseline'}
    return [c for c in ('random', 'marginal') if c in found]


def _optional_layout(constructions):
    paired = len(constructions) > 1
    if paired:
        fig, axes = plt.subplots(2, 2, figsize=(8, 4),
                                 sharex='col', sharey='row', squeeze=False)
    else:
        fig, arr = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
        axes = np.asarray(arr).reshape(2, 1)
    return fig, axes


def plot_optional_nx(rows, audit, out):
    """Primary-only baseline + five policies for EACH optional construction.

    With both constructions the old two metrics are shown as a 2x2 faceted
    figure: same plot style, aligned axes, common matched-instance baseline.
    Old random-only CSVs continue to render with the old compact two-panel appearance.
    """
    if not rows:
        return
    constructions = _constructions(rows)
    if not constructions:
        return
    sizes = sorted({int(r['N_x']) for r in rows})
    fig, axes = _optional_layout(constructions)
    metrics = (
        ('best_feasible_shot_primary', r'$H_{\mathrm{best}}^{(32)}$'),
        ('p_best_feasible_single_shot', r'$p_{\mathrm{opt}}$'),
    )
    for col, construction in enumerate(constructions):
        for row_idx, (key, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col]
            if len(constructions) > 1 and row_idx == 0:
                ax.set_title('Random' if construction == 'random' else 'Marginal-cost', fontsize=10)
            base = [r for r in rows if r['arm'] == 'baseline']
            if base:
                xx, yy, ee = [], [], []
                for n in sizes:
                    mean, sem, count = _stats([r for r in base if int(r['N_x']) == n],key)
                    if count:
                        xx.append(n); yy.append(mean); ee.append(sem)
                if xx:
                    ax.errorbar(xx, yy, yerr=ee, color='0.35', linestyle='-.',
                                marker='x', capsize=2, linewidth=1.,
                                label=r'$\lambda=0$')
            for arm, label, color, marker, ls in ARM_STYLES:
                policy = [r for r in rows if r['arm'] == arm and
                          str(r.get('construction') or 'random') == construction]
                if not policy:
                    continue
                xx, yy, ee = [], [], []
                for n in sizes:
                    mean, sem, count = _stats([r for r in policy if int(r['N_x']) == n],key)
                    if count:
                        xx.append(n); yy.append(mean); ee.append(sem)
                if xx:
                    ax.errorbar(xx, yy, yerr=ee, marker=marker,
                                linestyle=ls, capsize=2, color=color,
                                markersize=3.5, linewidth=1., label=label)
            if col == 0: ax.set_ylabel(ylabel)
            ax.set_xticks(sizes)
            ax.grid(alpha=.25)
        axes[-1, col].set_xlabel(r'$n$')
        # The absolute success probabilities in small-depth QAOA are often
        # below 0.1.  Start at zero but avoid flattening every policy under a
        # fixed 0--1 axis; preserve the SAME scale across both constructions.
        observed = [r for r in rows if r.get('arm') in ('baseline',) +
                    tuple(x[0] for x in ARM_STYLES)]
        probability_peak = max((float(r['p_best_feasible_single_shot'])
                                for r in observed if r.get('p_best_feasible_single_shot')
                                not in (None, '') and
                                math.isfinite(float(r['p_best_feasible_single_shot']))),
                               default=.01)
        axes[-1, col].set_ylim(0., min(1., max(.03, probability_peak*1.35)))
    _panel_labels(axes)
    # Keep the policy legend inside the plotting area, matching the theorem
    # figures' compact in-panel legend style.  The policy labels are shared by
    # both construction columns, so one de-duplicated legend is sufficient.
    _legend(axes[0, 0], fontsize=6.8 if len(constructions)>1 else 7.5, ncol=2)
    _savefig(fig, Path(out)/'optional_nx.pdf')


def _bound_xstats(rows):
    """Summarize heterogeneous actual lambda/order positions across instances."""
    per_inst = {}
    for row in rows:
        try:
            ratio = float(row['lambda_multiplier'])
            instance = int(row['instance'])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(ratio) and ratio > 0:
            per_inst[instance] = ratio
    if not per_inst:
        return math.nan, 0., 0.
    q25, med, q75 = np.quantile(list(per_inst.values()), [.25,.5,.75])
    return float(med), float(med-q25), float(q75-med)


def plot_optional_multiplier(rows, audits, out):
    """Two panels, exactly TWO errorbar series when both templates are selected.

    Each series is swept over lambda/lambda_order_exact = 2**power, using
    that construction's own exact ordering threshold.  Reference thresholds
    vary by instance and optional construction: vertical lines show medians;
    faint spans show their interquartile ranges.  Existing legacy
    bound-referenced data are intentionally not conflated with this protocol.
    """
    if not rows:
        return
    mode = str(rows[0].get('sweep_mode') or '').strip()
    if mode != 'ordering-tight':
        raise ValueError(
            'optional_multiplier.csv uses an older sweep protocol; rerun '
            'optional-multiplier to obtain the exact-ordering-normalized comparison'
        )
    constructions = _constructions(rows)
    if not constructions:
        return
    fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    metrics = (
        ('p_best_feasible_single_shot', r'$p_{\mathrm{opt}}$'),
        ('best_feasible_shot_regret_percent', r'$R_{\mathrm{best}}^{(32)}$ (%)'),
    )
    # Only random and marginal are experimental QAOA lines, at identical
    # multiplier coordinates.  The primary-only control is a horizontal line.
    # Match the rest of the manuscript plots: consecutive cycle colors for
    # independent data series, marker/linestyle redundancy for accessibility,
    # and neutral gray for the primary-only control.
    styles = {
        'random': ('Random', 'C0', 'o', '-'),
        'marginal': ('Marginal-cost', 'C1', 's', '--'),
    }
    baseline = [r for r in rows if r.get('arm') == 'baseline']
    positive_x = sorted({float(r['bound_multiplier']) for r in rows
                         if r.get('construction') in constructions and
                         r.get('arm') == 'ordering_tight' and
                         r.get('bound_multiplier') not in (None, '')})
    if not positive_x:
        raise ValueError('No exact-ordering multiplier rows found')
    # Per-template threshold ratios from one row per instance, NOT one per
    # multiplier.  The lines are annotations, not extra QAOA data series.
    threshold_specs = (
        ('lambda_order_practical_over_order', '--'),
        ('lambda_opt_struct_over_order', ':'),
        ('lambda_disp_exact_over_order', '-.'),
    )
    for panel, (key, ylabel) in enumerate(metrics):
        ax = axes[panel]
        ymax = 0.
        for construction in constructions:
            label, color, marker, series_ls = styles[construction]
            sub = [r for r in rows if r.get('construction') == construction and
                   r.get('arm') == 'ordering_tight']
            xx, means, errors = [], [], []
            for multiplier in positive_x:
                current = [r for r in sub if
                           np.isclose(float(r['bound_multiplier']), multiplier)]
                mean, sem, n = _stats(current, key)
                if n and math.isfinite(mean):
                    xx.append(multiplier)
                    means.append(mean)
                    errors.append(sem)
                    ymax = max(ymax, mean + sem)
            if xx:
                ax.errorbar(xx, means, yerr=errors, marker=marker,
                            color=color, linestyle=series_ls, capsize=2,
                            markersize=3.5, linewidth=1.0, elinewidth=.8,
                            label=label if panel == 0 else None)
            ref = [r for r in sub if
                   np.isclose(float(r['bound_multiplier']), positive_x[0])]
            # Median/IQR threshold markers, one median per construction and
            # bound type.  Lines/shading share the construction's color.
            for threshold, linestyle in threshold_specs:
                observed = []
                seen = set()
                for row in ref:
                    i = int(row['instance'])
                    if i in seen:
                        continue
                    seen.add(i)
                    try:
                        value = float(row[threshold])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if math.isfinite(value) and value > 0.:
                        observed.append(value)
                if not observed:
                    continue
                q25, median, q75 = np.quantile(observed, [.25, .5, .75])
                if q75 > q25 + 1e-12:
                    ax.axvspan(q25, q75, color=color, alpha=.07, zorder=0)
                ax.axvline(median, color=color, linestyle=linestyle,
                           alpha=.72, linewidth=1.0, zorder=1)
        if baseline:
            mean, sem, count = _stats(baseline, key)
            if count:
                ax.axhline(mean, color='0.4', linestyle='-.', linewidth=1.0,
                           label=r'$\lambda=0$' if panel==0 else None)
                ymax = max(ymax, mean + sem)
        ax.axvline(1., color='0.2', linestyle='-', linewidth=1.15, alpha=.85)
        ax.set_xscale('log', base=2)
        ax.set_ylabel(ylabel)
        if key == 'p_best_feasible_single_shot':
            ax.set_ylim(0., min(1., max(.03, ymax * 1.25)))
        else:
            ax.set_ylim(bottom=0.)
        ax.grid(alpha=.25)
    axes[-1].set_xlim(positive_x[0] * .80, positive_x[-1] * 1.22)
    axes[-1].set_xticks(positive_x)
    # Restore power-of-two tick labels with a limited number of major labels.
    powers = [int(round(np.log2(x))) for x in positive_x]
    stride = max(1, int(math.ceil((max(powers) - min(powers)) / 6)))
    major = [x for x, q in zip(positive_x, powers) if (q - min(powers)) % stride == 0]
    if 1.0 in positive_x and 1.0 not in major:
        major.append(1.0)
    if positive_x[-1] not in major:
        major.append(positive_x[-1])
    major = sorted(set(major))
    axes[-1].set_xticks(major)
    axes[-1].set_xticklabels([rf"$2^{{{int(round(np.log2(x)))}}}$" for x in major])
    axes[-1].set_xticks(positive_x, minor=True)
    axes[-1].tick_params(axis='x', which='minor', labelbottom=False)
    axes[-1].set_xlabel(r'$\lambda/\lambda_{\mathrm{ord}}^\star$')
    _panel_labels(axes)
    # Compact LaTeX legend: two experimental series, baseline, and the line
    # styles of the threshold references.  Reference colors follow the series.
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color=styles[c][1], marker=styles[c][2],
                     ls=styles[c][3], lw=1.0, markersize=3.5, label=styles[c][0])
              for c in constructions]
    if baseline:
        legend.append(Line2D([], [], color='0.4', ls='-.', lw=1.0,
                             label=r'$\lambda=0$'))
    legend.extend([
        Line2D([], [], color='0.3', ls='-', lw=.8,
               label=r'$\lambda_{\mathrm{ord}}^\star$'),
        Line2D([], [], color='0.4', ls='--', lw=.85,
               label=r'$\lambda_{\mathrm{ord,coef}}$'),
        Line2D([], [], color='0.4', ls=':', lw=.85,
               label=r'$\lambda_{\mathrm{disp,struct}}$'),
        Line2D([], [], color='0.4', ls='-.', lw=.85,
               label=r'$\lambda_{\mathrm{disp}}^\star$'),
    ])
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.1),
               fontsize=7, ncol=3, frameon=True, handlelength=2.0,
               handletextpad=.55, labelspacing=.35, borderpad=.4)
    _savefig(fig, Path(out)/'optional_multiplier.pdf')

# ---------------------------------------------------------------------------
# Unified manuscript-figure orchestration
# ---------------------------------------------------------------------------

EXPECTED = (
    "theorem_band.pdf", "theorem_local.pdf", "theorem_global.pdf",
    "theorem_ordering.pdf", "theorem_optima.pdf", "spectral.pdf",
    "conservativeness.pdf", "penalty_multiplier.pdf", "penalty_nx.pdf",
    "budget_heatmap.pdf", "optional_nx.pdf", "optional_multiplier.pdf",
)

OBSOLETE = (
    "practical_penalty.pdf", "scaling.pdf", "optional_practical.pdf",
    "qaoa_penalty.pdf", "spectral_gap.pdf", "spectral_mechanism.pdf",
)

# Clean up figure files produced by older revisions so a regenerated results
# directory contains only the current PDF outputs.
LEGACY_PNGS = tuple(Path(name).with_suffix(".png").name for name in (*EXPECTED, *OBSOLETE))


def generate_plots(results_dir: Path | str):
    """Regenerate the manuscript PDFs available from ``results_dir`` CSVs.

    Missing study families are skipped by their individual plotting functions.
    All manuscript plotting logic, including optional-term figures, lives in
    this single module.
    """
    out = Path(results_dir)
    for name in (*OBSOLETE, *LEGACY_PNGS):
        (out / name).unlink(missing_ok=True)

    plot_theorem_local(load_csv(out / "theorem_local.csv"), out)
    plot_theorem_global(load_csv(out / "theorem_global.csv"), out)
    plot_theorem_band(load_csv(out / "theorem_band.csv"), out)
    plot_theorem_ordering(load_csv(out / "theorem_ordering.csv"), out)
    plot_theorem_optima(
        load_csv(out / "theorem_optima.csv"),
        load_csv(out / "theorem_optima_audit.csv"),
        out,
    )
    plot_conservativeness(load_csv(out / "conservativeness.csv"), out)
    plot_spectral_gap(load_csv(out / "spectral_gap.csv"), out)
    penalty = load_csv(out / "qaoa_penalty.csv")
    plot_practical_penalty(penalty, out, qaoa_rows=penalty)
    plot_scaling_target(load_csv(out / "scaling.csv"), out)
    plot_budget_heatmap(load_csv(out / "budget_robustness.csv"), out)
    plot_optional_nx(
        load_csv(out / "optional_nx.csv"),
        load_csv(out / "optional_nx_audit.csv"),
        out,
    )
    plot_optional_multiplier(
        load_csv(out / "optional_multiplier.csv"),
        load_csv(out / "optional_multiplier_audit.csv"),
        out,
    )
    return [out / name for name in EXPECTED if (out / name).exists()]
