"""
rq1_utils.py - utilitários para RQ1 (Δfairness vs baseline)

Funções disponíveis:
- compute_delta_fairness_category: Δfairness por categoria (pre vs in)
- compute_delta_fairness_method: Δfairness por método individual
- compute_win_rate: proporção de datasets com Δfair > 0
- summarize_delta: resumo descritivo (n, mediana, IQR, min, max)
- save_delta_table: salva resumo em LaTeX
- plot_delta_fairness_category: scatter+mediana+IQR de Δfairness por categoria (horizontal)
- plot_delta_fairness_method: scatter+mediana+IQR de Δfairness por método (horizontal)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from .statistic_test import (
    aggregate_to_dataset_level,
    BASELINE_TYPICAL_LABEL,
    BASELINE_IDS,
    CATEGORY_MAPPING,
    EXP_ORDER,
    EXP_TO_CATEGORY,
)
from .analysis_utils import latex_and_save

# ── Configurações visuais centralizadas ─────────────────────────────────────
_VIZ = {
    "title_fontsize": 12,
    "label_fontsize": 11,
    "tick_fontsize": 10,
    "legend_fontsize": 9,
    # Posição da legenda: "above" | "below" | "inside"
    "legend_position": "above",
}


def _fit_ncol(fig, ax, handles, fontsize, columnspacing=1.2, handletextpad=0.4):
    """Calcula ncol máximo para que a legenda não exceda a largura do axes."""
    n = len(handles)
    if n <= 1:
        return 1
    renderer = fig.canvas.get_renderer()
    ax_width = ax.get_window_extent(renderer=renderer).width
    for ncol in range(n, 0, -1):
        leg = ax.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
            ncol=ncol, fontsize=fontsize, frameon=False,
            columnspacing=columnspacing, handletextpad=handletextpad,
        )
        fig.canvas.draw_idle()
        leg_w = leg.get_window_extent(renderer=renderer).width
        leg.remove()
        if leg_w <= ax_width * 1.02:
            return ncol
    return 1


def _place_legend(fig, ax, legend_handles, position=None):
    """Posiciona a legenda conforme _VIZ['legend_position']."""
    position = position or _VIZ.get("legend_position", "inside")
    fs = _VIZ["legend_fontsize"]
    if position == "above":
        ncol = _fit_ncol(fig, ax, legend_handles, fontsize=fs)
        nrows = -(-len(legend_handles) // ncol)
        pad = 35 + 15 * nrows
        title_obj = ax.title
        ax.set_title(title_obj.get_text(), fontsize=title_obj.get_fontsize(), pad=pad)
        ax.legend(
            handles=legend_handles, loc="lower center",
            bbox_to_anchor=(0.5, 1.02), ncol=ncol,
            fontsize=fs, frameon=False, columnspacing=1.2, handletextpad=0.4,
        )
    elif position == "below":
        ncol = _fit_ncol(fig, ax, legend_handles, fontsize=fs)
        ax.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.25), ncol=ncol,
            fontsize=fs, frameon=False, columnspacing=1.2, handletextpad=0.4,
        )
    else:
        ax.legend(handles=legend_handles, loc="best",
                  fontsize=fs)


def compute_delta_fairness_category(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    scope: str | List[str] = "all",
    baseline_ids: Optional[List[str]] = None,
    category_mapping: Optional[Dict] = None,
    min_baselines: int = 3,
) -> pd.DataFrame:
    """
    Calcula Δfairness por CATEGORIA (pre vs in) usando baseline como referência.

    Δ_fair = d_baseline - d_category
    """
    baseline_ids = baseline_ids or BASELINE_IDS
    category_mapping = category_mapping or CATEGORY_MAPPING

    gap_matrix = aggregate_to_dataset_level(
        df, metric, ideal_values, scope,
        baseline_ids=baseline_ids,
        category_mapping=category_mapping,
        analysis_type="category",
        min_baselines=min_baselines,
        return_raw_matrix=False
    )

    if gap_matrix.empty:
        return pd.DataFrame(columns=["dataset", "metric", "category", "delta_fair"])

    out = []
    for cat in ["pre_processing", "in_processing"]:
        if cat not in gap_matrix.columns:
            continue
        tmp = gap_matrix[["dataset", BASELINE_TYPICAL_LABEL, cat]].copy()
        tmp["delta_fair"] = tmp[BASELINE_TYPICAL_LABEL] - tmp[cat]
        tmp["category"] = cat
        tmp["metric"] = metric
        out.append(tmp[["dataset", "metric", "category", "delta_fair"]])

    if not out:
        return pd.DataFrame(columns=["dataset", "metric", "category", "delta_fair"])

    return pd.concat(out, ignore_index=True)


def compute_delta_fairness_method(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    scope: str | List[str] = "all",
    baseline_ids: Optional[List[str]] = None,
    exp_order: Optional[List[str]] = None,
    min_baselines: int = 3,
) -> pd.DataFrame:
    """
    Calcula Δfairness por MÉTODO (exp_id) usando baseline como referência.
    """
    baseline_ids = baseline_ids or BASELINE_IDS
    exp_order = exp_order or EXP_ORDER

    gap_matrix = aggregate_to_dataset_level(
        df, metric, ideal_values, scope,
        baseline_ids=baseline_ids,
        analysis_type="experiment",
        min_baselines=min_baselines,
        return_raw_matrix=False
    )

    if gap_matrix.empty:
        return pd.DataFrame(columns=["dataset", "metric", "method", "category", "delta_fair"])

    method_cols = [c for c in gap_matrix.columns if c not in ["dataset", BASELINE_TYPICAL_LABEL]]
    # preserve order if exp_order is provided
    method_cols = [m for m in exp_order if m in method_cols] + [m for m in method_cols if m not in exp_order]

    out = []
    for method in method_cols:
        tmp = gap_matrix[["dataset", BASELINE_TYPICAL_LABEL, method]].copy()
        tmp["delta_fair"] = tmp[BASELINE_TYPICAL_LABEL] - tmp[method]
        tmp["method"] = method
        tmp["category"] = EXP_TO_CATEGORY.get(method, "")
        tmp["metric"] = metric
        out.append(tmp[["dataset", "metric", "method", "category", "delta_fair"]])

    return pd.concat(out, ignore_index=True)


def summarize_delta(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str = "delta_fair",
) -> pd.DataFrame:
    """Resumo descritivo (n, mediana, IQR, min, max) para Δfairness."""
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["n", "median", "iqr", "min", "max"])

    def _iqr(x):
        return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

    summary = (
        df.groupby(group_cols)[value_col]
        .agg(
            n="count",
            median="median",
            iqr=_iqr,
            min="min",
            max="max",
        )
        .reset_index()
    )
    return summary


def save_delta_table(
    df: pd.DataFrame,
    group_cols: List[str],
    caption: str,
    label: str,
    result_dir: str,
    value_col: str = "delta_fair",
) -> pd.DataFrame:
    """
    Salva resumo de Δfairness em LaTeX usando o mesmo pipeline de tabelas.
    """
    summary = summarize_delta(df, group_cols=group_cols, value_col=value_col)
    latex_and_save(summary, caption=caption, label=label, result_dir=result_dir)
    return summary


# =========================================================================
# WIN RATE
# =========================================================================

def compute_win_rate(
    df: pd.DataFrame,
    group_col: str = "category",
    value_col: str = "delta_fair",
) -> pd.DataFrame:
    """
    Calcula win rate: proporção de casos (datasets) em que Δfair > 0.

    Args:
        df: DataFrame com coluna delta_fair (output de compute_delta_fairness_*)
        group_col: coluna de agrupamento ("category" ou "method")
        value_col: coluna de Δfairness

    Returns:
        DataFrame com colunas: group_col, n, wins, win_rate
    """
    if df.empty:
        return pd.DataFrame(columns=[group_col, "n", "wins", "win_rate"])

    result = (
        df.groupby(group_col)[value_col]
        .agg(
            n="count",
            wins=lambda x: (x > 0).sum(),
        )
        .reset_index()
    )
    result["win_rate"] = result["wins"] / result["n"]
    return result


# =========================================================================
# CORES E CONFIGURAÇÃO DE VISUALIZAÇÃO
# =========================================================================

_CATEGORY_COLORS = {
    "pre_processing": "#59A14F",
    "in_processing": "#F28E2B",
}

_EXP_COLORS = {
    "EDL": "#59A14F", "ERL": "#76B947",
    "IAD": "#f95a19",
    "IGLA": "#fa8733",
    "IGLU": "#fa772a",
    "IFG": "#f97f50",
    "IP": "#f96224",
    "IW": "#d04005",
}


# =========================================================================
# VISUALIZAÇÃO: Δfairness POR CATEGORIA
# =========================================================================

def plot_delta_fairness_category(
    delta_cat: pd.DataFrame,
    metric: str,
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 4),
    seed: int = 42,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Scatter + mediana (◆) + IQR whiskers de Δfairness por CATEGORIA.

    Orientação horizontal: categorias no eixo Y, Δfairness no eixo X.

    Args:
        delta_cat: output de compute_delta_fairness_category para uma métrica
        metric: nome da métrica (para título)
        translate: dicionário de tradução
        save_path: caminho para salvar (PDF/PNG)
        figsize: tamanho da figura
        seed: seed para jitter
        show: se True, exibe

    Returns:
        plt.Figure ou None se sem dados
    """
    translate = translate or {}
    order = ["pre_processing", "in_processing"]
    label_map = {"pre_processing": "Pre-processing", "in_processing": "In-processing"}

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize)

    for i, cat in enumerate(order):
        vals = delta_cat.loc[delta_cat["category"] == cat, "delta_fair"].dropna().values
        if len(vals) == 0:
            continue
        color = _CATEGORY_COLORS.get(cat, "#999999")

        # Pontos individuais (jitter vertical, valores no eixo X)
        y_jitter = rng.uniform(-0.15, 0.15, size=len(vals)) + i
        ax.scatter(
            vals, y_jitter, c=color, s=40, alpha=0.6,
            edgecolors="white", linewidths=0.5, zorder=2,
        )

        median = np.nanmedian(vals)
        q25 = np.nanpercentile(vals, 25)
        q75 = np.nanpercentile(vals, 75)

        # Mediana (diamante)
        ax.scatter(
            [median], [i], c=color, s=100, marker="D",
            edgecolors="black", linewidths=1.0, zorder=4,
        )

        # Whisker (IQR) — horizontal
        ax.errorbar(
            [median], [i],
            xerr=[[median - q25], [q75 - median]],
            fmt="none", ecolor=color, elinewidth=1.5,
            capsize=4, capthick=1.5, zorder=3,
        )

    _BASELINE_COLOR = "#4E79A7"
    ax.axvline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label_map[c] for c in order], fontsize=_VIZ["tick_fontsize"])
    ax.set_xlabel(r"$\Delta_{\mathrm{fair}}$ ($d_{\mathrm{baseline}}$ – $d_{\mathrm{método}}$)", fontsize=_VIZ["label_fontsize"])
    metric_label = translate.get(metric, metric)
    # ax.set_title(f"$\\Delta_{{\\text{{fair}}}}$ por categoria — {metric_label}", fontsize=_VIZ["title_fontsize"])

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.6, label="Conjunto de dados"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="Mediana ± IQR"),
        Line2D([0], [0], color=_BASELINE_COLOR, linestyle="--", linewidth=1,
               label="Sem efeito ($\\Delta$=0)"),
    ]
    _place_legend(fig, ax, legend_elements)
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3)

    # Separador decimal → vírgula
    from .analysis_utils import apply_comma_axes
    apply_comma_axes(ax)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =========================================================================
# VISUALIZAÇÃO: Δfairness POR MÉTODO
# =========================================================================

def plot_delta_fairness_method(
    delta_method: pd.DataFrame,
    metric: str,
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 6),
    seed: int = 42,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Scatter + mediana (◆) + IQR whiskers de Δfairness por MÉTODO individual.

    Orientação horizontal: métodos no eixo Y, Δfairness no eixo X.

    Args:
        delta_method: output de compute_delta_fairness_method para uma métrica
        metric: nome da métrica (para título)
        translate: dicionário de tradução
        save_path: caminho para salvar (PDF/PNG)
        figsize: tamanho da figura
        seed: seed para jitter
        show: se True, exibe

    Returns:
        plt.Figure ou None se sem dados
    """
    translate = translate or {}
    rng = np.random.default_rng(seed)

    # Ordem dos métodos (sem baselines), invertida para que o primeiro fique no topo
    methods_present = [m for m in EXP_ORDER
                       if m not in BASELINE_IDS
                       and m in delta_method["method"].unique()]
    if not methods_present:
        return None

    methods_rev = list(reversed(methods_present))

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods_rev):
        vals = delta_method.loc[
            delta_method["method"] == method, "delta_fair"
        ].dropna().values
        if len(vals) == 0:
            continue

        color = _EXP_COLORS.get(method, "#999999")

        # Pontos individuais (jitter vertical, valores no eixo X)
        y_jitter = rng.uniform(-0.18, 0.18, size=len(vals)) + i
        ax.scatter(
            vals, y_jitter, c=color, s=40, alpha=0.6,
            edgecolors="white", linewidths=0.5, zorder=2,
        )

        median = np.nanmedian(vals)
        q25 = np.nanpercentile(vals, 25)
        q75 = np.nanpercentile(vals, 75)

        # Mediana (diamante)
        ax.scatter(
            [median], [i], c=color, s=100, marker="D",
            edgecolors="black", linewidths=1.0, zorder=4,
        )

        # Whisker (IQR) — horizontal
        ax.errorbar(
            [median], [i],
            xerr=[[median - q25], [q75 - median]],
            fmt="none", ecolor=color, elinewidth=1.5,
            capsize=4, capthick=1.5, zorder=3,
        )

    _BASELINE_COLOR = "#4E79A7"
    ax.axvline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=1)

    # Fundo por categoria (pre vs in) — faixas horizontais
    _add_method_category_bg(ax, methods_rev)

    ax.set_yticks(range(len(methods_rev))) 
    ax.set_yticklabels(methods_rev, fontsize=_VIZ["tick_fontsize"])
    ax.set_xlabel(r"$\Delta_{\mathrm{fair}}$ ($d_{\mathrm{baseline}}$ – $d_{\mathrm{método}}$)", fontsize=_VIZ["label_fontsize"])
    metric_label = translate.get(metric, metric)
    # ax.set_title(f"$\\Delta_{{\\text{{fair}}}}$ por método — {metric_label}", fontsize=_VIZ["title_fontsize"])

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.6, label="Conjunto de dados"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="Mediana ± IQR"),
        Line2D([0], [0], color=_BASELINE_COLOR, linestyle="--", linewidth=1,
               label="Sem efeito ($\\Delta$=0)"),
    ]
    _place_legend(fig, ax, legend_elements)
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3)

    # Separador decimal → vírgula
    from .analysis_utils import apply_comma_axes
    apply_comma_axes(ax)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _add_method_category_bg(
    ax: plt.Axes, methods: List[str], alpha: float = 0.08
) -> None:
    """Adiciona fundo colorido por categoria atrás dos métodos (faixas horizontais)."""
    current_cat = None
    start_idx = 0

    for i, method in enumerate(methods + [None]):
        cat = EXP_TO_CATEGORY.get(method) if method else None
        if cat != current_cat:
            if current_cat is not None and current_cat != "baseline":
                color = _CATEGORY_COLORS.get(current_cat, "#999999")
                ax.axhspan(start_idx - 0.5, i - 0.5,
                           color=color, alpha=alpha, zorder=0)
            start_idx = i
            current_cat = cat
