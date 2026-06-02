"""
rq5_utils.py — Trade-off desempenho vs fairness (dataset-level)

Funções:
- compute_tradeoff_data: ΔF1 (gap-based) × Δfair por dataset e método
- plot_tradeoff_quadrant: scatter com quadrantes anotados + overlay mediana/IQR
- compute_quadrant_table: contagem de datasets por quadrante e método
- identify_pareto_front: métodos não-dominados (mediana)
- plot_pareto_front: scatter summary + fronteira de Pareto
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .statistic_test import (
    aggregate_to_dataset_level,
    BASELINE_IDS,
    BASELINE_TYPICAL_LABEL,
    EXP_ORDER,
    EXP_TO_CATEGORY,
)
from .analysis_utils import (
    latex_and_save,
    _LATEX_COL_NAMES,
    _escape_latex_cell,
    _apply_multirow,
    _insert_group_separators,
    _postprocess_latex,
    _slugify,
    format_caption,
    _replace_decimal_in_latex,
    apply_comma_axes,
)

# ── Configurações visuais centralizadas ─────────────────────────────────────
_VIZ = {
    "title_fontsize": 12,
    "label_fontsize": 11,
    "tick_fontsize": 10,
    "legend_fontsize": 12,
    "quadrant_fontsize": 14,
    "quadrant_color": "#555555",
    "quadrant_alpha": 0.45,
}

# ── Paleta e markers (reutilizados de tradeoff_utils) ──────────────────────
EXP_COLORS = {
    "BLRA": "#4E79A7", "BLRU": "#6B9AC4", "BRFA": "#89B4D4", "BRFU": "#A7CEE2",
    "EDL": "#59A14F", "ERL": "#76B947",
    "IAD": "#f95a19", "IGLA": "#fa8733", "IGLU": "#fa772a",
    "IFG": "#f97f50", "IP": "#f96224", "IW": "#d04005",
}

EXP_MARKERS = {
    "BLRA": "o", "BLRU": "s", "BRFA": "^", "BRFU": "D",
    "EDL": "p", "ERL": "H",
    "IAD": "X", "IGLA": "P", "IGLU": "*", "IFG": "h", "IP": "v", "IW": "<",
}

# Quadrante labels (upper-right → counter-clockwise)
_QUADRANT_LABELS = {
    "Q1": "Win–win\n(+desemp., +fair.)",
    "Q2": "Trade-off\n(−desemp., +fair.)",
    "Q3": "Lose–lose\n(−desemp., −fair.)",
    "Q4": "Só desempenho\n(+desemp., −fair.)",
}


# ═══════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

def compute_tradeoff_data(
    df: pd.DataFrame,
    fairness_metric: str,
    ideal_values: Dict,
    scope: str | List[str] = "all",
    baseline_ids: Optional[List[str]] = None,
    exp_order: Optional[List[str]] = None,
    min_baselines: int = 3,
) -> pd.DataFrame:
    """
    Calcula ΔF1 e Δfair ao nível de dataset para análise de trade-off.

    Convenção (gap-based, positivo = melhoria):
        delta_fair = gap_baseline_fair − gap_method_fair
        delta_f1   = gap_baseline_f1   − gap_method_f1

    Returns
    -------
    DataFrame tidy com colunas:
        dataset, method, category, delta_f1, delta_fair,
        gap_baseline_fair, gap_method_fair,
        gap_baseline_f1, gap_method_f1
    """
    baseline_ids = baseline_ids or BASELINE_IDS
    exp_order = exp_order or EXP_ORDER

    # ── 1. gap matrix para fairness ──
    fair_gap = aggregate_to_dataset_level(
        df, fairness_metric, ideal_values, scope,
        baseline_ids=baseline_ids,
        analysis_type="experiment",
        min_baselines=min_baselines,
    )
    if fair_gap.empty:
        return pd.DataFrame()

    # ── 2. gap matrix para F1 ──
    f1_gap = aggregate_to_dataset_level(
        df, "f1", ideal_values, scope,
        baseline_ids=baseline_ids,
        analysis_type="experiment",
        min_baselines=min_baselines,
    )
    if f1_gap.empty:
        return pd.DataFrame()

    # ── 3. métodos disponíveis (excluindo baselines) ──
    method_cols = [
        m for m in exp_order
        if m not in baseline_ids
        and m in fair_gap.columns
        and m in f1_gap.columns
    ]
    if not method_cols:
        return pd.DataFrame()

    # ── 4. melt para formato tidy ──
    rows = []
    for method in method_cols:
        for _, fr in fair_gap.iterrows():
            ds = fr["dataset"]

            # localizar mesma row em f1_gap
            f1_row = f1_gap.loc[f1_gap["dataset"] == ds]
            if f1_row.empty:
                continue
            f1_row = f1_row.iloc[0]

            gap_bl_fair = fr[BASELINE_TYPICAL_LABEL]
            gap_m_fair = fr[method]
            gap_bl_f1 = f1_row[BASELINE_TYPICAL_LABEL]
            gap_m_f1 = f1_row[method]

            if any(pd.isna(v) for v in [gap_bl_fair, gap_m_fair, gap_bl_f1, gap_m_f1]):
                continue

            rows.append({
                "dataset": ds,
                "method": method,
                "category": EXP_TO_CATEGORY.get(method, ""),
                "delta_f1": float(gap_bl_f1 - gap_m_f1),
                "delta_fair": float(gap_bl_fair - gap_m_fair),
                "gap_baseline_fair": float(gap_bl_fair),
                "gap_method_fair": float(gap_m_fair),
                "gap_baseline_f1": float(gap_bl_f1),
                "gap_method_f1": float(gap_m_f1),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# QUADRANT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _assign_quadrant(delta_f1: float, delta_fair: float) -> str:
    """Retorna o quadrante (Q1-Q4) dado ΔF1 e Δfair."""
    if delta_f1 >= 0 and delta_fair >= 0:
        return "Q1"
    elif delta_f1 < 0 and delta_fair >= 0:
        return "Q2"
    elif delta_f1 < 0 and delta_fair < 0:
        return "Q3"
    else:  # delta_f1 >= 0 and delta_fair < 0
        return "Q4"


def compute_quadrant_table(
    tradeoff_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conta datasets por quadrante para cada método.

    Returns
    -------
    DataFrame com colunas: method, category, Q1, Q2, Q3, Q4, n, win_rate_fair
    """
    if tradeoff_df.empty:
        return pd.DataFrame()

    data = tradeoff_df.copy()
    data["quadrant"] = data.apply(
        lambda r: _assign_quadrant(r["delta_f1"], r["delta_fair"]), axis=1
    )

    counts = (
        data.groupby(["method", "category", "quadrant"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        if q not in counts.columns:
            counts[q] = 0

    counts["n"] = counts[["Q1", "Q2", "Q3", "Q4"]].sum(axis=1)
    counts["win_rate_fair"] = (counts["Q1"] + counts["Q2"]) / counts["n"]
    counts["win_rate_winwin"] = counts["Q1"] / counts["n"]

    # ordenar por EXP_ORDER
    order_map = {m: i for i, m in enumerate(EXP_ORDER)}
    counts["_order"] = counts["method"].map(order_map).fillna(99)
    counts = counts.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return counts[["method", "category", "Q1", "Q2", "Q3", "Q4", "n",
                    "win_rate_fair", "win_rate_winwin"]]


# ═══════════════════════════════════════════════════════════════════════════
# PARETO FRONT
# ═══════════════════════════════════════════════════════════════════════════

def identify_pareto_front(
    tradeoff_df: pd.DataFrame,
) -> List[str]:
    """
    Identifica métodos na fronteira de Pareto (não-dominados).

    Usa medianas de (delta_f1, delta_fair) por método.
    Um método A domina B se A.delta_f1 >= B.delta_f1 AND A.delta_fair >= B.delta_fair
    com pelo menos uma desigualdade estrita.

    Returns
    -------
    Lista de method names na fronteira de Pareto.
    """
    if tradeoff_df.empty:
        return []

    medians = (
        tradeoff_df.groupby("method")[["delta_f1", "delta_fair"]]
        .median()
        .reset_index()
    )

    pareto = []
    for i, row_i in medians.iterrows():
        dominated = False
        for j, row_j in medians.iterrows():
            if i == j:
                continue
            # j domina i?
            if (row_j["delta_f1"] >= row_i["delta_f1"] and
                row_j["delta_fair"] >= row_i["delta_fair"] and
                (row_j["delta_f1"] > row_i["delta_f1"] or
                 row_j["delta_fair"] > row_i["delta_fair"])):
                dominated = True
                break
        if not dominated:
            pareto.append(row_i["method"])

    return pareto


def _compute_method_summary(
    tradeoff_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula mediana e IQR/2 de delta_f1 e delta_fair por método."""

    def _iqr2(x):
        q1, q3 = np.nanpercentile(x, [25, 75])
        return (q3 - q1) / 2.0

    summary = (
        tradeoff_df.groupby(["method", "category"])
        .agg(
            med_f1=("delta_f1", "median"),
            iqr2_f1=("delta_f1", _iqr2),
            med_fair=("delta_fair", "median"),
            iqr2_fair=("delta_fair", _iqr2),
            n=("delta_f1", "count"),
        )
        .reset_index()
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════

def _add_quadrant_labels(
    ax: plt.Axes,
    fontsize: int | None = None,
    alpha: float | None = None,
    color: str | None = None,
) -> None:
    """Adiciona labels dos quadrantes no background.

    Os defaults são lidos de _VIZ, permitindo controle global via notebook.
    """
    fontsize = fontsize if fontsize is not None else _VIZ["quadrant_fontsize"]
    alpha = alpha if alpha is not None else _VIZ["quadrant_alpha"]
    color = color if color is not None else _VIZ["quadrant_color"]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    positions = {
        "Q1": (xlim[1], ylim[1]),    # upper-right
        "Q2": (xlim[0], ylim[1]),    # upper-left
        "Q3": (xlim[0], ylim[0]),    # lower-left
        "Q4": (xlim[1], ylim[0]),    # lower-right
    }
    ha_map = {"Q1": "right", "Q2": "left", "Q3": "left", "Q4": "right"}
    va_map = {"Q1": "top", "Q2": "top", "Q3": "bottom", "Q4": "bottom"}

    for q, (x, y) in positions.items():
        # offset slightly from corners
        xoff = -0.02 * (xlim[1] - xlim[0]) if ha_map[q] == "right" else 0.02 * (xlim[1] - xlim[0])
        yoff = -0.03 * (ylim[1] - ylim[0]) if va_map[q] == "top" else 0.03 * (ylim[1] - ylim[0])
        ax.text(
            x + xoff, y + yoff,
            _QUADRANT_LABELS[q],
            ha=ha_map[q], va=va_map[q],
            fontsize=fontsize, alpha=alpha,
            style="italic", color=color,
        )


def plot_tradeoff_quadrant(
    tradeoff_df: pd.DataFrame,
    fairness_metric: str,
    translate: Optional[Dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10.5, 7.0),
    show: bool = True,
    seed: int = 42,
) -> Optional[plt.Figure]:
    """
    Scatter ΔF1 (x) vs Δfair (y) com:
    - 1 ponto por (dataset × método), cor/marker por método
    - Overlay: mediana (◆) + IQR/2 error bars por método
    - Quadrantes anotados
    """
    translate = translate or {}
    if tradeoff_df is None or tradeoff_df.empty:
        return None

    data = tradeoff_df.dropna(subset=["delta_f1", "delta_fair"]).copy()
    if data.empty:
        return None

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize)

    # ── Pontos individuais (dataset × método) ──
    methods_present = [m for m in EXP_ORDER if m not in BASELINE_IDS and m in data["method"].unique()]

    for method in methods_present:
        sub = data[data["method"] == method]
        color = EXP_COLORS.get(method, "#999999")
        marker = EXP_MARKERS.get(method, "o")

        ax.scatter(
            sub["delta_f1"], sub["delta_fair"],
            c=color, marker=marker, s=50, alpha=0.55,
            edgecolors="white", linewidths=0.4, zorder=2,
        )

    # ── Summary overlay: mediana + IQR/2 ──
    summary = _compute_method_summary(data)

    for _, row in summary.iterrows():
        method = row["method"]
        color = EXP_COLORS.get(method, "#999999")
        marker = EXP_MARKERS.get(method, "o")

        # Mediana (marker distinto por método, borda preta)
        ax.scatter(
            [row["med_f1"]], [row["med_fair"]],
            c=color, marker=marker, s=140,
            edgecolors="black", linewidths=1.0, zorder=5,
            label=method,
        )
        # Error bars (IQR/2)
        ax.errorbar(
            row["med_f1"], row["med_fair"],
            xerr=row["iqr2_f1"], yerr=row["iqr2_fair"],
            fmt="none", ecolor=color, elinewidth=1.3,
            capsize=3, capthick=1.0, alpha=0.75, zorder=4,
        )

    # ── Linhas de quadrante ──
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=1)

    # ── Labels ──
    fair_label = translate.get(fairness_metric, fairness_metric)
    ax.set_xlabel(r"$\Delta_{\mathrm{F1}}$ (+ = melhoria)", fontsize=_VIZ["label_fontsize"])
    ax.set_ylabel(r"$\Delta_{\mathrm{fair}}$ (+ = melhoria)", fontsize=_VIZ["label_fontsize"])
    # ax.set_title(f"Trade-off desempenho $\\leftrightarrow$ fairness — {fair_label}", fontsize=_VIZ["title_fontsize"])
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])

    # ── Quadrant annotations ──
    _add_quadrant_labels(ax)

    # ── Legenda (espaçamento entre itens) ──
    ax.legend(
        bbox_to_anchor=(1.02, 0.5), loc="center left",
        title="Método (mediana)", frameon=False, fontsize=_VIZ["legend_fontsize"],
        labelspacing=1.2, handletextpad=0.8,
    )

    ax.grid(True, alpha=0.25, linestyle="--")
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


def plot_pareto_front(
    tradeoff_df: pd.DataFrame,
    fairness_metric: str,
    translate: Optional[Dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10.5, 7.0),
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot summary (mediana ± IQR) com fronteira de Pareto destacada.
    """
    translate = translate or {}
    if tradeoff_df is None or tradeoff_df.empty:
        return None

    data = tradeoff_df.dropna(subset=["delta_f1", "delta_fair"]).copy()
    if data.empty:
        return None

    summary = _compute_method_summary(data)
    pareto_methods = identify_pareto_front(data)

    fig, ax = plt.subplots(figsize=figsize)

    # ── Pontos (mediana ± IQR) ──
    for _, row in summary.iterrows():
        method = row["method"]
        color = EXP_COLORS.get(method, "#999999")
        marker = EXP_MARKERS.get(method, "o")
        is_pareto = method in pareto_methods

        edge_color = "black" if is_pareto else "gray"
        edge_width = 1.5 if is_pareto else 0.8
        size = 160 if is_pareto else 100

        ax.scatter(
            [row["med_f1"]], [row["med_fair"]],
            c=color, marker=marker, s=size,
            edgecolors=edge_color, linewidths=edge_width,
            zorder=5, label=method,
        )

        # Error bars
        ax.errorbar(
            row["med_f1"], row["med_fair"],
            xerr=row["iqr2_f1"], yerr=row["iqr2_fair"],
            fmt="none", ecolor=color, elinewidth=1.2,
            capsize=3, capthick=1.0, alpha=0.7, zorder=4,
        )

    # ── Fronteira de Pareto ──
    if len(pareto_methods) >= 2:
        pareto_pts = summary[summary["method"].isin(pareto_methods)].copy()
        # Ordenar por delta_f1 para conectar a fronteira
        pareto_pts = pareto_pts.sort_values("med_f1")
        ax.plot(
            pareto_pts["med_f1"], pareto_pts["med_fair"],
            color="black", linestyle="-", linewidth=1.5, alpha=0.5,
            zorder=3, label="_nolegend_",
        )

    # ── Linhas de quadrante ──
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=1)

    # ── Labels ──
    fair_label = translate.get(fairness_metric, fairness_metric)
    ax.set_xlabel(r"$\Delta_{\mathrm{F1}}$ (+ = melhoria)", fontsize=_VIZ["label_fontsize"])
    ax.set_ylabel(r"$\Delta_{\mathrm{fair}}$ (+ = melhoria)", fontsize=_VIZ["label_fontsize"])
    # ax.set_title(
        # f"Fronteira de Pareto — {fair_label}\n"
        # f"(Pareto: {', '.join(pareto_methods)})",
        # fontsize=_VIZ["title_fontsize"],
    # )
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])

    _add_quadrant_labels(ax)

    # ── Legenda ──
    handles, labels = ax.get_legend_handles_labels()
    # Adicionar item para Pareto na legenda
    pareto_handle = Line2D(
        [0], [0], marker="D", color="w", markerfacecolor="gray",
        markeredgecolor="black", markersize=10, linewidth=0,
        label="Pareto-ótimo",
    )
    handles.append(pareto_handle)
    labels.append("Pareto-ótimo")

    ax.legend(
        handles, labels,
        bbox_to_anchor=(1.02, 0.5), loc="center left",
        title="Método", frameon=False, fontsize=_VIZ["legend_fontsize"],
        labelspacing=1.2, handletextpad=0.8,
    )

    ax.grid(True, alpha=0.25, linestyle="--")
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


# ═══════════════════════════════════════════════════════════════════════════
# TABELA DESCRITIVA DE TRADE-OFF (RQ5)
# ═══════════════════════════════════════════════════════════════════════════

def save_tradeoff_summary_latex(
    tradeoff_df: pd.DataFrame,
    fairness_metric: str,
    save_dir: str,
    translate: Dict = None,
) -> List[Path]:
    """Gera tabelas LaTeX de resumo do trade-off ΔF1 × Δfair.

    Produz duas tabelas:
    1. Por método (com coluna de categoria): mediana/IQR de Δfair e ΔF1
    2. Por categoria: mediana/IQR de Δfair e ΔF1

    Parameters
    ----------
    tradeoff_df : DataFrame
        Saída de ``compute_tradeoff_data`` com colunas
        ``dataset, method, category, delta_f1, delta_fair``.
    fairness_metric : str
        Nome da métrica de fairness (para caption e slug do arquivo).
    save_dir : str
        Diretório raiz de saída (``tables/``).
    translate : dict, optional
        Mapeamento de tradução para português.

    Returns
    -------
    List[Path]
        Caminhos dos arquivos .tex gerados.
    """
    translate = translate or {}
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = lambda x: translate.get(x, x)
    fair_label = t(fairness_metric)
    saved_paths: List[Path] = []

    if tradeoff_df.empty:
        return saved_paths

    # ── helpers ──
    def _agg(group_df):
        return pd.Series({
            "n": len(group_df),
            "median_delta_fair": group_df["delta_fair"].median(),
            "iqr_delta_fair": group_df["delta_fair"].quantile(0.75) - group_df["delta_fair"].quantile(0.25),
            "median_delta_f1": group_df["delta_f1"].median(),
            "iqr_delta_f1": group_df["delta_f1"].quantile(0.75) - group_df["delta_f1"].quantile(0.25),
        })

    def _format_and_save(df_agg, id_cols, filename, caption, label):
        df = df_agg.copy()

        # Traduzir colunas de texto
        for col in [c for c in ["method", "category"] if c in df.columns]:
            df[col] = df[col].apply(
                lambda x: _escape_latex_cell(t(x)) if isinstance(x, str) else x
            )

        # Formatar numéricos (vírgula como separador decimal)
        for col in ["median_delta_fair", "iqr_delta_fair", "median_delta_f1", "iqr_delta_f1"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{x:.4f}".replace(".", ",") if pd.notna(x) else "—"
                )
        if "n" in df.columns:
            df["n"] = df["n"].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else "—"
            )

        # Multirow categórico
        df_pre = df.copy()
        merge_rules = [{"col": c, "context": []} for c in id_cols if c in df.columns]
        df = _apply_multirow(df, merge_rules)

        # Renomear para PT
        df = df.rename(columns=_LATEX_COL_NAMES)

        path = out_dir / filename
        caption = format_caption(caption)
        latex = df.to_latex(
            index=False, escape=False,
            caption=caption,
            position='H',
            label=label,
        )
        group_col = "category" if "category" in df_pre.columns else id_cols[0]
        latex = _insert_group_separators(latex, df_pre, df, group_col)
        latex = _postprocess_latex(latex)
        latex = _replace_decimal_in_latex(latex)
        path.write_text(latex, encoding="utf-8")
        saved_paths.append(path)

    # ── 1. Tabela por método ──
    by_method = (
        tradeoff_df
        .groupby(["method", "category"], sort=False)
        .apply(_agg)
        .reset_index()
    )
    # Ordenar: Pre-Processing primeiro, depois In-Processing
    cat_order = {"pre_processing": 0, "in_processing": 1}
    by_method["_sort"] = by_method["category"].map(cat_order).fillna(2)
    by_method = by_method.sort_values(["_sort", "method"]).drop(columns=["_sort"])

    col_order = ["method", "category", "n",
                 "median_delta_fair", "iqr_delta_fair",
                 "median_delta_f1", "iqr_delta_f1"]
    by_method = by_method[[c for c in col_order if c in by_method.columns]]

    slug = _slugify(fairness_metric)
    _format_and_save(
        by_method,
        id_cols=["category"],
        filename=f"tradeoff_method_{slug}.tex",
        caption=(
            f"Resumo de trade-off por método: {fair_label}. "
            f"$\\Delta_{{\\text{{fair}}}} > 0$ indica melhoria de \\textit{{fairness}}; "
            f"$\\Delta_{{\\text{{F1}}}} > 0$ indica melhoria de desempenho."
        ),
        label=f"tab:tradeoff_method_{slug}",
    )

    # ── 2. Tabela por categoria ──
    # Agregar primeiro por dataset (mediana dos métodos de cada categoria),
    # para que cada dataset contribua com UMA observação por categoria.
    cat_by_ds = (
        tradeoff_df
        .groupby(["category", "dataset"], sort=False)
        .agg(delta_fair=("delta_fair", "median"), delta_f1=("delta_f1", "median"))
        .reset_index()
    )
    by_category = (
        cat_by_ds
        .groupby("category", sort=False)
        .apply(_agg)
        .reset_index()
    )
    by_category["_sort"] = by_category["category"].map(cat_order).fillna(2)
    by_category = by_category.sort_values("_sort").drop(columns=["_sort"])

    col_order_cat = ["category", "n",
                     "median_delta_fair", "iqr_delta_fair",
                     "median_delta_f1", "iqr_delta_f1"]
    by_category = by_category[[c for c in col_order_cat if c in by_category.columns]]

    _format_and_save(
        by_category,
        id_cols=["category"],
        filename=f"tradeoff_category_{slug}.tex",
        caption=(
            f"Resumo de trade-off por categoria: {fair_label}. "
            f"$\\Delta_{{\\text{{fair}}}} > 0$ indica melhoria de \\textit{{fairness}}; "
            f"$\\Delta_{{\\text{{F1}}}} > 0$ indica melhoria de desempenho."
        ),
        label=f"tab:tradeoff_category_{slug}",
    )

    return saved_paths
