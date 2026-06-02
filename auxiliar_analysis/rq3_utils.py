"""
rq3_utils.py — RQ3: A intensidade do viés afeta mitigação?

Funções de apoio para investigar se a eficácia de mitigação depende
da magnitude inicial do gap no baseline (proporcionalidade vs saturação).

PERGUNTA CENTRAL:
- Δ|gap| cresce linearmente com |gap|_baseline (proporcional)?
- Ou satura em gaps altos/baixos (teto/piso)?

BLOCOS:
1. build_intensity_scatter_data      — constrói dados (gap_baseline, delta_fair) + severidade
2. plot_intensity_scatter_category    — [DEPRECATED] scatter por categoria (pre vs in)
3. plot_intensity_scatter_method      — [DEPRECATED] scatter por método individual
4. plot_intensity_by_severity         — [DEPRECATED] scatter colorido por severidade
4b. plot_baseline_vs_mitigation            — scatter d_baseline × Δ_fair por método
4c. plot_baseline_vs_mitigation_by_dataset — dot-plot por dataset (ordenado por d_baseline)
4d. plot_baseline_vs_mitigation_faceted    — small-multiples facetado por método
5. compute_intensity_correlation_table — correlação Spearman/Pearson por grupo
6. build_intensity_summary_table      — resumo descritivo por severidade
7. plot_proportionality_ratio_boxplot  — [DEPRECATED] boxplot de Δ|gap| / |gap|_baseline
8. plot_ratio_heatmap                 — [DEPRECATED] heatmap método × métrica (cross-metric)
9. plot_ratio_strip_pooled            — [DEPRECATED] strip chart pooled cross-metric por método
10. plot_ratio_strip_faceted          — [DEPRECATED] small multiples strip chart per-metric
11. plot_severity_profile             — perfil de severidade: r_fair por faixa (line plot por métrica)
12. build_baseline_delta_latex_table  — tabela LaTeX d_baseline × Δ_fair por método e dataset

DEPENDÊNCIAS:
- classified (DataFrame): saída de classify_bias_patterns()
  Colunas necessárias: dataset, metric, severity_level, severity_label
- df (DataFrame principal): com colunas pattern_code, direction mergeadas
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats as sp_stats

from .statistic_test import (
    aggregate_to_dataset_level,
    BASELINE_TYPICAL_LABEL,
    BASELINE_IDS,
    EXP_ORDER,
    EXP_TO_CATEGORY,
)
from .rq1_utils import (
    _CATEGORY_COLORS,
    _EXP_COLORS,
    _add_method_category_bg,
)
from .analysis_utils import compute_correlation, apply_comma_axes, format_caption, _replace_decimal_in_latex


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

_SEVERITY_COLORS = {
    1: "#E15759",   # Alto (vermelho)
    2: "#EDC948",   # Médio (amarelo/dourado)
    3: "#76B7B2",   # Baixo (teal)
    0: "#BAB0AC",   # Inelegível (cinza)
}
_SEVERITY_LABELS = {1: "Alto", 2: "Médio", 3: "Baixo", 0: "Inelegível"}
_BASELINE_COLOR = "#4E79A7"
_REL_EPS = 1e-8

# ── Configurações visuais centralizadas ─────────────────────────────────────
_VIZ = {
    "title_fontsize": 12,
    "subtitle_fontsize": 10,
    "label_fontsize": 11,
    "tick_fontsize": 10,
    "legend_fontsize": 9,
    "annotation_fontsize": 8,
    "suptitle_fontsize": 12,
    "colorbar_label_fontsize": 10,
    "supxlabel_fontsize": 11,
    "facet_title_fontsize": 9,
    "facet_tick_fontsize": 9,
    "facet_suptitle_fontsize": 13,
    "severity_title_fontsize": 13,
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
        pad = 25 + 15 * nrows
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


def _place_legend_outside(
    ax: plt.Axes,
    fontsize: int = 9,
    ncol: int = 1,
    x_anchor: float = 1.02,
    extra_handles: Optional[List[Line2D]] = None,
    extra_labels: Optional[List[str]] = None,
) -> None:
    """Posiciona a legenda fora do eixo para evitar sobreposição no plot."""
    handles, labels = ax.get_legend_handles_labels()
    if extra_handles and extra_labels:
        handles = handles + list(extra_handles)
        labels = labels + list(extra_labels)
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(x_anchor, 1.0),
        borderaxespad=0.0,
        fontsize=fontsize,
        ncol=ncol,
        framealpha=0.9,
    )


def _add_outside_note(
    ax: plt.Axes,
    text: str,
    y: float,
    facecolor: str = "white",
    fontsize: int = 8,
) -> None:
    """Adiciona caixa de anotação fora do eixo (lado direito)."""
    ax.text(
        1.02,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="left",
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=facecolor, alpha=0.9),
    )


# =============================================================================
# BLOCO 1: CONSTRUÇÃO DOS DADOS PARA SCATTER
# =============================================================================

def build_intensity_scatter_data(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metrics: List[str],
    ideal_values: Dict[str, float],
    scope: str | List[str] = "all",
    level: Literal["method", "category"] = "method",
    baseline_ids: Optional[List[str]] = None,
    category_mapping: Optional[Dict] = None,
    min_baselines: int = 3,
) -> pd.DataFrame:
    """
    Constrói dados para scatter de intensidade vs mitigação.

    Para cada métrica:
    1. Chama aggregate_to_dataset_level() → gap_matrix
    2. Extrai gap_baseline = gap_matrix["baseline_typical"]
    3. Calcula delta_fair = gap_baseline − gap_method (redução absoluta)
       e mitigation_rel = delta_fair / (|gap_baseline| + ε) (fração removida)
    4. Merge com own_severity_level de *classified* (dataset × metric)

    Args:
        df: DataFrame principal (com folds × repetições).
        classified: Saída de classify_bias_patterns() (dataset × metric level).
        metrics: Lista de métricas a processar.
        ideal_values: Dict com valores ideais.
        scope: Filtro de padrão ("all", "A", "A1P", etc.).
        level: "method" (um ponto por método) ou "category" (pre vs in).
        baseline_ids: Lista de exp_ids baseline.
        category_mapping: Mapeamento categoria → exp_ids.
        min_baselines: Mínimo de baselines para formar baseline_typical.

    Returns:
        DataFrame com colunas:
            dataset, metric, method, category,
            gap_baseline, delta_fair, mitigation_rel,
            severity_level, severity_label
    """
    baseline_ids = baseline_ids or BASELINE_IDS

    # Preparar tabela de severidade (uma linha por dataset × metric)
    # Usa own_severity (severidade PRÓPRIA da métrica) em vez de severity_level
    # (que é herdada da métrica mais severa da categoria, inflando os dados).
    sev_cols_own = {
        "own_severity_level": "severity_level",
        "own_severity_label": "severity_label",
    }
    rename_map = {}
    pick_cols = ["dataset", "metric"]
    for own_col, target_col in sev_cols_own.items():
        if own_col in classified.columns:
            pick_cols.append(own_col)
            rename_map[own_col] = target_col
        elif target_col in classified.columns:
            pick_cols.append(target_col)

    severity_df = (
        classified[pick_cols]
        .drop_duplicates(subset=["dataset", "metric"])
        .rename(columns=rename_map)
    )

    all_parts: list[pd.DataFrame] = []

    for metric in metrics:
        # Escolher analysis_type conforme level
        analysis_type = "experiment" if level == "method" else "category"

        gap_matrix = aggregate_to_dataset_level(
            df,
            metric=metric,
            ideal_values=ideal_values,
            scope=scope,
            baseline_ids=baseline_ids,
            category_mapping=category_mapping,
            analysis_type=analysis_type,
            min_baselines=min_baselines,
            return_raw_matrix=False,
        )

        if gap_matrix.empty or BASELINE_TYPICAL_LABEL not in gap_matrix.columns:
            continue

        gap_baseline = gap_matrix[["dataset", BASELINE_TYPICAL_LABEL]].copy()
        gap_baseline = gap_baseline.rename(
            columns={BASELINE_TYPICAL_LABEL: "gap_baseline"}
        )

        if level == "method":
            # Colunas de métodos (excluir dataset e baseline_typical)
            method_cols = [
                c
                for c in gap_matrix.columns
                if c not in ("dataset", BASELINE_TYPICAL_LABEL)
                and c not in baseline_ids
            ]
            if not method_cols:
                continue

            # Melt → long format
            melted = gap_matrix.melt(
                id_vars="dataset",
                value_vars=method_cols,
                var_name="method",
                value_name="gap_method",
            )
            melted = melted.merge(gap_baseline, on="dataset", how="left")
            melted["delta_fair"] = melted["gap_baseline"] - melted["gap_method"]
            melted["category"] = melted["method"].map(EXP_TO_CATEGORY)

        else:  # level == "category"
            cat_cols = [
                c
                for c in ("pre_processing", "in_processing")
                if c in gap_matrix.columns
            ]
            if not cat_cols:
                continue

            melted = gap_matrix.melt(
                id_vars="dataset",
                value_vars=cat_cols,
                var_name="method",
                value_name="gap_method",
            )
            melted = melted.merge(gap_baseline, on="dataset", how="left")
            melted["delta_fair"] = melted["gap_baseline"] - melted["gap_method"]
            melted["category"] = melted["method"]  # já é pre/in

        melted["metric"] = metric

        # Merge severidade
        melted = melted.merge(severity_df, on=["dataset", "metric"], how="left")

        # Preencher NaN em severity
        if "severity_level" not in melted.columns:
            melted["severity_level"] = 0
        if "severity_label" not in melted.columns:
            melted["severity_label"] = "none"
        melted["severity_level"] = melted["severity_level"].fillna(0).astype(int)
        melted["severity_label"] = melted["severity_label"].fillna("none")

        # Mitigação relativa (fração removida), protegida contra baseline≈0
        # via epsilon no denominador (conforme documentação: r_fair = Δ_fair / (d_baseline + ε)).
        melted["mitigation_rel"] = (
            melted["delta_fair"] / (melted["gap_baseline"].abs() + _REL_EPS)
        )

        # Remover linhas com dados inválidos (NaN em gap_baseline).
        melted = melted.dropna(subset=["gap_baseline"])

        result_cols = [
            "dataset",
            "metric",
            "method",
            "category",
            "gap_baseline",
            "delta_fair",
            "mitigation_rel",
            "severity_level",
            "severity_label",
        ]
        all_parts.append(melted[[c for c in result_cols if c in melted.columns]])

    if not all_parts:
        return pd.DataFrame(
            columns=[
                "dataset", "metric", "method", "category",
                "gap_baseline", "delta_fair", "mitigation_rel",
                "severity_level", "severity_label",
            ]
        )

    return pd.concat(all_parts, ignore_index=True)


# =============================================================================
# BLOCO 2: SCATTER POR CATEGORIA (PRE VS IN)
# =============================================================================

def plot_intensity_scatter_category(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (10.3, 6),
    show: bool = True,
    seed: int = 42,
    add_regression: bool = True,
    add_correlation_box: bool = True,
) -> Optional[plt.Figure]:
    """
    Scatter |gap|_baseline vs mitigação relativa por CATEGORIA (pre vs in).

    Elementos visuais:
    - X: |gap|_baseline (contínuo)
    - Y: mitigation_rel (fração removida)
    - Pontos por categoria (verde=pre, laranja=in)
    - Linhas de referência: y=0 (sem efeito), y=1 (remoção total)
    - Regressão OLS por categoria (opcional)
    - Caixa de correlação Spearman ρ (opcional)

    Args:
        scatter_data: Output de build_intensity_scatter_data().
        metric: Nome da métrica.
        translate: Dict para tradução de labels.
        save_path: Caminho para salvar (PDF/PNG).
        figsize: Tamanho da figura.
        show: Se True, plt.show().
        seed: Seed para jitter (se necessário).
        add_regression: Se True, adiciona reta OLS por categoria.
        add_correlation_box: Se True, adiciona caixa com ρ.

    Returns:
        plt.Figure ou None se sem dados.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    data_m = scatter_data[scatter_data["metric"] == metric].copy()
    if data_m.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    cat_order = ["pre_processing", "in_processing"]
    label_map = {
        "pre_processing": "Pre-processing",
        "in_processing": "In-processing",
    }

    annotation_lines: list[str] = []

    for cat in cat_order:
        sub = data_m[data_m["category"] == cat]
        if sub.empty:
            continue

        color = _CATEGORY_COLORS.get(cat, "#999999")
        cat_label = label_map.get(cat, cat)

        ax.scatter(
            sub["gap_baseline"],
            sub["mitigation_rel"],
            c=color,
            s=78,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            label=cat_label,
            zorder=3,
        )

        # Diamante de mediana
        med_x = sub["gap_baseline"].median()
        med_y = sub["mitigation_rel"].median()
        ax.scatter(
            [med_x],
            [med_y],
            marker="D",
            s=175,
            c=color,
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
        )

        # Regressão OLS
        if add_regression:
            x = sub["gap_baseline"].values
            y = sub["mitigation_rel"].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                slope, intercept, r_value, _, _ = sp_stats.linregress(
                    x[mask], y[mask]
                )
                x_fit = np.linspace(x[mask].min(), x[mask].max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(
                    x_fit,
                    y_fit,
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                )
                annotation_lines.append(
                    f"{cat_label}: R²={r_value**2:.3f}, n={mask.sum()}"
                )

    # Linhas de referência
    ax.axhline(
        0, color=_BASELINE_COLOR, linestyle="--", linewidth=1,
        label="Sem efeito (mitigação=0)", zorder=1,
    )
    ax.axhline(
        1, color="gray", linestyle=":", linewidth=1,
        label="Remoção total (mitigação=1)", zorder=1,
    )

    reg_note_text: Optional[str] = None
    if annotation_lines:
        reg_note_text = "\n".join(annotation_lines)

    # Caixa de correlação (pooled)
    corr_note_text: Optional[str] = None
    if add_correlation_box and len(data_m) >= 4:
        x_all = data_m["gap_baseline"].values
        y_all = data_m["mitigation_rel"].values
        mask_all = np.isfinite(x_all) & np.isfinite(y_all)
        if mask_all.sum() >= 4:
            corr = compute_correlation(
                x_all[mask_all], y_all[mask_all], method="spearman"
            )
            rho = corr["spearman_r"]
            p = corr["spearman_p"]
            n = int(mask_all.sum())
            corr_note_text = f"Spearman ρ = {rho:.3f} (p={p:.3f}, n={n})"

    # Anotações fora do eixo (lado direito).
    if reg_note_text:
        _add_outside_note(ax, reg_note_text, y=0.58, facecolor="white", fontsize=_VIZ["annotation_fontsize"])
    if corr_note_text:
        _add_outside_note(
            ax, corr_note_text, y=0.26, facecolor="lightyellow", fontsize=_VIZ["annotation_fontsize"]
        )

    ax.set_xlabel(r"|gap|$_{\mathrm{baseline}}$ (distância ao ideal)", fontsize=_VIZ["label_fontsize"])
    ax.set_ylabel("Mitigação relativa (fração removida)", fontsize=_VIZ["label_fontsize"])
    ax.set_title(
        f"Intensidade do viés vs mitigação — {metric_label}", fontsize=_VIZ["title_fontsize"]
    )
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])
    _place_legend_outside(ax, fontsize=_VIZ["legend_fontsize"], ncol=1)
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout(rect=(0.0, 0.0, 0.70, 1.0))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 3: SCATTER POR MÉTODO INDIVIDUAL
# =============================================================================

def plot_intensity_scatter_method(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (13, 6),
    show: bool = True,
    seed: int = 42,
    add_regression: bool = True,
    add_correlation_box: bool = True,
) -> Optional[plt.Figure]:
    """
    Scatter |gap|_baseline vs mitigação relativa por MÉTODO individual.

    Pontos coloridos por método (usando _EXP_COLORS).
    Uma regressão global (não por método, dado N pequeno por método).

    Args:
        scatter_data: Output de build_intensity_scatter_data() (level="method").
        metric: Nome da métrica.
        translate: Dict para tradução de labels.
        save_path: Caminho para salvar.
        figsize: Tamanho da figura.
        show: Se True, plt.show().
        seed: Seed para jitter.
        add_regression: Se True, adiciona reta OLS global.
        add_correlation_box: Se True, adiciona caixa com ρ.

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    data_m = scatter_data[scatter_data["metric"] == metric].copy()
    if data_m.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Métodos na ordem padrão (sem baselines)
    methods_present = [
        m
        for m in EXP_ORDER
        if m not in BASELINE_IDS and m in data_m["method"].unique()
    ]
    marker_cycle = ["o", "s", "^", "v", "P", "X", "<", ">", "h", "8", "d", "*"]
    method_marker = {
        method: marker_cycle[i % len(marker_cycle)]
        for i, method in enumerate(methods_present)
    }

    for method in methods_present:
        sub = data_m[data_m["method"] == method]
        if sub.empty:
            continue
        color = _EXP_COLORS.get(method, "#999999")
        ax.scatter(
            sub["gap_baseline"],
            sub["mitigation_rel"],
            marker=method_marker[method],
            c=color,
            s=66,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            label=method,
            zorder=3,
        )
        # Diamante de mediana por método.
        med_x = sub["gap_baseline"].median()
        med_y = sub["mitigation_rel"].median()
        ax.scatter(
            [med_x],
            [med_y],
            marker="D",
            s=165,
            c=color,
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
        )

    # Regressão global
    if add_regression:
        x = data_m["gap_baseline"].values
        y = data_m["mitigation_rel"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3:
            slope, intercept, r_value, _, _ = sp_stats.linregress(
                x[mask], y[mask]
            )
            x_fit = np.linspace(x[mask].min(), x[mask].max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(
                x_fit,
                y_fit,
                color="black",
                linestyle="-",
                linewidth=1.5,
                alpha=0.6,
                zorder=2,
                label=f"OLS (R²={r_value**2:.3f})",
            )

    # Linhas de referência
    ax.axhline(
        0, color=_BASELINE_COLOR, linestyle="--", linewidth=1,
        label="Sem efeito (mitigação=0)", zorder=1,
    )
    ax.axhline(
        1, color="gray", linestyle=":", linewidth=1,
        label="Remoção total (mitigação=1)",
        zorder=1,
    )

    # Caixa de correlação
    corr_note_text: Optional[str] = None
    if add_correlation_box and len(data_m) >= 4:
        x_all = data_m["gap_baseline"].values
        y_all = data_m["mitigation_rel"].values
        mask_all = np.isfinite(x_all) & np.isfinite(y_all)
        if mask_all.sum() >= 4:
            corr = compute_correlation(
                x_all[mask_all], y_all[mask_all], method="spearman"
            )
            rho = corr["spearman_r"]
            p = corr["spearman_p"]
            n = int(mask_all.sum())
            corr_note_text = f"Spearman ρ = {rho:.3f} (p={p:.3f}, n={n})"

    if corr_note_text:
        _add_outside_note(ax, corr_note_text, y=0.40, facecolor="lightyellow", fontsize=_VIZ["annotation_fontsize"])

    ax.set_xlabel(r"|gap|$_{\mathrm{baseline}}$ (distância ao ideal)", fontsize=_VIZ["label_fontsize"])
    ax.set_ylabel("Mitigação relativa (fração removida)", fontsize=_VIZ["label_fontsize"])
    ax.set_title(
        f"Intensidade do viés vs mitigação por método — {metric_label}",
        fontsize=_VIZ["title_fontsize"],
    )
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])
    median_handle = Line2D(
        [],
        [],
        marker="D",
        markersize=8,
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="black",
        color="none",
    )
    _place_legend_outside(
        ax,
        fontsize=_VIZ["annotation_fontsize"],
        ncol=1,
        extra_handles=[median_handle],
        extra_labels=["Mediana (por método)"],
    )
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout(rect=(0.0, 0.0, 0.66, 1.0))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 4: SCATTER COLORIDO POR SEVERIDADE
# =============================================================================

def plot_intensity_by_severity(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (10.3, 6),
    show: bool = True,
    facet: bool = False,
) -> Optional[plt.Figure]:
    """
    Scatter |gap|_baseline vs mitigação relativa colorido/facetado por severity_level.

    Com N=7 datasets, facetas geram poucos pontos por painel.
    Por padrão (facet=False), usa scatter único com cor por severidade.

    Args:
        scatter_data: Output de build_intensity_scatter_data().
        metric: Nome da métrica.
        translate: Dict para tradução de labels.
        save_path: Caminho para salvar.
        figsize: Tamanho da figura.
        show: Se True, plt.show().
        facet: Se True, cria 3 subplots (alto|médio|baixo).

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    data_m = scatter_data[scatter_data["metric"] == metric].copy()
    # Excluir inelegíveis (severity_level == 0)
    data_m = data_m[data_m["severity_level"] > 0]
    if data_m.empty:
        return None

    sev_order = [1, 2, 3]  # alto, médio, baixo

    if facet:
        # ---- Facetado ----
        present = [s for s in sev_order if s in data_m["severity_level"].values]
        n_panels = len(present)
        if n_panels == 0:
            return None

        fig, axes = plt.subplots(
            1, n_panels, figsize=figsize, sharey=True, sharex=True
        )
        if n_panels == 1:
            axes = [axes]

        for ax, sev in zip(axes, present):
            sub = data_m[data_m["severity_level"] == sev]
            color = _SEVERITY_COLORS[sev]
            label = _SEVERITY_LABELS[sev]

            ax.scatter(
                sub["gap_baseline"],
                sub["mitigation_rel"],
                c=color,
                s=78,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

            ax.axhline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=1, zorder=1)
            ax.axhline(1, color="gray", linestyle=":", linewidth=1, zorder=1)

            n_pts = len(sub)
            ax.set_title(f"Severidade {label}\n(n={n_pts})", fontsize=_VIZ["subtitle_fontsize"])
            ax.set_xlabel(r"|gap|$_{\mathrm{baseline}}$", fontsize=_VIZ["label_fontsize"])
            ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])
            ax.xaxis.grid(True, alpha=0.3)
            ax.yaxis.grid(True, alpha=0.3)

        axes[0].set_ylabel("Mitigação relativa (fração removida)", fontsize=_VIZ["label_fontsize"])

        fig.suptitle(
            f"Intensidade vs mitigação por severidade — {metric_label}",
            fontsize=_VIZ["suptitle_fontsize"],
            y=1.02,
        )
        for a in fig.get_axes():
            apply_comma_axes(a)
        fig.tight_layout()

    else:
        # ---- Scatter único colorido ----
        fig, ax = plt.subplots(figsize=figsize)

        for sev in sev_order:
            sub = data_m[data_m["severity_level"] == sev]
            if sub.empty:
                continue
            color = _SEVERITY_COLORS[sev]
            label = f"Severidade {_SEVERITY_LABELS[sev]} (n={len(sub)})"

            ax.scatter(
                sub["gap_baseline"],
                sub["mitigation_rel"],
                c=color,
                s=78,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=label,
                zorder=3,
            )

        ax.axhline(
            0, color=_BASELINE_COLOR, linestyle="--", linewidth=1,
            label="Sem efeito (mitigação=0)", zorder=1,
        )
        ax.axhline(
            1, color="gray", linestyle=":", linewidth=1,
            label="Remoção total (mitigação=1)", zorder=1,
        )

        ax.set_xlabel(r"|gap|$_{\mathrm{baseline}}$ (distância ao ideal)", fontsize=_VIZ["label_fontsize"])
        ax.set_ylabel("Mitigação relativa (fração removida)", fontsize=_VIZ["label_fontsize"])
        ax.set_title(
            f"Intensidade vs mitigação por severidade — {metric_label}",
            fontsize=_VIZ["title_fontsize"],
        )
        ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])
        _place_legend_outside(ax, fontsize=_VIZ["legend_fontsize"], ncol=1)
        ax.xaxis.grid(True, alpha=0.3)
        ax.yaxis.grid(True, alpha=0.3)
        for a in fig.get_axes():
            apply_comma_axes(a)
        fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 4b: SCATTER CONTÍNUO — gap_baseline vs r_fair (sem discretizar severidade)
# =============================================================================

def plot_baseline_vs_mitigation(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (12, 5.5),
    show: bool = True,
) -> Optional[plt.Figure]:
    r"""Scatter: d\_baseline (X) × Δ\_fair (Y), colorido por método.

    Cada ponto = (dataset, método). Eixo X contínuo (sem faixas de
    severidade) para investigar correlação entre viés inicial e mitigação.

    Parameters
    ----------
    scatter_data : DataFrame
        Saída de :func:`build_intensity_scatter_data` (``level="method"``).
    metric : str
        Nome da métrica a plotar.
    translate : dict, optional
        Tradução de labels (métodos, métricas **e datasets**).
    save_path : str, optional
        Caminho para salvar (PDF/PNG).
    figsize : tuple
        Tamanho da figura.
    show : bool
        Se True, ``plt.show()``.

    Returns
    -------
    plt.Figure ou None se sem dados.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    data_m = scatter_data[scatter_data["metric"] == metric].copy()
    data_m = data_m.dropna(subset=["gap_baseline", "delta_fair"])
    if data_m.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    methods = [
        m for m in EXP_ORDER
        if m not in BASELINE_IDS and m in data_m["method"].unique()
    ]
    marker_cycle = ["o", "s", "^", "v", "P", "X", "<", ">", "h", "8", "d", "*"]
    legend_handles: List[Line2D] = []

    for idx, method in enumerate(methods):
        sub = data_m[data_m["method"] == method]
        if sub.empty:
            continue
        color = _EXP_COLORS.get(method, "#999999")
        glabel = translate.get(method, method)
        marker = marker_cycle[idx % len(marker_cycle)]

        ax.scatter(
            sub["gap_baseline"], sub["delta_fair"],
            marker=marker, c=color, s=72, alpha=0.80,
            edgecolors="black", linewidths=0.4, zorder=3,
        )
        legend_handles.append(
            Line2D([], [], marker=marker, color="none",
                   markerfacecolor=color, markeredgecolor="black",
                   markeredgewidth=0.4, markersize=7, label=glabel)
        )

    # ── Linhas de referência ────────────────────────────────────────────────
    ax.axhline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=1, zorder=1)
    legend_handles.append(
        Line2D([], [], color=_BASELINE_COLOR, linestyle="--",
               linewidth=1, label=r"Sem efeito ($\Delta_{\mathrm{fair}}$=0)")
    )

    # ── Correlação pooled ───────────────────────────────────────────────────
    x_all, y_all = data_m["gap_baseline"].values, data_m["delta_fair"].values
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    if mask.sum() >= 4:
        corr = compute_correlation(x_all[mask], y_all[mask], method="spearman")
        rho, p, n = corr["spearman_r"], corr["spearman_p"], int(mask.sum())
        legend_handles.append(
            Line2D([], [], color="none",
                   label=f"Spearman ρ={rho:.3f} (p={p:.3f}, n={n})")
        )

    # ── Eixos / título / legenda ────────────────────────────────────────────
    ax.set_xlabel(r"$d_{\mathrm{baseline}}$", fontsize=_VIZ["label_fontsize"])
    ax.set_ylabel(r"$\Delta_{\mathrm{fair}}$", fontsize=_VIZ["label_fontsize"])
    # ax.set_title(f"Viés baseline × mitigação absoluta — {metric_label}",
    #              fontsize=_VIZ["title_fontsize"])
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])
    _place_legend(fig, ax, legend_handles)
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_baseline_vs_mitigation_by_dataset(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (14, 5.5),
    show: bool = True,
) -> Optional[plt.Figure]:
    r"""Dot-plot: conjuntos de dados (X) × Δ\_fair (Y), colorido por método.

    O eixo X lista os conjuntos de dados ordenados por ``d_baseline``,
    com o valor entre parênteses.  Cada método é um marcador/cor distinto.

    Parameters
    ----------
    scatter_data : DataFrame
        Saída de :func:`build_intensity_scatter_data` (``level="method"``).
    metric : str
        Nome da métrica.
    translate : dict, optional
        Tradução de labels (métodos, métricas **e datasets**).
    save_path, figsize, show
        Mesmos de :func:`plot_baseline_vs_mitigation`.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    data_m = scatter_data[scatter_data["metric"] == metric].copy()
    data_m = data_m.dropna(subset=["gap_baseline", "delta_fair"])
    if data_m.empty:
        return None

    # ── Ordenar datasets por d_baseline ─────────────────────────────────────
    ds_order = (
        data_m.groupby("dataset")["gap_baseline"]
        .median()
        .sort_values()
    )
    ds_names = ds_order.index.tolist()
    ds_xpos = {ds: i for i, ds in enumerate(ds_names)}
    ds_labels = [
        f"{translate.get(ds, ds)}\n({ds_order[ds]:.3f})" for ds in ds_names
    ]

    fig, ax = plt.subplots(figsize=figsize)

    methods = [
        m for m in EXP_ORDER
        if m not in BASELINE_IDS and m in data_m["method"].unique()
    ]
    marker_cycle = ["o", "s", "^", "v", "P", "X", "<", ">", "h", "8", "d", "*"]
    legend_handles: List[Line2D] = []

    n_methods = len(methods)
    dodge_width = 0.5
    offsets = (
        np.linspace(-dodge_width / 2, dodge_width / 2, n_methods)
        if n_methods > 1 else [0.0]
    )

    for idx, method in enumerate(methods):
        sub = data_m[data_m["method"] == method]
        if sub.empty:
            continue
        color = _EXP_COLORS.get(method, "#999999")
        glabel = translate.get(method, method)
        marker = marker_cycle[idx % len(marker_cycle)]

        x_positions = [ds_xpos[ds] + offsets[idx] for ds in sub["dataset"]]

        ax.scatter(
            x_positions, sub["delta_fair"],
            marker=marker, c=color, s=72, alpha=0.80,
            edgecolors="black", linewidths=0.4, zorder=3,
        )
        legend_handles.append(
            Line2D([], [], marker=marker, color="none",
                   markerfacecolor=color, markeredgecolor="black",
                   markeredgewidth=0.4, markersize=7, label=glabel)
        )

    # ── Linhas de referência ────────────────────────────────────────────────
    ax.axhline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=1, zorder=1)
    legend_handles.append(
        Line2D([], [], color=_BASELINE_COLOR, linestyle="--",
               linewidth=1, label=r"Sem efeito ($\Delta_{\mathrm{fair}}$=0)")
    )

    # ── Eixos / título / legenda ────────────────────────────────────────────
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels(ds_labels, fontsize=_VIZ["tick_fontsize"] - 1,
                       rotation=45, ha="right")
    ax.set_ylabel(r"$\Delta_{\mathrm{fair}}$", fontsize=_VIZ["label_fontsize"])
    ax.set_xlabel(r"Conjunto de dados (ordenado por $d_{\mathrm{baseline}}$)",
                  fontsize=_VIZ["label_fontsize"])
    # ax.set_title(f"Mitigação por conjunto de dados — {metric_label}",
    #              fontsize=_VIZ["title_fontsize"])
    ax.tick_params(axis="y", labelsize=_VIZ["tick_fontsize"])
    _place_legend(fig, ax, legend_handles)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.3)
    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# Markers distintos por dataset (usados em faceted_baseline_delta)
_DS_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h"]
_DS_MARKER_COLOR = "#888888"  # cinza neutro (cor não compete com método)


def plot_baseline_vs_mitigation_faceted(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize_per_ax: Tuple = (4, 3.8),
    show: bool = True,
    ncols: int = 4,
    dataset_order: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    r"""Small-multiples (2 linhas): um painel por método.

    Cada faceta mostra ``d_baseline`` (X) × ``Δ_fair`` (Y) para um único
    método.  Cada dataset é representado por um **marker distinto**
    preenchido em cinza neutro, evitando sobreposição de rótulos textuais.
    Uma legenda horizontal abaixo do título mapeia marker → dataset.

    Parameters
    ----------
    scatter_data : DataFrame
        Saída de :func:`build_intensity_scatter_data` (``level="method"``).
    metric : str
        Nome da métrica.
    translate : dict, optional
        Tradução de labels (métodos, métricas **e datasets**).
    save_path : str, optional
        Caminho para salvar (PDF/PNG).
    figsize_per_ax : tuple
        Largura × altura de cada faceta individual.
    show : bool
        Se True, ``plt.show()``.
    ncols : int
        Colunas no grid (default 4 → 2 linhas para 8 métodos).
    dataset_order : list[str], optional
        Ordem dos datasets na legenda.  Se None, ordena alfabeticamente.
    """
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D

    translate = translate or {}
    metric_label = translate.get(metric, metric)

    data_m = scatter_data[scatter_data["metric"] == metric].copy()
    data_m = data_m.dropna(subset=["gap_baseline", "delta_fair"])
    if data_m.empty:
        return None

    methods = [
        m for m in EXP_ORDER
        if m not in BASELINE_IDS and m in data_m["method"].unique()
    ]
    if not methods:
        return None

    # Mapear dataset → marker
    all_ds = sorted(data_m["dataset"].unique())
    if dataset_order is not None:
        all_ds = [d for d in dataset_order if d in set(all_ds)]
    ds_marker_map = {
        ds: _DS_MARKERS[i % len(_DS_MARKERS)] for i, ds in enumerate(all_ds)
    }

    n = len(methods)
    nrows = -(-n // ncols)  # ceil division
    fw = figsize_per_ax[0] * ncols
    fh = figsize_per_ax[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False,
                             sharex=True, sharey=True)

    # Limites globais para eixos consistentes
    x_min, x_max = data_m["gap_baseline"].min(), data_m["gap_baseline"].max()
    y_min, y_max = data_m["delta_fair"].min(), data_m["delta_fair"].max()
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15

    # Fontes grandes
    _title_fs = 13
    _tick_fs = 11
    _label_fs = 16

    for idx, method in enumerate(methods):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = data_m[data_m["method"] == method].copy()
        color = _EXP_COLORS.get(method, "#999999")
        glabel = translate.get(method, method)

        # Plotar cada dataset com marker distinto
        for _, rd in sub.iterrows():
            mk = ds_marker_map.get(rd["dataset"], "o")
            ax.scatter(
                rd["gap_baseline"], rd["delta_fair"],
                marker=mk, s=72,
                facecolors=color, edgecolors="black",
                linewidths=0.5, alpha=0.90, zorder=3,
            )

        ax.axhline(0, color=_BASELINE_COLOR, linestyle="--",
                   linewidth=0.8, zorder=1)
        ax.set_title(glabel, fontsize=_title_fs, fontweight="bold")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.tick_params(axis="both", labelsize=_tick_fs)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.grid(True, alpha=0.3)
        ax.yaxis.grid(True, alpha=0.3)

    # Remover facetas vazias
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.supxlabel(r"$d_{\mathrm{baseline}}$", fontsize=_label_fs)
    fig.supylabel(r"$\Delta_{\mathrm{fair}}$", fontsize=_label_fs)
    # fig.suptitle(f"Viés baseline × mitigação por método — {metric_label}",
    #              fontsize=_title_fs + 2, y=1.0)

    # Legenda de datasets (markers em cinza) — centrada abaixo do título
    legend_handles = [
        Line2D(
            [], [], marker=ds_marker_map[ds], color="none",
            markerfacecolor=_DS_MARKER_COLOR, markeredgecolor="black",
            markeredgewidth=0.5, markersize=7,
            label=translate.get(ds, ds),
        )
        for ds in all_ds
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(all_ds), fontsize=12,
        frameon=False,
        handletextpad=0.3, columnspacing=1.0,
    )

    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout(rect=[0, 0, 1, 0.94])  # espaço para legenda + título

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =============================================================================
# BLOCO 5: TABELA DE CORRELAÇÃO
# =============================================================================

def compute_intensity_correlation_table(
    scatter_data: pd.DataFrame,
    group_col: Literal["category", "method"] = "category",
    translate: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Calcula correlação entre |gap|_baseline e mitigação relativa por grupo.

    ALERTA: Com N=7 datasets, correlações têm poder limitado.
    Focar em tamanho de efeito (ρ, r), não apenas p-valores.

    Args:
        scatter_data: Output de build_intensity_scatter_data().
        group_col: "category" (pre vs in) ou "method" (EDL, IAD, ...).
        translate: Dict para tradução de labels.

    Returns:
        DataFrame com colunas:
            metric, <group_col>, n, spearman_rho, spearman_p,
            pearson_r, pearson_p, interpretacao_N
    """
    translate = translate or {}
    results: list[dict] = []

    for (metric, group), sub in scatter_data.groupby(["metric", group_col]):
        mask = sub[["gap_baseline", "mitigation_rel"]].notna().all(axis=1)
        x = sub.loc[mask, "gap_baseline"].values
        y = sub.loc[mask, "mitigation_rel"].values
        n = len(x)

        if n < 3:
            continue

        corr = compute_correlation(x, y, method="both")

        results.append(
            {
                "metric": translate.get(str(metric), str(metric)),
                group_col: translate.get(str(group), str(group)),
                "n": n,
                "spearman_rho": round(corr["spearman_r"], 4),
                "spearman_p": round(corr["spearman_p"], 4),
                "pearson_r": round(corr["pearson_r"], 4),
                "pearson_p": round(corr["pearson_p"], 4),
            }
        )

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame(results)

    # Interpretação do N
    def _interpret_n(n: int) -> str:
        if n < 5:
            return "N<5: cautela extrema"
        if n < 10:
            return "N<10: focar em efeito"
        return "N adequado"

    df_result["interpretacao_N"] = df_result["n"].apply(_interpret_n)

    return df_result


# =============================================================================
# BLOCO 6: TABELA RESUMO POR SEVERIDADE
# =============================================================================

def build_intensity_summary_table(
    scatter_data: pd.DataFrame,
    group_col: Literal["severity_level", "category", "method"] = "severity_level",
    translate: Dict[str, str] = None,
    by_severity: bool = False,
) -> pd.DataFrame:
    """
    Tabela resumo descritiva por grupo.

    Para cada grupo:
    - n: número de observações
    - median_gap_baseline: mediana de |gap|_baseline
    - median_delta_fair: mediana de Δfair
    - iqr_delta_fair: IQR de Δfair
    - median_r_fair: mediana de r_fair = Δfair / (d_baseline + ε)
    - iqr_r_fair: IQR de r_fair
    - win_rate_pct: % de casos onde Δfair > 0 (melhora)

    Args:
        scatter_data: Output de build_intensity_scatter_data().
        group_col: Coluna de agrupamento principal.
        translate: Dict para tradução de labels.
        by_severity: Se True, adiciona severity_level como dimensão extra
            de agrupamento (ex: metric × category × severity).

    Returns:
        DataFrame pronto para display/LaTeX.
    """
    translate = translate or {}

    data = scatter_data.copy()

    # Ratio = mitigação relativa (fração removida), com epsilon no denominador.
    if "mitigation_rel" in data.columns:
        data["ratio"] = data["mitigation_rel"]
    else:
        data["ratio"] = data["delta_fair"] / (data["gap_baseline"].abs() + _REL_EPS)

    # Definir colunas de agrupamento
    group_keys = ["metric", group_col]
    if by_severity and group_col != "severity_level" and "severity_level" in data.columns:
        group_keys.append("severity_level")

    results: list[dict] = []

    for key_vals, sub in data.groupby(group_keys):
        # Desempacotar chaves (2 ou 3 dependendo de by_severity)
        if len(group_keys) == 3:
            metric, group, sev = key_vals
        else:
            metric, group = key_vals
            sev = None

        n = len(sub)
        if n == 0:
            continue

        # Label traduzido do grupo principal
        if group_col == "severity_level":
            group_label = _SEVERITY_LABELS.get(int(group), str(group))
        else:
            group_label = translate.get(str(group), str(group))

        q75_delta = sub["delta_fair"].quantile(0.75)
        q25_delta = sub["delta_fair"].quantile(0.25)
        q75_ratio = sub["ratio"].quantile(0.75)
        q25_ratio = sub["ratio"].quantile(0.25)

        row = {
            "metric": translate.get(str(metric), str(metric)),
            group_col: group_label,
        }

        # Coluna de severidade (quando by_severity=True)
        if sev is not None:
            row["severity_level"] = _SEVERITY_LABELS.get(int(sev), str(sev))

        row.update({
            "n": n,
            "median_gap_baseline": round(float(sub["gap_baseline"].median()), 4),
            "median_delta_fair": round(float(sub["delta_fair"].median()), 4),
            "iqr_delta_fair": round(float(q75_delta - q25_delta), 4),
            "median_r_fair": round(float(sub["ratio"].median()), 4)
            if sub["ratio"].notna().any()
            else np.nan,
            "iqr_r_fair": round(float(q75_ratio - q25_ratio), 4)
            if sub["ratio"].notna().any()
            else np.nan,
            "win_rate_pct": round(
                float((sub["delta_fair"] > 0).sum() / n * 100), 1
            ),
        })

        results.append(row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# =============================================================================
# BLOCO 7: BOXPLOT DE PROPORCIONALIDADE
# =============================================================================

def plot_proportionality_ratio_boxplot(
    scatter_data: pd.DataFrame,
    group_col: Literal["category", "method"] = "category",
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (10.3, 5),
    show: bool = True,
    seed: int = 42,
) -> Optional[plt.Figure]:
    """
    Boxplot de (Δ|gap| / |gap|_baseline) por grupo.

    Linha de referência em ratio=1 (redução proporcional perfeita).

    Interpretação:
    - ratio ≈ 1: método reduz gap proporcionalmente ao gap inicial
    - ratio < 1: sub-proporcional (gaps altos são mais difíceis)
    - ratio > 1: sobre-proporcional (gaps altos são mais fáceis — raro)

    Args:
        scatter_data: Output de build_intensity_scatter_data().
        group_col: "category" ou "method".
        translate: Dict para tradução de labels.
        save_path: Caminho para salvar.
        figsize: Tamanho da figura.
        show: Se True, plt.show().
        seed: Seed para jitter.

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}
    rng = np.random.default_rng(seed)

    data = scatter_data.copy()

    # Ratio protegido (fração removida), com epsilon no denominador.
    if "mitigation_rel" in data.columns:
        data["ratio"] = data["mitigation_rel"]
    else:
        data["ratio"] = data["delta_fair"] / (data["gap_baseline"].abs() + _REL_EPS)
    data = data.dropna(subset=["ratio"])
    # Filtrar ratios extremos para visualização legível
    data = data[np.isfinite(data["ratio"])]

    if data.empty:
        return None

    # Determinar grupos e ordem
    if group_col == "category":
        groups = [c for c in ["pre_processing", "in_processing"]
                  if c in data[group_col].values]
    elif group_col == "method":
        groups = [
            m
            for m in EXP_ORDER
            if m not in BASELINE_IDS and m in data[group_col].values
        ]
    else:
        groups = sorted(data[group_col].dropna().unique().tolist())

    if not groups:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    box_data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []

    for g in groups:
        sub = data[data[group_col] == g]
        if sub.empty:
            continue
        vals = sub["ratio"].values
        box_data.append(vals)
        labels.append(translate.get(str(g), str(g)))
        if group_col == "category":
            colors.append(_CATEGORY_COLORS.get(g, "#999999"))
        else:
            colors.append(_EXP_COLORS.get(g, "#999999"))

    if not box_data:
        plt.close(fig)
        return None

    bp = ax.boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Jitter points
    for i, (vals, color) in enumerate(zip(box_data, colors), start=1):
        x_jitter = rng.uniform(-0.15, 0.15, size=len(vals)) + i
        ax.scatter(
            x_jitter,
            vals,
            s=40,
            alpha=0.6,
            c=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Linha de referência: ratio = 1
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Proporcional perfeita (ratio=1)",
        zorder=1,
    )
    # Linha de referência: ratio = 0 (sem efeito)
    ax.axhline(
        0,
        color=_BASELINE_COLOR,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Sem efeito (ratio=0)",
        zorder=1,
    )

    ax.set_ylabel(r"$\Delta$ |gap| / |gap|$_{\mathrm{baseline}}$", fontsize=_VIZ["label_fontsize"])
    ax.set_xlabel("")
    ax.set_title("Proporcionalidade da mitigação", fontsize=_VIZ["title_fontsize"])
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])
    _place_legend_outside(ax, fontsize=_VIZ["legend_fontsize"], ncol=1)
    ax.yaxis.grid(True, alpha=0.3)

    # Fundo por categoria (se group_col == "method")
    if group_col == "method":
        _add_method_category_bg(ax, groups)

    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 8: HEATMAP MÉTODO × MÉTRICA  (mediana de mitigation_rel)
# =============================================================================

def plot_ratio_heatmap(
    scatter_data: pd.DataFrame,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (12, 5),
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Heatmap de método × métrica mostrando mediana de mitigation_rel.

    Cada célula = mediana da razão (Δ|gap| / |gap|_baseline) para aquele
    (método, métrica), agregada sobre os 6 datasets.

    Interpretação:
    - ≈ 1.0 → proporcional perfeita (remove 100% do gap)
    - 0 < ratio < 1 → sub-proporcional (remove fração do gap)
    - ratio > 1 → sobre-proporcional (reduz mais que o gap original)
    - ratio < 0 → piora (aumentou o gap)

    Args:
        scatter_data: Output de build_intensity_scatter_data() (level="method").
        translate: Dict para tradução de nomes de métricas.
        save_path: Caminho para salvar (PDF/PNG).
        figsize: Tamanho da figura.
        show: Se True, plt.show().

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}

    data = scatter_data.dropna(subset=["mitigation_rel"]).copy()
    if data.empty:
        return None

    # Agregar: mediana por (método, métrica)
    agg = (
        data.groupby(["method", "metric"])["mitigation_rel"]
        .median()
        .reset_index()
    )

    # Métodos na ordem padrão (sem baselines)
    methods = [
        m for m in EXP_ORDER
        if m not in BASELINE_IDS and m in agg["method"].unique()
    ]
    # Métricas presentes (manter ordem original do dataset)
    metrics_present = [
        m for m in data["metric"].unique()
        if m in agg["metric"].unique()
    ]

    if not methods or not metrics_present:
        return None

    # Pivot → matriz (métodos × métricas)
    pivot = agg.pivot(index="method", columns="metric", values="mitigation_rel")
    pivot = pivot.reindex(index=methods, columns=metrics_present)
    mat = pivot.values  # shape (n_methods, n_metrics)

    # Labels traduzidos para métricas
    metric_labels = [translate.get(m, m) for m in metrics_present]

    fig, ax = plt.subplots(figsize=figsize)

    # Limites simétricos centrados em 0
    vmax = np.nanmax(np.abs(mat))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(
        mat, aspect="auto", cmap="RdYlGn",
        vmin=-vmax, vmax=vmax,
    )

    # Anotar valores
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if pd.notna(val):
                text_color = (
                    "white"
                    if abs(val) > vmax * 0.65
                    else "black"
                )
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=_VIZ["facet_tick_fontsize"], fontweight="bold", color=text_color,
                )

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=_VIZ["facet_tick_fontsize"], rotation=35, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=_VIZ["tick_fontsize"])

    # Background de categoria nos y-labels
    _add_method_category_bg(ax, methods)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(
        r"Mediana mitigação relativa ($\Delta$|gap| / |gap|$_{\mathrm{baseline}}$)",
        fontsize=_VIZ["colorbar_label_fontsize"],
    )

    ax.set_title(
        "Mitigação relativa por método e métrica",
        fontsize=_VIZ["title_fontsize"],
    )

    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 9: STRIP CHART POOLED (mitigation_rel cross-metric)
# =============================================================================

def plot_ratio_strip_pooled(
    scatter_data: pd.DataFrame,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (9, 6),
    show: bool = True,
    seed: int = 42,
) -> Optional[plt.Figure]:
    """
    Strip chart horizontal de mitigation_rel pooled cross-metric por método.

    Para cada método: ~42 pontos (6 datasets × 7 métricas).
    Pontos individuais com jitter + diamante de mediana + whiskers IQR.

    Args:
        scatter_data: Output de build_intensity_scatter_data() (level="method").
        translate: Dict para tradução de labels.
        save_path: Caminho para salvar (PDF/PNG).
        figsize: Tamanho da figura.
        show: Se True, plt.show().
        seed: Seed para jitter.

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}
    rng = np.random.default_rng(seed)

    data = scatter_data.dropna(subset=["mitigation_rel"]).copy()
    if data.empty:
        return None

    # Métodos na ordem padrão (sem baselines), invertidos para top→bottom
    methods_present = [
        m for m in EXP_ORDER
        if m not in BASELINE_IDS and m in data["method"].unique()
    ]
    if not methods_present:
        return None

    methods_rev = list(reversed(methods_present))

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods_rev):
        vals = data.loc[
            data["method"] == method, "mitigation_rel"
        ].dropna().values
        if len(vals) == 0:
            continue

        color = _EXP_COLORS.get(method, "#999999")

        # Pontos individuais (jitter vertical, valores no eixo X)
        y_jitter = rng.uniform(-0.22, 0.22, size=len(vals)) + i
        ax.scatter(
            vals, y_jitter, c=color, s=36, alpha=0.55,
            edgecolors="white", linewidths=0.4, zorder=2,
        )

        median = np.nanmedian(vals)
        q25 = np.nanpercentile(vals, 25)
        q75 = np.nanpercentile(vals, 75)

        # Mediana (diamante)
        ax.scatter(
            [median], [i], c=color, s=110, marker="D",
            edgecolors="black", linewidths=1.0, zorder=4,
        )

        # Whisker (IQR) — horizontal
        ax.errorbar(
            [median], [i],
            xerr=[[median - q25], [q75 - median]],
            fmt="none", ecolor=color, elinewidth=1.5,
            capsize=4, capthick=1.5, zorder=3,
        )

    # Linhas de referência
    ax.axvline(
        0, color=_BASELINE_COLOR, linestyle="--", linewidth=1,
        label="Sem efeito (ratio=0)", zorder=1,
    )
    ax.axvline(
        1.0, color="red", linestyle=":", linewidth=1.5,
        label="Remoção total (ratio=1)", zorder=1,
    )

    # Fundo por categoria
    _add_method_category_bg(ax, methods_rev)

    ax.set_yticks(range(len(methods_rev)))
    ax.set_yticklabels(methods_rev, fontsize=_VIZ["tick_fontsize"])
    ax.set_xlabel(
        r"Mitigação relativa ($\Delta$|gap| / |gap|$_{\mathrm{baseline}}$)",
        fontsize=_VIZ["label_fontsize"],
    )
    ax.set_title(
        "Distribuição da mitigação relativa por método (pooled cross-metric)",
        fontsize=_VIZ["title_fontsize"],
    )
    ax.tick_params(axis="x", labelsize=_VIZ["tick_fontsize"])

    # Legenda
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.6, label="Dataset × métrica"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="Mediana ± IQR"),
        Line2D([0], [0], color=_BASELINE_COLOR, linestyle="--", linewidth=1,
               label="Sem efeito (ratio=0)"),
        Line2D([0], [0], color="red", linestyle=":", linewidth=1.5,
               label="Remoção total (ratio=1)"),
    ]
    _place_legend(fig, ax, legend_elements)

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3)
    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 10: STRIP CHART FACETADO POR MÉTRICA (small multiples)
# =============================================================================

def plot_ratio_strip_faceted(
    scatter_data: pd.DataFrame,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (20, 6),
    show: bool = True,
    seed: int = 42,
) -> Optional[plt.Figure]:
    """
    Small multiples de strip charts: 1 painel por métrica, métodos no eixo Y.

    Para cada painel/método: ~6 pontos (1 por dataset).
    Pontos individuais + diamante de mediana (sem whiskers, N=6 é pouco para IQR).

    Args:
        scatter_data: Output de build_intensity_scatter_data() (level="method").
        translate: Dict para tradução de nomes de métricas.
        save_path: Caminho para salvar (PDF/PNG).
        figsize: Tamanho da figura.
        show: Se True, plt.show().
        seed: Seed para jitter.

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}
    rng = np.random.default_rng(seed)

    data = scatter_data.dropna(subset=["mitigation_rel"]).copy()
    if data.empty:
        return None

    # Métricas presentes
    metrics_present = sorted(data["metric"].unique())
    n_metrics = len(metrics_present)
    if n_metrics == 0:
        return None

    # Métodos na ordem padrão, invertidos (top→bottom)
    methods_present = [
        m for m in EXP_ORDER
        if m not in BASELINE_IDS and m in data["method"].unique()
    ]
    if not methods_present:
        return None
    methods_rev = list(reversed(methods_present))

    fig, axes = plt.subplots(
        1, n_metrics, figsize=figsize, sharey=True,
    )
    if n_metrics == 1:
        axes = [axes]

    for ax_idx, metric in enumerate(metrics_present):
        ax = axes[ax_idx]
        data_m = data[data["metric"] == metric]

        for i, method in enumerate(methods_rev):
            vals = data_m.loc[
                data_m["method"] == method, "mitigation_rel"
            ].dropna().values
            if len(vals) == 0:
                continue

            color = _EXP_COLORS.get(method, "#999999")

            # Pontos individuais (jitter vertical)
            y_jitter = rng.uniform(-0.20, 0.20, size=len(vals)) + i
            ax.scatter(
                vals, y_jitter, c=color, s=30, alpha=0.6,
                edgecolors="white", linewidths=0.3, zorder=2,
            )

            # Mediana (diamante) — sem whiskers dado N≈6
            median = np.nanmedian(vals)
            ax.scatter(
                [median], [i], c=color, s=80, marker="D",
                edgecolors="black", linewidths=0.8, zorder=4,
            )

        # Linhas de referência
        ax.axvline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(1.0, color="red", linestyle=":", linewidth=1.0, zorder=1)

        metric_label = translate.get(metric, metric)
        ax.set_title(metric_label, fontsize=_VIZ["facet_title_fontsize"])
        ax.tick_params(axis="x", labelsize=_VIZ["facet_tick_fontsize"])
        ax.xaxis.grid(True, alpha=0.3)
        ax.yaxis.grid(False)

        if ax_idx == 0:
            ax.set_yticks(range(len(methods_rev)))
            ax.set_yticklabels(methods_rev, fontsize=_VIZ["facet_tick_fontsize"])
            # Background de categoria apenas no primeiro painel
            _add_method_category_bg(ax, methods_rev)
        else:
            _add_method_category_bg(ax, methods_rev)

    # Xlabel compartilhado
    fig.supxlabel(
        r"Mitigação relativa ($\Delta$|gap| / |gap|$_{\mathrm{baseline}}$)",
        fontsize=_VIZ["supxlabel_fontsize"],
    )
    fig.suptitle(
        "Mitigação relativa por método e métrica",
        fontsize=_VIZ["facet_suptitle_fontsize"], y=1.02,
    )

    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 11: PERFIL DE SEVERIDADE (line plot por métrica)
# =============================================================================

# Ordem canônica das severidades no eixo X (esquerda → direita)
_SEV_ORDER = [0, 3, 2, 1]  # Inelegível, Baixa, Média, Alta
_SEV_TICK_LABEL = {0: "Inelegível (0)", 3: "Baixa (3)", 2: "Média (2)", 1: "Alta (1)"}

# Marcadores por método para distinguir linhas sobrepostas
_METHOD_MARKERS = {
    "EDL": "o",
    "IAD": "s",
    "IGLA": "^",
    "IGLU": "v",
    "IFG": "P",
    "IP": "X",
    "IW": "D",
}


def plot_severity_profile(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 5),
    show: bool = True,
    seed: int = 42,
    level: Literal["method", "category"] = "method",
) -> Optional[plt.Figure]:
    """
    Perfil de severidade: r_fair (mediana) por faixa de severidade, para uma métrica.

    Para cada método (ou categoria), plota uma linha conectando a mediana de
    mitigation_rel nas faixas de severidade (baixa → média → alta).
    Pontos individuais (datasets) são mostrados com jitter horizontal.

    Interpretação direta:
    - Linha ascendente  → proporcionalidade (remove fração maior quando viés é severo)
    - Linha plana       → indiferença (fração removida constante)
    - Linha que achata  → saturação (teto de eficácia relativa)

    Args:
        scatter_data: Output de build_intensity_scatter_data().
        metric: Nome da métrica a plotar.
        translate: Dict para tradução de nomes.
        save_path: Caminho para salvar (PDF/PNG).
        figsize: Tamanho da figura.
        show: Se True, plt.show().
        seed: Seed para jitter.
        level: "method" (uma linha por método) ou "category" (pre vs in).

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}
    rng = np.random.default_rng(seed)

    data = scatter_data[scatter_data["metric"] == metric].copy()
    data = data.dropna(subset=["mitigation_rel"])

    if data.empty:
        return None

    # Determinar grupos (métodos ou categorias)
    if level == "method":
        group_col = "method"
        groups = [
            m for m in EXP_ORDER
            if m not in BASELINE_IDS and m in data[group_col].unique()
        ]
        color_map = _EXP_COLORS
        marker_map = _METHOD_MARKERS
    else:
        group_col = "category"
        groups = [
            c for c in ["pre_processing", "in_processing"]
            if c in data[group_col].unique()
        ]
        color_map = _CATEGORY_COLORS
        marker_map = {c: "o" for c in groups}

    _cat_label_map = {"pre_processing": "Pre-processing", "in_processing": "In-processing"}

    if not groups:
        return None

    # Faixas de severidade presentes (na ordem canônica, posições consecutivas)
    present = set(data["severity_level"].unique())
    sev_levels_present = [s for s in _SEV_ORDER if s in present]
    sev_xpos = {s: i for i, s in enumerate(sev_levels_present)}

    fig, ax = plt.subplots(figsize=figsize)

    # Dodge horizontal: afastar grupos para evitar sobreposição
    n_groups = len(groups)
    dodge_width = 0.12  # largura total do dodge
    dodge_offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n_groups) if n_groups > 1 else [0.0]

    for g_idx, group in enumerate(groups):
        sub = data[data[group_col] == group]
        color = color_map.get(group, "#999999")
        marker = marker_map.get(group, "o")
        dodge = dodge_offsets[g_idx]

        # Medianas por faixa
        medians = {}
        for sev in sev_levels_present:
            sev_sub = sub[sub["severity_level"] == sev]
            if sev_sub.empty:
                continue

            x_pos = sev_xpos[sev] + dodge
            vals = sev_sub["mitigation_rel"].values

            # Pontos individuais com jitter horizontal
            x_jitter = rng.uniform(-0.06, 0.06, size=len(vals)) + x_pos
            ax.scatter(
                x_jitter, vals,
                c=color, s=30, alpha=0.35,
                edgecolors="white", linewidths=0.3, zorder=2,
                marker=marker,
            )

            medians[x_pos] = np.nanmedian(vals)

        # Conectar medianas com linha
        if len(medians) >= 2:
            xs = sorted(medians.keys())
            ys = [medians[x] for x in xs]
            ax.plot(
                xs, ys,
                color=color, linewidth=2, alpha=0.8, zorder=3,
            )

        # Plotar medianas como marcadores maiores
        for x_pos, med_val in medians.items():
            ax.scatter(
                [x_pos], [med_val],
                c=color, s=100, marker=marker,
                edgecolors="black", linewidths=0.8, zorder=4,
                label=_cat_label_map.get(group, group) if x_pos == min(medians.keys()) else None,
            )

    # Linhas de referência
    ax.axhline(
        0, color=_BASELINE_COLOR, linestyle="--", linewidth=1,
        alpha=0.7, zorder=1, label="Sem efeito ($r=0$)",
    )
    ax.axhline(
        1.0, color="#E15759", linestyle=":", linewidth=1.5,
        alpha=0.7, zorder=1, label="Remoção total ($r=1$)",
    )

    # Eixo X — apenas categorias presentes, posições consecutivas
    xtick_positions = list(range(len(sev_levels_present)))
    xtick_labels = [_SEV_TICK_LABEL[s] for s in sev_levels_present]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=_VIZ["tick_fontsize"])
    ax.set_xlabel("Severidade do viés no baseline", fontsize=_VIZ["label_fontsize"])

    # Separação visual entre inelegível e elegíveis (só se ambos existirem)
    if 0 in present and len(present) > 1 and 0 in sev_xpos:
        ax.axvline(sev_xpos[0] + 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4, zorder=0)

    # Eixo Y
    ax.set_ylabel(
        r"$r_{\mathrm{fair}}$ (mitigação relativa)", fontsize=_VIZ["label_fontsize"],
    )

    # Título
    metric_label = translate.get(metric, metric)
    ax.set_title(metric_label, fontsize=_VIZ["severity_title_fontsize"])

    # Grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    ax.tick_params(axis="y", labelsize=_VIZ["tick_fontsize"])

    # Legenda
    _place_legend_outside(ax, fontsize=_VIZ["legend_fontsize"], ncol=1)

    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# BLOCO 12: TABELA LATEX  d_baseline × Δ_fair  POR MÉTODO E DATASET
# =============================================================================

def build_baseline_delta_latex_table(
    scatter_data: pd.DataFrame,
    metric: str,
    translate: Dict[str, str] = None,
    result_dir: Optional[str] = None,
    dataset_order: Optional[List[str]] = None,
) -> str:
    """
    Gera tabela LaTeX com ``d_{baseline}`` e ``\\Delta_{fair}`` por método
    e conjunto de dados, para uma métrica específica.

    Estrutura:
        Método (multirow) | Conjunto de dados | d_baseline | Δ_fair

    Linhas ordenadas por ``d_baseline`` (decrescente) dentro de cada método.

    Args:
        scatter_data: Saída de ``build_intensity_scatter_data()``.
        metric: Métrica a filtrar (chave interna, ex. ``"disparate_impact"``).
        translate: Dict para tradução de labels (métodos e datasets).
        result_dir: Se informado, salva o ``.tex`` neste diretório.
        dataset_order: Ordem fixa dos datasets (opcional; senão ordena
            por ``d_baseline`` dentro de cada método).

    Returns:
        String com o conteúdo LaTeX completo da tabela.
    """
    translate = translate or {}

    data = scatter_data.loc[scatter_data["metric"] == metric].copy()
    data = data.dropna(subset=["gap_baseline", "delta_fair"])

    if data.empty:
        return ""

    # Labels traduzidos
    data["method_label"] = data["method"].map(
        lambda m: translate.get(m, m)
    )
    data["dataset_label"] = data["dataset"].map(
        lambda d: translate.get(d, d)
    )

    # Ordenação: método (alfabético pelo label) → d_baseline decrescente
    data = data.sort_values(
        ["method_label", "gap_baseline"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Calcular r_fair se não existir
    if "mitigation_rel" in data.columns:
        data["r_fair"] = data["mitigation_rel"]
    else:
        data["r_fair"] = data["delta_fair"] / (data["gap_baseline"].abs() + 1e-12)

    # Arredondar
    data["d_bl_fmt"] = data["gap_baseline"].round(4)
    data["df_fmt"] = data["delta_fair"].round(4)
    data["rf_fmt"] = data["r_fair"].round(4)

    # ── Construir LaTeX ─────────────────────────────────────────────────
    metric_label = translate.get(metric, metric)

    label = r"tab_rq3_baseline_delta_" + metric.replace(" ", "_")
    header_row = (
        r"\textbf{Método} & \textbf{Conjunto de dados} "
        r"& $\boldsymbol{d_{\mathrm{baseline}}}$ "
        r"& $\boldsymbol{\Delta_{\mathrm{fair}}}$ "
        r"& $\boldsymbol{r_{\mathrm{fair}}}$ \\"
    )

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{longtable}{@{} l l r r r @{}}")
    lines.append(format_caption(
        r"\caption{Distância ao ideal no \textit{baseline} ($d_{\mathrm{baseline}}$) "
        r"e redução absoluta após mitigação ($\Delta_{\mathrm{fair}}$) "
        rf"para a métrica \textbf{{{metric_label}}}. "
        r"$r_{\mathrm{fair}} = \Delta_{\mathrm{fair}} / d_{\mathrm{baseline}}$ "
        r"indica a fração do viés removida (valores $>1$ indicam sobrecorreção). "
        r"Valores positivos de $\Delta_{\mathrm{fair}}$ indicam redução do viés; "
        r"valores negativos indicam aumento. "
        r"Linhas ordenadas por $d_{\mathrm{baseline}}$ decrescente dentro de cada método.}"
        r"\label{" + label + r"} \\"
    ))
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        r"\multicolumn{5}{l}{\footnotesize\itshape Continuação da Tabela~\ref{"
        + label + r"}} \\"
    )
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{r}{\footnotesize\itshape Continua na próxima página} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    prev_method = None
    for idx, row in data.iterrows():
        m_label = row["method_label"]
        ds_label = row["dataset_label"]
        d_bl = _replace_decimal_in_latex(f"{row['d_bl_fmt']:.4f}")
        d_f = _replace_decimal_in_latex(f"{row['df_fmt']:.4f}")
        r_f = _replace_decimal_in_latex(f"{row['rf_fmt']:.4f}")

        if m_label != prev_method:
            # Inserir midrule entre métodos (exceto o primeiro)
            if prev_method is not None:
                lines.append(r"\midrule")

            n_rows = int((data["method_label"] == m_label).sum())
            method_cell = rf"\multirow{{{n_rows}}}{{*}}{{{m_label}}}"
            prev_method = m_label
        else:
            method_cell = ""

        lines.append(
            f"{method_cell} & {ds_label} & {d_bl} & {d_f} & {r_f} " + r"\\"
        )

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")

    tex = "\n".join(lines)

    # Salvar se diretório informado
    if result_dir:
        out_dir = Path(result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"tab_rq3_baseline_delta_{metric.replace(' ', '_')}.tex"
        (out_dir / fname).write_text(tex, encoding="utf-8")

    return tex
