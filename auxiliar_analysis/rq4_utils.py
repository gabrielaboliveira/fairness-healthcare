"""
rq4_utils.py — RQ4: O sentido do viés altera a mitigação?

Funções de apoio para analisar se a direção do viés no baseline (U/P)
influencia a eficácia das estratégias de mitigação.

CONCEITOS:
- U (Unprivileged): viés contra o grupo não-privilegiado
- P (Privileged): viés contra o grupo privilegiado
- O rótulo U/P é propriedade do (dataset × métrica) no baseline
- Categorias A e C possuem direção interpretável
- Categoria B não possui direção (excluída por padrão)

BLOCOS:
1. build_direction_coverage_table — cobertura U/P por categoria
   build_severity_by_direction_table — composição de severidade por estrato U/P
   build_direction_rfair_data — r_fair estratificado por direção
   detect_overcorrection — detecção de inversão de sentido do viés
   build_overcorrection_summary — tabela de frequência de over-correction
2. plot_delta_by_direction_category — facetas U|P por categoria (pre vs in)
   plot_delta_by_direction_method  — facetas U|P por método individual
3. compute_win_rate_by_direction   — win rate separado por U/P
4. run_stats_by_direction          — Friedman + Holm separado por U/P
5. build_method_direction_comparison — tabela resumo método × direção
   plot_method_direction_heatmap     — heatmap método × direção

DEPENDÊNCIAS:
- classified (DataFrame): saída de classify_bias_patterns()
  Colunas necessárias: dataset, metric, category, direction, eligible, pattern_code
- df (DataFrame principal): com colunas pattern_code, direction mergeadas
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .analysis_utils import apply_comma_axes, format_caption, _replace_decimal_in_latex
from .rq1_utils import (
    compute_delta_fairness_category,
    compute_delta_fairness_method,
    compute_win_rate,
    _CATEGORY_COLORS,
    _EXP_COLORS,
    _add_method_category_bg,
)
from .statistic_test import (
    EXP_ORDER,
    BASELINE_IDS,
    EXP_TO_CATEGORY,
    BASELINE_TYPICAL_LABEL,
    aggregate_to_dataset_level,
    run_all_category_analysis,
    run_all_experiment_analysis,
    save_results_latex,
    save_descriptive_latex,
)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# ── Configurações visuais centralizadas ─────────────────────────────────────
_VIZ = {
    "title_fontsize": 12,
    "subtitle_fontsize": 10,
    "label_fontsize": 11,
    "tick_fontsize": 10,
    "legend_fontsize": 9,
    "annotation_fontsize": 10,
    "suptitle_fontsize": 12,
    "colorbar_label_fontsize": 10,
    "paired_legend_fontsize": 8,
    # Posição da legenda nos paired scatter: "above" | "below" | "inside"
    "paired_legend_position": "above",
}

# Categorias com direção interpretável
DIRECTIONAL_CATEGORIES = ["A", "C"]

# Labels de direção
DIRECTION_LABELS = {
    "U": "Contra não-privilegiado (U)",
    "P": "Contra privilegiado (P)",
}


# =============================================================================
# BLOCO 1: COBERTURA E DISTRIBUIÇÃO U/P
# =============================================================================

def build_direction_coverage_table(
    classified: pd.DataFrame,
    categories: List[str] = None,
    translate: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Constrói tabela de cobertura mostrando quantos datasets são U, P ou
    inelegíveis, por categoria e por métrica.

    Args:
        classified: DataFrame da saída de classify_bias_patterns().
            Colunas necessárias: dataset, metric, category, direction,
            eligible, pattern_code.
        categories: Lista de categorias a incluir (default: ["A", "C"]).
        translate: Dicionário de tradução de nomes de métricas.

    Returns:
        DataFrame com colunas:
            Categoria, Métrica, Total, U, P, Inelegível
    """
    categories = categories or DIRECTIONAL_CATEGORIES
    translate = translate or {}

    rows = []

    for cat in categories:
        cat_data = classified[classified["category"] == cat].copy()

        if cat_data.empty:
            continue

        # Iterar por métrica dentro da categoria
        for metric in sorted(cat_data["metric"].unique()):
            metric_data = cat_data[cat_data["metric"] == metric]

            # Deduplica para nível de dataset (um rótulo por dataset×métrica)
            ds_level = metric_data.drop_duplicates(subset=["dataset"])

            n_total = len(ds_level)
            n_eligible = ds_level["eligible"].sum()
            n_ineligible = n_total - n_eligible

            eligible_only = ds_level[ds_level["eligible"]]
            n_u = (eligible_only["direction"] == "U").sum()
            n_p = (eligible_only["direction"] == "P").sum()
            n_no_dir = n_eligible - n_u - n_p  # elegível mas sem direção

            metric_label = translate.get(metric, metric)

            rows.append({
                "Categoria": cat,
                "Métrica": metric_label,
                "Total": n_total,
                "U (não-priv.)": int(n_u),
                "P (priv.)": int(n_p),
                "Sem direção": int(n_no_dir),
                "Inelegível": int(n_ineligible),
            })

        # Linha resumo por categoria (nível dataset, deduplicado)
        cat_ds = cat_data.drop_duplicates(subset=["dataset", "category"])
        n_total_cat = len(cat_ds)
        n_eligible_cat = cat_ds["eligible"].sum()
        eligible_cat = cat_ds[cat_ds["eligible"]]
        n_u_cat = (eligible_cat["direction"] == "U").sum()
        n_p_cat = (eligible_cat["direction"] == "P").sum()
        n_no_dir_cat = n_eligible_cat - n_u_cat - n_p_cat

        rows.append({
            "Categoria": cat,
            "Métrica": f"— Total {cat} —",
            "Total": n_total_cat,
            "U (não-priv.)": int(n_u_cat),
            "P (priv.)": int(n_p_cat),
            "Sem direção": int(n_no_dir_cat),
            "Inelegível": int(n_total_cat - n_eligible_cat),
        })

    result = pd.DataFrame(rows)
    return result


def get_datasets_by_direction(
    classified: pd.DataFrame,
    category: str,
    direction: str,
) -> List[str]:
    """
    Retorna lista de datasets elegíveis com dada direção e categoria.

    Útil para filtrar o DataFrame principal antes de chamar
    compute_delta_fairness_*.

    Args:
        classified: DataFrame da saída de classify_bias_patterns().
        category: "A" ou "C".
        direction: "U" ou "P".

    Returns:
        Lista de nomes de datasets.
    """
    mask = (
        (classified["category"] == category)
        & (classified["eligible"] == True)
        & (classified["direction"] == direction)
    )
    return sorted(
        classified.loc[mask, "dataset"].drop_duplicates().tolist()
    )


def build_direction_rfair_data(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metrics: List[str],
    category: str,
    ideal_values: Dict,
    level: str = "method",
    baseline_ids: Optional[List[str]] = None,
    category_mapping: Optional[Dict] = None,
    min_baselines: int = 3,
) -> pd.DataFrame:
    """
    Constrói DataFrame com r_fair estratificado por direção U/P.

    Combina build_intensity_scatter_data() (de rq3_utils) com
    get_datasets_by_direction() para produzir dados de mitigação relativa
    separados por sentido do viés.

    Args:
        df: DataFrame principal (com folds × repetições).
        classified: Saída de classify_bias_patterns().
        metrics: Lista de métricas a processar.
        category: Categoria de padrão ("A" ou "C").
        ideal_values: Dict com valores ideais.
        level: "method" (um ponto por método) ou "category" (pre vs in).
        baseline_ids: Lista de exp_ids baseline.
        category_mapping: Mapeamento categoria → exp_ids.
        min_baselines: Mínimo de baselines para baseline_typical.

    Returns:
        DataFrame com colunas:
            dataset, metric, method, category, gap_baseline,
            delta_fair, mitigation_rel, severity_level,
            severity_label, direction
    """
    from .rq3_utils import build_intensity_scatter_data

    scatter_data = build_intensity_scatter_data(
        df, classified, metrics, ideal_values,
        scope=category, level=level,
        baseline_ids=baseline_ids,
        category_mapping=category_mapping,
        min_baselines=min_baselines,
    )

    if scatter_data.empty:
        return scatter_data

    # Direção herdada por (dataset, category)
    dir_info = (
        classified[classified["category"] == category]
        [["dataset", "direction"]]
        .drop_duplicates(subset=["dataset"])
    )
    scatter_data = scatter_data.merge(dir_info, on="dataset", how="left")

    return scatter_data[scatter_data["direction"].isin(["U", "P"])].copy()


def build_severity_by_direction_table(
    classified: pd.DataFrame,
    categories: List[str] = None,
    translate: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Tabela cruzada: Padrão × Sentido × Severidade → n datasets.

    Mostra a composição de severidade dentro de cada estrato de direção (U/P),
    permitindo avaliar se um sentido concentra mais casos severos.

    Args:
        classified: DataFrame da saída de classify_bias_patterns().
            Colunas necessárias: dataset, category, direction, eligible,
            severity_level, severity_label.
        categories: Lista de categorias a incluir (default: ["A", "C"]).
        translate: Dicionário de tradução (não usado aqui, reservado).

    Returns:
        DataFrame com colunas:
            Categoria, Sentido, Alto, Médio, Baixo, Total
    """
    categories = categories or DIRECTIONAL_CATEGORIES

    eligible = classified[
        (classified["eligible"] == True)
        & (classified["direction"].isin(["U", "P"]))
        & (classified["category"].isin(categories))
    ].copy()

    if eligible.empty:
        return pd.DataFrame(
            columns=["Categoria", "Sentido", "Alto", "Médio", "Baixo", "Total"]
        )

    # Deduplica para nível (dataset, category) — herança de severidade
    ds_level = eligible.drop_duplicates(subset=["dataset", "category"]).copy()

    severity_label_map = {1: "Alto", 2: "Médio", 3: "Baixo"}
    ds_level["sev_label"] = ds_level["severity_level"].map(severity_label_map)

    rows = []
    for cat in categories:
        cat_data = ds_level[ds_level["category"] == cat]
        for direction in ["U", "P"]:
            dir_data = cat_data[cat_data["direction"] == direction]
            counts = dir_data["sev_label"].value_counts()
            rows.append({
                "Categoria": cat,
                "Sentido": direction,
                "Alto": int(counts.get("Alto", 0)),
                "Médio": int(counts.get("Médio", 0)),
                "Baixo": int(counts.get("Baixo", 0)),
                "Total": len(dir_data),
            })

    return pd.DataFrame(rows)


def _compute_signed_delta(value: float, ideal: float, metric: str) -> float:
    """Computa δ = value - ideal com sinal (sem módulo)."""
    if metric == "bias_amplification":
        # ideal ≤ 0; retorna valor diretamente (positivo = viés)
        return value
    if pd.isna(ideal):
        return np.nan
    return value - ideal


def detect_overcorrection(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metrics: List[str],
    category: str,
    ideal_values: Dict,
    baseline_ids: Optional[List[str]] = None,
    min_baselines: int = 3,
) -> pd.DataFrame:
    """
    Detecta se a mitigação inverte o sentido do viés (over-correction).

    Para cada (dataset, método, métrica):
    1. Computa δ_baseline = mediana(m_baseline) - ideal (com sinal)
    2. Computa δ_método = mediana(m_método) - ideal (com sinal)
    3. Classifica:
       - "mitigação": |δ_método| < |δ_baseline| sem inversão de sinal
       - "over-correction": sign(δ_método) ≠ sign(δ_baseline) e ambos ≠ 0
       - "degradação": |δ_método| > |δ_baseline| sem inversão de sinal
       - "neutro": δ_método ≈ 0 (dentro de tolerância 1e-6)

    Filtra apenas datasets elegíveis para a categoria e com direção U/P.

    Args:
        df: DataFrame principal com colunas value_num, exp_id, etc.
        classified: Saída de classify_bias_patterns().
        metrics: Lista de métricas a analisar.
        category: "A" ou "C".
        ideal_values: Dicionário de valores ideais.
        baseline_ids: IDs dos baselines (default: BASELINE_IDS).
        min_baselines: Mínimo de baselines para baseline.

    Returns:
        DataFrame com colunas:
            dataset, method, metric, direction,
            delta_baseline, delta_method, classification
    """
    baseline_ids = baseline_ids or BASELINE_IDS
    TOLERANCE = 1e-6

    # Datasets elegíveis com direção para a categoria
    ds_dir = (
        classified[
            (classified["category"] == category)
            & (classified["eligible"] == True)
            & (classified["direction"].isin(["U", "P"]))
        ][["dataset", "direction"]]
        .drop_duplicates(subset=["dataset"])
    )

    if ds_dir.empty:
        return pd.DataFrame(columns=[
            "dataset", "method", "metric", "direction",
            "delta_baseline", "delta_method", "classification",
        ])

    eligible_datasets = set(ds_dir["dataset"])

    rows = []
    for metric in metrics:
        ideal = ideal_values.get(metric, np.nan)

        # Obter raw matrix via aggregate_to_dataset_level
        _, raw_matrix = aggregate_to_dataset_level(
            df, metric, ideal_values, scope=category,
            baseline_ids=baseline_ids,
            analysis_type="experiment",
            min_baselines=min_baselines,
            return_raw_matrix=True,
        )

        if raw_matrix.empty or BASELINE_TYPICAL_LABEL not in raw_matrix.columns:
            continue

        method_cols = [
            c for c in raw_matrix.columns
            if c not in ["dataset", BASELINE_TYPICAL_LABEL]
            and c not in baseline_ids
        ]

        for _, row in raw_matrix.iterrows():
            dataset = row["dataset"]
            if dataset not in eligible_datasets:
                continue

            direction = ds_dir.loc[
                ds_dir["dataset"] == dataset, "direction"
            ].iloc[0]

            baseline_raw = row[BASELINE_TYPICAL_LABEL]
            if pd.isna(baseline_raw):
                continue

            delta_bl = _compute_signed_delta(baseline_raw, ideal, metric)
            if pd.isna(delta_bl):
                continue

            for method in method_cols:
                method_raw = row.get(method, np.nan)
                if pd.isna(method_raw):
                    continue

                delta_m = _compute_signed_delta(method_raw, ideal, metric)

                # Classificar
                if abs(delta_m) < TOLERANCE:
                    classification = "neutro"
                elif abs(delta_bl) < TOLERANCE:
                    # Baseline já é neutro; qualquer desvio é degradação
                    classification = "degradação"
                elif np.sign(delta_m) != np.sign(delta_bl):
                    classification = "over-correction"
                elif abs(delta_m) < abs(delta_bl):
                    classification = "mitigação"
                else:
                    classification = "degradação"

                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "direction": direction,
                    "delta_baseline": delta_bl,
                    "delta_method": delta_m,
                    "classification": classification,
                })

    if not rows:
        return pd.DataFrame(columns=[
            "dataset", "method", "metric", "direction",
            "delta_baseline", "delta_method", "classification",
        ])

    return pd.DataFrame(rows)


def build_overcorrection_summary(
    oc_df: pd.DataFrame,
    by: str = "direction",
) -> pd.DataFrame:
    """
    Tabela de frequência de classificações de over-correction.

    Args:
        oc_df: Saída de detect_overcorrection().
        by: "direction" (estrato U/P) ou "method" (método × estrato).

    Returns:
        DataFrame com contagens e percentuais por classificação.
    """
    if oc_df.empty:
        return pd.DataFrame()

    CLASS_ORDER = ["mitigação", "over-correction", "degradação", "neutro"]

    if by == "direction":
        group_cols = ["direction"]
    elif by == "method":
        group_cols = ["method", "direction"]
    else:
        group_cols = [by]

    counts = (
        oc_df.groupby(group_cols + ["classification"])
        .size()
        .reset_index(name="n")
    )

    totals = (
        counts.groupby(group_cols)["n"]
        .sum()
        .reset_index(name="total")
    )

    counts = counts.merge(totals, on=group_cols)
    counts["pct"] = (counts["n"] / counts["total"] * 100).round(1)

    # Ordenar classificações
    counts["classification"] = pd.Categorical(
        counts["classification"], categories=CLASS_ORDER, ordered=True,
    )
    counts = counts.sort_values(group_cols + ["classification"]).reset_index(drop=True)

    return counts


# =============================================================================
# BLOCO 2: Δfairness ESTRATIFICADO POR DIREÇÃO (FACETAS U | P)
# =============================================================================

_BASELINE_COLOR = "#4E79A7"
_DIRECTION_LABELS_SHORT = {"U": "Viés contra não-priv. (U)", "P": "Viés contra priv. (P)"}


def _render_scatter_on_ax(
    ax: plt.Axes,
    items: List[str],
    delta_df: pd.DataFrame,
    group_col: str,
    color_map: Dict[str, str],
    seed: int = 42,
    jitter_range: float = 0.15,
) -> None:
    """
    Renderiza scatter + mediana(◆) + IQR whiskers em um Axes já existente.

    Parâmetros:
        ax: Axes matplotlib
        items: lista ordenada de valores do group_col (eixo Y)
        delta_df: DataFrame com colunas [group_col, "delta_fair"]
        group_col: "category" ou "method"
        color_map: dicionário de cores por item
        seed: seed para jitter
        jitter_range: amplitude do jitter vertical
    """
    rng = np.random.default_rng(seed)

    for i, item in enumerate(items):
        vals = delta_df.loc[delta_df[group_col] == item, "delta_fair"].dropna().values
        if len(vals) == 0:
            continue
        color = color_map.get(item, "#999999")

        # Pontos individuais
        y_jitter = rng.uniform(-jitter_range, jitter_range, size=len(vals)) + i
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

        # Whisker (IQR)
        ax.errorbar(
            [median], [i],
            xerr=[[median - q25], [q75 - median]],
            fmt="none", ecolor=color, elinewidth=1.5,
            capsize=4, capthick=1.5, zorder=3,
        )


def plot_delta_by_direction_category(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metric: str,
    category: str,
    ideal_values: Dict,
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 4),
    seed: int = 42,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Facetas U | P de Δfairness por CATEGORIA (pre vs in).

    Dois subplots lado a lado: esquerdo = U, direito = P.
    Mesma escala X compartilhada para comparabilidade.

    Args:
        df: DataFrame principal com colunas pattern_code, direction etc.
        classified: saída de classify_bias_patterns() (para obter datasets por direção).
        metric: nome da métrica.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        translate: tradução de nomes de métricas.
        save_path: caminho para salvar (PDF/PNG).
        figsize: tamanho total da figura (ambos painéis).
        seed: seed para jitter.
        show: se True, exibe.

    Returns:
        plt.Figure ou None se dados insuficientes.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)
    order = ["pre_processing", "in_processing"]
    label_map = {"pre_processing": "Pre-processing", "in_processing": "In-processing"}

    # Obter datasets por direção
    directions_data = {}
    for d in ["U", "P"]:
        ds_list = get_datasets_by_direction(classified, category, d)
        if not ds_list:
            continue
        df_sub = df[df["dataset"].isin(ds_list)]
        delta = compute_delta_fairness_category(
            df_sub, metric=metric, ideal_values=ideal_values, scope=category,
        )
        if not delta.empty:
            directions_data[d] = (delta, ds_list)

    if not directions_data:
        return None

    n_panels = len(directions_data)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True, sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (direction, (delta_df, ds_list)) in zip(axes, directions_data.items()):
        _render_scatter_on_ax(
            ax, order, delta_df, group_col="category",
            color_map=_CATEGORY_COLORS, seed=seed,
        )

        ax.axvline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([label_map[c] for c in order])
        ax.set_xlabel(r"$\Delta_{\mathrm{fair}}$", fontsize=_VIZ["label_fontsize"])
        dir_label = _DIRECTION_LABELS_SHORT.get(direction, direction)
        n_ds = len(ds_list)
        # ax.set_title(f"{dir_label}\n(n={n_ds} datasets)", fontsize=_VIZ["subtitle_fontsize"])
        ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])

        ax.yaxis.grid(False)
        ax.xaxis.grid(True, alpha=0.3)

    # Legenda compartilhada
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.6, label="Conjunto de dados"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="Mediana ± IQR"),
        Line2D([0], [0], color=_BASELINE_COLOR, linestyle="--", linewidth=1,
               label="Sem efeito ($\\Delta$=0)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=_VIZ["legend_fontsize"], bbox_to_anchor=(0.5, -0.02),
    )

    # fig.suptitle(
        # f"$\\Delta_{{\\text{{fair}}}}$ por categoria — {metric_label} (Padrão {category})",
        # fontsize=_VIZ["suptitle_fontsize"], y=1.02,
    # )
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


def plot_delta_by_direction_method(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metric: str,
    category: str,
    ideal_values: Dict,
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 6),
    seed: int = 42,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Facetas U | P de Δfairness por MÉTODO individual.

    Dois subplots lado a lado: esquerdo = U, direito = P.
    Mesma escala X compartilhada para comparabilidade.

    Args:
        df: DataFrame principal com colunas pattern_code, direction etc.
        classified: saída de classify_bias_patterns() (para obter datasets por direção).
        metric: nome da métrica.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        translate: tradução de nomes de métricas.
        save_path: caminho para salvar (PDF/PNG).
        figsize: tamanho total da figura (ambos painéis).
        seed: seed para jitter.
        show: se True, exibe.

    Returns:
        plt.Figure ou None se dados insuficientes.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    # Métodos (sem baselines), invertidos para topo-primeiro
    methods_all = [m for m in EXP_ORDER if m not in BASELINE_IDS]

    # Obter datasets por direção
    directions_data = {}
    for d in ["U", "P"]:
        ds_list = get_datasets_by_direction(classified, category, d)
        if not ds_list:
            continue
        df_sub = df[df["dataset"].isin(ds_list)]
        delta = compute_delta_fairness_method(
            df_sub, metric=metric, ideal_values=ideal_values, scope=category,
        )
        if not delta.empty:
            # Filtrar métodos presentes
            methods_present = [m for m in methods_all if m in delta["method"].unique()]
            methods_rev = list(reversed(methods_present))
            directions_data[d] = (delta, ds_list, methods_rev)

    if not directions_data:
        return None

    # Usar a lista de métodos mais completa para Y compartilhado
    all_methods_rev = []
    for _, (_, _, mr) in directions_data.items():
        for m in mr:
            if m not in all_methods_rev:
                all_methods_rev.append(m)
    # Reordenar conforme EXP_ORDER (invertido)
    methods_order = [m for m in reversed(methods_all) if m in all_methods_rev]

    n_panels = len(directions_data)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True, sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (direction, (delta_df, ds_list, _)) in zip(axes, directions_data.items()):
        _render_scatter_on_ax(
            ax, methods_order, delta_df, group_col="method",
            color_map=_EXP_COLORS, seed=seed, jitter_range=0.18,
        )

        ax.axvline(0, color=_BASELINE_COLOR, linestyle="--", linewidth=1)

        # Fundo por categoria
        _add_method_category_bg(ax, methods_order)

        ax.set_yticks(range(len(methods_order)))
        ax.set_yticklabels(methods_order, fontsize=_VIZ["tick_fontsize"])
        ax.set_xlabel(r"$\Delta_{\mathrm{fair}}$", fontsize=_VIZ["label_fontsize"])
        dir_label = _DIRECTION_LABELS_SHORT.get(direction, direction)
        n_ds = len(ds_list)
        # ax.set_title(f"{dir_label}\n(n={n_ds} datasets)", fontsize=_VIZ["subtitle_fontsize"])
        ax.tick_params(axis="x", labelsize=_VIZ["tick_fontsize"])

        ax.yaxis.grid(False)
        ax.xaxis.grid(True, alpha=0.3)

    # Legenda compartilhada
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.6, label="Conjunto de dados"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="Mediana ± IQR"),
        Line2D([0], [0], color=_BASELINE_COLOR, linestyle="--", linewidth=1,
               label="Sem efeito ($\\Delta$=0)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=_VIZ["legend_fontsize"], bbox_to_anchor=(0.5, -0.02),
    )

    # fig.suptitle(
        # f"$\\Delta_{{\\text{{fair}}}}$ por método — {metric_label} (Padrão {category})",
        # fontsize=_VIZ["suptitle_fontsize"], y=1.02,
    # )
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
# BLOCO 3: WIN RATE POR DIREÇÃO (U vs P)
# =============================================================================

def compute_win_rate_by_direction(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metrics: List[str],
    category: str,
    ideal_values: Dict,
    group_col: str = "category",
    translate: Dict = None,
) -> pd.DataFrame:
    """
    Calcula win rate separado por direção U e P, para comparação.

    Para cada métrica e cada direção, filtra os datasets correspondentes,
    calcula Δfairness e depois o win rate (proporção de Δ > 0).

    Args:
        df: DataFrame principal (com pattern_code, direction mergeados).
        classified: saída de classify_bias_patterns().
        metrics: lista de métricas a processar.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        group_col: "category" (pre/in) ou "method" (por experimento).
        translate: tradução de nomes de métricas.

    Returns:
        DataFrame com colunas:
            metric_label, <group_col>, direction, n, wins, win_rate_pct
    """
    translate = translate or {}

    compute_fn = (
        compute_delta_fairness_category if group_col == "category"
        else compute_delta_fairness_method
    )

    rows = []
    for direction in ["U", "P"]:
        ds_list = get_datasets_by_direction(classified, category, direction)
        if not ds_list:
            continue

        df_sub = df[df["dataset"].isin(ds_list)]

        for m in metrics:
            delta = compute_fn(
                df_sub, metric=m, ideal_values=ideal_values, scope=category,
            )
            if delta.empty:
                continue

            wr = compute_win_rate(delta, group_col=group_col)
            wr["metric"] = m
            wr["direction"] = direction
            wr["metric_label"] = translate.get(m, m)
            rows.append(wr)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["win_rate_pct"] = (result["win_rate"] * 100).round(1)

    # Reordenar colunas para exibição
    display_cols = ["metric_label", group_col, "direction", "n", "wins", "win_rate_pct"]
    existing = [c for c in display_cols if c in result.columns]
    return result[existing]


# =============================================================================
# BLOCO 4: TESTES ESTATÍSTICOS POR DIREÇÃO (Friedman + Holm, U vs P)
# =============================================================================

def run_stats_by_direction(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metrics: List[str],
    category: str,
    ideal_values: Dict,
    analysis_type: str = "category",
    save_dir: Optional[str] = None,
    translate: Dict = None,
    alternative: str = "less",
    cols: List[str] = None,
) -> pd.DataFrame:
    """
    Executa testes estatísticos (Friedman + Holm) separados por direção U/P.

    Para cada direção, filtra o DataFrame principal pelos datasets com aquela
    direção na categoria informada, e então roda a análise existente
    (run_all_category_analysis ou run_all_experiment_analysis) com scope=categoria.

    Args:
        df: DataFrame principal (com pattern_code, direction mergeados).
        classified: saída de classify_bias_patterns().
        metrics: lista de métricas a testar.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        analysis_type: "category" (pre/in vs baseline) ou "experiment" (cada método).
        save_dir: diretório para salvar LaTeX (None = não salvar).
        translate: tradução de nomes de métricas.
        alternative: hipótese alternativa ("less" = método melhor que baseline).

    Returns:
        DataFrame consolidado com coluna extra "direction" (U/P).
    """
    translate = translate or {}

    run_fn = (
        run_all_category_analysis if analysis_type == "category"
        else run_all_experiment_analysis
    )

    parts = []
    for direction in ["U", "P"]:
        ds_list = get_datasets_by_direction(classified, category, direction)
        if not ds_list:
            continue

        df_sub = df[df["dataset"].isin(ds_list)].copy()

        stats = run_fn(
            df_sub,
            ideal_values=ideal_values,
            metrics=metrics,
            scopes=[category],
            print_results=False,
            save_dir=None,          # salvamos separadamente, se pedido
            translate=translate,
            alternative=alternative,
        )

        if stats.empty:
            continue

        stats["direction"] = direction
        stats["n_datasets_dir"] = len(ds_list)
        stats["datasets"] = ", ".join(ds_list)
        parts.append(stats)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)

    # Salvar LaTeX descritivo (P e U mesclados na mesma tabela) se solicitado
    type_tag = "cat" if analysis_type == "category" else "exp"
    if save_dir:
        for m in result["metric"].unique():
            sub_m = result[result["metric"] == m]
            save_descriptive_latex(
                sub_m,
                metric=m,
                analysis_type=analysis_type,
                save_dir=save_dir,
                translate=translate,
                cols=cols,
                scope_suffix=f"{type_tag}_{category}",
            )

    return result


# =============================================================================
# BLOCO 5: COMPARAÇÃO POR MÉTODO — U vs P
# =============================================================================

def build_method_direction_comparison(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metrics: List[str],
    category: str,
    ideal_values: Dict,
    translate: Dict = None,
    value_col: str = "delta_fair",
    precomputed: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Constrói tabela comparativa: para cada método, mostra valor em U e P.

    Colunas resultantes:
        method, category, metric_label,
        n_U, median_U, iqr_U, wr_U,
        n_P, median_P, iqr_P, wr_P,
        diff_median  (median_P − median_U)

    diff_median > 0 → método funciona melhor em P que em U.

    Args:
        df: DataFrame principal (com pattern_code, direction mergeados).
        classified: saída de classify_bias_patterns().
        metrics: lista de métricas.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        translate: tradução de nomes de métricas.
        value_col: coluna de valor a usar ("delta_fair" ou "mitigation_rel").
        precomputed: DataFrame pré-computado com colunas direction, method, metric
            e value_col (saída de build_direction_rfair_data). Se fornecido,
            não computa delta internamente.

    Returns:
        DataFrame pronto para display / LaTeX.
    """
    translate = translate or {}

    # Métodos (sem baselines)
    methods_all = [m for m in EXP_ORDER if m not in BASELINE_IDS]

    rows = []
    for metric in metrics:
        metric_label = translate.get(metric, metric)

        # Computar valores por método para cada direção
        dir_data = {}  # direction -> DataFrame
        if precomputed is not None:
            for direction in ["U", "P"]:
                sub = precomputed[
                    (precomputed["metric"] == metric)
                    & (precomputed["direction"] == direction)
                ]
                if not sub.empty:
                    dir_data[direction] = sub
        else:
            for direction in ["U", "P"]:
                ds_list = get_datasets_by_direction(classified, category, direction)
                if not ds_list:
                    continue
                df_sub = df[df["dataset"].isin(ds_list)]
                delta = compute_delta_fairness_method(
                    df_sub, metric=metric, ideal_values=ideal_values, scope=category,
                )
                if not delta.empty:
                    dir_data[direction] = delta

        # Montar uma linha por método
        for method in methods_all:
            row = {
                "method": method,
                "category": EXP_TO_CATEGORY.get(method, ""),
                "metric": metric,
                "metric_label": metric_label,
            }

            for direction in ["U", "P"]:
                sfx = f"_{direction}"
                if direction in dir_data:
                    vals = dir_data[direction]
                    m_vals = vals.loc[vals["method"] == method, value_col].dropna()
                    n = len(m_vals)
                    if n > 0:
                        row[f"n{sfx}"] = n
                        row[f"median{sfx}"] = round(float(np.nanmedian(m_vals)), 4)
                        q25 = np.nanpercentile(m_vals, 25)
                        q75 = np.nanpercentile(m_vals, 75)
                        row[f"iqr{sfx}"] = round(float(q75 - q25), 4)
                        row[f"wr{sfx}"] = round(float((m_vals > 0).sum() / n * 100), 1)
                    else:
                        row[f"n{sfx}"] = 0
                        row[f"median{sfx}"] = np.nan
                        row[f"iqr{sfx}"] = np.nan
                        row[f"wr{sfx}"] = np.nan
                else:
                    row[f"n{sfx}"] = 0
                    row[f"median{sfx}"] = np.nan
                    row[f"iqr{sfx}"] = np.nan
                    row[f"wr{sfx}"] = np.nan

            # Diferença: P − U (positivo = melhor em P)
            med_u = row.get("median_U", np.nan)
            med_p = row.get("median_P", np.nan)
            if pd.notna(med_u) and pd.notna(med_p):
                row["diff_median"] = round(med_p - med_u, 4)
            else:
                row["diff_median"] = np.nan

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Ordenar: por método na ordem padrão
    method_order = {m: i for i, m in enumerate(methods_all)}
    result["_order"] = result["method"].map(method_order)
    result = result.sort_values(["metric", "_order"]).drop(columns="_order")

    display_cols = [
        "metric", "metric_label", "method", "category",
        "n_U", "median_U", "iqr_U", "wr_U",
        "n_P", "median_P", "iqr_P", "wr_P",
        "diff_median",
    ]
    existing = [c for c in display_cols if c in result.columns]
    return result[existing].reset_index(drop=True)


def build_category_direction_comparison(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metrics: List[str],
    category: str,
    ideal_values: Dict,
    translate: Dict = None,
    value_col: str = "delta_fair",
    precomputed: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Constrói tabela comparativa por CATEGORIA (pre/in): valor em U e P.

    Colunas resultantes:
        group, metric_label,
        n_U, median_U, iqr_U, wr_U,
        n_P, median_P, iqr_P, wr_P,
        diff_median  (median_P − median_U)

    diff_median > 0 → categoria funciona melhor em P que em U.

    Args:
        df: DataFrame principal (com pattern_code, direction mergeados).
        classified: saída de classify_bias_patterns().
        metrics: lista de métricas.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        translate: tradução de nomes de métricas.
        value_col: coluna de valor a usar ("delta_fair" ou "mitigation_rel").
        precomputed: DataFrame pré-computado com colunas direction, category, metric
            e value_col (saída de build_direction_rfair_data com level="category").
            Se fornecido, não computa delta internamente.

    Returns:
        DataFrame pronto para display / LaTeX.
    """
    translate = translate or {}

    groups = ["pre_processing", "in_processing"]
    group_labels = {"pre_processing": "Pre-processing", "in_processing": "In-processing"}

    rows = []
    for metric in metrics:
        metric_label = translate.get(metric, metric)

        # Computar valores por categoria para cada direção
        dir_data = {}
        if precomputed is not None:
            for direction in ["U", "P"]:
                sub = precomputed[
                    (precomputed["metric"] == metric)
                    & (precomputed["direction"] == direction)
                ]
                if not sub.empty:
                    dir_data[direction] = sub
        else:
            for direction in ["U", "P"]:
                ds_list = get_datasets_by_direction(classified, category, direction)
                if not ds_list:
                    continue
                df_sub = df[df["dataset"].isin(ds_list)]
                delta = compute_delta_fairness_category(
                    df_sub, metric=metric, ideal_values=ideal_values, scope=category,
                )
                if not delta.empty:
                    dir_data[direction] = delta

        # Montar uma linha por grupo (pre/in)
        for group in groups:
            row = {
                "group": group,
                "group_label": group_labels.get(group, group),
                "metric": metric,
                "metric_label": metric_label,
            }

            for direction in ["U", "P"]:
                sfx = f"_{direction}"
                if direction in dir_data:
                    vals = dir_data[direction]
                    g_vals = vals.loc[vals["category"] == group, value_col].dropna()
                    n = len(g_vals)
                    if n > 0:
                        row[f"n{sfx}"] = n
                        row[f"median{sfx}"] = round(float(np.nanmedian(g_vals)), 4)
                        q25 = np.nanpercentile(g_vals, 25)
                        q75 = np.nanpercentile(g_vals, 75)
                        row[f"iqr{sfx}"] = round(float(q75 - q25), 4)
                        row[f"wr{sfx}"] = round(float((g_vals > 0).sum() / n * 100), 1)
                    else:
                        row[f"n{sfx}"] = 0
                        row[f"median{sfx}"] = np.nan
                        row[f"iqr{sfx}"] = np.nan
                        row[f"wr{sfx}"] = np.nan
                else:
                    row[f"n{sfx}"] = 0
                    row[f"median{sfx}"] = np.nan
                    row[f"iqr{sfx}"] = np.nan
                    row[f"wr{sfx}"] = np.nan

            # Diferença: P − U (positivo = melhor em P)
            med_u = row.get("median_U", np.nan)
            med_p = row.get("median_P", np.nan)
            if pd.notna(med_u) and pd.notna(med_p):
                row["diff_median"] = round(med_p - med_u, 4)
            else:
                row["diff_median"] = np.nan

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    display_cols = [
        "metric", "metric_label", "group", "group_label",
        "n_U", "median_U", "iqr_U", "wr_U",
        "n_P", "median_P", "iqr_P", "wr_P",
        "diff_median",
    ]
    existing = [c for c in display_cols if c in result.columns]
    return result[existing].reset_index(drop=True)


def plot_category_direction_paired(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metric: str,
    category: str,
    ideal_values: Dict,
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 3),
    seed: int = 42,
    show: bool = True,
    value_col: str = "delta_fair",
    ylabel: Optional[str] = None,
    precomputed: Optional[pd.DataFrame] = None,
) -> Optional[plt.Figure]:
    """
    Paired scatter U vs P por CATEGORIA (pre/in), com dumbbell conectando medianas.

    Visualização unificada (sem facetas) que mostra, para cada categoria:
    - Distribuição U (acima) e P (abaixo) com scatter + mediana + IQR
    - Linha dumbbell conectando medianas para evidenciar a diferença

    Args:
        df: DataFrame principal com colunas pattern_code, direction etc.
        classified: saída de classify_bias_patterns().
        metric: nome da métrica.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        translate: tradução de nomes de métricas.
        save_path: caminho para salvar (PDF/PNG).
        figsize: tamanho total da figura.
        seed: seed para jitter.
        show: se True, exibe.
        value_col: coluna de valor ("delta_fair" ou "mitigation_rel").
        ylabel: label do eixo X (auto-derivado de value_col se None).
        precomputed: DataFrame pré-computado (saída de build_direction_rfair_data
            com level="category"). Se fornecido, não computa delta internamente.

    Returns:
        plt.Figure ou None se dados insuficientes.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    if ylabel is None:
        ylabel = (r"$r_{\mathrm{fair}}$" if value_col == "mitigation_rel"
                  else r"$\Delta_{\mathrm{fair}}$")

    groups = ["pre_processing", "in_processing"]
    group_labels = {"pre_processing": "Pre-processing", "in_processing": "In-processing"}

    # Obter dados por direção
    dir_data = {}
    dir_n = {}
    if precomputed is not None:
        for direction in ["U", "P"]:
            sub = precomputed[
                (precomputed["metric"] == metric)
                & (precomputed["direction"] == direction)
            ]
            if not sub.empty:
                dir_data[direction] = sub
                dir_n[direction] = sub["dataset"].nunique()
    else:
        for direction in ["U", "P"]:
            ds_list = get_datasets_by_direction(classified, category, direction)
            if not ds_list:
                continue
            df_sub = df[df["dataset"].isin(ds_list)]
            delta = compute_delta_fairness_category(
                df_sub, metric=metric, ideal_values=ideal_values, scope=category,
            )
            if not delta.empty:
                dir_data[direction] = delta
                dir_n[direction] = len(ds_list)

    if not dir_data:
        return None

    # Grupos presentes em pelo menos uma direção
    groups_present = []
    for g in groups:
        for d in dir_data.values():
            if g in d["category"].unique():
                groups_present.append(g)
                break
    groups_rev = list(reversed(groups_present))

    if not groups_rev:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    for i, group in enumerate(groups_rev):
        vals_u = np.array([])
        vals_p = np.array([])

        if "U" in dir_data:
            vals_u = dir_data["U"].loc[
                dir_data["U"]["category"] == group, value_col
            ].dropna().values
        if "P" in dir_data:
            vals_p = dir_data["P"].loc[
                dir_data["P"]["category"] == group, value_col
            ].dropna().values

        _render_paired_scatter_on_ax(
            ax, y_pos=i, vals_u=vals_u, vals_p=vals_p, seed=seed,
        )

    # Linha de referência =0
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1, zorder=0)

    ax.set_yticks(range(len(groups_rev)))
    ax.set_yticklabels([group_labels.get(g, g) for g in groups_rev],
                       fontsize=_VIZ["tick_fontsize"])
    ax.set_xlabel(ylabel, fontsize=_VIZ["label_fontsize"])
    ax.tick_params(axis="x", labelsize=_VIZ["tick_fontsize"])

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3)

    # Legenda
    n_u = dir_n.get("U", 0)
    n_p = dir_n.get("P", 0)
    legend_elements = [
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=_DIRECTION_COLORS["U"],
               markeredgecolor="black", markersize=8,
               label=f"U — viés contra não-priv. (n={n_u})"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=_DIRECTION_COLORS["P"],
               markeredgecolor="black", markersize=8,
               label=f"P — viés contra priv. (n={n_p})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.5, label="Conjunto de dados"),
        Line2D([0], [0], color="#888888", linestyle="-", linewidth=1,
               alpha=0.6, label="Diferença U–P"),
        Line2D([0], [0], color="#999999", linestyle="--", linewidth=1,
               label="Sem efeito (=0)"),
    ]

    val_label = ("$r_{\\text{fair}}$" if value_col == "mitigation_rel"
                 else "$\\Delta_{\\text{fair}}$")
    # ax.set_title(
        # f"{val_label} U vs P por categoria — {metric_label} (Padrão {category})",
        # fontsize=_VIZ["title_fontsize"],
    # )

    _place_paired_legend(fig, ax, legend_elements)

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


def plot_method_direction_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = None,
    value_col: str = "median",
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 5),
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Heatmap de método × direção mostrando mediana de Δfairness (ou win rate).

    Permite comparar visualmente quais métodos se destacam em U vs P.

    Args:
        comparison_df: saída de build_method_direction_comparison() (pode ser
            filtrada para uma métrica específica).
        metric: se fornecido, filtra comparison_df por essa métrica.
        value_col: "median" para mediana de Δfairness, "wr" para win rate (%).
        translate: tradução de nomes.
        save_path: caminho para salvar.
        figsize: tamanho da figura.
        show: se True, exibe.

    Returns:
        plt.Figure ou None.
    """
    translate = translate or {}

    sub = comparison_df.copy()
    if metric is not None:
        sub = sub[sub["metric"] == metric]

    if sub.empty:
        return None

    metric_label = sub["metric_label"].iloc[0] if "metric_label" in sub.columns else metric or ""

    # Construir a matriz método × direção
    col_u = f"{value_col}_U"
    col_p = f"{value_col}_P"

    if col_u not in sub.columns or col_p not in sub.columns:
        return None

    methods = sub["method"].tolist()
    mat = sub[[col_u, col_p]].values  # shape (n_methods, 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Determinar limites simétricos para colormap centrado em 0
    vmax = np.nanmax(np.abs(mat))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0

    cmap = "RdYlGn"  # vermelho = piora, verde = melhora
    if value_col == "wr":
        cmap = "RdYlGn"
        vmin, vmax_cbar = 0, 100
    else:
        vmin, vmax_cbar = -vmax, vmax

    im = ax.imshow(
        mat, aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax_cbar,
    )

    # Anotar valores
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if pd.notna(val):
                fmt = f"{val:.1f}%" if value_col == "wr" else f"{val:.3f}"
                text_color = "white" if abs(val - (vmin + vmax_cbar) / 2) > (vmax_cbar - vmin) * 0.35 else "black"
                ax.text(j, i, fmt, ha="center", va="center",
                        fontsize=_VIZ["annotation_fontsize"], fontweight="bold", color=text_color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        _DIRECTION_LABELS_SHORT.get("U", "U"),
        _DIRECTION_LABELS_SHORT.get("P", "P"),
    ], fontsize=_VIZ["tick_fontsize"])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=_VIZ["tick_fontsize"])

    # Background de categoria nos y-labels
    _add_method_category_bg(ax, methods)

    unit_label = "Win rate (%)" if value_col == "wr" else r"Mediana $\Delta_{\mathrm{fair}}$"
    ax.set_xlabel("")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(unit_label, fontsize=_VIZ["colorbar_label_fontsize"])

    title_metric = f" — {metric_label}" if metric_label else ""
    # ax.set_title(f"Comparação U vs P por método{title_metric}", fontsize=_VIZ["title_fontsize"])

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
# BLOCO 6: PAIRED SCATTER — U vs P NO MESMO EIXO
# =============================================================================

# Cores fixas para direções (distintas das cores de categoria/método)
_DIRECTION_COLORS = {"U": "#4E79A7", "P": "#E15759"}


def _fit_ncol(
    fig: plt.Figure,
    ax: plt.Axes,
    legend_elements: list,
    fontsize: float,
    columnspacing: float = 1.2,
    handletextpad: float = 0.4,
) -> int:
    """
    Calcula o ncol máximo para que a legenda não exceda a largura do axes.

    Cria uma legenda temporária com ncol=len(elements), mede a largura,
    e reduz ncol até caber. Retorna ncol ideal (mínimo 1).
    """
    n = len(legend_elements)
    if n <= 1:
        return 1

    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    ax_width = ax_bbox.width  # pixels

    for ncol in range(n, 0, -1):
        test_leg = ax.legend(
            handles=legend_elements, loc="lower center",
            bbox_to_anchor=(0.5, 1.02), ncol=ncol,
            fontsize=fontsize, frameon=False,
            columnspacing=columnspacing, handletextpad=handletextpad,
        )
        fig.canvas.draw_idle()
        leg_bbox = test_leg.get_window_extent(renderer=renderer)
        test_leg.remove()
        if leg_bbox.width <= ax_width * 1.02:  # 2% de tolerância
            return ncol

    return 1


def _place_paired_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    legend_elements: list,
    position: str | None = None,
) -> None:
    """
    Posiciona a legenda dos paired scatter conforme _VIZ["paired_legend_position"].

    Modos:
        "above"  — horizontal entre o título e a área do gráfico
        "below"  — horizontal abaixo do xlabel
        "inside" — dentro do gráfico (lower right), comportamento anterior

    No modo "above"/"below" o ncol é calculado automaticamente para que
    a legenda não exceda a largura do axes, distribuindo em múltiplas
    linhas se necessário.
    """
    position = position or _VIZ.get("paired_legend_position", "inside")
    fs = _VIZ["paired_legend_fontsize"]

    if position == "above":
        ncol = _fit_ncol(fig, ax, legend_elements, fontsize=fs)
        nrows = -(-len(legend_elements) // ncol)  # ceil division
        pad = 25 + 15 * nrows  # mais linhas → mais pad

        title_obj = ax.title
        ax.set_title(title_obj.get_text(),
                     fontsize=title_obj.get_fontsize(), pad=pad)

        ax.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            fontsize=fs,
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.4,
        )
    elif position == "below":
        ncol = _fit_ncol(fig, ax, legend_elements, fontsize=fs)
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=ncol,
            fontsize=fs,
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.4,
        )
    else:  # "inside"
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=fs,
            framealpha=0.9,
        )


def _render_paired_scatter_on_ax(
    ax: plt.Axes,
    y_pos: float,
    vals_u: np.ndarray,
    vals_p: np.ndarray,
    offset: float = 0.14,
    seed: int = 42,
    jitter_range: float = 0.08,
) -> None:
    """
    Renderiza scatter pareado (U acima, P abaixo) com dumbbell conectando medianas.

    Para uma única linha do eixo Y (um método), plota:
    - Pontos individuais U com jitter acima de y_pos
    - Pontos individuais P com jitter abaixo de y_pos
    - Diamante (mediana) + whiskers (IQR) para cada grupo
    - Linha dumbbell conectando as medianas U e P

    Args:
        ax: Axes matplotlib.
        y_pos: posição Y central (e.g., 0, 1, 2...).
        vals_u: valores delta_fair para direção U.
        vals_p: valores delta_fair para direção P.
        offset: deslocamento vertical de cada grupo em relação a y_pos.
        seed: seed para jitter.
        jitter_range: amplitude do jitter vertical dentro de cada sub-faixa.
    """
    rng = np.random.default_rng(seed)
    color_u = _DIRECTION_COLORS["U"]
    color_p = _DIRECTION_COLORS["P"]

    medians = {}

    for vals, color, y_off, direction in [
        (vals_u, color_u, +offset, "U"),
        (vals_p, color_p, -offset, "P"),
    ]:
        if len(vals) == 0:
            continue

        y_center = y_pos + y_off
        y_jitter = rng.uniform(-jitter_range, jitter_range, size=len(vals)) + y_center

        # Pontos individuais
        ax.scatter(
            vals, y_jitter, c=color, s=30, alpha=0.5,
            edgecolors="white", linewidths=0.4, zorder=2,
        )

        median = float(np.nanmedian(vals))
        q25 = float(np.nanpercentile(vals, 25))
        q75 = float(np.nanpercentile(vals, 75))
        medians[direction] = median

        # Mediana (diamante)
        ax.scatter(
            [median], [y_center], c=color, s=90, marker="D",
            edgecolors="black", linewidths=0.8, zorder=4,
        )

        # Whisker (IQR)
        ax.errorbar(
            [median], [y_center],
            xerr=[[median - q25], [q75 - median]],
            fmt="none", ecolor=color, elinewidth=1.5,
            capsize=3, capthick=1.2, zorder=3,
        )

    # Linha dumbbell conectando medianas
    if "U" in medians and "P" in medians:
        ax.plot(
            [medians["U"], medians["P"]],
            [y_pos + offset, y_pos - offset],
            color="#888888", linewidth=1.0, linestyle="-",
            alpha=0.6, zorder=1,
        )


def plot_method_direction_paired(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metric: str,
    category: str,
    ideal_values: Dict,
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 6),
    seed: int = 42,
    show: bool = True,
    value_col: str = "delta_fair",
    ylabel: Optional[str] = None,
    precomputed: Optional[pd.DataFrame] = None,
) -> Optional[plt.Figure]:
    """
    Paired scatter U vs P por método, com dumbbell conectando medianas.

    Visualização unificada (sem facetas) que mostra, para cada método:
    - Distribuição U (acima) e P (abaixo) com scatter + mediana + IQR
    - Linha dumbbell conectando medianas para evidenciar a diferença

    Args:
        df: DataFrame principal com colunas pattern_code, direction etc.
        classified: saída de classify_bias_patterns().
        metric: nome da métrica.
        category: "A" ou "C".
        ideal_values: dicionário de valores ideais.
        translate: tradução de nomes de métricas.
        save_path: caminho para salvar (PDF/PNG).
        figsize: tamanho total da figura.
        seed: seed para jitter.
        show: se True, exibe.
        value_col: coluna de valor ("delta_fair" ou "mitigation_rel").
        ylabel: label do eixo X (auto-derivado de value_col se None).
        precomputed: DataFrame pré-computado (saída de build_direction_rfair_data
            com level="method"). Se fornecido, não computa delta internamente.

    Returns:
        plt.Figure ou None se dados insuficientes.
    """
    translate = translate or {}
    metric_label = translate.get(metric, metric)

    if ylabel is None:
        ylabel = (r"$r_{\mathrm{fair}}$" if value_col == "mitigation_rel"
                  else r"$\Delta_{\mathrm{fair}}$")

    methods_all = [m for m in EXP_ORDER if m not in BASELINE_IDS]

    # Obter dados por direção
    dir_data = {}
    dir_n = {}
    if precomputed is not None:
        for direction in ["U", "P"]:
            sub = precomputed[
                (precomputed["metric"] == metric)
                & (precomputed["direction"] == direction)
            ]
            if not sub.empty:
                dir_data[direction] = sub
                dir_n[direction] = sub["dataset"].nunique()
    else:
        for direction in ["U", "P"]:
            ds_list = get_datasets_by_direction(classified, category, direction)
            if not ds_list:
                continue
            df_sub = df[df["dataset"].isin(ds_list)]
            delta = compute_delta_fairness_method(
                df_sub, metric=metric, ideal_values=ideal_values, scope=category,
            )
            if not delta.empty:
                dir_data[direction] = delta
                dir_n[direction] = len(ds_list)

    if not dir_data:
        return None

    # Métodos presentes em pelo menos uma direção
    methods_present = []
    for m in methods_all:
        for d in dir_data.values():
            if m in d["method"].unique():
                methods_present.append(m)
                break
    methods_rev = list(reversed(methods_present))

    if not methods_rev:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods_rev):
        vals_u = np.array([])
        vals_p = np.array([])

        if "U" in dir_data:
            vals_u = dir_data["U"].loc[
                dir_data["U"]["method"] == method, value_col
            ].dropna().values
        if "P" in dir_data:
            vals_p = dir_data["P"].loc[
                dir_data["P"]["method"] == method, value_col
            ].dropna().values

        _render_paired_scatter_on_ax(
            ax, y_pos=i, vals_u=vals_u, vals_p=vals_p, seed=seed,
        )

    # Linha de referência =0
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1, zorder=0)

    # Background por categoria
    _add_method_category_bg(ax, methods_rev)

    ax.set_yticks(range(len(methods_rev)))
    ax.set_yticklabels(methods_rev, fontsize=_VIZ["tick_fontsize"])
    ax.set_xlabel(ylabel, fontsize=_VIZ["label_fontsize"])
    ax.tick_params(axis="x", labelsize=_VIZ["tick_fontsize"])

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3)

    # Legenda
    n_u = dir_n.get("U", 0)
    n_p = dir_n.get("P", 0)
    legend_elements = [
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=_DIRECTION_COLORS["U"],
               markeredgecolor="black", markersize=8,
               label=f"U — viés contra não-priv. (n={n_u})"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=_DIRECTION_COLORS["P"],
               markeredgecolor="black", markersize=8,
               label=f"P — viés contra priv. (n={n_p})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.5, label="Conjunto de dados"),
        Line2D([0], [0], color="#888888", linestyle="-", linewidth=1,
               alpha=0.6, label="Diferença U–P"),
        Line2D([0], [0], color="#999999", linestyle="--", linewidth=1,
               label="Sem efeito (=0)"),
    ]

    val_label = ("$r_{\\text{fair}}$" if value_col == "mitigation_rel"
                 else "$\\Delta_{\\text{fair}}$")
    # ax.set_title(
        # f"{val_label} U vs P por método — {metric_label} (Padrão {category})",
        # fontsize=_VIZ["title_fontsize"],
    # )

    _place_paired_legend(fig, ax, legend_elements)

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
# BLOCO 8: BAR CHART — Δfair POR DATASET AGRUPADO POR DIREÇÃO (U / P)
# =============================================================================

# Cores específicas do bar chart (não sobrescrevem _DIRECTION_COLORS do Bloco 6)
_BAR_DIRECTION_COLORS = {
    "U": "#F4CCCC",   # salmon/pink
    "P": "#FFF2CC",   # light yellow/gold
}
_BAR_DIRECTION_EDGE_COLORS = {
    "U": "#E06666",   # darker pink border
    "P": "#F1C232",   # darker gold border
}


def _get_datasets_by_metric_direction(
    classified: pd.DataFrame,
    category: str,
    metric: str,
    direction: str,
) -> List[str]:
    """
    Retorna datasets elegíveis com dada direção para uma métrica específica.

    Diferente de get_datasets_by_direction(), filtra por (category, metric, direction)
    em vez de apenas (category, direction), evitando que um dataset apareça
    em múltiplas direções quando métricas diferentes da mesma categoria
    possuem sentidos opostos.

    Args:
        classified: DataFrame da saída de classify_bias_patterns().
        category: "A" ou "C".
        metric: nome da métrica específica.
        direction: "U" ou "P".

    Returns:
        Lista de nomes de datasets.
    """
    mask = (
        (classified["category"] == category)
        & (classified["metric"] == metric)
        & (classified["eligible"] == True)
        & (classified["direction"] == direction)
    )
    return sorted(
        classified.loc[mask, "dataset"].drop_duplicates().tolist()
    )


def plot_rfair_bar_by_direction_faceted(
    df: pd.DataFrame,
    classified: pd.DataFrame,
    metric: str,
    category: str,
    ideal_values: Dict,
    translate: Dict = None,
    save_path: Optional[str] = None,
    figsize_per_ax: Tuple[float, float] = (4, None),
    show: bool = True,
    ncols: int = 4,
) -> Optional[plt.Figure]:
    r"""Faceted horizontal bar chart: r\_fair por dataset × método, agrupado por direção U/P.

    Um subplot por método de mitigação.  Em cada subplot, datasets elegíveis
    são divididos em U (salmon) e P (gold), separados por um gap visual.
    O eixo X mostra *r\_fair* (fração de mitigação relativa) e o rótulo Y
    inclui o *Δ\_fair* do método correspondente:  ``Dataset (Δ=0.12)``.

    Parameters
    ----------
    df : DataFrame
        DataFrame principal (folds × repetições).
    classified : DataFrame
        Saída de :func:`classify_bias_patterns`.
    metric : str
        Nome da métrica.
    category : str
        ``"A"`` ou ``"C"``.
    ideal_values : dict
        Valores ideais por métrica.
    translate : dict, optional
        Tradução de nomes (métricas, métodos, datasets).
    save_path : str, optional
        Caminho para salvar (PDF/PNG).
    figsize_per_ax : tuple
        ``(largura, altura)`` por faceta.  Se altura=None, calcula
        automaticamente a partir do número de datasets.
    show : bool
        Se True, ``plt.show()``.
    ncols : int
        Colunas no grid (default 4).

    Returns
    -------
    plt.Figure or None
        Retorna None se não houver dados suficientes.
    """
    from .rq3_utils import build_intensity_scatter_data

    translate = translate or {}
    metric_label = translate.get(metric, metric)

    # ── 1. Obter datasets elegíveis por direção (nível métrica) ──────────
    ds_u = _get_datasets_by_metric_direction(classified, category, metric, "U")
    ds_p = _get_datasets_by_metric_direction(classified, category, metric, "P")
    all_eligible = set(ds_u) | set(ds_p)
    if not all_eligible:
        return None

    # Mapear dataset → direção
    ds_direction: Dict[str, str] = {}
    for ds in ds_u:
        ds_direction[ds] = "U"
    for ds in ds_p:
        ds_direction[ds] = "P"

    # ── 2. Construir scatter data (por método) ──────────────────────────
    scatter = build_intensity_scatter_data(
        df, classified, [metric], ideal_values,
        scope=category, level="method",
    )
    if scatter.empty:
        return None

    scatter = scatter[scatter["dataset"].isin(all_eligible)].copy()
    scatter["direction"] = scatter["dataset"].map(ds_direction)
    scatter = scatter.dropna(subset=["mitigation_rel", "direction"])
    if scatter.empty:
        return None

    # ── 3. Determinar métodos e ordem ────────────────────────────────────
    methods = [
        m for m in EXP_ORDER
        if m not in BASELINE_IDS and m in scatter["method"].unique()
    ]
    if not methods:
        return None

    # ── 4. Montar ordem do eixo Y: P (bottom) → gap → U (top) ──────────
    # Ordenar por mediana de r_fair (ascendente) dentro de cada grupo
    med_rfair = (
        scatter.groupby(["dataset", "direction"])["mitigation_rel"]
        .median()
        .reset_index()
    )

    p_ds = sorted(
        [d for d in all_eligible if ds_direction.get(d) == "P"],
        key=lambda d: med_rfair.loc[med_rfair["dataset"] == d, "mitigation_rel"].values[0]
        if len(med_rfair.loc[med_rfair["dataset"] == d]) > 0 else 0,
    )
    u_ds = sorted(
        [d for d in all_eligible if ds_direction.get(d) == "U"],
        key=lambda d: med_rfair.loc[med_rfair["dataset"] == d, "mitigation_rel"].values[0]
        if len(med_rfair.loc[med_rfair["dataset"] == d]) > 0 else 0,
    )

    gap_size = 0.8
    positions: Dict[str, float] = {}
    bar_colors: Dict[str, str] = {}
    bar_edge_colors: Dict[str, str] = {}
    pos = 0.0
    for ds in p_ds:
        positions[ds] = pos
        bar_colors[ds] = _BAR_DIRECTION_COLORS["P"]
        bar_edge_colors[ds] = _BAR_DIRECTION_EDGE_COLORS["P"]
        pos += 1.0
    if p_ds and u_ds:
        pos += gap_size
    for ds in u_ds:
        positions[ds] = pos
        bar_colors[ds] = _BAR_DIRECTION_COLORS["U"]
        bar_edge_colors[ds] = _BAR_DIRECTION_EDGE_COLORS["U"]
        pos += 1.0

    ds_order = p_ds + u_ds  # bottom to top

    # ── 5. Grid de subplots ─────────────────────────────────────────────
    n = len(methods)
    nrows = -(-n // ncols)  # ceil division

    n_bars = len(ds_order)
    ax_h = figsize_per_ax[1] if figsize_per_ax[1] is not None else max(2.5, 0.5 * n_bars + 1.0)
    fw = figsize_per_ax[0] * ncols
    fh = ax_h * nrows

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fw, fh), squeeze=False,
        sharey=True, sharex=True,
    )

    bar_height = 0.7

    for idx, method in enumerate(methods):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        sub = scatter[scatter["method"] == method]

        # Para cada dataset, plotar barra
        y_positions = []
        rfair_values = []
        colors_list = []
        edge_list = []
        y_labels = []

        for ds in ds_order:
            ds_row = sub[sub["dataset"] == ds]
            if ds_row.empty:
                rfair_val = 0.0
                delta_val = np.nan
            else:
                rfair_val = ds_row["mitigation_rel"].values[0]
                delta_val = ds_row["delta_fair"].values[0]

            y_positions.append(positions[ds])
            rfair_values.append(rfair_val)
            colors_list.append(bar_colors[ds])
            edge_list.append(bar_edge_colors[ds])

            ds_label = translate.get(ds, ds)
            if np.isnan(delta_val):
                y_labels.append(ds_label)
            else:
                y_labels.append(
                    fr"{ds_label} ($\Delta_{{\mathrm{{fair}}}}$={delta_val:.2f})"
                )

        ax.barh(
            y_positions, rfair_values, height=bar_height,
            color=colors_list, edgecolor=edge_list, linewidth=1.0,
        )

        ax.axvline(0, color="#555555", linewidth=0.8, zorder=1)

        method_label = translate.get(method, method)
        ax.set_title(method_label, fontsize=_VIZ["title_fontsize"], fontweight="bold")

        ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])
        ax.xaxis.grid(True, alpha=0.3)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Y-axis labels only on first column
        if col == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=_VIZ["tick_fontsize"] + 3)
        else:
            ax.set_yticks(y_positions)

    # ── Ocultar facetas vazias ──────────────────────────────────────────
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # ── Labels globais ──────────────────────────────────────────────────
    fig.supxlabel(r"$r_{\mathrm{fair}}$", fontsize=_VIZ["label_fontsize"] + 1)

    # ── Legenda global (U/P) — acima dos subplots ───────────────────────
    legend_handles = []
    for d_key, d_label in [("U", "U"), ("P", "P")]:
        ds_list = ds_u if d_key == "U" else ds_p
        if ds_list:
            legend_handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=_BAR_DIRECTION_COLORS[d_key],
                    edgecolor=_BAR_DIRECTION_EDGE_COLORS[d_key],
                    linewidth=1.0,
                    label=d_label,
                )
            )

    # fig.suptitle(
        # f"{metric_label}",
        # fontsize=_VIZ["suptitle_fontsize"] + 1, y=1.02,
    # )
    fig.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(legend_handles), fontsize=_VIZ["legend_fontsize"] + 1,
        frameon=False,
    )

    for a in fig.get_axes():
        apply_comma_axes(a)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
