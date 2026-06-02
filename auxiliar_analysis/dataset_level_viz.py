"""
dataset_level_viz.py - VisualizaÃ§Ãµes dataset-level para anÃ¡lise de fairness

PRINCÃPIO: Evitar pseudo-replicaÃ§Ã£o
- Unidade de anÃ¡lise: DATASET (nÃ£o foldÃ—repetiÃ§Ã£oÃ—dataset)
- Para cada dataset: agregar foldsÃ—repetiÃ§Ãµes via mediana
- Plotar um ponto por dataset + resumo entre datasets (mediana + IQR)

CORES PADRONIZADAS:
- Baseline: Azul (#4E79A7)
- Pre-processing: Verde (#59A14F)
- In-processing: Laranja (#F28E2B)

CONFIGURAÃ‡ÃƒO DE FONTES:
- Todas as configuraÃ§Ãµes de fonte estÃ£o centralizadas em VIZ_CONFIG
- Use update_viz_config() para alterar configuraÃ§Ãµes globalmente
- Ou passe font_config={} para funÃ§Ãµes individuais
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Literal, Union
from scipy.stats import iqr
import warnings

from .analysis_utils import apply_comma_axes, format_caption, _replace_decimal_in_latex

# =============================================================================
# CONFIGURAÃ‡ÃƒO DE CORES E ESTILO
# =============================================================================

CATEGORY_COLORS = {
    "baseline": "#4E79A7",
    "baseline_typical": "#4E79A7",
    "pre_processing": "#59A14F",
    "in_processing": "#F28E2B",
}

EXP_COLORS = {
    "BLRA": "#4E79A7",
    "BLRU": "#6B9AC4",
    "BRFA": "#89B4D4",
    "BRFU": "#A7CEE2",
    "EDL": "#59A14F",
    "ERL": "#76B947",
    "IAD": "#f95a19",
    "IGLA": "#fa8733",
    "IGLU": "#fa772a",
    "IFG": "#f97f50",
    "IP": "#f96224",
    "IW": "#d04005",
}

EXP_MARKERS = {
    "BLRA": "o",
    "BLRU": "s",
    "BRFA": "^",
    "BRFU": "D",
    "EDL": "p",
    "ERL": "H",
    "IAD": "X",
    "IGLA": "P",
    "IGLU": "*",
    "IFG": "h",
    "IP": "v",
    "IW": "<",
}

EXP_ORDER = [
    "BLRA",
    "BLRU",
    "BRFA",
    "BRFU",
    "EDL",
    "ERL",
    "IAD",
    "IGLA",
    "IGLU",
    "IFG",
    "IP",
    "IW",
]
BASELINE_IDS = ["BLRA", "BLRU", "BRFA", "BRFU"]

CATEGORY_MAPPING = {
    "baseline": ["BLRA", "BLRU", "BRFA", "BRFU"],
    "pre_processing": ["EDL", "ERL"],
    "in_processing": ["IAD", "IGLA", "IGLU", "IFG", "IP", "IW"],
}

EXP_TO_CATEGORY = {exp: cat for cat, exps in CATEGORY_MAPPING.items() for exp in exps}

# =============================================================================
# CONFIGURAÃ‡ÃƒO CENTRALIZADA DE VISUALIZAÃ‡ÃƒO (inclui TODAS as fontes)
# =============================================================================

VIZ_CONFIG = {
    # --- Marcadores e pontos ---
    "point_alpha": 0.6,
    "point_size": 50,
    "summary_marker_size": 150,
    "whisker_linewidth": 2.5,
    "whisker_capsize": 8,
    # --- Linha ideal ---
    "ideal_color": "red",
    "ideal_linestyle": "--",
    "ideal_linewidth": 1.5,
    # --- Grid ---
    "grid_alpha": 0.3,
    # --- FONTES (centralizadas) ---
    "title_fontsize": 14,
    "axis_label_fontsize": 14,
    "xtick_fontsize": 12,
    "ytick_fontsize": 12,
    "legend_fontsize": 11,
    "legend_title_fontsize": 12,
    "annotation_fontsize": 11,
    # --- Legenda separada ---
    "legend_figsize": (3, 4),
    "legend_ncol": 1,
    "legend_markerscale": 1.2,
}


def update_viz_config(**kwargs) -> None:
    """
    Atualiza configuraÃ§Ãµes globais de visualizaÃ§Ã£o.

    Exemplo:
        update_viz_config(
            title_fontsize=18,
            axis_label_fontsize=16,
            xtick_fontsize=14,
            ytick_fontsize=14,
            legend_fontsize=12
        )
    """
    for key, value in kwargs.items():
        if key in VIZ_CONFIG:
            VIZ_CONFIG[key] = value
        else:
            warnings.warn(f"Chave '{key}' nÃ£o existe em VIZ_CONFIG. Ignorada.")


def get_font_config(override: Dict = None) -> Dict:
    """
    Retorna configuraÃ§Ãµes de fonte, com possibilidade de override local.

    Args:
        override: Dict com valores para sobrescrever temporariamente

    Returns:
        Dict com todas as configuraÃ§Ãµes de fonte

    Exemplo:
        fonts = get_font_config({"title_fontsize": 20})
        ax.set_title("Title", fontsize=fonts["title_fontsize"])
    """
    font_keys = [
        "title_fontsize",
        "axis_label_fontsize",
        "xtick_fontsize",
        "ytick_fontsize",
        "legend_fontsize",
        "legend_title_fontsize",
        "annotation_fontsize",
    ]
    config = {k: VIZ_CONFIG[k] for k in font_keys}

    if override:
        config.update(override)

    return config


def _slugify(s: str) -> str:
    """Converte string para slug."""
    import re

    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s[:120]


# =============================================================================
# FUNÃ‡ÃƒO AUXILIAR: SALVAR LEGENDA SEPARADAMENTE
# =============================================================================


def _save_legend_separately(
    ax: plt.Axes,
    save_path: Union[str, Path],
    legend_figsize: Tuple[float, float] = None,
    legend_ncol: int = None,
    legend_markerscale: float = None,
    legend_fontsize: float = None,
    dpi: int = 300,
    format: str = "pdf",
) -> Path:
    """
    Salva a legenda de um eixo como arquivo separado.

    Args:
        ax: Eixo matplotlib com a legenda
        save_path: Caminho do arquivo principal (a legenda serÃ¡ salva com sufixo _legend)
        legend_figsize: Tamanho da figura da legenda
        legend_ncol: NÃºmero de colunas na legenda
        legend_markerscale: Escala dos marcadores na legenda
        legend_fontsize: Tamanho da fonte na legenda
        dpi: ResoluÃ§Ã£o
        format: Formato do arquivo (pdf, png, etc.)

    Returns:
        Path do arquivo da legenda salvo
    """
    # Usar valores do VIZ_CONFIG se nÃ£o especificados
    legend_figsize = legend_figsize or VIZ_CONFIG.get("legend_figsize", (3, 4))
    legend_ncol = legend_ncol or VIZ_CONFIG.get("legend_ncol", 1)
    legend_markerscale = legend_markerscale or VIZ_CONFIG.get("legend_markerscale", 1.2)
    legend_fontsize = legend_fontsize or VIZ_CONFIG.get("legend_fontsize", 11)

    # Obter handles e labels do eixo
    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        warnings.warn("Nenhum item de legenda encontrado no eixo.")
        return None

    # Construir caminho da legenda
    save_path = Path(save_path)
    legend_path = save_path.with_name(f"{save_path.stem}_legend{save_path.suffix}")

    # Criar figura separada para a legenda
    fig_leg = plt.figure(figsize=legend_figsize)
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")

    # Criar legenda centralizada
    legend = ax_leg.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        markerscale=legend_markerscale,
    )

    for a in fig_leg.get_axes():
        apply_comma_axes(a)
    fig_leg.tight_layout()

    # Salvar
    legend_path.parent.mkdir(parents=True, exist_ok=True)
    fig_leg.savefig(legend_path, dpi=dpi, bbox_inches="tight", format=format)
    plt.close(fig_leg)

    return legend_path


# =============================================================================
# AGREGAÃ‡ÃƒO DATASET-LEVEL
# =============================================================================


def aggregate_to_dataset_level(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "exp_id",
    value_col: str = "value_num",
    ideal_values: Optional[Dict] = None,
    use_gap: bool = False,  # backwards compatibility
    transform: Literal["raw", "abs", "gap"] = "raw",
) -> pd.DataFrame:
    """
    Agrega dados ao nível de dataset.

    Para cada (dataset, grupo):
    1) Calcula a mediana sobre folds×repetições
    2) Opcionalmente transforma o valor:
       - raw: valor bruto (assinado/razão)
       - abs: |valor| (útil p/ métricas assinadas com ideal=0, e.g., SPD, FNRD)
       - gap: distância ao ideal:
           * Para métricas de diferença (ideal=0): |valor|
           * Para métricas de razão (ideal=1): |valor - 1|
           * Para bias_amplification: max(valor, 0) (one-sided)

    Args:
        df: DataFrame com dados brutos
        metric: Métrica a filtrar
        group_col: Coluna de agrupamento ("exp_id" ou "category")
        value_col: Coluna de valores
        ideal_values: Dict com valores ideais por métrica
        use_gap: DEPRECATED - usar transform="gap"
        transform: Tipo de transformação ("raw", "abs", "gap")

    Returns:
        DataFrame com colunas: dataset, group, value
    """
    ideal_values = ideal_values or {}

    # Backwards compatibility
    if use_gap:
        warnings.warn(
            "use_gap is deprecated, use transform='gap' instead", DeprecationWarning
        )
        transform = "gap"

    df_metric = df[df["metric"] == metric].copy()
    if df_metric.empty:
        return pd.DataFrame(columns=["dataset", "group", "value"])

    # Agregar por dataset × grupo: mediana
    agg = (
        df_metric.groupby(["dataset", group_col], as_index=False)
        .agg(value=(value_col, "median"))
        .rename(columns={group_col: "group"})
    )

    # Transformação pós-agregação
    if transform == "raw":
        return agg

    if transform == "abs":
        # ABS: valor absoluto simples
        # Adequado para métricas de diferença com ideal=0 (SPD, FNRD, ERD, etc.)
        # AVISO: para métricas de razão (DI, consistency), abs não representa
        # distância ao ideal - use transform="gap" para essas métricas
        if metric == "bias_amplification":
            # BA: one-sided (amplificação = valores > 0)
            agg["value"] = agg["value"].clip(lower=0)
        else:
            agg["value"] = agg["value"].abs()
        return agg

    if transform == "gap":
        ideal = ideal_values.get(metric, 0)

        if metric == "bias_amplification":
            # BA: ideal ≤ 0, penalizar apenas amplificação (valores > 0)
            agg["value"] = agg["value"].clip(lower=0)
        else:
            # Distância ao ideal: |valor - ideal|
            agg["value"] = (agg["value"] - ideal).abs()
        return agg

    raise ValueError(f"Unknown transform='{transform}'. Use 'raw', 'abs' or 'gap'.")


def _get_ylabel(
    metric: str,
    transform: Literal["raw", "abs", "gap"],
    translate: Dict = None,
    ideal_values: Dict = None,
) -> str:
    """
    Gera o label do eixo Y baseado na métrica e transformação.

    Args:
        metric: Nome da métrica
        transform: Tipo de transformação aplicada
        translate: Dict de tradução de nomes
        ideal_values: Dict com valores ideais (para mostrar no label de gap)

    Returns:
        String formatada para ylabel
    """
    translate = translate or {}
    ideal_values = ideal_values or {}

    metric_label = translate.get(metric, metric)

    if transform == "raw":
        return metric_label

    if transform == "abs":
        return f"|{metric_label}|"

    if transform == "gap":
        ideal = ideal_values.get(metric)
        if ideal is not None:
            return f"Gap to ideal ({ideal}): {metric_label}"
        return f"Gap to ideal: {metric_label}"

    return metric_label


def aggregate_by_category(
    df: pd.DataFrame,
    metric: str,
    value_col: str = "value_num",
    ideal_values: Dict = None,
    use_gap: bool = False,
    min_baselines: int = 3,
) -> pd.DataFrame:
    """
    Agrega dados ao nÃ­vel de dataset por CATEGORIA.

    Para baseline_typical: mediana dos 4 baselines por split, depois mediana por dataset.
    Para pre/in_processing: mediana dos mÃ©todos da categoria por split, depois mediana por dataset.

    Args:
        df: DataFrame com dados brutos
        metric: MÃ©trica a filtrar
        value_col: Coluna de valores
        ideal_values: Dict com valores ideais
        use_gap: Se True, calcula gap ao ideal
        min_baselines: MÃ­nimo de baselines por split para formar baseline_typical

    Returns:
        DataFrame com colunas: dataset, category, value
    """
    ideal_values = ideal_values or {}

    df_metric = df[df["metric"] == metric].copy()

    if df_metric.empty:
        return pd.DataFrame(columns=["dataset", "category", "value"])

    # Garantir coluna category
    if "category" not in df_metric.columns:
        df_metric["category"] = df_metric["exp_id"].map(EXP_TO_CATEGORY)

    split_cols = ["dataset", "fold", "repetition_id"]

    results = []

    for dataset in df_metric["dataset"].unique():
        df_ds = df_metric[df_metric["dataset"] == dataset]

        # BASELINE_TYPICAL: mediana dos baselines por split, depois mediana entre splits
        baseline_data = df_ds[df_ds["exp_id"].isin(BASELINE_IDS)]
        if not baseline_data.empty:
            baseline_by_split = baseline_data.groupby(split_cols, as_index=False).agg(
                split_median=(value_col, "median"), n_baselines=(value_col, "count")
            )
            # Invalidar splits com poucos baselines
            baseline_by_split.loc[
                baseline_by_split["n_baselines"] < min_baselines, "split_median"
            ] = np.nan
            baseline_value = baseline_by_split["split_median"].median()

            results.append(
                {
                    "dataset": dataset,
                    "category": "baseline_typical",
                    "value": baseline_value,
                }
            )

        # PRE_PROCESSING e IN_PROCESSING
        for cat_name in ["pre_processing", "in_processing"]:
            cat_ids = CATEGORY_MAPPING.get(cat_name, [])
            cat_data = df_ds[df_ds["exp_id"].isin(cat_ids)]

            if cat_data.empty:
                continue

            # Mediana da categoria por split, depois mediana entre splits
            cat_by_split = cat_data.groupby(split_cols, as_index=False).agg(
                split_median=(value_col, "median")
            )
            cat_value = cat_by_split["split_median"].median()

            results.append(
                {"dataset": dataset, "category": cat_name, "value": cat_value}
            )

    result_df = pd.DataFrame(results)

    # Calcular gap se solicitado
    if use_gap and not result_df.empty:
        ideal = ideal_values.get(metric, 0)
        if metric == "bias_amplification":
            result_df["value"] = result_df["value"].clip(lower=0)
        else:
            result_df["value"] = (result_df["value"] - ideal).abs()

    return result_df


def compute_group_summary(
    df_agg: pd.DataFrame, group_col: str = "group"
) -> pd.DataFrame:
    """
    Calcula resumo estatÃ­stico entre datasets para cada grupo.

    Args:
        df_agg: DataFrame agregado por dataset (output de aggregate_to_dataset_level)
        group_col: Coluna de agrupamento

    Returns:
        DataFrame com: group, median, q25, q75, iqr, n_datasets
    """
    summary = df_agg.groupby(group_col, as_index=False).agg(
        median=("value", "median"),
        q25=("value", lambda x: np.nanpercentile(x, 25)),
        q75=("value", lambda x: np.nanpercentile(x, 75)),
        n_datasets=("value", "count"),
    )
    summary["iqr"] = summary["q75"] - summary["q25"]
    return summary


# =============================================================================
# VISUALIZAÃ‡Ã•ES POR CATEGORIA (DATASET-LEVEL)
# =============================================================================


def plot_category_dataset_level(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict = None,
    translate: Dict = None,
    use_gap: bool = False,
    category_order: List[str] = None,
    y_lim: Tuple = None,
    figsize: Tuple = (8, 6),
    title: str = None,
    save_path: str = None,
    show: bool = True,
    seed: int = 42,
    font_config: Dict = None,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """
    Boxplot/strip por CATEGORIA com agregaÃ§Ã£o dataset-level.

    - Um ponto por dataset (mediana intra-dataset)
    - Marcador maior = mediana entre datasets
    - Whisker = IQR (Q25-Q75)

    Args:
        df: DataFrame com dados brutos
        metric: MÃ©trica a plotar
        ideal_values: Dict com valores ideais
        translate: Dict de traduÃ§Ã£o
        use_gap: Se True, plota gap ao ideal
        category_order: Ordem das categorias no eixo X
        y_lim: Limites do eixo Y
        figsize: Tamanho da figura
        title: TÃ­tulo customizado
        save_path: Caminho para salvar
        show: Se True, exibe o grÃ¡fico
        seed: Seed para jitter
        font_config: Dict para sobrescrever configuraÃ§Ãµes de fonte

    Returns:
        Tuple (figura, DataFrame com dados agregados)
    """
    ideal_values = ideal_values or {}
    translate = translate or {}
    fonts = get_font_config(font_config)

    if category_order is None:
        category_order = ["baseline_typical", "pre_processing", "in_processing"]

    rng = np.random.default_rng(seed)

    # Agregar por categoria
    df_agg = aggregate_by_category(
        df, metric, ideal_values=ideal_values, use_gap=use_gap
    )

    if df_agg.empty:
        warnings.warn(f"Sem dados para mÃ©trica '{metric}'")
        return None, pd.DataFrame()

    # Calcular resumo por categoria
    summary = compute_group_summary(df_agg, group_col="category")

    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)

    # Plotar cada categoria
    for i, cat in enumerate(category_order):
        cat_data = df_agg[df_agg["category"] == cat]["value"].dropna().values
        cat_summary = summary[summary["category"] == cat]

        if len(cat_data) == 0:
            continue

        color = CATEGORY_COLORS.get(cat, "#999999")

        # Pontos individuais (um por dataset) com jitter
        x_jitter = rng.uniform(-0.15, 0.15, size=len(cat_data)) + i
        ax.scatter(
            x_jitter,
            cat_data,
            c=color,
            s=VIZ_CONFIG["point_size"],
            alpha=VIZ_CONFIG["point_alpha"],
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

        # Marcador de mediana (maior)
        if len(cat_summary) > 0:
            median = cat_summary["median"].values[0]
            q25 = cat_summary["q25"].values[0]
            q75 = cat_summary["q75"].values[0]

            ax.scatter(
                [i],
                [median],
                c=color,
                s=VIZ_CONFIG["summary_marker_size"],
                marker="D",
                edgecolors="black",
                linewidths=1.5,
                zorder=4,
            )

            # Whisker (IQR)
            ax.errorbar(
                [i],
                [median],
                yerr=[[median - q25], [q75 - median]],
                fmt="none",
                ecolor=color,
                elinewidth=VIZ_CONFIG["whisker_linewidth"],
                capsize=VIZ_CONFIG["whisker_capsize"],
                capthick=VIZ_CONFIG["whisker_linewidth"],
                zorder=3,
            )

    # Linha de valor ideal
    if not use_gap and metric in ideal_values:
        ideal = ideal_values[metric]
        ax.axhline(
            ideal,
            color=VIZ_CONFIG["ideal_color"],
            linestyle=VIZ_CONFIG["ideal_linestyle"],
            linewidth=VIZ_CONFIG["ideal_linewidth"],
            label=f"Ideal ({ideal})",
            zorder=1,
        )
        ax.legend(loc="upper right", fontsize=fonts["legend_fontsize"])

    # ConfiguraÃ§Ã£o dos eixos
    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(
        [translate.get(c, c) for c in category_order],
        rotation=0,
        fontsize=fonts["xtick_fontsize"],
    )

    if y_lim:
        ax.set_ylim(y_lim)

    metric_label = translate.get(metric, metric)
    ylabel = f"Gap to ideal: {metric_label}" if use_gap else metric_label
    ax.set_ylabel(ylabel, fontsize=fonts["axis_label_fontsize"])
    ax.tick_params(axis="y", labelsize=fonts["ytick_fontsize"])

    if title:
        ax.set_title(title, fontsize=fonts["title_fontsize"])

    ax.grid(axis="y", alpha=VIZ_CONFIG["grid_alpha"])
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Legenda explicativa
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            alpha=0.6,
            label="Individual Dataset",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            label="Median ± IQR",
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right", fontsize=fonts["legend_fontsize"]
    )

    for a in fig.get_axes():
        apply_comma_axes(a)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, df_agg


def plot_category_comparison_panel(
    df: pd.DataFrame,
    metrics: List[str],
    ideal_values: Dict = None,
    translate: Dict = None,
    use_gap: bool = False,
    category_order: List[str] = None,
    ncols: int = 3,
    figsize_per_panel: Tuple = (4, 4),
    save_path: str = None,
    show: bool = True,
    seed: int = 42,
    font_config: Dict = None,
) -> Optional[plt.Figure]:
    """
    Painel de mÃºltiplas mÃ©tricas comparando categorias (dataset-level).
    """
    ideal_values = ideal_values or {}
    translate = translate or {}
    fonts = get_font_config(font_config)

    if category_order is None:
        category_order = ["baseline_typical", "pre_processing", "in_processing"]

    rng = np.random.default_rng(seed)

    n_metrics = len(metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        df_agg = aggregate_by_category(
            df, metric, ideal_values=ideal_values, use_gap=use_gap
        )

        if df_agg.empty:
            ax.set_title(
                f"{translate.get(metric, metric)}\n(sem dados)",
                fontsize=fonts["title_fontsize"],
            )
            ax.set_visible(False)
            continue

        summary = compute_group_summary(df_agg, group_col="category")

        for i, cat in enumerate(category_order):
            cat_data = df_agg[df_agg["category"] == cat]["value"].dropna().values
            cat_summary = summary[summary["category"] == cat]

            if len(cat_data) == 0:
                continue

            color = CATEGORY_COLORS.get(cat, "#999999")

            # Pontos individuais
            x_jitter = rng.uniform(-0.12, 0.12, size=len(cat_data)) + i
            ax.scatter(
                x_jitter,
                cat_data,
                c=color,
                s=30,
                alpha=0.5,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )

            # Mediana + IQR
            if len(cat_summary) > 0:
                median = cat_summary["median"].values[0]
                q25 = cat_summary["q25"].values[0]
                q75 = cat_summary["q75"].values[0]

                ax.scatter(
                    [i],
                    [median],
                    c=color,
                    s=80,
                    marker="D",
                    edgecolors="black",
                    linewidths=1,
                    zorder=4,
                )
                ax.errorbar(
                    [i],
                    [median],
                    yerr=[[median - q25], [q75 - median]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=2,
                    capsize=5,
                    zorder=3,
                )

        # Linha ideal
        if not use_gap and metric in ideal_values:
            ax.axhline(
                ideal_values[metric],
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

        ax.set_xticks(range(len(category_order)))
        ax.set_xticklabels(
            [c[0].upper() for c in category_order], fontsize=fonts["xtick_fontsize"]
        )
        ax.set_title(translate.get(metric, metric), fontsize=fonts["title_fontsize"])
        ax.tick_params(axis="y", labelsize=fonts["ytick_fontsize"])
        ax.grid(axis="y", alpha=0.3)

    # Esconder eixos vazios
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    for a in fig.get_axes():
        apply_comma_axes(a)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# =============================================================================
# VISUALIZAÃ‡Ã•ES POR MÃ‰TODO (DATASET-LEVEL)
# =============================================================================


def plot_method_dataset_level(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict = None,
    translate: Dict = None,
    use_gap: bool = False,
    exp_order: List[str] = None,
    y_lim: Tuple = None,
    figsize: Tuple = (12, 6),
    title: str = None,
    save_path: str = None,
    show: bool = True,
    seed: int = 42,
    transform: Literal["raw", "abs", "gap"] = "raw",
    font_config: Dict = None,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """
    Boxplot/strip por MÉTODO com agregação dataset-level.

    - Um ponto por dataset (mediana intra-dataset)
    - Marcador maior = mediana entre datasets
    - Whisker = IQR (Q25-Q75)

    Args:
        df: DataFrame com dados brutos
        metric: Métrica a plotar
        ideal_values: Dict com valores ideais
        translate: Dict de tradução
        use_gap: DEPRECATED - usar transform="gap"
        exp_order: Ordem dos experimentos no eixo X
        y_lim: Limites do eixo Y
        figsize: Tamanho da figura
        title: Título customizado
        save_path: Caminho para salvar
        show: Se True, exibe o gráfico
        seed: Seed para jitter
        transform: Tipo de transformação ("raw", "abs", "gap")
        font_config: Dict para sobrescrever configurações de fonte

    Returns:
        Tuple (figura, DataFrame com dados agregados)
    """
    ideal_values = ideal_values or {}
    translate = translate or {}
    fonts = get_font_config(font_config)

    # Backwards compatibility
    if use_gap:
        warnings.warn(
            "use_gap is deprecated, use transform='gap' instead", DeprecationWarning
        )
        transform = "gap"

    if exp_order is None:
        exp_order = EXP_ORDER

    rng = np.random.default_rng(seed)

    # Agregar por método
    df_agg = aggregate_to_dataset_level(
        df,
        metric,
        group_col="exp_id",
        ideal_values=ideal_values,
        transform=transform,
    )

    if df_agg.empty:
        warnings.warn(f"No data for metric '{metric}'")
        return None, pd.DataFrame()

    # Filtrar métodos presentes
    present_methods = [m for m in exp_order if m in df_agg["group"].unique()]

    if not present_methods:
        warnings.warn(f"No methods found for '{metric}'")
        return None, pd.DataFrame()

    # Calcular resumo
    summary = compute_group_summary(df_agg, group_col="group")

    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(present_methods):
        method_data = df_agg[df_agg["group"] == method]["value"].dropna().values
        method_summary = summary[summary["group"] == method]

        if len(method_data) == 0:
            continue

        color = EXP_COLORS.get(method, "#999999")
        marker = EXP_MARKERS.get(method, "o")

        # Pontos individuais (um por dataset) com jitter
        x_jitter = rng.uniform(-0.2, 0.2, size=len(method_data)) + i
        ax.scatter(
            x_jitter,
            method_data,
            c=color,
            s=VIZ_CONFIG["point_size"],
            marker=marker,
            alpha=VIZ_CONFIG["point_alpha"],
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

        # Marcador de mediana + IQR
        if len(method_summary) > 0:
            median = method_summary["median"].values[0]
            q25 = method_summary["q25"].values[0]
            q75 = method_summary["q75"].values[0]

            ax.scatter(
                [i],
                [median],
                c=color,
                s=VIZ_CONFIG["summary_marker_size"],
                marker=marker,
                edgecolors="black",
                linewidths=1.5,
                zorder=4,
            )

            ax.errorbar(
                [i],
                [median],
                yerr=[[median - q25], [q75 - median]],
                fmt="none",
                ecolor=color,
                elinewidth=VIZ_CONFIG["whisker_linewidth"],
                capsize=VIZ_CONFIG["whisker_capsize"],
                capthick=VIZ_CONFIG["whisker_linewidth"],
                zorder=3,
            )

    # Linha de valor ideal (apenas para raw)
    if transform == "raw" and metric in ideal_values:
        ideal = ideal_values[metric]
        ax.axhline(
            ideal,
            color=VIZ_CONFIG["ideal_color"],
            linestyle=VIZ_CONFIG["ideal_linestyle"],
            linewidth=VIZ_CONFIG["ideal_linewidth"],
            label=f"Ideal ({ideal})",
            zorder=1,
        )
        ax.legend(loc="upper right", fontsize=fonts["legend_fontsize"])

    # Configuração dos eixos
    ax.set_xticks(range(len(present_methods)))
    ax.set_xticklabels(present_methods, rotation=0, fontsize=fonts["xtick_fontsize"])

    if y_lim:
        ax.set_ylim(y_lim)

    # Label do eixo Y dinâmico
    ylabel = _get_ylabel(metric, transform, translate, ideal_values)
    ax.set_ylabel(ylabel, fontsize=fonts["axis_label_fontsize"])
    ax.tick_params(axis="y", labelsize=fonts["ytick_fontsize"])

    if title:
        ax.set_title(title, fontsize=fonts["title_fontsize"])

    ax.grid(axis="y", alpha=VIZ_CONFIG["grid_alpha"])

    # Colorir fundo por categoria
    _add_category_background(ax, present_methods)

    for a in fig.get_axes():
        apply_comma_axes(a)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, df_agg


def _add_category_background(ax, methods: List[str], alpha: float = 0.08):
    """Adiciona fundo colorido por categoria."""
    current_cat = None
    start_idx = 0

    for i, method in enumerate(methods + [None]):  # None para fechar Ãºltimo grupo
        if method is None:
            cat = None
        else:
            cat = EXP_TO_CATEGORY.get(method)

        if cat != current_cat:
            if current_cat is not None:
                color = CATEGORY_COLORS.get(current_cat, "#999999")
                ax.axvspan(start_idx - 0.5, i - 0.5, color=color, alpha=alpha, zorder=0)
            start_idx = i
            current_cat = cat


# =============================================================================
# VISUALIZAÃ‡Ã•ES TRADEOFF FAIRNESSÃ—UTILITY (DATASET-LEVEL)
# =============================================================================


def aggregate_tradeoff_dataset_level(
    df: pd.DataFrame,
    fairness_metric: str,
    performance_metric: str = "f1",
    group_col: str = "exp_id",
    ideal_values: Dict = None,
    use_gap_fairness: bool = True,
) -> pd.DataFrame:
    """
    Agrega dados de tradeoff ao nÃ­vel de dataset.

    Para cada (dataset, grupo):
    - Calcula mediana de fairness e performance sobre foldsÃ—repetiÃ§Ãµes

    Returns:
        DataFrame com: dataset, group, fairness, performance
    """
    ideal_values = ideal_values or {}

    # Fairness
    df_fair = df[df["metric"] == fairness_metric].copy()
    fair_agg = df_fair.groupby(["dataset", group_col], as_index=False).agg(
        fairness=("value_num", "median")
    )

    # Performance
    df_perf = df[df["metric"] == performance_metric].copy()
    perf_agg = df_perf.groupby(["dataset", group_col], as_index=False).agg(
        performance=("value_num", "median")
    )

    # Merge
    merged = pd.merge(fair_agg, perf_agg, on=["dataset", group_col], how="inner")
    merged = merged.rename(columns={group_col: "group"})

    # Calcular gap de fairness se solicitado
    if use_gap_fairness:
        ideal = ideal_values.get(fairness_metric, 0)
        if fairness_metric == "bias_amplification":
            merged["fairness"] = merged["fairness"].clip(lower=0)
        else:
            merged["fairness"] = (merged["fairness"] - ideal).abs()

    return merged


def plot_tradeoff_quadrant_dataset_level(
    df: pd.DataFrame,
    fairness_metric: str,
    performance_metric: str = "f1",
    ideal_values: Dict = None,
    translate: Dict = None,
    use_gap_fairness: bool = True,
    group_by: Literal["exp_id", "category"] = "exp_id",
    figsize: Tuple = (10, 8),
    save_path: str = None,
    show: bool = True,
    seed: int = 42,
    font_config: Dict = None,
    # --- Novos parÃ¢metros para legenda separada ---
    save_legend_separately: bool = False,
    legend_figsize: Tuple[float, float] = None,
    legend_ncol: int = None,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """
    Quadrante de tradeoff fairnessÃ—utility com agregaÃ§Ã£o dataset-level.

    - Um ponto por dataset
    - Marcador maior = mediana entre datasets
    - Crosshair = IQR em ambas as dimensÃµes

    Args:
        df: DataFrame com dados brutos
        fairness_metric: MÃ©trica de fairness
        performance_metric: MÃ©trica de performance
        ideal_values: Dict com valores ideais
        translate: Dict de traduÃ§Ã£o
        use_gap_fairness: Se True, usa gap ao ideal para fairness
        group_by: Agrupar por "exp_id" ou "category"
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe o grÃ¡fico
        seed: Seed para jitter
        font_config: Dict para sobrescrever configuraÃ§Ãµes de fonte
        save_legend_separately: Se True, salva a legenda como arquivo separado
        legend_figsize: Tamanho da figura da legenda (se salva separadamente)
        legend_ncol: NÃºmero de colunas na legenda (se salva separadamente)

    Returns:
        Tuple (figura, DataFrame com dados agregados)
    """
    ideal_values = ideal_values or {}
    translate = translate or {}
    fonts = get_font_config(font_config)

    rng = np.random.default_rng(seed)

    # Agregar ao nÃ­vel de dataset
    if group_by == "category":
        df_agg = _aggregate_tradeoff_by_category(
            df,
            fairness_metric,
            performance_metric,
            ideal_values=ideal_values,
            use_gap_fairness=use_gap_fairness,
        )
        groups = ["baseline_typical", "pre_processing", "in_processing"]
        color_map = CATEGORY_COLORS
        marker_map = {
            "baseline_typical": "s",
            "pre_processing": "^",
            "in_processing": "o",
        }
    else:
        df_agg = aggregate_tradeoff_dataset_level(
            df,
            fairness_metric,
            performance_metric,
            group_col="exp_id",
            ideal_values=ideal_values,
            use_gap_fairness=use_gap_fairness,
        )
        groups = [m for m in EXP_ORDER if m in df_agg["group"].unique()]
        color_map = EXP_COLORS
        marker_map = EXP_MARKERS

    if df_agg.empty:
        warnings.warn(
            f"Sem dados para tradeoff {fairness_metric} Ã— {performance_metric}"
        )
        return None, pd.DataFrame()

    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)

    for group in groups:
        group_data = df_agg[df_agg["group"] == group]

        if group_data.empty:
            continue

        x_vals = group_data["fairness"].dropna().values
        y_vals = group_data["performance"].dropna().values

        if len(x_vals) == 0:
            continue

        color = color_map.get(group, "#999999")
        marker = marker_map.get(group, "o")

        # Pontos individuais (um por dataset)
        ax.scatter(
            x_vals,
            y_vals,
            c=color,
            marker=marker,
            s=VIZ_CONFIG["point_size"],
            alpha=VIZ_CONFIG["point_alpha"],
            edgecolors="white",
            linewidths=0.5,
            label=translate.get(group, group),
            zorder=2,
        )

        # Mediana + crosshair IQR
        med_x = np.median(x_vals)
        med_y = np.median(y_vals)
        q25_x, q75_x = np.percentile(x_vals, [25, 75])
        q25_y, q75_y = np.percentile(y_vals, [25, 75])

        # Marcador central
        ax.scatter(
            [med_x],
            [med_y],
            c=color,
            marker=marker,
            s=VIZ_CONFIG["summary_marker_size"],
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )

        # Crosshair (IQR em X e Y)
        ax.errorbar(
            [med_x],
            [med_y],
            xerr=[[med_x - q25_x], [q75_x - med_x]],
            yerr=[[med_y - q25_y], [q75_y - med_y]],
            fmt="none",
            ecolor=color,
            elinewidth=2,
            capsize=5,
            capthick=2,
            zorder=3,
        )

    # Labels
    fair_label = translate.get(fairness_metric, fairness_metric)
    perf_label = translate.get(performance_metric, performance_metric)

    xlabel = f"Gap to the ideal: {fair_label}" if use_gap_fairness else fair_label
    ax.set_xlabel(xlabel, fontsize=fonts["axis_label_fontsize"])
    ax.set_ylabel(perf_label, fontsize=fonts["axis_label_fontsize"])
    ax.set_title(
        f"Trade-off {fair_label} — {perf_label}\n(Dataset-level: median ± IQR)",
        fontsize=fonts["title_fontsize"],
    )
    ax.tick_params(axis="both", labelsize=fonts["xtick_fontsize"])

    ax.grid(True, alpha=VIZ_CONFIG["grid_alpha"])

    # Legenda: no plot ou salva separadamente
    if save_legend_separately and save_path:
        # Salvar legenda separadamente
        legend_path = _save_legend_separately(
            ax,
            save_path,
            legend_figsize=legend_figsize,
            legend_ncol=legend_ncol,
            legend_fontsize=fonts["legend_fontsize"],
        )
        # Remover legenda do plot principal (se existir)
        legend = ax.get_legend()
        if legend:
            legend.remove()
    else:
        # Legenda no plot
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=fonts["legend_fontsize"],
        )

    for a in fig.get_axes():
        apply_comma_axes(a)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, df_agg


def _aggregate_tradeoff_by_category(
    df: pd.DataFrame,
    fairness_metric: str,
    performance_metric: str,
    ideal_values: Dict = None,
    use_gap_fairness: bool = True,
) -> pd.DataFrame:
    """Agrega tradeoff por categoria ao nÃ­vel de dataset."""
    ideal_values = ideal_values or {}

    # Fairness por categoria
    fair_agg = aggregate_by_category(
        df, fairness_metric, ideal_values=ideal_values, use_gap=use_gap_fairness
    )
    fair_agg = fair_agg.rename(columns={"value": "fairness"})

    # Performance por categoria (sem gap)
    perf_agg = aggregate_by_category(
        df, performance_metric, ideal_values={}, use_gap=False
    )
    perf_agg = perf_agg.rename(columns={"value": "performance"})

    # Merge
    merged = pd.merge(fair_agg, perf_agg, on=["dataset", "category"], how="inner")
    merged = merged.rename(columns={"category": "group"})

    return merged


def plot_tradeoff_relative_quadrant_dataset_level(
    df: pd.DataFrame,
    fairness_metric: str,
    performance_metric: str = "f1",
    ideal_values: Dict = None,
    translate: Dict = None,
    use_gap_fairness: bool = True,
    figsize: Tuple = (10, 8),
    save_path: str = None,
    show: bool = True,
    seed: int = 42,
    font_config: Dict = None,
    percentual_reference: bool = False,
    # --- Parâmetro para anotação de quadrante ---
    fairness_direction: Literal[None, "standard", "inverted"] = None,
    # --- Novos parâmetros para legenda separada ---
    save_legend_separately: bool = False,
    legend_figsize: Tuple[float, float] = None,
    legend_ncol: int = None,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """
    Quadrante de tradeoff RELATIVO ao baseline (dataset-level).

    Diferença = valor_método - mediana_baseline (por dataset)

    Args:
        df: DataFrame com dados brutos
        fairness_metric: Métrica de fairness
        performance_metric: Métrica de performance
        ideal_values: Dict com valores ideais
        translate: Dict de tradução
        use_gap_fairness: Se True, usa gap ao ideal para fairness
        figsize: Tamanho da figura
        save_path: Caminho para salvar
        show: Se True, exibe o gráfico
        seed: Seed para jitter
        font_config: Dict para sobrescrever configurações de fonte
        percentual_reference: Se True, calcula diferença percentual
        fairness_direction: Controla a exibição das anotações de quadrante:
            - None: não exibe anotações (padrão)
            - "standard": Δ gap < 0 significa melhor fairness (redução do gap)
            - "inverted": Δ gap > 0 significa melhor fairness (ex: viés invertido)
        save_legend_separately: Se True, salva a legenda como arquivo separado
        legend_figsize: Tamanho da figura da legenda (se salva separadamente)
        legend_ncol: Número de colunas na legenda (se salva separadamente)

    Returns:
        Tuple (figura, DataFrame com dados filtrados)

    """
    VIZ_CONFIG = {
        # --- Marcadores e pontos ---
        "point_alpha": 0.5,
        "point_size": 80,
        "summary_marker_size": 500,
        "whisker_linewidth": 2.5,
        "whisker_capsize": 8,
        # --- Linha ideal ---
        "ideal_color": "red",
        "ideal_linestyle": "--",
        "ideal_linewidth": 1.5,
        # --- Grid ---
        "grid_alpha": 0.3,
        # --- FONTES (centralizadas) ---
        "title_fontsize": 14,
        "axis_label_fontsize": 14,
        "xtick_fontsize": 12,
        "ytick_fontsize": 12,
        "legend_fontsize": 11,
        "legend_title_fontsize": 12,
        "annotation_fontsize": 11,
        # --- Legenda separada ---
        "legend_figsize": (3, 4),
        "legend_ncol": 1,
        "legend_markerscale": 1.2,
    }

    ideal_values = ideal_values or {}
    translate = translate or {}
    fonts = get_font_config(font_config)

    rng = np.random.default_rng(seed)

    # Agregar tradeoff completo
    df_agg = aggregate_tradeoff_dataset_level(
        df,
        fairness_metric,
        performance_metric,
        group_col="exp_id",
        ideal_values=ideal_values,
        use_gap_fairness=use_gap_fairness,
    )

    if df_agg.empty:
        warnings.warn(f"Sem dados para tradeoff relativo")
        return None, pd.DataFrame()

    # Calcular baseline por dataset
    baseline_df = df_agg[df_agg["group"].isin(BASELINE_IDS)]
    baseline_by_dataset = baseline_df.groupby("dataset", as_index=False).agg(
        baseline_fairness=("fairness", "median"),
        baseline_performance=("performance", "median"),
    )

    # Calcular diferença relativa
    df_merged = df_agg.merge(baseline_by_dataset, on="dataset", how="left")
    if percentual_reference:
        # Diferença percentual
        df_merged["diff_fairness"] = (
            (df_merged["fairness"] - df_merged["baseline_fairness"])
            / df_merged["baseline_fairness"].replace(0, np.nan)
        ) * 100
        df_merged["diff_performance"] = (
            (df_merged["performance"] - df_merged["baseline_performance"])
            / df_merged["baseline_performance"].replace(0, np.nan)
        ) * 100
    else:
        df_merged["diff_fairness"] = (
            df_merged["fairness"] - df_merged["baseline_fairness"]
        )
        df_merged["diff_performance"] = (
            df_merged["performance"] - df_merged["baseline_performance"]
        )

    # Filtrar apenas métodos (não baseline)
    methods = [m for m in EXP_ORDER if m not in BASELINE_IDS]
    df_methods = df_merged[df_merged["group"].isin(methods)]

    if df_methods.empty:
        warnings.warn("Sem dados para plotar")
        return None, pd.DataFrame()

    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        method_data = df_methods[df_methods["group"] == method]

        if method_data.empty:
            continue

        x_vals = method_data["diff_fairness"].dropna().values
        y_vals = method_data["diff_performance"].dropna().values

        if len(x_vals) == 0:
            continue

        color = EXP_COLORS.get(method, "#999999")
        marker = EXP_MARKERS.get(method, "o")

        # Pontos individuais
        ax.scatter(
            x_vals,
            y_vals,
            c=color,
            marker=marker,
            s=VIZ_CONFIG["point_size"],
            alpha=VIZ_CONFIG["point_alpha"],
            edgecolors="white",
            linewidths=0.5,
            label=method,
            zorder=2,
        )

        # Mediana + crosshair
        med_x = np.median(x_vals)
        med_y = np.median(y_vals)
        q25_x, q75_x = np.percentile(x_vals, [25, 75])
        q25_y, q75_y = np.percentile(y_vals, [25, 75])

        ax.scatter(
            [med_x],
            [med_y],
            c=color,
            marker=marker,
            s=250,
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )
        ax.errorbar(
            [med_x],
            [med_y],
            xerr=[[med_x - q25_x], [q75_x - med_x]],
            yerr=[[med_y - q25_y], [q75_y - med_y]],
            fmt="none",
            ecolor=color,
            elinewidth=2,
            capsize=4,
            zorder=3,
        )

    # Linhas de referência (baseline = 0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Labels
    fair_label = translate.get(fairness_metric, fairness_metric)
    perf_label = translate.get(performance_metric, performance_metric)

    ax.set_xlabel(
        rf"$\Delta$ Gap {fair_label} (vs.baseline)",
        fontsize=fonts["axis_label_fontsize"],
    )

    ax.set_ylabel(
        rf"$\Delta$ {perf_label} (vs.baseline)", fontsize=fonts["axis_label_fontsize"]
    )
    ax.tick_params(axis="both", labelsize=fonts["xtick_fontsize"])
    ax.set_title(
        f"Relative trade-off: {fair_label} — {perf_label}",
        fontsize=fonts["title_fontsize"],
    )

    # Anotações de quadrante (apenas se fairness_direction estiver definido)
    # Performance (Y): sempre top = better, bottom = worse
    # Fairness (X): depende do sentido do viés
    if fairness_direction is not None:
        if fairness_direction == "standard":
            # Δ gap < 0 = better fairness → melhor no lado esquerdo
            # Better/Better: top-left | Worse/Worse: bottom-right
            better_pos = (0.02, 0.98)
            better_ha, better_va = "left", "top"
            worse_pos = (0.98, 0.02)
            worse_ha, worse_va = "right", "bottom"
        elif fairness_direction == "inverted":
            # Δ gap > 0 = better fairness → melhor no lado direito
            # Better/Better: top-right | Worse/Worse: bottom-left
            better_pos = (0.98, 0.98)
            better_ha, better_va = "right", "top"
            worse_pos = (0.02, 0.02)
            worse_ha, worse_va = "left", "bottom"
        else:
            raise ValueError(
                f"fairness_direction deve ser None, 'standard' ou 'inverted', "
                f"recebido: {fairness_direction}"
            )

        ax.annotate(
            "Better fairness\nBetter performance",
            xy=better_pos,
            xycoords="axes fraction",
            ha=better_ha,
            va=better_va,
            fontsize=fonts["annotation_fontsize"],
            color="green",
            alpha=0.7,
        )

        ax.annotate(
            "Worse fairness\nWorse performance",
            xy=worse_pos,
            xycoords="axes fraction",
            ha=worse_ha,
            va=worse_va,
            fontsize=fonts["annotation_fontsize"],
            color="red",
            alpha=0.7,
        )

    ax.grid(True, alpha=VIZ_CONFIG["grid_alpha"])

    # Legenda: no plot ou salva separadamente
    if save_legend_separately and save_path:
        # Salvar legenda separadamente
        legend_path = _save_legend_separately(
            ax,
            save_path,
            legend_figsize=legend_figsize,
            legend_ncol=legend_ncol,
            legend_fontsize=fonts["legend_fontsize"],
        )
        # Remover legenda do plot principal (se existir)
        legend = ax.get_legend()
        if legend:
            legend.remove()
    else:
        # Legenda no plot
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=fonts["legend_fontsize"],
            title_fontsize=fonts["legend_title_fontsize"],
        )

    for a in fig.get_axes():
        apply_comma_axes(a)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return


# =============================================================================
# FUNÃ‡Ã•ES UTILITÃRIAS
# =============================================================================


def generate_all_category_plots(
    df: pd.DataFrame,
    metrics: List[str],
    ideal_values: Dict,
    translate: Dict = None,
    use_gap: bool = False,
    save_dir: str = None,
    show: bool = True,
    font_config: Dict = None,
) -> Dict[str, str]:
    """Gera plots por categoria para mÃºltiplas mÃ©tricas."""
    translate = translate or {}
    saved_paths = {}

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        save_path = None
        if save_dir:
            suffix = "_gap" if use_gap else ""
            save_path = (
                f"{save_dir}/category_dataset_level_{_slugify(metric)}{suffix}.pdf"
            )

        fig, _ = plot_category_dataset_level(
            df,
            metric,
            ideal_values=ideal_values,
            translate=translate,
            use_gap=use_gap,
            save_path=save_path,
            show=show,
            font_config=font_config,
        )

        if save_path and fig:
            saved_paths[metric] = save_path
            plt.close(fig)

    return saved_paths


def generate_all_method_plots(
    df: pd.DataFrame,
    metrics: List[str],
    ideal_values: Dict,
    translate: Dict = None,
    use_gap: bool = False,
    save_dir: str = None,
    show: bool = True,
    font_config: Dict = None,
) -> Dict[str, str]:
    """Gera plots por mÃ©todo para mÃºltiplas mÃ©tricas."""
    translate = translate or {}
    saved_paths = {}

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        save_path = None
        if save_dir:
            suffix = "_gap" if use_gap else ""
            save_path = (
                f"{save_dir}/method_dataset_level_{_slugify(metric)}{suffix}.pdf"
            )

        fig, _ = plot_method_dataset_level(
            df,
            metric,
            ideal_values=ideal_values,
            translate=translate,
            use_gap=use_gap,
            save_path=save_path,
            show=show,
            font_config=font_config,
        )

        if save_path and fig:
            saved_paths[metric] = save_path
            plt.close(fig)

    return saved_paths


def print_dataset_level_summary(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict = None,
    group_by: Literal["category", "exp_id"] = "category",
) -> pd.DataFrame:
    """Imprime resumo estatÃ­stico dataset-level."""
    ideal_values = ideal_values or {}

    if group_by == "category":
        df_agg = aggregate_by_category(
            df, metric, ideal_values=ideal_values, use_gap=True
        )
        group_col = "category"
    else:
        df_agg = aggregate_to_dataset_level(
            df, metric, group_col="exp_id", ideal_values=ideal_values, use_gap=True
        )
        group_col = "group"

    if df_agg.empty:
        print(f"Sem dados para {metric}")
        return pd.DataFrame()

    summary = df_agg.groupby(group_col, as_index=False).agg(
        n_datasets=("value", "count"),
        median_gap=("value", "median"),
        q25=("value", lambda x: np.percentile(x, 25)),
        q75=("value", lambda x: np.percentile(x, 75)),
        iqr=("value", lambda x: iqr(x)),
    )

    print(f"\nðŸ“Š Resumo Dataset-Level: {metric}")
    print("=" * 60)
    print(summary.to_markdown(index=False, floatfmt=".4f"))

    return summary
