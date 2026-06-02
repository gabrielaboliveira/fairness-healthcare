"""
rq0_utils.py — utilitários para RQ0.1 (diagnóstico de fairness no baseline)

Funções disponíveis:
- METRIC_SPECS: configuração central das métricas de fairness do paper
- build_metric_characterization_table: tabela de caracterização com limiares de severidade
- build_baseline_diagnostic_table: tabela descritiva do baseline por métrica
- compute_gating_coverage: cobertura do gating por métrica/categoria
- plot_baseline_gap_overview: panorama de d_baseline (distância ao ideal) por métrica (horizontal)
- plot_baseline_components_gap: comparação dos 4 componentes do baseline (d ao ideal, horizontal)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .analysis_utils import latex_and_save
from .bias_pattern_classifier import (
    compute_distance,
    DEFAULT_RATIO_METRICS,
)


# =========================================================================
# CONFIGURAÇÃO CENTRAL DAS MÉTRICAS
# =========================================================================

METRIC_SPECS: Dict[str, Dict] = {
    # --- Grupo A: baseadas em paridade ---
    "disparate_impact": {
        "group": "A",
        "group_label": "Paridade estatística",
        "range": r"$[0, \infty)$",
        "ideal": 1.0,
        "threshold_type": "ratio_symmetric",
        # IQR fixo = 0.25: com medium_units=1.0, d = 0.25 → 1/(1.25) = 0.80
        # (four-fifths rule: impacto moderado a partir de DI = 0.8)
        "fixed_iqr": 0.25,
    },
    "bias_amplification": {
        "group": "A",
        "group_label": "Paridade estatística",
        "range": r"$(-\infty, \infty)$",
        "ideal": 0.0,
        "ideal_note": r"$\leq 0$",
        "threshold_type": "above_zero",
    },
    # --- Grupo B: fairness individual ---
    "consistency": {
        "group": "B",
        "group_label": "Fairness individual",
        "range": r"$[0, 1]$",
        "ideal": 1.0,
        "threshold_type": "below_ideal",
    },
    "generalized_entropy_index": {
        "group": "B",
        "group_label": "Fairness individual",
        "range": r"$[0, \infty)$",
        "ideal": 0.0,
        "threshold_type": "above_zero",
    },
    # --- Grupo C: baseadas em matriz de confusão ---
    "false_negative_rate_difference": {
        "group": "C",
        "group_label": "Matriz de confusão",
        "range": r"$(-\infty, \infty)$",
        "ideal": 0.0,
        "threshold_type": "abs_symmetric",
    },
    "false_discovery_rate_difference": {
        "group": "C",
        "group_label": "Matriz de confusão",
        "range": r"$(-\infty, \infty)$",
        "ideal": 0.0,
        "threshold_type": "abs_symmetric",
    },
    "error_rate_difference": {
        "group": "C",
        "group_label": "Matriz de confusão",
        "range": r"$(-\infty, \infty)$",
        "ideal": 0.0,
        "threshold_type": "abs_symmetric",
    },
}

# Ordem canônica de apresentação (grupo A → B → C)
METRIC_ORDER = [
    "disparate_impact",
    "bias_amplification",
    "consistency",
    "generalized_entropy_index",
    "false_negative_rate_difference",
    "false_discovery_rate_difference",
    "error_rate_difference",
]


# =========================================================================
# CONVERSÃO DISTÂNCIA → LIMIARES EM VALORES BRUTOS
# =========================================================================

def _distance_to_raw_thresholds(
    d_lo: float,
    d_hi: float,
    ideal: float,
    threshold_type: str,
) -> str:
    """
    Converte uma faixa de distância [d_lo, d_hi) em limiares expressos nos
    valores brutos da métrica, respeitando a semântica de cada tipo.

    Args:
        d_lo: limite inferior da distância (inclusive)
        d_hi: limite superior da distância (exclusive), ou None para ≥
        ideal: valor ideal da métrica
        threshold_type: tipo de conversão:
            - "abs_symmetric": |valor| em torno de 0 → ex: |0.07 a 0.14|
            - "above_zero": ideal ≤ 0, severidade cresce acima de 0 → ex: 0.17 a 0.35
            - "below_ideal": ideal=1, faixa [0,1], severidade cresce abaixo → ex: 0.78 a 0.89
            - "ratio_symmetric": ideal=1, simétrico → ex: 0.80 a 0.90 / 1.10 a 1.20

    Returns:
        String formatada para exibição na tabela
    """
    if np.isnan(d_lo):
        return "—"

    if threshold_type == "abs_symmetric":
        # Diferenças centradas em 0: severidade em |valor|
        # Faixa [d_lo, d_hi) em módulo
        if d_hi is not None:
            return f"|{d_lo:.2f} a {d_hi:.2f}|"
        return f"> |{d_lo:.2f}|"

    elif threshold_type == "above_zero":
        # DFBA / GEI: ideal=0 (ou ≤0), severidade cresce acima de 0
        if d_hi is not None:
            return f"{d_lo:.2f} a {d_hi:.2f}"
        return f"> {d_lo:.2f}"

    elif threshold_type == "below_ideal":
        # Consistência: ideal=1, [0,1], severidade = ideal - valor
        # distância d = ideal - valor → valor = ideal - d
        val_hi = ideal - d_lo   # menor distância → valor mais alto
        val_lo = ideal - d_hi if d_hi is not None else None
        if val_lo is not None:
            return f"{val_lo:.2f} a {val_hi:.2f}"
        return f"< {val_hi:.2f}"

    elif threshold_type == "ratio_symmetric":
        # DI: ideal=1, distância multiplicativa: d = max(v/ideal, ideal/v) - 1
        # Inversão:
        #   acima do ideal: v = ideal * (1 + d)
        #   abaixo do ideal: v = ideal / (1 + d)
        below_hi = ideal / (1 + d_lo)   # menor distância → mais perto de 1
        below_lo = ideal / (1 + d_hi) if d_hi is not None else None
        above_lo = ideal * (1 + d_lo)
        above_hi = ideal * (1 + d_hi) if d_hi is not None else None

        if d_hi is not None:
            return (
                f"{below_lo:.2f} a {below_hi:.2f}\n"
                f"{above_lo:.2f} a {above_hi:.2f}"
            )
        return f"< {below_hi:.2f}\nou > {above_lo:.2f}"

    # fallback
    if d_hi is not None:
        return f"{d_lo:.2f} a {d_hi:.2f}"
    return f"> {d_lo:.2f}"


# =========================================================================
# TABELA DE CARACTERIZAÇÃO DE MÉTRICAS + LIMIARES DE SEVERIDADE
# =========================================================================

def build_metric_characterization_table(
    baseline_ds: pd.DataFrame,
    translate: Dict[str, str] = None,
    ratio_metrics: Set[str] = None,
    min_units: float = 0.5,
    medium_units: float = 1.0,
    high_units: float = 1.5,
    iqr_overrides: Dict[str, float] = None,
    value_col: str = "value_num",
    result_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gera tabela de caracterização das métricas de fairness com limiares de
    severidade expressos em **valores brutos** da métrica.

    Reproduz a tabela da imagem de referência (Tabela 6.3):
    | Grupo | Faixa | Ideal | Métricas | n | Mediana | IQR |
    | Baixo impacto | Impacto moderado | Impacto severo |

    Os limiares são calculados como:
      1. S = IQR das distâncias ao ideal entre datasets (ou IQR fixo, se fornecido)
      2. Faixas em unidades de S: baixo [min_units, medium_units),
         moderado [medium_units, high_units), severo ≥ high_units
      3. Distâncias convertidas em valores brutos conforme a semântica:
         - abs_symmetric (diferenças): |d_lo a d_hi|
         - above_zero (DFBA, GEI): d_lo a d_hi (acima de 0)
         - below_ideal (consistência): ideal-d_hi a ideal-d_lo
         - ratio_symmetric (DI): ideal±d com faixas acima e abaixo

    Args:
        baseline_ds: DataFrame dataset-level com colunas [dataset, metric, value_col]
        translate: dicionário de tradução de nomes
        ratio_metrics: métricas que usam distância multiplicativa
        min_units / medium_units / high_units: limiares em unidades de IQR
        iqr_overrides: IQR fixo por métrica (ex: {"disparate_impact": 0.2})
            Se None, usa METRIC_SPECS["fixed_iqr"] quando definido,
            senão calcula empiricamente.
        value_col: coluna de valores
        result_dir: diretório para salvar via latex_and_save

    Returns:
        DataFrame com colunas de display + colunas auxiliares (_*)
    """
    translate = translate or {}
    ratio_metrics = ratio_metrics if ratio_metrics is not None else DEFAULT_RATIO_METRICS
    iqr_overrides = iqr_overrides or {}

    rows = []
    for metric in METRIC_ORDER:
        spec = METRIC_SPECS.get(metric)
        if spec is None:
            continue

        vals = baseline_ds.loc[baseline_ds["metric"] == metric, value_col].dropna()
        if vals.empty:
            continue

        ideal = spec["ideal"]
        use_mult = metric in ratio_metrics
        threshold_type = spec["threshold_type"]

        # Calcular distância ao ideal para cada dataset
        distances = vals.apply(lambda v: compute_distance(v, ideal, use_mult))

        n = len(vals)
        median_val = float(np.nanmedian(vals))
        iqr_val = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))

        # Escala S: IQR override > METRIC_SPECS.fixed_iqr > empírico
        if metric in iqr_overrides:
            d_iqr = float(iqr_overrides[metric])
        elif "fixed_iqr" in spec:
            d_iqr = float(spec["fixed_iqr"])
        else:
            d_iqr = float(
                np.nanpercentile(distances, 75) - np.nanpercentile(distances, 25)
            )

        # Limiares em distância
        if d_iqr > 0:
            low_lo = d_iqr * min_units
            low_hi = d_iqr * medium_units
            mod_lo = d_iqr * medium_units
            mod_hi = d_iqr * high_units
            sev_lo = d_iqr * high_units
        else:
            low_lo = low_hi = mod_lo = mod_hi = sev_lo = np.nan

        # Converter distâncias → valores brutos
        low_str = _distance_to_raw_thresholds(low_lo, low_hi, ideal, threshold_type)
        mod_str = _distance_to_raw_thresholds(mod_lo, mod_hi, ideal, threshold_type)
        sev_str = _distance_to_raw_thresholds(sev_lo, None, ideal, threshold_type)

        metric_label = translate.get(metric, metric)
        ideal_display = spec.get("ideal_note", str(spec["ideal"]))

        rows.append({
            "Grupo": spec["group_label"],
            "Categoria": spec["group"],
            "Faixa": spec["range"],
            "Ideal": ideal_display,
            "Métrica": metric_label,
            "n": n,
            "Mediana": round(median_val, 2),
            "IQR": round(iqr_val, 2),
            "Baixo impacto": low_str,
            "Impacto moderado": mod_str,
            "Impacto severo": sev_str,
            # Colunas auxiliares (uso programático)
            "_metric": metric,
            "_d_iqr": d_iqr,
            "_sev_low_lo": low_lo,
            "_sev_low_hi": low_hi,
            "_sev_mod_lo": mod_lo,
            "_sev_mod_hi": mod_hi,
            "_sev_hi_lo": sev_lo,
        })

    table = pd.DataFrame(rows)

    # Salvar se solicitado
    display_cols = [
        "Grupo", "Categoria", "Faixa", "Ideal", "Métrica", "n",
        "Mediana", "IQR", "Baixo impacto", "Impacto moderado", "Impacto severo",
    ]

    if result_dir is not None:
        latex_and_save(
            table[display_cols],
            caption=(
                f"Faixas de severidade e medidas-resumo por grupo de métricas. "
                f"Limiares em unidades de IQR da distância ao ideal: "
                f"baixo [{min_units}–{medium_units}), "
                f"moderado [{medium_units}–{high_units}), "
                f"severo ≥ {high_units}."
            ),
            label="tab_rq0_metric_characterization",
            result_dir=result_dir,
        )

    return table


# =========================================================================
# TABELA DESCRITIVA DO BASELINE POR MÉTRICA × DATASET
# =========================================================================

def build_baseline_diagnostic_table(
    baseline_ds: pd.DataFrame,
    translate: Dict[str, str] = None,
    value_col: str = "value_num",
    result_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Tabela descritiva do baseline: n, mediana, IQR, min, max por métrica.

    Args:
        baseline_ds: DataFrame dataset-level
        translate: dicionário de tradução
        value_col: coluna de valores
        result_dir: diretório para salvar LaTeX

    Returns:
        DataFrame com estatísticas descritivas
    """
    translate = translate or {}

    rows = []
    for metric in METRIC_ORDER:
        vals = baseline_ds.loc[baseline_ds["metric"] == metric, value_col].dropna()
        if vals.empty:
            continue
        rows.append({
            "Métrica": translate.get(metric, metric),
            "_metric": metric,
            "n": len(vals),
            "Mediana": round(float(np.nanmedian(vals)), 4),
            "IQR": round(float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)), 4),
            "Min": round(float(np.nanmin(vals)), 4),
            "Max": round(float(np.nanmax(vals)), 4),
        })

    table = pd.DataFrame(rows)

    if result_dir is not None:
        display_cols = ["Métrica", "n", "Mediana", "IQR", "Min", "Max"]
        latex_and_save(
            table[display_cols],
            caption="Estatísticas descritivas do baseline por métrica de fairness",
            label="tab_rq0_baseline_diagnostic",
            result_dir=result_dir,
        )

    return table


# =========================================================================
# COBERTURA DO GATING
# =========================================================================

def compute_gating_coverage(
    classified_df: pd.DataFrame,
    translate: Dict[str, str] = None,
    result_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reporta cobertura do gating: quantos datasets passam o limiar de elegibilidade
    por métrica e por categoria (A/B/C).

    Args:
        classified_df: output de classify_bias_patterns
        translate: dicionário de tradução
        result_dir: diretório para salvar LaTeX

    Returns:
        DataFrame com cobertura por métrica
    """
    translate = translate or {}

    rows = []
    for metric in METRIC_ORDER:
        m_data = classified_df[classified_df["metric"] == metric]
        if m_data.empty:
            continue
        n_total = len(m_data)
        n_eligible = int(m_data["own_eligible"].sum())
        spec = METRIC_SPECS.get(metric, {})

        rows.append({
            "Métrica": translate.get(metric, metric),
            "Categoria": spec.get("group", "?"),
            "Total datasets": n_total,
            "Elegíveis": n_eligible,
            "_metric": metric,
        })

    table = pd.DataFrame(rows)

    # Calcular elegíveis por categoria: número de datasets distintos elegíveis
    # em pelo menos uma métrica da categoria (união).
    cat_eligible_count = {}
    for cat in table["Categoria"].unique():
        cat_metrics = table.loc[table["Categoria"] == cat, "_metric"].tolist()
        cat_data = classified_df[
            (classified_df["metric"].isin(cat_metrics))
            & (classified_df["own_eligible"])
        ]
        cat_eligible_count[cat] = int(cat_data["dataset"].nunique())

    table["Eleg. cat."] = table["Categoria"].map(cat_eligible_count)
    n_datasets = table["Total datasets"].iloc[0] if len(table) > 0 else 1
    table["Cobertura (%)"] = table["Eleg. cat."].apply(
        lambda x: round(x / n_datasets * 100, 1)
    )

    if result_dir is not None:
        _save_gating_coverage_latex(table, result_dir)

    # Exibir markdown
    display_cols = [
        "Categoria", "Métrica", "Elegíveis", "Eleg. cat.", "Cobertura (%)",
    ]
    print(table[display_cols].to_markdown(index=False))

    return table


def _save_gating_coverage_latex(
    table: pd.DataFrame, result_dir: str,
) -> None:
    """Gera LaTeX customizado para a tabela de cobertura do gating."""
    from pathlib import Path

    results_dir = Path(result_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(
        r"\caption{Cobertura por categoria de \textit{fairness}. "
        r"As categorias são A~(paridade estatística), B~(\textit{fairness} individual) "
        r"e C~(matriz de confusão). "
        r"\emph{Elegíveis} indica quantos conjuntos de dados ultrapassam o limiar "
        r"de disparidade ($d(m) \geq d_{\min}$) para cada métrica; "
        r"\emph{Eleg.\ cat.} é o número de conjuntos de dados elegíveis em pelo menos "
        r"uma métrica da categoria (união); "
        r"\emph{Cobertura} é a fração correspondente do total de conjuntos de dados.}"
    )
    lines.append(r"\label{tab_rq0_gating_coverage}")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{clccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Categoria & Métrica & Elegíveis & Eleg.\ cat. & Cobertura (\%) \\"
    )
    lines.append(r"\midrule")

    # Agrupar por categoria mantendo ordem original
    seen_cats = []
    for _, row in table.iterrows():
        if row["Categoria"] not in seen_cats:
            seen_cats.append(row["Categoria"])

    for cat in seen_cats:
        cat_rows = table[table["Categoria"] == cat]
        n_rows = len(cat_rows)
        eleg_cat = int(cat_rows["Eleg. cat."].iloc[0])
        cov = cat_rows["Cobertura (%)"].iloc[0]

        for i, (_, row) in enumerate(cat_rows.iterrows()):
            parts = []
            if i == 0:
                parts.append(
                    rf"\multirow{{{n_rows}}}{{*}}{{{cat}}}"
                )
            else:
                parts.append("")

            parts.append(row["Métrica"])
            parts.append(str(int(row["Elegíveis"])))

            if i == 0:
                parts.append(
                    rf"\multirow{{{n_rows}}}{{*}}{{{eleg_cat}}}"
                )
                parts.append(
                    rf"\multirow{{{n_rows}}}{{*}}{{{cov:.1f}}}"
                )
            else:
                parts.append("")
                parts.append("")

            lines.append(" & ".join(parts) + r" \\")

        # Separador entre categorias (exceto após a última)
        if cat != seen_cats[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines) + "\n"
    from .analysis_utils import _replace_decimal_in_latex
    latex_str = _replace_decimal_in_latex(latex_str)
    latex_path = results_dir / "tab_rq0_gating_coverage.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"Tabela salva em: {latex_path}")


# =========================================================================
# VISUALIZAÇÕES DE PANORAMA DO BASELINE
# =========================================================================

# Tons de azul para contexto de baseline (reserva-se azul/verde/laranja
# para baseline vs pre-processing vs in-processing nas RQs seguintes).
# Grupo A: azul escuro, Grupo B: azul médio, Grupo C: azul claro
_BASELINE_GROUP_COLORS = {
    "A": "#1B4F72",  # azul escuro
    "B": "#2E86C1",  # azul médio
    "C": "#85C1E9",  # azul claro
}

# Componentes do baseline — tons mais escuros para RF, mais claros para LR;
# ligeiramente mais escuro para Aware.
_COMP_COLORS = {
    "BLRA":  "#2C5F8A",  # LR Aware — azul médio-escuro
    "BLRU":  "#6CA6CD",  # LR Unaware — azul médio-claro
    "BRFA": "#1A3A5C",  # RF Aware — azul bem escuro
    "BRFU": "#3D7EAA",  # RF Unaware — azul escuro
}
_COMP_LABELS = {
    "BLRA":  "BLRA",
    "BLRU":  "BLRU",
    "BRFA": "BRFA",
    "BRFU": "BRFU",
}
_COMP_MARKERS = {
    "BLRA":  "o",   # circle
    "BLRU":  "s",   # square
    "BRFA": "^",   # triangle up
    "BRFU": "D",   # diamond
}

# Configurações visuais consistentes com o restante do projeto
_VIZ = {
    "point_alpha": 0.6,
    "point_size": 50,
    "summary_marker_size": 130,
    "whisker_linewidth": 2.5,
    "whisker_capsize": 8,
    "jitter_range": 0.15,
    "title_fontsize": 15,
    "label_fontsize": 14,
    "tick_fontsize": 14,
    "legend_fontsize": 13,
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
        pad = 40 + 15 * nrows
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
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=fs, framealpha=0.9)


def _compute_gap_series(
    ds: pd.DataFrame,
    ratio_metrics: Set[str],
    value_col: str = "value_num",
) -> pd.DataFrame:
    """
    Adiciona coluna 'gap' ao DataFrame dataset-level.
    Gap = distância ao ideal (multiplicativa para ratio_metrics, aditiva para o resto).
    """
    ds = ds.copy()
    gaps = []
    for _, row in ds.iterrows():
        m = row["metric"]
        spec = METRIC_SPECS.get(m, {})
        ideal = spec.get("ideal", 0.0)
        use_mult = m in ratio_metrics
        gaps.append(compute_distance(row[value_col], ideal, use_mult))
    ds["gap"] = gaps
    return ds


def plot_baseline_gap_overview(
    baseline_ds: pd.DataFrame,
    translate: Dict[str, str] = None,
    ratio_metrics: Set[str] = None,
    value_col: str = "value_num",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Panorama da distância ao ideal do baseline para todas as métricas.

    Cada ponto = 1 dataset.  Scatter + mediana (◆) + IQR whiskers,
    orientação horizontal (métricas no eixo Y, d_baseline no eixo X).
    Cores: tons de azul por categoria (A/B/C).

    Args:
        baseline_ds: DataFrame dataset-level (baseline)
        translate: dicionário de tradução
        ratio_metrics: métricas com distância multiplicativa
        value_col: coluna de valores
        figsize: tamanho da figura
        save_path: caminho para salvar (PDF/PNG)

    Returns:
        matplotlib Figure
    """
    from matplotlib.lines import Line2D

    translate = translate or {}
    ratio_metrics = ratio_metrics if ratio_metrics is not None else DEFAULT_RATIO_METRICS

    ds = _compute_gap_series(baseline_ds, ratio_metrics, value_col)

    # Ordenar métricas conforme METRIC_ORDER (apenas as presentes), invertido para horizontal
    present = [m for m in METRIC_ORDER if m in ds["metric"].unique()]
    present_rev = list(reversed(present))  # para que a primeira fique no topo
    labels = [translate.get(m, m) for m in present_rev]
    groups = [METRIC_SPECS.get(m, {}).get("group", "?") for m in present_rev]

    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(len(present_rev))
    rng = np.random.default_rng(42)

    for i, (m, g) in enumerate(zip(present_rev, groups)):
        vals = ds.loc[ds["metric"] == m, "gap"].dropna().values
        color = _BASELINE_GROUP_COLORS.get(g, "#999999")

        if len(vals) == 0:
            continue

        median = float(np.nanmedian(vals))
        q25 = float(np.nanpercentile(vals, 25))
        q75 = float(np.nanpercentile(vals, 75))

        # Pontos individuais (scatter) — jitter vertical
        jitter = rng.uniform(-_VIZ["jitter_range"], _VIZ["jitter_range"], size=len(vals))
        ax.scatter(
            vals, positions[i] + jitter,
            c=color, s=_VIZ["point_size"], alpha=_VIZ["point_alpha"],
            edgecolors="white", linewidths=0.5, zorder=2,
        )

        # IQR whisker — horizontal
        ax.errorbar(
            [median], [positions[i]],
            xerr=[[median - q25], [q75 - median]],
            fmt="none",
            ecolor=color, elinewidth=_VIZ["whisker_linewidth"],
            capsize=_VIZ["whisker_capsize"],
            capthick=_VIZ["whisker_linewidth"],
            zorder=3,
        )

        # Mediana (diamond)
        ax.scatter(
            [median], [positions[i]],
            c=color, s=_VIZ["summary_marker_size"], marker="D",
            edgecolors="black", linewidths=1.5, zorder=4,
        )

    # Linha de referência d = 0
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$d_{\mathrm{baseline}}$", fontsize=_VIZ["label_fontsize"])
    # ax.set_title(
        # "Distância ao ideal no baseline por métrica",
        # fontsize=_VIZ["title_fontsize"],
    # )
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])

    # Legenda de grupos
    legend_handles = []
    seen = set()
    for m, g in zip(present_rev, groups):
        if g not in seen:
            seen.add(g)
            glabel = METRIC_SPECS.get(m, {}).get("group_label", g)
            legend_handles.append(
                Line2D([0], [0], marker="D", color="w",
                       markerfacecolor=_BASELINE_GROUP_COLORS.get(g, "#999"),
                       markeredgecolor="black", markersize=10,
                       label=f"{g}: {glabel}")
            )
    # Legenda de formato
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="gray", markersize=8, alpha=0.6,
               label="Conjunto de dados")
    )
    _place_legend(fig, ax, legend_handles)

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3)

    # Separador decimal → vírgula
    from .analysis_utils import apply_comma_axes
    apply_comma_axes(ax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura salva em: {save_path}")

    return fig


def plot_baseline_components_gap(
    df_fairness_baseline: pd.DataFrame,
    translate: Dict[str, str] = None,
    ratio_metrics: Set[str] = None,
    baseline_ids: List[str] = None,
    value_col: str = "value_num",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compara os 4 componentes do baseline (BLRA, BLRU, BRFA, BRFU) em termos
    da distância ao ideal, para cada métrica.

    Orientação horizontal (métricas no eixo Y, d_baseline no eixo X).
    Cores: tons de azul — mais escuros para RF, mais claros para LR;
    ligeiramente mais escuro para Aware.

    Args:
        df_fairness_baseline: DataFrame bruto do baseline (com folds/reps)
        translate: dicionário de tradução
        ratio_metrics: métricas com distância multiplicativa
        baseline_ids: IDs dos 4 componentes
        value_col: coluna de valores
        figsize: tamanho da figura
        save_path: caminho para salvar

    Returns:
        matplotlib Figure
    """
    from matplotlib.lines import Line2D

    translate = translate or {}
    ratio_metrics = ratio_metrics if ratio_metrics is not None else DEFAULT_RATIO_METRICS
    baseline_ids = baseline_ids or ["BLRA", "BLRU", "BRFA", "BRFU"]

    # Agregar ao nível dataset POR componente
    fb = df_fairness_baseline[df_fairness_baseline["exp_id"].isin(baseline_ids)].copy()
    comp_ds = (
        fb.groupby(["exp_id", "dataset", "metric"], as_index=False)
        .agg({value_col: "median"})
    )
    comp_ds = comp_ds[comp_ds["metric"].isin(METRIC_SPECS)].copy()
    comp_ds = _compute_gap_series(comp_ds, ratio_metrics, value_col)

    # Métricas presentes (invertidas para horizontal)
    present = [m for m in METRIC_ORDER if m in comp_ds["metric"].unique()]
    present_rev = list(reversed(present))
    n_metrics = len(present_rev)

    fig, ax = plt.subplots(figsize=figsize)

    n_comp = len(baseline_ids)
    spacing = 0.18
    offsets = np.linspace(
        -(n_comp - 1) * spacing / 2,
        (n_comp - 1) * spacing / 2,
        n_comp,
    )

    for j, comp_id in enumerate(baseline_ids):
        comp_data = comp_ds[comp_ds["exp_id"] == comp_id]
        color = _COMP_COLORS.get(comp_id, "#999999")
        marker = _COMP_MARKERS.get(comp_id, "o")

        rng = np.random.default_rng(42 + j)

        for i, m in enumerate(present_rev):
            vals = comp_data.loc[comp_data["metric"] == m, "gap"].dropna().values
            if len(vals) == 0:
                continue

            y_base = i + offsets[j]

            median = float(np.nanmedian(vals))
            q25 = float(np.nanpercentile(vals, 25))
            q75 = float(np.nanpercentile(vals, 75))

            # Pontos individuais
            jitter = rng.uniform(-spacing * 0.3, spacing * 0.3, size=len(vals))
            ax.scatter(
                vals, y_base + jitter,
                c=color, s=_VIZ["point_size"] * 0.7, alpha=_VIZ["point_alpha"],
                marker=marker,
                edgecolors="white", linewidths=0.4, zorder=2,
            )

            # IQR whisker — horizontal
            ax.errorbar(
                [median], [y_base],
                xerr=[[median - q25], [q75 - median]],
                fmt="none",
                ecolor=color, elinewidth=_VIZ["whisker_linewidth"] * 0.8,
                capsize=_VIZ["whisker_capsize"] * 0.6,
                capthick=_VIZ["whisker_linewidth"] * 0.8,
                zorder=3,
            )

            # Mediana 
            ax.scatter(
                [median], [y_base],
                c=color, s=_VIZ["summary_marker_size"] * 0.7,
                edgecolors="black", linewidths=1.2, zorder=4,
                marker=marker
            )

    # Referência d = 0
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Separadores visuais entre métricas
    for i in range(1, n_metrics):
        ax.axhline(i - 0.5, color="#DDDDDD", linewidth=0.5, zorder=0)

    labels = [translate.get(m, m) for m in present_rev]
    ax.set_yticks(np.arange(n_metrics))
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$d_{\mathrm{baseline}}$", fontsize=_VIZ["label_fontsize"])
    # ax.set_title(
        # "Componentes do baseline — distância ao ideal por métrica",
        # fontsize=_VIZ["title_fontsize"],
    # )
    ax.tick_params(axis="both", labelsize=_VIZ["tick_fontsize"])

    # Legenda dos componentes
    baseline_ids = sorted(
    baseline_ids,
    key=lambda x: _COMP_LABELS.get(x, x),
    reverse=True
    )
    legend_handles = [
        Line2D([0], [0], marker=_COMP_MARKERS.get(c, "o"), color="w",
               markerfacecolor=_COMP_COLORS.get(c, "#999"),
               markeredgecolor="black", markersize=9,
               label=_COMP_LABELS.get(c, c))
        for c in baseline_ids
    ]
    _place_legend(fig, ax, legend_handles)

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.3)

    # Separador decimal → vírgula
    from .analysis_utils import apply_comma_axes
    apply_comma_axes(ax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura salva em: {save_path}")

    return fig


# =========================================================================
# TABELA DE ELEGIBILIDADE E SENTIDO POR DATASET
# =========================================================================

def build_eligibility_direction_table(
    classified_df: pd.DataFrame,
    baseline_ds: pd.DataFrame,
    translate: Dict[str, str] = None,
    value_col: str = "value_num",
    result_dir: Optional[str] = None,
    dataset_order: Optional[List[str]] = None,
    min_units: float = 0.5,
) -> pd.DataFrame:
    r"""
    Tabela de distância ao ideal $d(m)$ por métrica e dataset, com limiar
    de elegibilidade e sentido de viés.

    Células mostram $d(m)$ arredondado a 2 casas decimais.  Se $d(m) \geq
    d_{\min}$, acrescenta-se sufixo com o sentido do viés (U, P ou
    $\checkmark$ para categoria B).  Se $d(m) < d_{\min}$, apenas o
    valor numérico.  Se $d(m) = 0$, exibe ``---``.

    Parameters
    ----------
    classified_df : pd.DataFrame
        Saída de ``classify_bias_patterns()``.  Deve conter as colunas
        ``distance``, ``scale_iqr``, ``own_eligible``, ``own_direction``.
    baseline_ds : pd.DataFrame
        Dados dataset-level do baseline (não utilizado diretamente na nova
        versão, mantido por compatibilidade de assinatura).
    translate : dict, optional
        Tradução de nomes de métricas e datasets.
    value_col : str
        Coluna de valores em ``baseline_ds`` (mantido por compatibilidade).
    result_dir : str, optional
        Diretório para salvar LaTeX.
    dataset_order : list[str], optional
        Ordem dos datasets nas colunas.  Se None, ordena alfabeticamente.
    min_units : float
        Limiar de elegibilidade em unidades de IQR (default 0.5).

    Returns
    -------
    pd.DataFrame
        DataFrame pronto para display.
    """
    from pathlib import Path
    from .bias_pattern_classifier import METRICS_WITH_DIRECTION

    translate = translate or {}

    # Se o DataFrame não possui scale_iqr, recalcular a partir de distance
    if "scale_iqr" not in classified_df.columns:
        _iqr_map = (
            classified_df.groupby("metric")["distance"]
            .apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
            .to_dict()
        )
        classified_df = classified_df.copy()
        classified_df["scale_iqr"] = classified_df["metric"].map(_iqr_map)

    # Datasets presentes
    all_datasets = sorted(classified_df["dataset"].unique())
    if dataset_order is not None:
        all_datasets = [d for d in dataset_order if d in set(all_datasets)]

    rows = []
    for metric in METRIC_ORDER:
        spec = METRIC_SPECS.get(metric)
        if spec is None:
            continue

        m_data = classified_df[classified_df["metric"] == metric]
        if m_data.empty:
            continue

        # Escala S (IQR das distâncias) — constante por métrica
        scale_iqr = float(m_data["scale_iqr"].iloc[0])
        d_min = round(min_units * scale_iqr, 2)

        ideal_display = spec.get("ideal_note", str(spec["ideal"]))
        metric_label = translate.get(metric, metric)
        cat_label = spec["group"]

        # Célula por dataset: d(m) + sufixo se elegível
        ds_cells = {}
        for ds in all_datasets:
            row_data = m_data[m_data["dataset"] == ds]
            if row_data.empty:
                ds_cells[ds] = "---"
                continue

            row_data = row_data.iloc[0]
            dist = float(row_data["distance"])
            dist_round = round(dist, 2)

            if dist_round == 0.0:
                ds_cells[ds] = "---"
            elif row_data["own_eligible"]:
                # Determinar sufixo de sentido
                if metric not in METRICS_WITH_DIRECTION:
                    suffix = "✓"
                else:
                    d = row_data["own_direction"]
                    suffix = d if d in ("U", "P") else "✓"
                ds_cells[ds] = (dist_round, suffix)
            else:
                ds_cells[ds] = (dist_round, None)

        row = {
            "category": cat_label,
            "metric": metric_label,
            "ideal": ideal_display,
            "d_min": d_min,
        }
        for ds in all_datasets:
            ds_label = translate.get(ds, ds)
            row[ds_label] = ds_cells[ds]

        rows.append(row)

    table = pd.DataFrame(rows)

    # Salvar LaTeX
    if result_dir is not None:
        _save_eligibility_direction_latex(
            table, all_datasets, translate, result_dir,
        )

    # Display (encoding-safe para terminais Windows)
    try:
        print(table.to_markdown(index=False))
    except UnicodeEncodeError:
        print(table.to_markdown(index=False).encode("utf-8", errors="replace").decode("utf-8"))

    return table


def _save_eligibility_direction_latex(
    table: pd.DataFrame,
    all_datasets: List[str],
    translate: Dict[str, str],
    result_dir: str,
) -> None:
    """Gera LaTeX customizado para a tabela de distância ao ideal."""
    from pathlib import Path

    results_dir = Path(result_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ds_headers = [translate.get(ds, ds) for ds in all_datasets]
    n_ds = len(ds_headers)

    # c para Categoria, l para Métrica, c para Ideal, c para d_min, c×n_ds
    col_spec = "@{} c l c c " + " c" * n_ds + " @{}"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(
        r"\caption{Distância ao ideal $d(m)$ no \textit{baseline}"
        r"por métrica e conjunto de dados. "
        r"Categorias: A~(paridade estatística), B~(fairness individual) e "
        r"C~(matriz de confusão). "
        r"A coluna $d_{\min}$ indica o limiar mínimo de distância para "
        r"elegibilidade ($0{,}5 \cdot S_m$). "
        r"Valores acima do limiar são acompanhados do sentido do viés: "
        r"U = contra o grupo não privilegiado; "
        r"P = contra o grupo privilegiado; "
        r"$(\checkmark)$ = elegível sem sentido direcional (categoria~B). "
        r"Valores abaixo do limiar não recebem marcação; "
        r"--- indica distância nula.}"
    )
    lines.append(r"\label{tab_rq0_eligibility_direction}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    hdr = [
        r"\textbf{Cat.}",
        r"\textbf{Métrica}",
        r"\textbf{Ideal}",
        r"$\boldsymbol{d_{\min}}$",
    ]
    hdr += [rf"\textbf{{{h}}}" for h in ds_headers]
    lines.append(" & ".join(hdr) + r" \\")
    lines.append(r"\midrule")

    # Body — multirow por categoria
    prev_cat = None
    for _, r in table.iterrows():
        cat_raw = r["category"]

        if cat_raw != prev_cat:
            n_rows_cat = int((table["category"] == cat_raw).sum())
            cat_cell = rf"\multirow{{{n_rows_cat}}}{{*}}{{\textbf{{{cat_raw}}}}}"
            if prev_cat is not None:
                lines.append(r"\midrule")
            prev_cat = cat_raw
        else:
            cat_cell = ""

        # Formatar células de dataset
        ds_vals = []
        for ds in all_datasets:
            ds_label = translate.get(ds, ds)
            v = r[ds_label]
            if v == "---":
                ds_vals.append("---")
            elif isinstance(v, tuple):
                dist_val, suffix = v
                if suffix is None:
                    ds_vals.append(f"{dist_val:.2f}")
                elif suffix == "✓":
                    ds_vals.append(
                        rf"{dist_val:.2f}$^{{\checkmark}}$"
                    )
                else:
                    ds_vals.append(
                        rf"{dist_val:.2f}$^{{\text{{{suffix}}}}}$"
                    )
            else:
                ds_vals.append(str(v))

        cells = [
            cat_cell,
            str(r["metric"]),
            str(r["ideal"]),
            f"{r['d_min']:.2f}",
        ] + ds_vals

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    latex_path = results_dir / "tab_rq0_eligibility_direction.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Tabela salva em: {latex_path}")
