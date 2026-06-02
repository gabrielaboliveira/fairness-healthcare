"""
analysis_utils.py - Funções utilitárias para análise de resultados de fairness

CONFIGURAÇÕES DE VISUALIZAÇÃO PADRONIZADAS:
--------------------------------------------
- Baseline: Azul (#4E79A7)
- Pre-processing: Verde (#59A14F)
- In-processing: Laranja (#F28E2B)
- Medianas: Pretas, espessura=2
- Valor ideal: Vermelho tracejado, espessura=1.5
- Jitter points: alfa=0.35
- Fundo boxplots: alfa=0.4
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import re
import textwrap
from typing import Dict, List, Union
from .config import CATEGORY_MAP, TRANSLATE

# =============================================================================
# CONFIGURAÇÃO DE ESTILO
# =============================================================================

EXPERIMENT_COLORS = {
    "baseline": "#4E79A7",
    "pre_processing": "#59A14F",
    "in_processing": "#F28E2B"
}

VIZ_CONFIG = {
    "median_color": "black",
    "median_linewidth": 2,
    "ideal_color": "red",
    "ideal_linestyle": "--",
    "ideal_linewidth": 1.5,
    "jitter_alpha": 0.35,
    "jitter_alpha_outlier": 0.12,
    "box_alpha": 0.4,
    "show_xgrid_boxplot": False,
    "show_ygrid_boxplot": True
}


def set_plot_style(
    palette=None,
    background="whitegrid",
    font="Arial",
    font_scale=1.0,
    line_width=1.5,
    rc_extra=None
):
    """Configura tema e cores padrão para todos os gráficos."""
    if palette is None:
        palette = [
            "#4E79A7", "#76B7B2", "#59A14F", "#F28E2B", "#E15759",
            "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
        ]
    sns.set_theme(style=background, font=font, font_scale=font_scale)
    sns.set_palette(palette)
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(color=palette),
        "axes.linewidth": line_width,
        "grid.alpha": 0.3,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })
    # Registrar ScalarFormatter customizado que troca ponto por vírgula.
    # Cada função de plot pode chamar apply_comma_axes(ax) para ativar.
    if rc_extra:
        plt.rcParams.update(rc_extra)


category_map = CATEGORY_MAP
translate = TRANSLATE


# =============================================================================
# FORMATAÇÃO DE CAPTIONS E SEPARADOR DECIMAL
# =============================================================================

# Termos em inglês que devem aparecer em itálico nos captions LaTeX.
_ENGLISH_TERMS = [
    "baseline", "fairness", "trade-off", "in-processing", "pre-processing",
    "aware", "unaware", "folds", "win-win", "lose-lose", "gating", "outlier",
    "gap",
]


def format_caption(raw_caption: str) -> str:
    r"""Aplica regras de padronização a um caption LaTeX.

    Regras aplicadas (conforme checklist 9a–9i):
    - Remove "típico" / "tipico" do texto (ex: "Baseline típico" → "Baseline")
    - Substitui "dataset(s)" / "Dataset(s)" por "conjunto(s) de dados"
    - Coloca termos em inglês em ``\textit{}`` (exceto os que já estão)
    """
    s = raw_caption

    # 9a: remover "típico"
    s = re.sub(r"\s+típico\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+tipico\b", "", s, flags=re.IGNORECASE)

    # 9h: dataset → conjunto de dados
    s = re.sub(r"\bDatasets\b", "conjuntos de dados", s)
    s = re.sub(r"\bdatasets\b", "conjuntos de dados", s)
    s = re.sub(r"\bDataset\b", "Conjunto de dados", s)
    s = re.sub(r"\bdataset\b", "conjunto de dados", s)

    # 9c: termos em inglês em itálico
    for term in _ENGLISH_TERMS:
        # Não envolver se já está dentro de \textit{} ou \textit{...term...}
        # Padrão: palavra isolada que NÃO está precedida por { ou seguida por }
        pattern = rf"(?<!\\textit{{)(?<!\{{)\b({re.escape(term)})\b(?!\}})"
        replacement = rf"\\textit{{\1}}"
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    # Limpar itálicos duplicados: \textit{\textit{x}} → \textit{x}
    s = re.sub(r"\\textit\{\\textit\{([^}]+)\}\}", r"\\textit{\1}", s)

    return s


def _comma_decimal_formatter(x, pos):
    """FuncFormatter callback: troca ponto por vírgula em tick labels."""
    # Formatar sem trailing zeros desnecessários
    txt = f"{x:g}"
    return txt.replace(".", ",")


def comma_formatter():
    """Retorna um FuncFormatter que usa vírgula como separador decimal."""
    return mticker.FuncFormatter(_comma_decimal_formatter)


def _axis_has_text_labels(axis) -> bool:
    """Detecta se um eixo tem labels de texto (não numéricos).

    Verifica o formatter interno: se for um partial wrapping
    ``_format_with_dict`` (criado por ``set_xticklabels``/``set_yticklabels``),
    ou se for FixedFormatter, os labels são textuais e não devem ser
    sobrescritos pelo comma formatter.
    """
    import functools
    import matplotlib.ticker as _mticker

    fmt = axis.get_major_formatter()

    # FixedFormatter é texto explícito
    if isinstance(fmt, _mticker.FixedFormatter):
        return True

    # set_yticklabels cria FuncFormatter(partial(_format_with_dict, {…}))
    if isinstance(fmt, _mticker.FuncFormatter):
        inner = getattr(fmt, 'func', None)
        if isinstance(inner, functools.partial):
            fname = getattr(inner.func, '__name__', '')
            if '_format_with_dict' in fname:
                return True

    return False


def apply_comma_axes(ax):
    """Aplica separador decimal vírgula apenas nos eixos numéricos de um Axes.

    Não sobrescreve eixos que já têm labels de texto (ex: nomes de métricas).
    """
    fmt = comma_formatter()
    if not _axis_has_text_labels(ax.xaxis):
        ax.xaxis.set_major_formatter(fmt)
    if not _axis_has_text_labels(ax.yaxis):
        ax.yaxis.set_major_formatter(fmt)


def _replace_decimal_in_latex(latex_str: str) -> str:
    r"""Substitui ponto decimal por vírgula em números dentro de tabelas LaTeX.

    Atua apenas em valores numéricos (ex: 0.1234, -3.14, 1.5e-3),
    preservando comandos LaTeX e texto.
    """
    # Padrão: número com ponto decimal, opcionalmente negativo, com parte inteira ou não
    # Captura contextos como "0.1234", "-0.05", ".999", "1.5e-3"
    # Evita substituir pontos em comandos LaTeX como \textit{...}
    def _replace(match):
        return match.group(0).replace(".", ",")

    return re.sub(
        r"(?<![a-zA-Z\\{])(-?\d*\.\d+(?:[eE][-+]?\d+)?)",
        _replace,
        latex_str,
    )


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

def load_results(
    datasets_to_run: dict,
    dataset_configs: dict,
    results_base: Path = None
) -> pd.DataFrame:
    """Carrega resultados de experimentos de múltiplos datasets."""
    if results_base is None:
        results_base = Path("results")
    
    dfs = []
    for key, run_value in datasets_to_run.items():
        if run_value and key in dataset_configs:
            dataset_code = dataset_configs[key]["dataset_code"]
            path_results = results_base / dataset_code
            if path_results.exists():
                for arquivo in path_results.glob("*.csv"):
                    dfs.append(pd.read_csv(arquivo))
    
    if not dfs:
        raise ValueError("Nenhum arquivo de resultado encontrado")
    return pd.concat(dfs, ignore_index=True)


def convert_value_to_numeric(
    df: pd.DataFrame,
    value_col: str = "value"
) -> pd.DataFrame:
    """Converte coluna de valores para numérico, tratando listas e strings."""
    import ast
    
    def to_number(x):
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = ast.literal_eval(s)
                if isinstance(arr, (list, tuple)) and len(arr) > 0:
                    return float(arr[0])
            except Exception:
                return np.nan
        s = s.replace(",", ".")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group()) if m else np.nan
    
    df = df.copy()
    df["value_num"] = df[value_col].map(to_number)
    return df



def add_experiment_metadata(
    df: pd.DataFrame,
    category_map: dict = category_map,
    exp_id_col: str = "exp_id",
    unknown: str = "raise",             
    group_labels: tuple[str, str] = ("baseline", "method"),  
) -> pd.DataFrame:

    # 0) validação simples: um mesmo exp_id não pode estar em duas categorias
    seen = {}
    duplicates = []
    for cat, ids in category_map.items():
        for _id in ids:
            if _id in seen and seen[_id] != cat:
                duplicates.append((_id, seen[_id], cat))
            seen[_id] = cat
    if duplicates:
        raise ValueError(f"exp_id presente em múltiplas categorias: {duplicates}")

    exp_to_category = {exp_id: cat for cat, ids in category_map.items() for exp_id in ids}

    df = df.copy()
    df["category"] = df[exp_id_col].map(exp_to_category)

    if unknown == "raise":
        missing = df.loc[df["category"].isna(), exp_id_col].unique()
        if len(missing) > 0:
            raise ValueError(
                f"{len(missing)} exp_id(s) sem categoria no category_map: {missing.tolist()}"
            )

    baseline_label, treatment_label = group_labels
    df["exp_type"] = np.where(df["category"].eq("baseline"), baseline_label, treatment_label)

    # compatibilidade com nomes antigos (opcional)
    df["experiment_pipeline_type"] = df["category"]

    return df


# =============================================================================
# ESTATÍSTICAS
# =============================================================================

def iqr_standard(x):
    """IQR padrão."""
    return x.quantile(0.75) - x.quantile(0.25)


def iqr_truncated(x, lower_q=0.01, upper_q=0.99):
    """IQR truncado para lidar com outliers extremos."""
    x = x.dropna()
    low, high = x.quantile(lower_q), x.quantile(upper_q)
    x_trim = x[(x >= low) & (x <= high)]
    if len(x_trim) < 2:
        return np.nan
    return x_trim.quantile(0.75) - x_trim.quantile(0.25)


def pct_gap_to_ideal(value: float, ideal: float) -> float:
    """Calcula gap percentual em relação ao valor ideal."""
    if pd.isna(ideal) or pd.isna(value):
        return np.nan
    if abs(ideal) < 1e-12:
        return abs(value) * 100.0
    return abs((value - ideal) / ideal) * 100.0


def compute_correlation(x, y, method="both"):
    """Calcula correlação entre duas séries."""
    result = {}
    if method in ("pearson", "both"):
        r, p = pearsonr(x, y)
        result["pearson_r"], result["pearson_p"] = r, p
    if method in ("spearman", "both"):
        r, p = spearmanr(x, y)
        result["spearman_r"], result["spearman_p"] = r, p
    return result


# =============================================================================
# CLASSIFICAÇÃO DE TIPOS DE PROBLEMA (BIAS PATTERNS)
# =============================================================================

PARITY_METRICS = ["disparate_impact", "statistical_parity_difference", "bias_amplification"]
INDIVIDUAL_METRICS = ["generalized_entropy_index", "consistency"]
CONFUSION_METRICS = ["false_negative_rate_difference", "false_discovery_rate_difference", "error_rate_difference"]

BIAS_PATTERN_RULES = {
    "A1": {
        "datasets": ["4_heart_disease", "11_arrhythmia"],
        "metrics": PARITY_METRICS,
        "severity": 1,
        "bias_direction": "unprivileged",
        "description": "Viés severo em paridade estatística contra grupo desprivilegiado"
    },
    "A2": {
        "datasets": ["15_mental", "16_diabetes", "6_aids"],
        "metrics": PARITY_METRICS,
        "severity": 2,
        "bias_direction": "privileged",
        "description": "Viés moderado em paridade estatística contra grupo privilegiado"
    },
    "A3": {
        "datasets": ["7_obesity"],
        "metrics": PARITY_METRICS,
        "severity": 3,
        "bias_direction": "privileged",
        "description": "Viés baixo em paridade estatística contra grupo privilegiado"
    },
    "B1": {
        "datasets": ["15_mental"],
        "metrics": INDIVIDUAL_METRICS,
        "severity": 1,
        "bias_direction": None,
        "description": "Problema severo de fairness individual"
    },
    "B2": {
        "datasets": ["4_heart_disease", "11_arrhythmia", "16_diabetes", "6_aids"],
        "metrics": INDIVIDUAL_METRICS,
        "severity": 2,
        "bias_direction": None,
        "description": "Problema moderado de fairness individual"
    },
    "C1": {
        "datasets": ["15_mental", "6_aids"],
        "metrics": CONFUSION_METRICS,
        "severity": 1,
        "bias_direction": "privileged",
        "description": "Viés em métricas de erro contra grupo privilegiado"
    },
    "C2": {
        "datasets": ["4_heart_disease"],
        "metrics": CONFUSION_METRICS,
        "severity": 2,
        "bias_direction": "unprivileged",
        "description": "Viés moderado em métricas de erro contra grupo desprivilegiado"
    },
    "C3": {
        "datasets": ["11_arrhythmia"],
        "metrics": CONFUSION_METRICS,
        "severity": 3,
        "bias_direction": "unprivileged",
        "description": "Viés baixo em métricas de erro contra grupo desprivilegiado"
    }
}


def classify_bias_patterns(
    df: pd.DataFrame,
    rules: dict = None
) -> pd.DataFrame:
    """Classifica cada linha do DataFrame por padrão de viés."""
    if rules is None:
        rules = BIAS_PATTERN_RULES
    
    df = df.copy()
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["dataset"] = df["dataset"].astype(str)
    df["problem_type"] = None
    
    for ptype, config in rules.items():
        mask = (
            df["dataset"].isin(config["datasets"]) &
            df["metric"].isin(config["metrics"])
        )
        df.loc[mask, "problem_type"] = ptype
    
    return df


def get_bias_pattern_info(pattern_code: str) -> dict:
    """Retorna informações sobre um padrão de viés específico."""
    return BIAS_PATTERN_RULES.get(pattern_code, {})

def build_bias_pattern_table(
    classified: pd.DataFrame,
    translate: dict = None,
    caption: str = "Conjuntos de dados por padrão final de viés (categoria, severidade e sentido).",
    label: str = "tab_rq0_bias_patterns_datasets",
    result_dir: str = "results",
) -> str:
    """Gera tabela LaTeX cruzando padrões de viés × conjuntos de dados.

    Lê o DataFrame ``classified`` (saída de ``classify_bias_patterns``)
    e produz uma tabela no formato:

        Padrão | Severidade | Sentido | DS1 | DS2 | …

    com ``\\multirow`` na coluna de padrão e ``x`` nas células onde
    o dataset pertence àquela combinação (categoria, severidade, sentido).

    Apenas combinações elegíveis (``eligible == True``) são incluídas.

    Parameters
    ----------
    classified : pd.DataFrame
        Saída de ``classify_bias_patterns()``.  Colunas esperadas:
        ``dataset``, ``category``, ``severity_level``, ``direction``,
        ``eligible``.
    translate : dict, optional
        Tradução de nomes de dataset
        (ex.: ``{"4_heart_disease": "Heart Disease"}``).
    caption, label, result_dir : str
        Metadados da tabela LaTeX gerada.

    Returns
    -------
    str
        Código LaTeX completo da tabela.
    """
    translate = translate or {}

    _CATEGORY_LABELS = {
        "A": r"Paridade Estatística (A)",
        "B": r"Individual (B)",
        "C": r"Matriz de Confusão (C)",
    }

    # ── 1. Filtrar apenas combinações elegíveis ──
    elig = classified[classified["eligible"]].copy()

    # Nível dataset × categoria: 1 linha por (dataset, category)
    # com a severidade e direção herdadas (já calculadas pelo classificador)
    ds_cat = (
        elig.groupby(["dataset", "category"], as_index=False)
        .agg(severity_level=("severity_level", "first"),
             direction=("direction", "first"))
    )

    # ── 2. Coletar todos os datasets (ordenados) ──
    all_datasets = sorted(ds_cat["dataset"].unique())
    ds_headers = [translate.get(ds, ds) for ds in all_datasets]

    # ── 3. Agrupar por (category, severity, direction) ──
    ds_cat["dir_label"] = ds_cat["direction"].map(
        {"U": "U", "P": "P"}
    ).fillna("---")

    groups = (
        ds_cat
        .sort_values(["category", "severity_level", "dir_label"])
        .groupby(["category", "severity_level", "dir_label"], sort=False)
        ["dataset"]
        .apply(set)
        .reset_index()
    )

    # Organizar por categoria
    from collections import OrderedDict
    categories: dict[str, list] = OrderedDict()
    for _, row in groups.iterrows():
        cat = row["category"]
        ds_set = row["dataset"]
        categories.setdefault(cat, []).append({
            "severity": int(row["severity_level"]),
            "direction": row["dir_label"],
            "marks": ["x" if ds in ds_set else "" for ds in all_datasets],
        })

    # ── 4. Construir linhas LaTeX ──
    n_ds = len(all_datasets)
    col_spec = "@{} l c c " + " c" * n_ds + " @{}"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(rf"\caption{{{format_caption(caption)}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Cabeçalho
    hdr_cells = [
        r"\textbf{Padrão de viés}",
        r"\textbf{Severidade}",
        r"\textbf{Sentido}",
    ]
    hdr_cells += [rf"\textbf{{{h}}}" for h in ds_headers]
    lines.append(" & ".join(hdr_cells) + r" \\")
    lines.append(r"\midrule")

    cat_keys = list(categories.keys())
    for cat_idx, cat in enumerate(cat_keys):
        rows = categories[cat]
        n_rows = len(rows)
        cat_label = _CATEGORY_LABELS.get(cat, cat)

        for i, row in enumerate(rows):
            cells = []
            if i == 0:
                cells.append(
                    rf"\multirow{{{n_rows}}}{{*}}{{\textbf{{{cat_label}}}}}"
                )
            else:
                cells.append("")

            cells.append(str(row["severity"]))
            cells.append(row["direction"])
            cells.extend(row["marks"])
            lines.append(" & ".join(cells) + r" \\")

        # Separador entre categorias (exceto após a última)
        if cat_idx < len(cat_keys) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    latex = _replace_decimal_in_latex(latex)

    # ── 5. Salvar ──
    results_dir = Path(result_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    latex_path = results_dir / f"{label}.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Tabela salva em: {latex_path}")

    return latex


def apply_translate_to_columns(
    df: pd.DataFrame,
    cols: list[str],
    translate: dict = translate
) -> pd.DataFrame:
    """Traduz valores de colunas categóricas para fins de exibição."""
    if not translate:
        return df
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: translate.get(x, x))
    return out



# =============================================================================
# HELPERS LATEX (compartilhados entre analysis_utils, statistic_test, rq*_utils)
# =============================================================================

# Mapeamento de nomes internos de colunas para cabeçalhos LaTeX em português.
# Colunas com símbolos matemáticos usam raw strings; escape=False é usado no
# to_latex para que esses símbolos sejam renderizados corretamente.
_LATEX_COL_NAMES = {
    "metric":            r"Métrica",
    "metric_label":      r"Métrica",
    "method":            r"Método",
    "category":          r"Categoria",
    "statistic":         r"Estatística",
    "p_value":           r"\textit{p}-valor",
    "p_raw":             r"\textit{p}-valor",
    "p_holm":            r"\textit{p} (Holm)",
    "n":                 r"$n$",
    "median_gap":        r"Mediana $d$",
    "iqr_gap":           r"IQR($d$)",
    "median_delta_fair": r"Mediana $\Delta_{\text{fair}}$",
    "iqr_delta_fair":    r"IQR($\Delta_{\text{fair}}$)",
    "median_r_fair":     r"Mediana $r_{\text{fair}}$",
    "iqr_r_fair":        r"IQR($r_{\text{fair}}$)",
    "median_delta_f1":   r"Mediana $\Delta_{\text{F1}}$",
    "iqr_delta_f1":      r"IQR($\Delta_{\text{F1}}$)",
    "median_raw":        r"Mediana (bruto)",
    "iqr_raw":           r"IQR (bruto)",
    "avg_rank":          r"Rank médio",
    "control_rank":      r"Rank controle",
    "delta_rank":        r"$\Delta_{\text{rank}}$",
    "z":                 r"$z$",
    "reject_holm":       r"Rejeita",
    "scope":             r"Escopo",
    "test":              r"Teste",
    "exp_id":            r"Experimento",
    "direction":         r"Sentido",
    "wins":              r"Vitórias",
    "win_rate_pct":      r"Win rate (\\%)",
    "win_rate_fair":     r"Win rate fair",
    "win_rate_winwin":   r"Win rate win-win",
    "Q1":                r"Q1",
    "Q2":                r"Q2",
    "Q3":                r"Q3",
    "Q4":                r"Q4",
    "median_gap_baseline": r"Mediana $|d|_{\text{base}}$",
    "median_ratio":      r"Mediana $r_{\text{fair}}$",  # legado, usar median_r_fair
    "spearman_rho":      r"Spearman $\rho$",
    "spearman_p":        r"$p$ (Spearman)",
    "pearson_r":         r"Pearson $r$",
    "pearson_p":         r"$p$ (Pearson)",
    "interpretacao_N":   r"Nota",
    "severity_level":    r"Severidade",
    "median":            r"Mediana",
    "iqr":               r"IQR",
    "min":               r"Mín",
    "max":               r"Máx",
    "mean":              r"Média",
    "std":               r"Desvio padrão",
    "median_gap_pct":    r"Gap mediano (\\%)",
    "ideal_value":       r"Ideal",
    "dataset":           r"Conjunto",
    "exp_id":            r"Método",
    "experiment_pipeline_type": r"Categoria",
}


def _escape_latex_cell(val: str) -> str:
    """Escapa caracteres especiais do LaTeX em valores de célula."""
    for ch, repl in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        val = val.replace(ch, repl)
    return val


def _postprocess_latex(latex: str) -> str:
    r"""Insere \centering e \scriptsize no LaTeX gerado pelo pandas."""
    latex = latex.replace(
        r"\begin{tabular}",
        r"\centering" + "\n" + r"\scriptsize" + "\n" + r"\begin{tabular}",
    )
    return latex


def _find_multirow_spans(df_multirow: pd.DataFrame) -> Dict[int, List[tuple]]:
    r"""Identifica spans de ``\multirow`` em cada coluna do DataFrame processado.

    Returns
    -------
    dict[int, list[tuple[int,int]]]
        Mapeia índice de coluna para lista de ``(start_row, end_row)`` inclusivo.
    """
    spans: Dict[int, List[tuple]] = {}
    for col_idx in range(len(df_multirow.columns)):
        col_spans = []
        for row_idx in range(len(df_multirow)):
            cell = str(df_multirow.iloc[row_idx, col_idx])
            m = re.match(r"\\multirow\{(\d+)\}", cell)
            if m:
                size = int(m.group(1))
                col_spans.append((row_idx, row_idx + size - 1))
        spans[col_idx] = col_spans
    return spans


def _insert_group_separators(
    latex: str,
    df_original: pd.DataFrame,
    df_multirow: pd.DataFrame,
    group_col: Union[str, List[str]],
) -> str:
    r"""Insere ``\cmidrule`` entre linhas onde *group_col* muda de valor.

    As linhas são posicionadas nas fronteiras de grupo e evitam cortar
    colunas que possuem ``\multirow`` ativo cruzando aquela fronteira.

    Parameters
    ----------
    group_col : str ou list[str]
        Coluna(s) cujas mudanças de valor definem fronteiras de grupo.
        Se uma lista, insere separador onde **qualquer** coluna muda.
    """
    # Normaliza para lista
    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = list(group_col)

    # Filtra colunas que existem
    group_cols = [c for c in group_cols if c in df_original.columns]
    if not group_cols:
        return latex

    n_rows = len(df_original)
    n_cols = len(df_multirow.columns)

    # Fronteiras: índice da linha onde QUALQUER coluna de grupo muda
    boundaries = [
        i for i in range(1, n_rows)
        if any(
            df_original[c].iloc[i] != df_original[c].iloc[i - 1]
            for c in group_cols
        )
    ]
    if not boundaries:
        return latex

    # Spans de multirow por coluna
    spans = _find_multirow_spans(df_multirow)

    def _span_crosses(col_idx: int, boundary: int) -> bool:
        for start, end in spans.get(col_idx, []):
            if start < boundary <= end:
                return True
        return False

    lines = latex.split("\n")
    midrule_idx = next(
        (i for i, l in enumerate(lines) if l.strip() == r"\midrule"), None
    )
    if midrule_idx is None:
        return latex
    data_start = midrule_idx + 1

    for boundary in reversed(boundaries):
        ranges: List[tuple] = []
        range_start = None
        for c in range(n_cols):
            if not _span_crosses(c, boundary):
                if range_start is None:
                    range_start = c + 1
            else:
                if range_start is not None:
                    ranges.append((range_start, c))
                    range_start = None
        if range_start is not None:
            ranges.append((range_start, n_cols))

        if ranges:
            separator = " ".join(
                rf"\cmidrule(lr){{{s}-{e}}}" for s, e in ranges
            )
            lines.insert(data_start + boundary, separator)

    return "\n".join(lines)


def _apply_multirow(df: pd.DataFrame, merge_rules: List[dict]) -> pd.DataFrame:
    r"""Aplica ``\multirow`` a colunas com valores consecutivos repetidos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame já formatado (valores como strings).
    merge_rules : list[dict]
        Cada elemento é um dict com:
        - ``col``     : nome da coluna a mesclar.
        - ``context`` : lista de colunas que delimitam o escopo de merge.

    Returns
    -------
    pd.DataFrame
        Cópia do DataFrame com ``\multirow{N}{*}{valor}`` na primeira
        linha de cada bloco e strings vazias nas linhas subsequentes.
    """
    df_out = df.copy()
    original = df.copy()

    for rule in merge_rules:
        col = rule["col"]
        context_cols = rule.get("context", [])

        if col not in df_out.columns:
            continue

        values = original[col].tolist()
        n = len(values)

        if context_cols:
            ctx_keys = [
                tuple(original[c].iloc[i] for c in context_cols if c in original.columns)
                for i in range(n)
            ]
        else:
            ctx_keys = [None] * n

        col_idx = df_out.columns.get_loc(col)
        i = 0
        while i < n:
            j = i + 1
            while (
                j < n
                and values[j] == values[i]
                and ctx_keys[j] == ctx_keys[i]
            ):
                j += 1

            span = j - i
            if span > 1:
                df_out.iloc[i, col_idx] = (
                    rf"\multirow{{{span}}}{{*}}{{{values[i]}}}"
                )
                for k in range(i + 1, j):
                    df_out.iloc[k, col_idx] = ""

            i = j

    return df_out


def _slugify(s: str) -> str:
    """Converte string para slug."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s[:120]


# =============================================================================
# EXPORTAÇÃO
# =============================================================================

def _tabular_to_longtable(
    latex: str,
    caption: str,
    label: str,
    n_cols: int,
    scriptsize: bool = True,
) -> str:
    r"""Converte LaTeX tabular padrão para ``longtable`` com cabeçalho repetido.

    O longtable permite que a tabela se estenda por múltiplas páginas,
    repetindo o cabeçalho em cada nova página.
    """
    lines = latex.split("\n")

    # Localizar posições-chave
    toprule_idx = next(i for i, l in enumerate(lines) if r"\toprule" in l)
    midrule_idx = next(i for i, l in enumerate(lines) if l.strip() == r"\midrule")
    bottomrule_idx = next(
        i for i, l in enumerate(lines) if r"\bottomrule" in l
    )

    # Extrair linhas de cabeçalho e dados
    header_lines = lines[toprule_idx + 1 : midrule_idx]
    data_lines = lines[midrule_idx + 1 : bottomrule_idx]

    # Detectar column_format da tabular original
    for l in lines:
        m = re.match(r".*\\begin\{tabular\}\{([^}]+)\}", l)
        if m:
            col_format = m.group(1)
            break
    else:
        col_format = "l" * n_cols

    out = []
    if scriptsize:
        out.append(r"{\scriptsize")
    out.append(rf"\begin{{longtable}}{{{col_format}}}")
    out.append(rf"\caption{{{caption}}}")
    out.append(rf"\label{{{label}}} \\")
    out.append(r"\toprule")
    out.extend(header_lines)
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append("")
    out.append(
        rf"\multicolumn{{{n_cols}}}{{c}}"
        r"{{\tablename\ \thetable{} -- continuação}} \\"
    )
    out.append(r"\toprule")
    out.extend(header_lines)
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append("")
    out.append(r"\midrule")
    out.append(
        rf"\multicolumn{{{n_cols}}}{{r}}"
        r"{{Continua na próxima página}} \\"
    )
    out.append(r"\endfoot")
    out.append("")
    out.append(r"\bottomrule")
    out.append(r"\endlastfoot")
    out.append("")
    out.extend(data_lines)
    out.append(r"\end{longtable}")
    if scriptsize:
        out.append(r"}")

    return "\n".join(out)


def latex_and_save(
    df: pd.DataFrame,
    caption: str = "caption",
    label: str = "label",
    result_dir: str = "results",
    float_format: str = "%.4f",
    centering: bool = True,
    scriptsize: bool = True,
    translate: Dict = None,
    cols: List[str] = None,
    multirow_cols: List[str] = None,
    group_col: Union[str, List[str]] = None,
    col_names: Dict = None,
    longtable: bool = False,
) -> str:
    """Salva DataFrame como tabela LaTeX e exibe em Markdown.

    Parameters
    ----------
    translate : dict, optional
        Dicionário de tradução para valores de colunas categóricas.
    cols : list[str], optional
        Colunas a incluir (e sua ordem). Se None, usa todas.
    multirow_cols : list[str], optional
        Colunas categóricas para aplicar \\multirow em valores consecutivos
        repetidos. Exemplo: ``["metric_label", "category"]``.
    group_col : str ou list[str], optional
        Coluna(s) para inserir separadores ``\\cmidrule`` entre grupos.
        Se lista, insere separador onde qualquer coluna muda de valor.
    col_names : dict, optional
        Mapeamento coluna→cabeçalho LaTeX. Se None, usa ``_LATEX_COL_NAMES``.
    longtable : bool, default False
        Se True, gera ``longtable`` com cabeçalho repetido a cada página.
    """
    results_dir = Path(result_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()

    # 1. Selecionar colunas
    if cols is not None:
        cols = [c for c in cols if c in df_out.columns]
        df_out = df_out[cols]

    # 2. Traduzir valores categóricos
    if translate:
        t = lambda x: translate.get(x, x)
        cat_cols = df_out.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            df_out[col] = df_out[col].apply(
                lambda x: _escape_latex_cell(t(x)) if isinstance(x, str) else x
            )

    # 3. Multirow
    df_pre = df_out.copy()
    if multirow_cols:
        merge_rules = [{"col": c, "context": []} for c in multirow_cols if c in df_out.columns]
        df_out = _apply_multirow(df_out, merge_rules)

    # 4. Renomear colunas para português
    rename_map = col_names if col_names is not None else _LATEX_COL_NAMES
    df_out = df_out.rename(columns=rename_map)

    # 5. Formatar caption
    caption = format_caption(caption)

    # 6. Gerar LaTeX (tabular intermediário)
    latex_table = df_out.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        position='H',
        column_format="l" * len(df_out.columns),
        escape=False
    )

    # 7. Separadores de grupo (cmidrule)
    if group_col and multirow_cols:
        latex_table = _insert_group_separators(latex_table, df_pre, df_out, group_col)

    if longtable:
        # 7a. Converter tabular → longtable com cabeçalho repetido
        latex_table = _tabular_to_longtable(
            latex_table, caption, label, len(df_out.columns), scriptsize,
        )
    else:
        # 7b. Centering + scriptsize (apenas tabular padrão)
        extra = ""
        if centering:
            extra += r"\centering" + "\n"
        if scriptsize:
            extra += r"\scriptsize" + "\n"
        if extra:
            latex_table = latex_table.replace(
                r"\begin{tabular}", extra + r"\begin{tabular}"
            )

    # Substituir separador decimal: ponto → vírgula
    latex_table = _replace_decimal_in_latex(latex_table)

    latex_path = results_dir / f"{label}.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"Tabela salva em: {latex_path}")
    print(df.to_markdown(index=False))
    return latex_table


# =============================================================================
# FUNÇÕES AUXILIARES DE VISUALIZAÇÃO
# =============================================================================

def add_ideal_line(ax, ideal_value, label=None):
    """Adiciona linha de valor ideal ao gráfico."""
    if label is None:
        label = f"Ideal ({ideal_value})"
    ax.axhline(
        ideal_value,
        color=VIZ_CONFIG["ideal_color"],
        linestyle=VIZ_CONFIG["ideal_linestyle"],
        linewidth=VIZ_CONFIG["ideal_linewidth"],
        label=label
    )


def configure_boxplot_grid(ax):
    """Configura grid padrão para boxplots."""
    ax.grid(axis="y", visible=VIZ_CONFIG["show_ygrid_boxplot"])
    ax.grid(axis="x", visible=VIZ_CONFIG["show_xgrid_boxplot"])


def get_experiment_palette(order=None):
    """Retorna paleta de cores padronizada para experimentos."""
    if order is None:
        return EXPERIMENT_COLORS.copy()
    return [EXPERIMENT_COLORS[g] for g in order if g in EXPERIMENT_COLORS]


def _wrap_label(text: str, width: int = 15) -> str:
    """Quebra texto para labels de gráficos."""
    return "\n".join(textwrap.wrap(str(text).replace("_", " ").title(), width))


def _whisker_bounds(vals: np.ndarray, whis: float = 1.5):
    """Calcula limites de whisker para classificação de outliers."""
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    return q1 - whis * iqr, q3 + whis * iqr


def _slugify(s: str) -> str:
    """Converte string para formato slug (filename safe)."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s[:120]


# =============================================================================
# VISUALIZAÇÕES - GAP PLOTS (BASELINE)
# =============================================================================

def plot_baseline_relative_gap(
    df: pd.DataFrame,
    ideal_values: dict,
    translate: dict = None,
    metric_col: str = "metric",
    value_col: str = "value_num",
    exp_type_col: str = "exp_type",
    baseline_tag: str = "baseline",
    title: str = "Diferença relativa (%) até o valor ideal das métricas de fairness do baseline (escala logarítmica)",
    save_path: str = None,
    figsize: tuple = (16, 6),
    font_scale: float = 1.15,
    jitter_width: float = 0.18,
    spacing: float = 1.8,
    box_width: float = 0.46,
    seed: int = 42
):
    """Boxplots + pontos do gap percentual relativo ao ideal para o baseline."""
    translate = translate or {}
    
    fb = df[df[exp_type_col] == baseline_tag].copy()
    
    # Truncar negativos para bias amplification
    mask_ba = fb[metric_col] == "bias_amplification"
    fb.loc[mask_ba, value_col] = fb.loc[mask_ba, value_col].clip(lower=0)
    
    # Calcular gap percentual
    fb["pct_gap_to_ideal"] = fb.apply(
        lambda r: pct_gap_to_ideal(r[value_col], ideal_values.get(r[metric_col], np.nan)),
        axis=1
    )
    
    # Ordenar métricas pela mediana
    med_order = (
        fb.groupby(metric_col)["pct_gap_to_ideal"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    
    data, medians = [], []
    for m in med_order:
        s = fb.loc[fb[metric_col] == m, "pct_gap_to_ideal"].dropna().clip(lower=1e-3)
        data.append(s.values)
        medians.append(float(np.median(s)) if len(s) else np.nan)
    
    metrics_translated = [_wrap_label(translate.get(m, m), 18) for m in med_order]
    
    base_font = 12 * font_scale
    with plt.rc_context({
        "font.size": base_font,
        "axes.titlesize": base_font * 1.15,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font * 0.95,
        "ytick.labelsize": base_font * 0.95,
    }):
        fig, ax = plt.subplots(figsize=figsize)
        rng = np.random.default_rng(seed)
        
        n = len(data)
        positions = np.arange(n) * spacing + 1.0
        
        bp = ax.boxplot(
            data, positions=positions, widths=box_width,
            labels=metrics_translated, patch_artist=True, showfliers=False,
            medianprops=dict(color=VIZ_CONFIG["median_color"], linewidth=VIZ_CONFIG["median_linewidth"]),
            boxprops=dict(facecolor="none", edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            zorder=2
        )
        
        color = EXPERIMENT_COLORS["baseline"]
        
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(VIZ_CONFIG["box_alpha"])
        
        for pos, vals in zip(positions, data):
            if len(vals) > 0:
                x_jitter = rng.uniform(-jitter_width, jitter_width, size=len(vals)) + pos
                ax.scatter(x_jitter, vals, s=10, alpha=VIZ_CONFIG["jitter_alpha"],
                           color=color, zorder=1, edgecolors="black", linewidths=0.2)
        
        x_offset = 0.20 * spacing
        for pos, med in zip(positions, medians):
            if not np.isnan(med):
                ax.text(pos + x_offset, med, f"{med:.1f}%", va="center", ha="left", fontsize=base_font)
        
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics_translated, rotation=0, ha="center")
        configure_boxplot_grid(ax)
        
        m = max(0.05, 0.5) * spacing
        ax.set_xlim(positions[0] - m, positions[-1] + m)
        fig.tight_layout(pad=0.6)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        
        plt.show()


def plot_gap_boxplot_for_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    fairness_metrics_order: list,
    translate: dict = None,
    yaxis_config: dict = None,
    output_dir: str = "results",
    rng_seed: int = 42,
    wrap_width: int = 16,
    font_scale: float = 1.0,
    bottom_margin: float = 0.26
):
    """Plota boxplot do gap percentual para um dataset específico."""
    translate = translate or {}
    rng = np.random.default_rng(rng_seed)
    
    title_fs = 12 * font_scale
    label_fs = 10 * font_scale
    tick_fs = 9 * font_scale
    annot_fs = 9 * font_scale
    
    df_ds = df[df["dataset"] == dataset_name].copy()
    
    mask_ba = df_ds["metric"] == "bias_amplification"
    df_ds.loc[mask_ba, "pct_gap_to_ideal"] = df_ds.loc[mask_ba, "pct_gap_to_ideal"].clip(lower=0)
    
    data, medians = [], []
    for metric in fairness_metrics_order:
        vals = df_ds.loc[df_ds["metric"] == metric, "pct_gap_to_ideal"].dropna().values
        medians.append(np.median(vals) if len(vals) else np.nan)
        data.append(np.clip(vals, 1e-3, None))
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    raw_labels = [translate.get(m, m) for m in fairness_metrics_order]
    tick_labels = [
        textwrap.fill(lbl, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        for lbl in raw_labels
    ]
    
    bp = ax.boxplot(
        data, patch_artist=True, tick_labels=tick_labels, showfliers=False,
        medianprops=dict(color=VIZ_CONFIG["median_color"], linewidth=VIZ_CONFIG["median_linewidth"])
    )
    
    exp_types = df_ds["exp_type"].dropna().unique().tolist()
    exp_type = exp_types[0] if exp_types else None
    pal = get_experiment_palette([exp_type]) if exp_type else []
    color = pal[0] if pal else "#B0B0B0"
    
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(VIZ_CONFIG["box_alpha"])
    
    for i, vals in enumerate(data, start=1):
        if len(vals) > 0:
            jitter = rng.uniform(-0.15, 0.15, size=len(vals)) + i
            ax.scatter(jitter, vals, s=10, alpha=VIZ_CONFIG["jitter_alpha"], c=color, linewidths=0.3)
    
    for i, med in enumerate(medians, start=1):
        if not np.isnan(med):
            ax.text(i + 0.25, med, f"{med:.1f}%", ha="left", va="center", fontsize=annot_fs)
    
    ax.set_yscale("log")
    vals_all = df_ds["pct_gap_to_ideal"].dropna()
    if not vals_all.empty:
        raw_ymin, raw_ymax = np.percentile(vals_all, [1, 99])
        raw_ymin = max(raw_ymin, 1e-3)
        raw_ymax = max(raw_ymax, raw_ymin * 10)
    else:
        raw_ymin, raw_ymax = 1e-2, 1e2
    
    ymin, ymax_mult = raw_ymin, 3
    if yaxis_config:
        for key, cfg in yaxis_config.items():
            if key.lower() in dataset_name.lower():
                ymin = cfg.get("ymin", ymin)
                ymax_mult = cfg.get("ymax_mult", ymax_mult)
    ax.set_ylim(ymin, raw_ymax * ymax_mult)
    
    ds_title = translate.get(dataset_name, dataset_name)
    ax.set_title(f"Conjunto de dados: {ds_title}", fontsize=title_fs)
    ax.set_ylabel("Diferença relativa para o ideal (%)", fontsize=label_fs)
    ax.tick_params(axis="x", rotation=0, labelsize=tick_fs, pad=15)
    ax.tick_params(axis="y", labelsize=tick_fs)
    
    n = len(data)
    ax.set_xlim(0.5, n + 0.8)
    fig.subplots_adjust(bottom=bottom_margin, right=0.98)
    configure_boxplot_grid(ax)
    
    plt.show()
    
    save_path = f"{output_dir}/baseline_gap_boxplots_{dataset_name}.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    
    return save_path


def generate_all_gap_plots(
    fairness_baseline: pd.DataFrame,
    ideal_values: dict,
    translate: dict = None,
    yaxis_config: dict = None,
    output_dir: str = None,
    datasets: list = None,
    wrap_width: int = 16,
    font_scale: float = 1.0,
    apply_style_once: bool = True,
    bottom_margin: float = 0.26
):
    """Orquestra a geração de boxplots por dataset (baseline)."""
    translate = translate or {}
    df = fairness_baseline.copy()
    
    if apply_style_once:
        set_plot_style(font_scale=font_scale)
    
    ideals = df["metric"].map(ideal_values).astype("float64")
    vals = pd.to_numeric(df["value_num"], errors="coerce").astype("float64")
    denom = ideals.abs().where(ideals.abs() > 1e-12, 1.0)
    df["pct_gap_to_ideal"] = ((vals - ideals).abs() / denom) * 100.0
    df.loc[ideals.isna() | vals.isna(), "pct_gap_to_ideal"] = np.nan
    
    order = (
        df.assign(_gap=df["pct_gap_to_ideal"].clip(lower=1e-3))
        .groupby("metric")["_gap"].median()
        .sort_values(ascending=False).index.tolist()
    )
    
    if datasets is None:
        datasets = sorted(df["dataset"].dropna().unique().tolist())
    
    out_dir = Path(output_dir) if output_dir else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ds in datasets:
        plot_gap_boxplot_for_dataset(
            df=df, dataset_name=ds, fairness_metrics_order=order,
            translate=translate, yaxis_config=yaxis_config, output_dir=str(out_dir),
            wrap_width=wrap_width, font_scale=font_scale, bottom_margin=bottom_margin
        )
    
    return df, order


# =============================================================================
# VISUALIZAÇÕES - BOXPLOTS SEGMENTADOS
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_fairness_boxpanels_segmented(
    df: pd.DataFrame,
    metrics: list = None,
    translate: dict = None,
    ideal_values: dict = None,
    metric_type: str = "fairness",
    hue_order: list = None,
    y_lim: tuple = None,
    custom_lim: dict = None,
    figsize: tuple = (10, 5),
    wrap_width: int = 25,
    title_fontsize: int = 12,
    save_dir: str = None,
    dpi: int = 300,
    show: bool = True,
    close: bool = True,
    filename_template: str = "fairness_boxpanel_{metric}.pdf",
    seed: int = 42,
    show_median: bool = True,
    median_decimals: int = 3,
    # ---- novos: colunas ----
    category_col: str = "category",
    metric_type_col: str = "metric_type",
    metric_col: str = "metric",
    value_col: str = "value_num",
    drop_unknown_categories: bool = True,
) -> dict:
    """
    Cria boxplots lado a lado (baseline, pre, in) por métrica.

    Nota: Para testes estatísticos formais (p-valores), usar compute_wilcoxon_table()
    do módulo tradeoff_utils, que implementa o teste Wilcoxon pareado corretamente.
    """
    translate = translate or {}
    ideal_values = ideal_values or {}
    custom_lim = custom_lim or {}

    if hue_order is None:
        hue_order = ["baseline", "pre_processing", "in_processing"]

    rng = np.random.default_rng(seed)

    fairness_df = df.loc[df[metric_type_col] == metric_type].copy()

    if metrics is None:
        metrics = fairness_df[metric_col].unique().tolist()

    if category_col not in fairness_df.columns:
        raise KeyError(f"Coluna '{category_col}' não encontrada no dataframe.")

    fairness_df["group"] = fairness_df[category_col]

    if drop_unknown_categories:
        fairness_df = fairness_df.loc[fairness_df["group"].isin(hue_order)].copy()

    saved_paths = {}

    for metric_name in metrics:
        g = fairness_df.loc[fairness_df[metric_col] == metric_name]

        fig, ax = plt.subplots(figsize=figsize)

        data, labels, colors = [], [], []
        for grp in hue_order:
            vals = g.loc[g["group"] == grp, value_col].dropna().values
            if len(vals) > 0:
                data.append(vals)

                med = float(np.median(vals))
                grp_label = translate.get(grp, grp)
                if show_median:
                    grp_label = f"{grp_label}\nmed={med:.{median_decimals}f}"
                labels.append(grp_label)

                colors.append(EXPERIMENT_COLORS.get(grp, "#999999"))

        if not data:
            plt.close(fig)
            continue

        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
            medianprops=dict(
                color=VIZ_CONFIG["median_color"],
                linewidth=VIZ_CONFIG["median_linewidth"]
            ),
            boxprops=dict(edgecolor="dimgray"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black")
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(VIZ_CONFIG["box_alpha"])

        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            x = rng.normal(loc=i, scale=0.08, size=len(vals))
            ax.scatter(x, vals, s=12, alpha=VIZ_CONFIG["jitter_alpha"], color=color, zorder=3)

        if y_lim:
            ax.set_ylim(y_lim)

        if metric_name in ideal_values:
            add_ideal_line(ax, ideal_values[metric_name])
            ax.legend(loc="upper right", fontsize=10)

        if metric_name in custom_lim:
            ax.set_ylim(custom_lim[metric_name])

        metric_label = translate.get(metric_name, metric_name)
        ax.set_title(_wrap_label(metric_label, wrap_width), fontsize=title_fontsize)
        ax.set_ylabel(None)
        configure_boxplot_grid(ax)

        plt.setp(ax.get_xticklabels(), fontsize=13)

        # Ajustar espaço para labels com múltiplas linhas
        if show_median:
            fig.subplots_adjust(bottom=0.2)

        fig.tight_layout()

        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            metric_slug = _slugify(metric_name)
            save_path = out_dir / filename_template.format(metric=metric_slug)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, format="pdf")
            saved_paths[metric_name] = str(save_path)

        if show:
            plt.show()

        if close:
            plt.close(fig)

    return saved_paths


# =============================================================================
# VISUALIZAÇÕES - FACETGRID E BOXPLOTS POR MÉTRICA
# =============================================================================

def plot_fairness_compact_broken_x(
    fairness_baseline: pd.DataFrame,
    translate: dict = None,
    ideal_values: dict = None,
    output_path: str = "results/fairness_compact_broken_x.png",
    font_scale: float = 1.0,
    exp_type: str = None,
    wrap_width: int = 26,
    height_per_metric: float = 0.55,
    xlim_left: tuple = (-0.5, 1.3),
    xlim_right: tuple = (1.3, 4.6),
    rng_seed: int = 42
):
    """Boxplot horizontal com eixo X quebrado (zoom + cauda)."""
    translate = translate or {}
    df = fairness_baseline.copy()

    set_plot_style(font_scale=font_scale)
    rng = np.random.default_rng(rng_seed)

    df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

    mask_ba = df["metric"] == "bias_amplification"
    df.loc[mask_ba, "value_num"] = df.loc[mask_ba, "value_num"].clip(lower=0)

    metric_order = (
        df.groupby("metric")["value_num"]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )

    if exp_type is None:
        exp_types = df.get("exp_type", pd.Series([], dtype="object")).dropna().unique().tolist()
        exp_type = exp_types[0] if exp_types else "baseline"
    color = EXPERIMENT_COLORS.get(exp_type, EXPERIMENT_COLORS.get("baseline", "#4E79A7"))

    data_list = []
    for m in metric_order:
        vals = df.loc[df["metric"] == m, "value_num"].dropna().values
        data_list.append(vals)

    n = len(metric_order)
    fig_h = max(3.0, height_per_metric * n)

    fig, (axL, axR) = plt.subplots(
        1, 2,
        sharey=True,
        figsize=(12, fig_h),
        gridspec_kw={"width_ratios": [3.2, 1.8], "wspace": 0.05}
    )

    y_pos = np.arange(1, n + 1)

    def draw(ax):
        bp = ax.boxplot(
            data_list,
            vert=False,
            positions=y_pos,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=VIZ_CONFIG["median_color"], linewidth=VIZ_CONFIG["median_linewidth"]),
            boxprops=dict(facecolor=color, alpha=VIZ_CONFIG["box_alpha"], edgecolor="black"),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )

        for i, vals in enumerate(data_list, start=1):
            if len(vals) == 0:
                continue
            yj = rng.uniform(-0.22, 0.22, size=len(vals)) + i
            ax.scatter(
                vals, yj,
                s=10,
                alpha=VIZ_CONFIG["jitter_alpha"],
                c=color,
                linewidths=0.2,
                zorder=1
            )

        if ideal_values is not None:
            for i, m in enumerate(metric_order, start=1):
                ideal = ideal_values.get(m)
                if ideal is None or pd.isna(ideal):
                    continue
                ax.scatter(
                    [ideal], [i],
                    marker="|",
                    s=520,
                    color=VIZ_CONFIG.get("ideal_color", "red"),
                    linewidths=2.2,
                    zorder=3
                )

        ax.grid(axis="x", alpha=0.25)

    draw(axL)
    draw(axR)

    axL.set_xlim(*xlim_left)
    axR.set_xlim(*xlim_right)

    ylabels = [textwrap.fill(translate.get(m, m), width=wrap_width,
                             break_long_words=False, break_on_hyphens=False)
               for m in metric_order]
    axL.set_yticks(y_pos)
    axL.set_yticklabels(ylabels, fontsize=9 * font_scale)
    axL.invert_yaxis()

    axR.tick_params(labelleft=False, left=False)

    axL.set_xlabel("Valor da métrica", fontsize=10 * font_scale)
    axR.set_xlabel("")

    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    d = 0.012
    kwargs = dict(color="black", clip_on=False, linewidth=1.0)
    axL.plot((1 - d, 1 + d), (-d, +d), transform=axL.transAxes, **kwargs)
    axL.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axL.transAxes, **kwargs)
    axR.plot((-d, +d), (-d, +d), transform=axR.transAxes, **kwargs)
    axR.plot((-d, +d), (1 - d, 1 + d), transform=axR.transAxes, **kwargs)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.show()
    plt.close(fig)
    return output_path


def plot_fairness_facetgrid_boxplot(
    fairness_baseline: pd.DataFrame,
    translate: dict = None,
    output_path: str = "results/fairness_facetgrid_boxplot.png",
    font_scale: float = 1.0,
    exp_type: str = None,
    wrap_titles: int = None
):
    """Plota FacetGrid com boxplots por métrica de fairness (com jitter)."""
    translate = translate or {}
    df = fairness_baseline.copy()

    set_plot_style(font_scale=font_scale)

    mask_ba = df["metric"] == "bias_amplification"
    df.loc[mask_ba, "value_num"] = pd.to_numeric(df.loc[mask_ba, "value_num"], errors="coerce").clip(lower=0)

    df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

    df["metric_translated"] = df["metric"].map(lambda m: translate.get(m, m))
    order = (
        df.groupby("metric")["value_num"]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )
    order_translated = [translate.get(m, m) for m in order]
    df["metric_translated"] = pd.Categorical(df["metric_translated"], categories=order_translated, ordered=True)

    if exp_type is None:
        exp_types = df.get("exp_type", pd.Series([], dtype="object")).dropna().unique().tolist()
        exp_type = exp_types[0] if exp_types else "baseline"

    pal = get_experiment_palette([exp_type])
    base_color = pal[0] if pal else EXPERIMENT_COLORS.get("baseline", "#4E79A7")

    g = sns.FacetGrid(
        df,
        col="metric_translated",
        col_wrap=3,
        height=3.3,
        sharex=False,
        sharey=False
    )

    def jitter_boxplot(x, color):
        ax = plt.gca()
        vals = x.dropna().values
        if len(vals) == 0:
            ax.set_xticks([])
            return

        jitter = np.random.uniform(-0.08, 0.08, size=len(vals)) + 1
        ax.scatter(
            jitter, vals,
            s=11,
            alpha=VIZ_CONFIG["jitter_alpha"],
            color=color,
            zorder=1,
            linewidths=0.3
        )

        bp = ax.boxplot(
            [vals],
            positions=[1],
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(
                linewidth=VIZ_CONFIG["median_linewidth"],
                color=VIZ_CONFIG["median_color"]
            ),
            boxprops=dict(
                facecolor=color,
                alpha=VIZ_CONFIG["box_alpha"],
                edgecolor="black"
            ),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black")
        )

        ax.set_xticks([])

    g.map(jitter_boxplot, "value_num", color=base_color)

    if wrap_titles is not None and isinstance(wrap_titles, int) and wrap_titles > 0:
        for ax in g.axes.flatten():
            t = ax.get_title().replace("metric_translated = ", "")
            ax.set_title(textwrap.fill(t, width=wrap_titles))
    else:
        g.set_titles("{col_name}")

    g.set_xlabels("")
    sns.despine(left=True)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    print(f"Figura salva em: {output_path}")
    plt.show()
    return output_path


def plot_baseline_boxplots_by_metric(
    fairness_baseline: pd.DataFrame,
    translate: dict = None,
    out_dir: str = "results",
    figsize: tuple = (8, 4.2),
    annotate_median: bool = True,
    seed: int = 42,
    font_scale: float = 1.0,
    exp_type: str = None,
    x_tick_pad: int = 10,
    xlim_right_pad: float = 0.8,
    apply_style: bool = True
):
    """Gera e salva um boxplot por métrica (um arquivo por métrica)."""
    translate = translate or {}
    df = fairness_baseline.copy()

    if apply_style:
        set_plot_style(font_scale=font_scale)

    df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")
    mask_ba = df["metric"] == "bias_amplification"
    df.loc[mask_ba, "value_num"] = df.loc[mask_ba, "value_num"].clip(lower=0)

    metric_order = (
        df.groupby("metric")["value_num"]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    tick_labels = [translate.get(ds, ds) for ds in datasets]

    if exp_type is None:
        exp_types = df.get("exp_type", pd.Series([], dtype="object")).dropna().unique().tolist()
        exp_type = exp_types[0] if exp_types else "baseline"
    color = EXPERIMENT_COLORS.get(exp_type, EXPERIMENT_COLORS.get("baseline", "#4E79A7"))

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_fs = 12 * font_scale
    tick_fs = 9 * font_scale
    annot_fs = 9 * font_scale

    saved = {}
    for metric in metric_order:
        df_m = df[df["metric"] == metric].copy()

        data, meds = [], []
        for ds in datasets:
            vals = df_m.loc[df_m["dataset"] == ds, "value_num"].dropna().to_numpy()
            data.append(vals)
            meds.append(float(np.median(vals)) if vals.size else np.nan)

        fig, ax = plt.subplots(figsize=figsize)

        for i, vals in enumerate(data, start=1):
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                x = rng.uniform(-0.15, 0.15, size=len(vals)) + i
                ax.scatter(
                    x, vals,
                    s=10,
                    alpha=VIZ_CONFIG["jitter_alpha"],
                    c=color,
                    linewidths=0.2,
                    zorder=1
                )

        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(
                color=VIZ_CONFIG["median_color"],
                linewidth=VIZ_CONFIG["median_linewidth"]
            ),
            boxprops=dict(
                facecolor=color,
                alpha=VIZ_CONFIG["box_alpha"],
                edgecolor="black"
            ),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            zorder=2
        )
        for med_line in bp["medians"]:
            med_line.set_zorder(4)

        ax.set_xticks(np.arange(1, len(datasets) + 1))
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=tick_fs)
        ax.tick_params(axis="x", pad=x_tick_pad)
        ax.tick_params(axis="y", labelsize=tick_fs)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(translate.get(metric, metric), fontsize=title_fs)

        configure_boxplot_grid(ax)

        ax.set_xlim(0.5, len(datasets) + xlim_right_pad)

        if annotate_median:
            for i, med in enumerate(meds, start=1):
                if not np.isnan(med):
                    ax.annotate(
                        f"{med:.2f}",
                        xy=(i, med),
                        xytext=(25, -5),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=annot_fs,
                        zorder=5
                    )

        fig.tight_layout(pad=0.35)

        save_path = out_dir / f"baseline_boxplot_metric_{metric}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        saved[metric] = str(save_path)

        plt.show()
        plt.close(fig)

    return saved


def plot_fairness_one_axis_per_metric(
    df: pd.DataFrame,
    metrics: list | None = None,
    translate: dict | None = None,
    ideal_values: dict | None = None,
    hue_order: list | None = None,
    figsize: tuple = (14, 4),
    save_path: str | None = None,
    show: bool = True,
    seed: int = 42,
    # ---- novos: colunas ----
    category_col: str = "category",
    metric_type_col: str = "metric_type",
    metric_type_value: str = "fairness",
    metric_col: str = "metric",
    value_col: str = "value_num",
    drop_unknown_categories: bool = True,
):
    """Plota todas as métricas lado a lado em um único eixo por métrica (uma boxplot por grupo)."""
    translate = translate or {}
    ideal_values = ideal_values or {}

    if hue_order is None:
        hue_order = ["baseline", "pre_processing", "in_processing"]

    # filtro de fairness
    fairness_df = df.loc[df[metric_type_col] == metric_type_value].copy()

    if metrics is None:
        metrics = fairness_df[metric_col].unique().tolist()

    # usa category diretamente como group
    if category_col not in fairness_df.columns:
        raise KeyError(f"Coluna '{category_col}' não encontrada no dataframe.")

    fairness_df["group"] = fairness_df[category_col]

    if drop_unknown_categories:
        fairness_df = fairness_df.loc[fairness_df["group"].isin(hue_order)].copy()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    rng = np.random.default_rng(seed)

    for ax, metric_name in zip(axes, metrics):
        g = fairness_df.loc[fairness_df[metric_col] == metric_name]

        data, labels, colors = [], [], []
        for grp in hue_order:
            vals = g.loc[g["group"] == grp, value_col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(grp[0].upper())  # B, P, I
                colors.append(EXPERIMENT_COLORS.get(grp, "#999999"))

        if not data:
            continue

        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
            medianprops=dict(
                color=VIZ_CONFIG["median_color"],
                linewidth=VIZ_CONFIG["median_linewidth"]
            ),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(VIZ_CONFIG["box_alpha"])

        for i, (vals, color) in enumerate(zip(data, colors), start=1):
            x = rng.normal(loc=i, scale=0.08, size=len(vals))
            ax.scatter(x, vals, s=8, alpha=VIZ_CONFIG["jitter_alpha"], color=color, zorder=3)

        if metric_name in ideal_values:
            ax.axhline(ideal_values[metric_name], color="red", linestyle="--", linewidth=1.5, alpha=0.8)

        metric_label = translate.get(metric_name, metric_name)
        ax.set_title(metric_label, fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        configure_boxplot_grid(ax)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# =============================================================================
# TABELA RESUMO COM ESTATÍSTICAS
# =============================================================================

def save_table(
    data: pd.DataFrame,
    ideal_values: dict,
    group_list: list = None,
    value_col: str = "value_num",
    stats: tuple = ("n", "median", "iqr", "median_gap_pct"),
    caption: str = "caption",
    label: str = "label_summary",
    result_dir: str = "results",
    translate: Dict = None,
    multirow_cols: List[str] = None,
    group_col: Union[str, List[str]] = None,
    longtable: bool = False,
) -> pd.DataFrame:
    """Cria tabela resumo com estatísticas selecionáveis, incluindo gap percentual.

    Parameters
    ----------
    translate : dict, optional
        Dicionário de tradução para valores categóricos e cabeçalhos.
    multirow_cols : list[str], optional
        Colunas categóricas para mesclar linhas iguais consecutivas (multirow).
    group_col : str ou list[str], optional
        Coluna(s) para inserir separadores cmidrule entre grupos.
    longtable : bool, default False
        Se True, gera longtable com cabeçalho repetido a cada página.
    """
    if group_list is None:
        group_list = ["metric"]

    df = data.copy()
    df["ideal_value"] = df["metric"].map(ideal_values)
    df["gap_pct"] = df.apply(
        lambda r: pct_gap_to_ideal(r[value_col], r["ideal_value"]), axis=1
    )

    # ── Pré-agregação ao nível de dataset ──
    # Quando há múltiplos datasets e a análise NÃO é por dataset,
    # primeiro agrega (mediana) por dataset para que cada dataset
    # contribua com UMA observação por grupo.
    # Quando "dataset" já está no group_list, pula a pré-agregação
    # para preservar a variabilidade entre folds/repetições.
    if "dataset" in df.columns and df["dataset"].nunique() > 1 and "dataset" not in group_list:
        agg_keys = list(dict.fromkeys(["dataset"] + group_list))
        agg_dict = {value_col: "median", "gap_pct": "median"}
        for extra in ["ideal_value", "metric_group"]:
            if extra in df.columns:
                agg_dict[extra] = "first"
        df = df.groupby(agg_keys, as_index=False).agg(agg_dict)

    def iqr_adaptive(subdf):
        if "metric_group" not in subdf.columns:
            return iqr_standard(subdf[value_col])
        mg = subdf["metric_group"].iloc[0]
        return iqr_truncated(subdf[value_col], 0.01, 0.99) if mg == "ratio_based" else iqr_standard(subdf[value_col])

    STAT_FUNCS = {
        "n": lambda s: s[value_col].count(),
        "mean": lambda s: s[value_col].mean(),
        "median": lambda s: s[value_col].median(),
        "std": lambda s: s[value_col].std(),
        "min": lambda s: s[value_col].min(),
        "max": lambda s: s[value_col].max(),
        "iqr": iqr_adaptive,
        "ideal_value": lambda s: s["ideal_value"].iloc[0],
        "median_gap_pct": lambda s: s["gap_pct"].median()
    }

    invalid = set(stats) - STAT_FUNCS.keys()
    if invalid:
        raise ValueError(f"Estatísticas inválidas: {invalid}")

    def summarize(subdf):
        return pd.Series({stat: STAT_FUNCS[stat](subdf) for stat in stats})

    data_summary = df.groupby(group_list).apply(summarize).reset_index()

    latex_and_save(
        data_summary,
        caption=caption,
        label=label,
        result_dir=result_dir,
        translate=translate,
        multirow_cols=multirow_cols,
        group_col=group_col,
        longtable=longtable,
    )
    return data_summary


# =============================================================================
# ANÁLISE DE CONSISTÊNCIA (REPORT)
# =============================================================================

def generate_consistency_report(df: pd.DataFrame) -> dict:
    """Gera relatório consolidado de consistência dos resultados."""
    report = {}
    
    total_by_metric = df.groupby("metric").size()
    nan_by_metric = df[df["value"].isna()].groupby("metric").size()
    report["nan_by_metric"] = (nan_by_metric / total_by_metric * 100).round(2).fillna(0).to_dict()
    
    total_by_dataset = df.groupby("dataset").size()
    nan_by_dataset = df[df["value"].isna()].groupby("dataset").size()
    report["nan_by_dataset"] = (nan_by_dataset / total_by_dataset * 100).round(2).fillna(0).to_dict()
    
    if "value_num" in df.columns:
        report["inf_count"] = int(np.isinf(df["value_num"]).sum())
    else:
        report["inf_count"] = 0
    
    report["execution_counts"] = df.groupby("exp_id").size().to_dict()
    
    return report


def print_consistency_report(report: dict):
    """Imprime relatório de consistência de forma legível."""
    print("=" * 60)
    print("RELATÓRIO DE CONSISTÊNCIA DOS RESULTADOS")
    print("=" * 60)
    
    print("\n📊 NaN por Métrica (%):")
    for k, v in sorted(report["nan_by_metric"].items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  • {k}: {v}%")
    if all(v == 0 for v in report["nan_by_metric"].values()):
        print("  ✓ Nenhum NaN encontrado")
    
    print("\n📊 NaN por Dataset (%):")
    for k, v in sorted(report["nan_by_dataset"].items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  • {k}: {v}%")
    if all(v == 0 for v in report["nan_by_dataset"].values()):
        print("  ✓ Nenhum NaN encontrado")
    
    print(f"\n⚠️  Valores Infinitos: {report['inf_count']}")
    
    print("\n📈 Execuções por Experimento:")
    for k, v in sorted(report["execution_counts"].items()):
        print(f"  • {k}: {v}")
    
    print("=" * 60)