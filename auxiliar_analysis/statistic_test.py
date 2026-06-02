"""
statistic_test.py - Testes estatísticos para comparação de métodos de mitigação de viés

METODOLOGIA:
- Unidade de análise: DATASET (N = número de datasets no escopo)
- Folds e repetições são usados apenas para estimar um valor estável por dataset
- Teste omnibus: Friedman (não-paramétrico, por ranks)
- Pós-hoc: comparações vs controle (baseline típico) com correção Holm (FWER)

ANÁLISES SUPORTADAS:
[1] Categorias vs baseline típico (scope="all")
[2] Categorias vs baseline típico por padrão de viés (scope="A1", etc.)
[3] Cada experimento vs baseline típico (scope="all")
[4] Cada experimento vs baseline típico por padrão de viés (scope="A1", etc.)

FUNÇÕES

Utilitárias:
_slugify(s): 
    Converte uma string em um slug "seguro" (minúsculas, sem caracteres especiais) para uso em nomes de arquivos/labels.
compute_gap_to_ideal: 
    Calcula o gap ao valor ideal da métrica (base para ranqueamento): em geral |valor âˆ’ ideal|; para bias_amplification, usa max(valor, 0) (ideal é "â‰¤ 0", então só penaliza valores positivos).
_format_pvalue(p): 
    Formata p-valores para impressão (inclui "asteriscos" de significância e fallback para ausentes).

Agregação e diagnóstico:
aggregate_to_dataset_level
    Transforma os dados brutos (folds/repetições) em uma matriz dataset × métodos apropriada para testes não paramétricos.
compute_completeness_report(data_matrix, method_cols)
    Gera um relatório de completude antes do complete-case (quantos datasets "sobram" com todos os métodos, presença por método, lista de datasets removidos, etc.).

Teste estatístico
run_wilcoxon_paired (data_matrix, method1, method2, alternative="less"): 
    Executa Wilcoxon signed-rank (pareado) entre dois métodos, com tratamento de casos degenerados (ex.: diferenças todas zero). Retorna estatísticas, p-valor, N e mediana das diferenças.
run_friedman_test(data_matrix, method_cols, control_method="baseline_typical", alternative="less")
    Executa Friedman em complete-case (datasets com todos os métodos).
    Se k=2, cai automaticamente para Wilcoxon (usando control_method como referência).
run_posthoc_vs_control(friedman_result, control_method="baseline_typical", alpha=0.05, alternative="less")
    Executa pós-hoc vs controle usando z-test em ranks médios e aplica correção de Holm (FWER).
    Interpretação típica: alternative="less" significa "método melhor que o controle" via rank médio menor (consistente com "gap menor é melhor").

Rotinas de análise
run_category_tests(df, metric, ideal_values, scope="all", baseline_ids=None, category_mapping=None, alpha=0.05, run_posthoc="always", alternative="less", min_baselines=3)
    Roda o pipeline completo por categoria
run_category_analysis(df, metric, ideal_values, scopes=None, baseline_ids=None, alpha=0.05, run_posthoc="always", alternative="less", save_dir=None, translate=None)
    Executa a análise por categoria para múltiplos scopes e devolve um DataFrame "tidy" consolidado (linhas de omnibus e linhas de pós-hoc). Opcionalmente salva LaTeX.
run_all_category_analysis(df, ideal_values, metrics=None, metric_type="fairness", scopes=None, print_results=True, save_dir=None, translate=None, alternative="less")
    Executa run_category_analysis para todas as métricas (filtra por metric_type se existir) e concatena os resultados
run_experiment_tests(df, metric, ideal_values, scope="all", baseline_ids=None, exp_order=None, alpha=0.05, run_posthoc="always", alternative="less", min_baselines=3)
    Roda o pipeline completo por experimento
run_experiment_analysis(df, metric, ideal_values, scopes=None, baseline_ids=None, alpha=0.05, run_posthoc="always", alternative="less", save_dir=None, translate=None)
    Versão "multi-scope" da análise por experimento
run_all_experiment_analysis(df, ideal_values, metrics=None, metric_type="fairness", scopes=None, print_results=True, save_dir=None, translate=None, alternative="less")
    Executa run_experiment_analysis para todas as métricas e concatena os resultados.

Impressão:
print_category_results(stats_df)
print_experiment_results(stats_df)
save_results_latex(stats_df, metric, analysis_type, save_dir, translate=None)
    
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Literal, Tuple
from scipy.stats import friedmanchisquare, norm, rankdata, iqr
from statsmodels.stats.multitest import multipletests
import re

from .analysis_utils import (
    classify_bias_patterns,
    _LATEX_COL_NAMES,
    _escape_latex_cell,
    _postprocess_latex,
    _find_multirow_spans,
    _insert_group_separators,
    _apply_multirow,
    _slugify,
    format_caption,
    _replace_decimal_in_latex,
    _tabular_to_longtable,
)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASELINE_IDS = ["BLRA", "BLRU", "BRFA", "BRFU"]

EXP_ORDER = ["BLRA", "BLRU", "BRFA", "BRFU", "EDL", "ERL", "IAD", "IGLA", "IGLU", "IFG", "IP", "IW"]

CATEGORY_MAPPING = {
    "baseline": ["BLRA", "BLRU", "BRFA", "BRFU"],
    "pre_processing": ["EDL","ERL"],
    "in_processing": ["IAD", "IGLA", "IGLU", "IFG", "IP", "IW"]
}

EXP_TO_CATEGORY = {}
for category, exp_ids in CATEGORY_MAPPING.items():
    for exp_id in exp_ids:
        EXP_TO_CATEGORY[exp_id] = category

BASELINE_TYPICAL_LABEL = "baseline_typical"

# Epsilon para mitigação relativa (evita divisão por zero)
_REL_EPS = 1e-8


# =============================================================================
# GAP TO IDEAL
# =============================================================================

def compute_gap_to_ideal(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    value_col: str = "value_num"
) -> pd.Series:
    """
    Calcula gap ao ideal para uma métrica específica.
    
    Para bias_amplification: ideal â‰¤ 0, então gap = max(value, 0)
    Para outras métricas: gap = |value - ideal|
    """
    mask = df["metric"] == metric
    values = df.loc[mask, value_col].copy()
    
    if metric == "bias_amplification":
        return values.clip(lower=0)
    
    ideal = ideal_values.get(metric, np.nan)
    if pd.isna(ideal):
        return pd.Series(np.nan, index=values.index)
    
    return (values - ideal).abs()


# =============================================================================
# FUNÇÕES AUXILIARES PARA FILTRO POR SCOPE (PU-AWARE)
# =============================================================================

def _ensure_problem_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o DataFrame tenha a coluna `problem_type`.
    
    Estratégia:
    1. Se já existir `problem_type`, retorna sem alterações
    2. Se existir `pattern_code`, usa como base para `problem_type`
       - Se houver coluna `direction` separada, combina: pattern_code + direction
       - Normaliza direção para P/U (aceita "privileged"/"unprivileged" ou "P"/"U")
    3. Caso contrário, chama classify_bias_patterns (legado)
    
    Returns:
        DataFrame com coluna `problem_type` garantida
    """
    if "problem_type" in df.columns:
        return df
    
    df = df.copy()
    
    if "pattern_code" in df.columns:
        # Mapa de normalização de direção
        direction_map = {
            "privileged": "P", "P": "P", "p": "P",
            "unprivileged": "U", "U": "U", "u": "U",
        }
        
        def combine_pattern_direction(row):
            pattern = row.get("pattern_code", None)
            
            if pd.isna(pattern) or pattern is None:
                return None
            
            pattern = str(pattern).strip()
            
            # Se pattern já termina com P ou U, não adicionar direção
            if pattern and pattern[-1] in ("P", "U"):
                return pattern
            
            # Tentar adicionar direção da coluna separada (se existir)
            if "direction" in row.index:
                direction = row.get("direction", None)
                if pd.notna(direction) and direction:
                    dir_code = direction_map.get(str(direction).strip(), "")
                    if dir_code:
                        return f"{pattern}{dir_code}"
            
            return pattern
        
        if "direction" in df.columns:
            df["problem_type"] = df.apply(combine_pattern_direction, axis=1)
        else:
            df["problem_type"] = df["pattern_code"]
    else:
        # Fallback: usar classify_bias_patterns (legado)
        df = classify_bias_patterns(df)
    
    return df


def _filter_by_scope(
    df: pd.DataFrame,
    scope,
    problem_type_col: str = "problem_type"
) -> pd.DataFrame:
    """
    Filtra DataFrame por scope com suporte a direção (P/U).
    
    Regras de matching:
    - scope="all" → sem filtro (retorna df completo)
    - scope="A" (só letra) → prefixo, inclui A, A1, A1P, A1U, A2, A2P, etc.
    - scope="A1" (letra+número) → inclui A1, A1P, A1U (com e sem direção)
    - scope="A1P" (com direção) → match exato apenas A1P
    - scope=["A1", "B2"] (lista) → OR entre os scopes
    
    Args:
        df: DataFrame a filtrar
        scope: Escopo de filtro ("all", str, ou lista de str)
        problem_type_col: Nome da coluna de problem_type
    
    Returns:
        DataFrame filtrado (cópia)
    """
    if scope == "all":
        return df
    
    if problem_type_col not in df.columns:
        warnings.warn(
            f"Coluna '{problem_type_col}' não encontrada. "
            "Retornando DataFrame sem filtro.",
            UserWarning
        )
        return df
    
    # Normalizar scope para lista
    if isinstance(scope, (list, tuple, set)):
        scopes = [str(s) for s in scope]
    else:
        scopes = [str(scope)]
    
    masks = []
    for s in scopes:
        s = s.strip()
        
        # Caso 1: apenas letra (A, B, C) → prefixo
        if re.fullmatch(r"[A-Za-z]", s):
            mask = df[problem_type_col].astype(str).str.startswith(s, na=False)
        
        # Caso 2: letra + número sem direção (A1, B2, C3) → inclui variantes com direção
        elif re.fullmatch(r"[A-Za-z]\d+", s):
            # Regex: ^A1$ ou ^A1[PU]$
            pattern = rf"^{re.escape(s)}[PU]?$"
            mask = df[problem_type_col].astype(str).str.match(pattern, na=False)
        
        # Caso 3: padrão completo (A1P, A1U, ou qualquer outro) → match exato
        else:
            mask = df[problem_type_col].astype(str) == s
        
        masks.append(mask)
    
    # Combinar máscaras com OR
    if not masks:
        return df
    
    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = combined_mask | m
    
    return df[combined_mask].copy()


# =============================================================================
# AGREGAÇÃO PARA DATASET-LEVEL
# =============================================================================

def aggregate_to_dataset_level(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    scope: str | List[str] = "all",
    baseline_ids: List[str] = None,
    category_mapping: Dict = None,
    analysis_type: Literal["category", "experiment"] = "experiment",
    min_baselines: int = 3,
    return_raw_matrix: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrega dados ao nível de DATASET para análise Friedman.
    
    Pipeline:
    1. Filtrar por métrica
    2. Garantir problem_type (com suporte a pattern_code + direction)
    3. Filtrar por scope (PU-aware)
    4. Calcular gap
    5. Agregar por split: mediana dos gaps por método/categoria
    6. Agregar para dataset-level: mediana sobre splits
    
    Args:
        df: DataFrame com dados brutos
        metric: Métrica a analisar
        ideal_values: Dict com valores ideais
        scope: Escopo de filtro. Opções:
            - "all": sem filtro
            - "A", "B", "C": por categoria (prefixo)
            - "A1", "B2": por padrão (inclui A1, A1P, A1U)
            - "A1P", "A1U": por padrão+direção (match exato)
            - Lista de scopes: OR entre eles
        baseline_ids: IDs dos baselines
        category_mapping: Mapeamento categoria -> exp_ids
        analysis_type: "category" (3 categorias) ou "experiment" (cada exp_id)
        min_baselines: Mínimo de baselines por split para formar baseline típico
        return_raw_matrix: Se True, retorna também a matriz com valores brutos (raw)
    
    Returns:
        DataFrame pivotado: dataset × métodos, com gap agregado
        Se return_raw_matrix=True: tupla (gap_matrix, raw_matrix)
    """
    baseline_ids = baseline_ids or BASELINE_IDS
    category_mapping = category_mapping or CATEGORY_MAPPING
    
    # 1. Filtrar por métrica
    df_metric = df[df["metric"] == metric].copy()
    
    if df_metric.empty:
        if return_raw_matrix:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()
    
    # 2. Garantir problem_type (com suporte a pattern_code + direction)
    if scope != "all":
        df_metric = _ensure_problem_type(df_metric)
    
    # 3. Filtrar por scope (PU-aware)
    if scope != "all":
        df_metric = _filter_by_scope(df_metric, scope)
    
    if df_metric.empty:
        if return_raw_matrix:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()
    
    # 4. Calcular gap E manter valor bruto
    df_metric["gap"] = compute_gap_to_ideal(df_metric, metric, ideal_values)
    df_metric["raw"] = df_metric["value_num"].copy()
    
    # 5. Agregar ao nível de split primeiro
    split_cols = ["dataset", "fold", "repetition_id"]
    
    # Agregar gap
    df_split_gap = (
        df_metric.groupby(split_cols + ["exp_id"], as_index=False)
        .agg(gap=("gap", "median"))
    )
    
    # Agregar raw (valores brutos)
    df_split_raw = (
        df_metric.groupby(split_cols + ["exp_id"], as_index=False)
        .agg(raw=("raw", "median"))
    )
    
    # Baseline típico por split - GAP
    baseline_by_split_gap = (
        df_split_gap[df_split_gap["exp_id"].isin(baseline_ids)]
        .groupby(split_cols, as_index=False)
        .agg(
            baseline_gap=("gap", "median"),
            n_baselines=("gap", "count")
        )
    )
    
    # Baseline típico por split - RAW
    baseline_by_split_raw = (
        df_split_raw[df_split_raw["exp_id"].isin(baseline_ids)]
        .groupby(split_cols, as_index=False)
        .agg(
            baseline_raw=("raw", "median"),
            n_baselines=("raw", "count")
        )
    )
    
    # Validar baselines
    invalid_mask = baseline_by_split_gap["n_baselines"] < min_baselines
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        examples = baseline_by_split_gap.loc[invalid_mask, split_cols].head(5).to_dict("records")
        warnings.warn(
            f"[aggregate_to_dataset_level] baseline_typical invalidado em {n_invalid} splits "
            f"(n_baselines < {min_baselines}). Exemplos: {examples}",
            RuntimeWarning,
        )
    baseline_by_split_gap.loc[baseline_by_split_gap["n_baselines"] < min_baselines, "baseline_gap"] = np.nan
    baseline_by_split_raw.loc[baseline_by_split_raw["n_baselines"] < min_baselines, "baseline_raw"] = np.nan

    # 6. Agregar para dataset-level
    if analysis_type == "category":
        result_rows_gap = []
        result_rows_raw = []
        
        for dataset in df_split_gap["dataset"].unique():
            row_gap = {"dataset": dataset}
            row_raw = {"dataset": dataset}
            
            # Baseline típico
            bl_splits_gap = baseline_by_split_gap[baseline_by_split_gap["dataset"] == dataset]["baseline_gap"]
            bl_splits_raw = baseline_by_split_raw[baseline_by_split_raw["dataset"] == dataset]["baseline_raw"]
            row_gap[BASELINE_TYPICAL_LABEL] = bl_splits_gap.median() if len(bl_splits_gap.dropna()) > 0 else np.nan
            row_raw[BASELINE_TYPICAL_LABEL] = bl_splits_raw.median() if len(bl_splits_raw.dropna()) > 0 else np.nan
            
            for category_name in ["pre_processing", "in_processing"]:
                category_ids = category_mapping.get(category_name, [])
                
                # GAP
                category_splits_gap = df_split_gap[
                    (df_split_gap["dataset"] == dataset) & 
                    (df_split_gap["exp_id"].isin(category_ids))
                ]
                
                # RAW
                category_splits_raw = df_split_raw[
                    (df_split_raw["dataset"] == dataset) & 
                    (df_split_raw["exp_id"].isin(category_ids))
                ]
                
                if category_splits_gap.empty:
                    row_gap[category_name] = np.nan
                    row_raw[category_name] = np.nan
                else:
                    category_by_split_gap = (
                        category_splits_gap.groupby(split_cols, as_index=False)
                        .agg(category_gap_split=("gap", "median"))
                    )
                    row_gap[category_name] = category_by_split_gap["category_gap_split"].median()
                    
                    category_by_split_raw = (
                        category_splits_raw.groupby(split_cols, as_index=False)
                        .agg(category_raw_split=("raw", "median"))
                    )
                    row_raw[category_name] = category_by_split_raw["category_raw_split"].median()
            
            result_rows_gap.append(row_gap)
            result_rows_raw.append(row_raw)
        
        result_gap = pd.DataFrame(result_rows_gap)
        result_raw = pd.DataFrame(result_rows_raw)
        
    else:  # experiment
        # GAP
        baseline_by_dataset_gap = (
            baseline_by_split_gap.groupby("dataset", as_index=False)
            .agg(baseline_gap_median=("baseline_gap", "median"))
        )
        baseline_by_dataset_gap = baseline_by_dataset_gap.rename(columns={"baseline_gap_median": BASELINE_TYPICAL_LABEL})
        
        # RAW
        baseline_by_dataset_raw = (
            baseline_by_split_raw.groupby("dataset", as_index=False)
            .agg(baseline_raw_median=("baseline_raw", "median"))
        )
        baseline_by_dataset_raw = baseline_by_dataset_raw.rename(columns={"baseline_raw_median": BASELINE_TYPICAL_LABEL})
        
        exp_ids = [e for e in EXP_ORDER if e in df_split_gap["exp_id"].unique() and e not in baseline_ids]
        
        if not exp_ids:
            if return_raw_matrix:
                return pd.DataFrame(), pd.DataFrame()
            return pd.DataFrame()
        
        # GAP
        exp_by_dataset_gap = (
            df_split_gap[df_split_gap["exp_id"].isin(exp_ids)]
            .groupby(["dataset", "exp_id"], as_index=False)
            .agg(gap=("gap", "median"))
            .pivot(index="dataset", columns="exp_id", values="gap")
            .reset_index()
        )
        
        # RAW
        exp_by_dataset_raw = (
            df_split_raw[df_split_raw["exp_id"].isin(exp_ids)]
            .groupby(["dataset", "exp_id"], as_index=False)
            .agg(raw=("raw", "median"))
            .pivot(index="dataset", columns="exp_id", values="raw")
            .reset_index()
        )
        
        result_gap = pd.merge(baseline_by_dataset_gap, exp_by_dataset_gap, on="dataset", how="outer")
        result_raw = pd.merge(baseline_by_dataset_raw, exp_by_dataset_raw, on="dataset", how="outer")
    
    if return_raw_matrix:
        return result_gap, result_raw
    return result_gap


# =============================================================================
# DIAGNÓSTICO DE COMPLETUDE
# =============================================================================

def compute_completeness_report(
    data_matrix: pd.DataFrame,
    method_cols: List[str]
) -> Dict:
    """Gera relatório de completude antes do complete-case."""
    valid_cols = [c for c in method_cols if c in data_matrix.columns]
    
    if data_matrix.empty or not valid_cols:
        return {
            "n_datasets_total": 0,
            "n_datasets_complete": 0,
            "method_presence": {},
            "datasets_dropped": [],
            "methods_with_missing": []
        }
    
    n_total = len(data_matrix)
    
    method_presence = {}
    for col in valid_cols:
        n_present = data_matrix[col].notna().sum()
        method_presence[col] = {
            "n_present": int(n_present),
            "pct_present": float(n_present / n_total * 100) if n_total > 0 else 0
        }
    
    df_complete = data_matrix[["dataset"] + valid_cols].dropna()
    n_complete = len(df_complete)
    
    complete_datasets = set(df_complete["dataset"].tolist())
    all_datasets = set(data_matrix["dataset"].tolist())
    datasets_dropped = list(all_datasets - complete_datasets)
    
    methods_with_missing = [
        col for col, info in method_presence.items() 
        if info["pct_present"] < 100
    ]
    
    return {
        "n_datasets_total": n_total,
        "n_datasets_complete": n_complete,
        "method_presence": method_presence,
        "datasets_dropped": datasets_dropped,
        "methods_with_missing": methods_with_missing
    }


# =============================================================================
# TESTE WILCOXON PAREADO (PARA K=2)
# =============================================================================

def run_wilcoxon_paired(
    data_matrix: pd.DataFrame,
    method1: str,
    method2: str,
    alternative: Literal["two-sided", "less", "greater"] = "less"
) -> Dict:
    """Executa teste Wilcoxon signed-rank para comparação binária (k=2)."""
    from scipy.stats import wilcoxon as scipy_wilcoxon
    
    if method1 not in data_matrix.columns or method2 not in data_matrix.columns:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_datasets": 0,
            "median_diff": np.nan,
            "valid": False
        }
    
    df_complete = data_matrix[["dataset", method1, method2]].dropna()
    n_datasets = len(df_complete)
    
    if n_datasets < 2:
        warnings.warn(
            f"Wilcoxon requer pelo menos 2 datasets. Encontrados: {n_datasets}.",
            UserWarning
        )
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_datasets": n_datasets,
            "median_diff": np.nan,
            "valid": False
        }
    
    diffs = df_complete[method2].values - df_complete[method1].values
    median_diff = float(np.median(diffs))
    
    nonzero_diffs = diffs[diffs != 0]
    
    if len(nonzero_diffs) < 2:
        warnings.warn(
            f"Wilcoxon: apenas {len(nonzero_diffs)} diferenças não-zero. "
            "Resultado pode ser impreciso.",
            UserWarning
        )
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_datasets": n_datasets,
            "median_diff": median_diff,
            "valid": False
        }
    
    try:
        stat, p_value = scipy_wilcoxon(nonzero_diffs, alternative=alternative)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "n_datasets": n_datasets,
            "median_diff": median_diff,
            "valid": True
        }
    except Exception as e:
        warnings.warn(f"Erro no teste Wilcoxon: {e}", UserWarning)
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_datasets": n_datasets,
            "median_diff": median_diff,
            "valid": False
        }


# =============================================================================
# TESTE FRIEDMAN (OMNIBUS)
# =============================================================================

def run_friedman_test(
    data_matrix: pd.DataFrame,
    method_cols: List[str],
    control_method: str = BASELINE_TYPICAL_LABEL,
    alternative: str = "less",
    raw_matrix: pd.DataFrame = None
) -> Dict:
    """
    Executa teste de Friedman nos dados.
    
    Se k=2, automaticamente usa Wilcoxon signed-rank.
    
    Args:
        data_matrix: DataFrame com valores de gap (usado para o teste)
        method_cols: Lista de colunas de métodos
        control_method: Método de controle
        alternative: Alternativa do teste
        raw_matrix: DataFrame com valores brutos (para estatísticas descritivas)
    
    Returns:
        Dict com: statistic, p_value, n_datasets, k_methods, avg_ranks,
                  descriptives_gap, descriptives_raw, descriptives_delta_fair,
                  descriptives_r_fair, completeness
    """
    valid_cols = [c for c in method_cols if c in data_matrix.columns]
    
    cols_with_data = [c for c in valid_cols if data_matrix[c].notna().any()]
    
    if len(cols_with_data) < len(valid_cols):
        removed = set(valid_cols) - set(cols_with_data)
        warnings.warn(
            f"Métodos removidos (sem dados): {removed}. "
            f"Continuando com {len(cols_with_data)} métodos.",
            UserWarning
        )
    
    valid_cols = cols_with_data
    completeness = compute_completeness_report(data_matrix, valid_cols)
    
    if len(valid_cols) < 2:
        warnings.warn(
            f"Teste requer pelo menos 2 métodos com dados. Encontrados: {len(valid_cols)}. "
            "Retornando resultados vazios.",
            UserWarning
        )
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_datasets": 0,
            "k_methods": len(valid_cols),
            "avg_ranks": {},
            "descriptives_gap": {},
            "descriptives_raw": {},
            "descriptives_delta_fair": {},
            "descriptives_r_fair": {},
            "valid": False,
            "test_used": "none",
            "completeness": completeness
        }

    df_complete = data_matrix[["dataset"] + valid_cols].dropna()
    n_datasets = len(df_complete)
    k_methods = len(valid_cols)
    
    if n_datasets < 2:
        warnings.warn(
            f"Teste requer pelo menos 2 datasets com dados completos. "
            f"Encontrados: {n_datasets} (de {completeness['n_datasets_total']} totais). "
            f"Datasets removidos: {completeness['datasets_dropped']}",
            UserWarning
        )
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n_datasets": n_datasets,
            "k_methods": k_methods,
            "avg_ranks": {},
            "descriptives_gap": {},
            "descriptives_raw": {},
            "descriptives_delta_fair": {},
            "descriptives_r_fair": {},
            "valid": False,
            "test_used": "none",
            "completeness": completeness
        }

    # Calcular estatísticas descritivas para GAP (mediana, IQR, N)
    descriptives_gap = {}
    for col in valid_cols:
        values = df_complete[col].values
        descriptives_gap[col] = {
            "median": float(np.median(values)),
            "iqr": float(iqr(values)),
            "n": int(len(values))
        }

    # Calcular estatísticas descritivas para DELTA_FAIR e R_FAIR
    # Δfair = d_baseline - d_method
    # r_fair = Δfair / (d_baseline + ε)  (mitigação relativa)
    descriptives_delta_fair = {}
    descriptives_r_fair = {}
    if control_method in valid_cols:
        baseline_values = df_complete[control_method].values
        for col in valid_cols:
            if col == control_method:
                # Delta_fair e r_fair do baseline consigo mesmo são 0
                descriptives_delta_fair[col] = {
                    "median": 0.0,
                    "iqr": 0.0,
                    "n": int(len(baseline_values))
                }
                descriptives_r_fair[col] = {
                    "median": 0.0,
                    "iqr": 0.0,
                    "n": int(len(baseline_values))
                }
            else:
                delta_fair_values = baseline_values - df_complete[col].values
                r_fair_values = delta_fair_values / (baseline_values + _REL_EPS)
                descriptives_delta_fair[col] = {
                    "median": float(np.median(delta_fair_values)),
                    "iqr": float(iqr(delta_fair_values)),
                    "n": int(len(delta_fair_values))
                }
                descriptives_r_fair[col] = {
                    "median": float(np.median(r_fair_values)),
                    "iqr": float(iqr(r_fair_values)),
                    "n": int(len(r_fair_values))
                }
    else:
        for col in valid_cols:
            descriptives_delta_fair[col] = {
                "median": np.nan,
                "iqr": np.nan,
                "n": 0
            }
            descriptives_r_fair[col] = {
                "median": np.nan,
                "iqr": np.nan,
                "n": 0
            }
    
    # Calcular estatísticas descritivas para RAW (valores brutos)
    descriptives_raw = {}
    if raw_matrix is not None and not raw_matrix.empty:
        # Alinhar com os mesmos datasets do complete-case
        complete_datasets = set(df_complete["dataset"].tolist())
        raw_complete = raw_matrix[raw_matrix["dataset"].isin(complete_datasets)]
        
        for col in valid_cols:
            if col in raw_complete.columns:
                raw_values = raw_complete[col].dropna().values
                if len(raw_values) > 0:
                    descriptives_raw[col] = {
                        "median": float(np.median(raw_values)),
                        "iqr": float(iqr(raw_values)),
                        "n": int(len(raw_values))
                    }
                else:
                    descriptives_raw[col] = {
                        "median": np.nan,
                        "iqr": np.nan,
                        "n": 0
                    }
            else:
                descriptives_raw[col] = {
                    "median": np.nan,
                    "iqr": np.nan,
                    "n": 0
                }
    else:
        # Se não tiver raw_matrix, deixar vazio
        for col in valid_cols:
            descriptives_raw[col] = {
                "median": np.nan,
                "iqr": np.nan,
                "n": 0
            }
    
    # FALLBACK: k=2 â†’ Wilcoxon signed-rank
    if k_methods == 2:
        warnings.warn(
            f"k=2 métodos detectado. Usando Wilcoxon signed-rank em vez de Friedman.",
            UserWarning
        )
        
        if control_method in valid_cols:
            treatment = [m for m in valid_cols if m != control_method][0]
        else:
            control_method = valid_cols[0]
            treatment = valid_cols[1]
        
        wilcox_result = run_wilcoxon_paired(
            data_matrix, control_method, treatment, alternative
        )
        
        values = df_complete[valid_cols].values
        ranks = np.array([rankdata(row) for row in values])
        avg_ranks = {col: ranks[:, i].mean() for i, col in enumerate(valid_cols)}
        
        return {
            "statistic": wilcox_result["statistic"],
            "p_value": wilcox_result["p_value"],
            "n_datasets": wilcox_result["n_datasets"],
            "k_methods": k_methods,
            "avg_ranks": avg_ranks,
            "descriptives_gap": descriptives_gap,
            "descriptives_raw": descriptives_raw,
            "descriptives_delta_fair": descriptives_delta_fair,
            "descriptives_r_fair": descriptives_r_fair,
            "valid": wilcox_result["valid"],
            "test_used": "wilcoxon",
            "median_diff": wilcox_result.get("median_diff", np.nan),
            "completeness": completeness
        }
    
    # k >= 3: Friedman
    values = df_complete[valid_cols].values
    ranks = np.array([rankdata(row) for row in values])
    avg_ranks = {col: ranks[:, i].mean() for i, col in enumerate(valid_cols)}
    
    try:
        stat, p_value = friedmanchisquare(*[values[:, i] for i in range(k_methods)])
    except Exception as e:
        warnings.warn(f"Erro no teste de Friedman: {e}", UserWarning)
        stat, p_value = np.nan, np.nan
    
    return {
        "statistic": float(stat) if not np.isnan(stat) else np.nan,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "n_datasets": n_datasets,
        "k_methods": k_methods,
        "avg_ranks": avg_ranks,
        "descriptives_gap": descriptives_gap,
        "descriptives_raw": descriptives_raw,
        "descriptives_delta_fair": descriptives_delta_fair,
        "descriptives_r_fair": descriptives_r_fair,
        "valid": True,
        "test_used": "friedman",
        "completeness": completeness
    }


# =============================================================================
# PÓS-HOC: COMPARAÇÕES VS CONTROLE COM HOLM
# =============================================================================

def run_posthoc_vs_control(
    friedman_result: Dict,
    control_method: str = BASELINE_TYPICAL_LABEL,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "less"
) -> pd.DataFrame:
    """
    Executa comparações pós-hoc vs controle usando z-test em ranks médios.
    
    Fórmula: z = (R_i - R_ctrl) / sqrt(k(k+1)/(6N))
    """
    if not friedman_result.get("valid", False):
        return pd.DataFrame()
    
    avg_ranks = friedman_result["avg_ranks"]
    n = friedman_result["n_datasets"]
    k = friedman_result["k_methods"]
    
    if control_method not in avg_ranks:
        warnings.warn(
            f"Método de controle '{control_method}' não encontrado nos ranks. "
            "Retornando resultados vazios.",
            UserWarning
        )
        return pd.DataFrame()
    
    control_rank = avg_ranks[control_method]
    methods = [m for m in avg_ranks.keys() if m != control_method]
    
    if len(methods) == 0:
        warnings.warn(
            "Nenhum método além do controle para comparar. "
            "Retornando resultados vazios.",
            UserWarning
        )
        return pd.DataFrame()
    
    se = np.sqrt(k * (k + 1) / (6 * n))
    
    results = []
    for method in methods:
        method_rank = avg_ranks[method]
        delta_rank = method_rank - control_rank
        z = delta_rank / se
        
        if alternative == "less":
            p_raw = norm.cdf(z)
        elif alternative == "greater":
            p_raw = 1 - norm.cdf(z)
        else:
            p_raw = 2 * (1 - norm.cdf(abs(z)))
        
        results.append({
            "method": method,
            "avg_rank": method_rank,
            "control_rank": control_rank,
            "delta_rank": delta_rank,
            "z": z,
            "p_raw": p_raw
        })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        reject, p_holm, _, _ = multipletests(
            df_results["p_raw"].values,
            alpha=alpha,
            method="holm"
        )
        df_results["p_holm"] = p_holm
        df_results["reject_holm"] = reject
    else:
        df_results["p_holm"] = []
        df_results["reject_holm"] = []
    
    df_results["n_datasets"] = n
    df_results["k_methods"] = k
    df_results["control"] = control_method
    
    return df_results


# =============================================================================
# ANÁLISE POR CATEGORIA
# =============================================================================

def run_category_tests(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    scope: str | List[str] = "all",
    baseline_ids: List[str] = None,
    category_mapping: Dict = None,
    alpha: float = 0.05,
    run_posthoc: Literal["always", "if_significant", "never"] = "always",
    alternative: Literal["two-sided", "less", "greater"] = "less",
    min_baselines: int = 3
) -> Dict:
    """
    Executa análise Friedman + pós-hoc para CATEGORIAS de algoritmo.
    
    Compara: baseline_typical vs pre_processing vs in_processing
    
    Args:
        scope: Escopo de filtro (PU-aware). Opções:
            - "all": sem filtro
            - "A", "B", "C": por categoria (prefixo)
            - "A1", "B2": por padrão (inclui A1, A1P, A1U)
            - "A1P", "A1U": por padrão+direção (match exato)
            - Lista de scopes: OR entre eles
    """
    # Obter tanto gap_matrix quanto raw_matrix
    gap_matrix, raw_matrix = aggregate_to_dataset_level(
        df, metric, ideal_values, scope, baseline_ids, category_mapping,
        analysis_type="category", min_baselines=min_baselines,
        return_raw_matrix=True
    )
    
    if gap_matrix.empty:
        return {
            "scope": scope,
            "metric": metric,
            "analysis_type": "category",
            "friedman": {"valid": False, "completeness": {}},
            "posthoc": pd.DataFrame(),
            "data_matrix": pd.DataFrame(),
            "raw_matrix": pd.DataFrame()
        }
    
    method_cols = [BASELINE_TYPICAL_LABEL, "pre_processing", "in_processing"]
    
    friedman_result = run_friedman_test(
        gap_matrix, method_cols, 
        control_method=BASELINE_TYPICAL_LABEL,
        alternative=alternative,
        raw_matrix=raw_matrix
    )
    
    posthoc_df = pd.DataFrame()
    if run_posthoc == "always" or (
        run_posthoc == "if_significant" and 
        friedman_result.get("p_value", 1) < alpha
    ):
        posthoc_df = run_posthoc_vs_control(
            friedman_result, BASELINE_TYPICAL_LABEL, alpha, alternative
        )
    
    return {
        "scope": scope,
        "metric": metric,
        "analysis_type": "category",
        "friedman": friedman_result,
        "posthoc": posthoc_df,
        "data_matrix": gap_matrix,
        "raw_matrix": raw_matrix
    }


def run_category_analysis(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    scopes: List[str] = None,
    baseline_ids: List[str] = None,
    alpha: float = 0.05,
    run_posthoc: str = "always",
    alternative: str = "less",
    save_dir: Optional[str] = None,
    translate: Dict = None
) -> pd.DataFrame:
    """Análise completa por CATEGORIA para múltiplos scopes."""
    scopes = scopes or ["all"]
    
    all_results = []
    
    for scope in scopes:
        result = run_category_tests(
            df, metric, ideal_values, scope, baseline_ids,
            alpha=alpha, run_posthoc=run_posthoc, alternative=alternative
        )
        
        friedman = result["friedman"]
        posthoc = result["posthoc"]
        completeness = friedman.get("completeness", {})
        descriptives_gap = friedman.get("descriptives_gap", {})
        descriptives_raw = friedman.get("descriptives_raw", {})
        descriptives_delta_fair = friedman.get("descriptives_delta_fair", {})
        descriptives_r_fair = friedman.get("descriptives_r_fair", {})

        # Dados comuns do teste omnibus
        omnibus_common = {
            "scope": scope,
            "metric": metric,
            "analysis_type": "category",
            "test": friedman.get("test_used", "friedman"),
            "statistic": friedman.get("statistic", np.nan),
            "p_value": friedman.get("p_value", np.nan),
            "n_datasets": friedman.get("n_datasets", 0),
            "n_datasets_total": completeness.get("n_datasets_total", 0),
            "k_methods": friedman.get("k_methods", 0),
        }

        # Uma linha por categoria (unpivot) com estatísticas GAP, RAW, DELTA_FAIR e R_FAIR
        for category_name in descriptives_gap.keys():
            stats_gap = descriptives_gap.get(category_name, {})
            stats_raw = descriptives_raw.get(category_name, {})
            stats_delta = descriptives_delta_fair.get(category_name, {})
            stats_rfair = descriptives_r_fair.get(category_name, {})

            omnibus_row = {
                **omnibus_common,
                "category": category_name,
                # Estatísticas de GAP (usadas no teste)
                "median_gap": stats_gap.get("median", np.nan),
                "iqr_gap": stats_gap.get("iqr", np.nan),
                # Estatísticas DELTA_FAIR (Δfair = d_baseline - d_method)
                "median_delta_fair": stats_delta.get("median", np.nan),
                "iqr_delta_fair": stats_delta.get("iqr", np.nan),
                # Estatísticas R_FAIR (r_fair = Δfair / (d_baseline + ε))
                "median_r_fair": stats_rfair.get("median", np.nan),
                "iqr_r_fair": stats_rfair.get("iqr", np.nan),
                # Estatísticas RAW (valores brutos)
                "median_raw": stats_raw.get("median", np.nan),
                "iqr_raw": stats_raw.get("iqr", np.nan),
                "n": stats_gap.get("n", 0),
            }
            all_results.append(omnibus_row)

        # Linhas do pós-hoc
        if not posthoc.empty:
            for _, row in posthoc.iterrows():
                method_name = row["method"]
                stats_gap = descriptives_gap.get(method_name, {})
                stats_raw = descriptives_raw.get(method_name, {})
                stats_delta = descriptives_delta_fair.get(method_name, {})
                stats_rfair = descriptives_r_fair.get(method_name, {})

                posthoc_row = {
                    "scope": scope,
                    "metric": metric,
                    "analysis_type": "category",
                    "test": "posthoc_holm",
                    "category": method_name,
                    "avg_rank": row["avg_rank"],
                    "control_rank": row["control_rank"],
                    "delta_rank": row["delta_rank"],
                    "z": row["z"],
                    "p_raw": row["p_raw"],
                    "p_holm": row["p_holm"],
                    "reject_holm": row["reject_holm"],
                    "n_datasets": row["n_datasets"],
                    "k_methods": row["k_methods"],
                    # Estatísticas de GAP
                    "median_gap": stats_gap.get("median", np.nan),
                    "iqr_gap": stats_gap.get("iqr", np.nan),
                    # Estatísticas DELTA_FAIR
                    "median_delta_fair": stats_delta.get("median", np.nan),
                    "iqr_delta_fair": stats_delta.get("iqr", np.nan),
                    # Estatísticas R_FAIR
                    "median_r_fair": stats_rfair.get("median", np.nan),
                    "iqr_r_fair": stats_rfair.get("iqr", np.nan),
                    # Estatísticas RAW
                    "median_raw": stats_raw.get("median", np.nan),
                    "iqr_raw": stats_raw.get("iqr", np.nan),
                }
                all_results.append(posthoc_row)

    stats_df = pd.DataFrame(all_results)
    return stats_df


# =============================================================================
# ANÁLISE POR EXPERIMENTO
# =============================================================================

def run_experiment_tests(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    scope: str | List[str] = "all",
    baseline_ids: List[str] = None,
    exp_order: List[str] = None,
    alpha: float = 0.05,
    run_posthoc: Literal["always", "if_significant", "never"] = "always",
    alternative: Literal["two-sided", "less", "greater"] = "less",
    min_baselines: int = 3
) -> Dict:
    """
    Executa análise Friedman + pós-hoc para CADA experimento vs baseline.
    
    Args:
        scope: Escopo de filtro (PU-aware). Opções:
            - "all": sem filtro
            - "A", "B", "C": por categoria (prefixo)
            - "A1", "B2": por padrão (inclui A1, A1P, A1U)
            - "A1P", "A1U": por padrão+direção (match exato)
            - Lista de scopes: OR entre eles
    """
    baseline_ids = baseline_ids or BASELINE_IDS
    exp_order = exp_order or EXP_ORDER
    
    # Obter tanto gap_matrix quanto raw_matrix
    gap_matrix, raw_matrix = aggregate_to_dataset_level(
        df, metric, ideal_values, scope, baseline_ids,
        analysis_type="experiment", min_baselines=min_baselines,
        return_raw_matrix=True
    )
    
    if gap_matrix.empty:
        return {
            "scope": scope,
            "metric": metric,
            "analysis_type": "experiment",
            "friedman": {"valid": False, "completeness": {}},
            "posthoc": pd.DataFrame(),
            "data_matrix": pd.DataFrame(),
            "raw_matrix": pd.DataFrame()
        }
    
    exp_ids = [e for e in exp_order if e in gap_matrix.columns and e not in baseline_ids]
    method_cols = [BASELINE_TYPICAL_LABEL] + exp_ids
    
    friedman_result = run_friedman_test(
        gap_matrix, method_cols,
        control_method=BASELINE_TYPICAL_LABEL,
        alternative=alternative,
        raw_matrix=raw_matrix
    )
    
    posthoc_df = pd.DataFrame()
    if run_posthoc == "always" or (
        run_posthoc == "if_significant" and 
        friedman_result.get("p_value", 1) < alpha
    ):
        posthoc_df = run_posthoc_vs_control(
            friedman_result, BASELINE_TYPICAL_LABEL, alpha, alternative
        )
        
        if not posthoc_df.empty:
            posthoc_df["category"] = posthoc_df["method"].map(EXP_TO_CATEGORY)
    
    return {
        "scope": scope,
        "metric": metric,
        "analysis_type": "experiment",
        "friedman": friedman_result,
        "posthoc": posthoc_df,
        "data_matrix": gap_matrix,
        "raw_matrix": raw_matrix
    }


def run_experiment_analysis(
    df: pd.DataFrame,
    metric: str,
    ideal_values: Dict,
    scopes: List[str] = None,
    baseline_ids: List[str] = None,
    alpha: float = 0.05,
    run_posthoc: str = "always",
    alternative: str = "less",
    save_dir: Optional[str] = None,
    translate: Dict = None,
    cols_omnibus: List[str] = None,
    cols_posthoc: List[str] = None,
) -> pd.DataFrame:
    """Análise completa por EXPERIMENTO para múltiplos scopes."""
    scopes = scopes or ["all"]
    
    all_results = []
    
    for scope in scopes:
        result = run_experiment_tests(
            df, metric, ideal_values, scope, baseline_ids,
            alpha=alpha, run_posthoc=run_posthoc, alternative=alternative
        )
        
        friedman = result["friedman"]
        posthoc = result["posthoc"]
        completeness = friedman.get("completeness", {})
        descriptives_gap = friedman.get("descriptives_gap", {})
        descriptives_raw = friedman.get("descriptives_raw", {})
        descriptives_delta_fair = friedman.get("descriptives_delta_fair", {})
        descriptives_r_fair = friedman.get("descriptives_r_fair", {})

        # Dados comuns do teste omnibus
        omnibus_common = {
            "scope": scope,
            "metric": metric,
            "analysis_type": "experiment",
            "test": friedman.get("test_used", "friedman"),
            "statistic": friedman.get("statistic", np.nan),
            "p_value": friedman.get("p_value", np.nan),
            "n_datasets": friedman.get("n_datasets", 0),
            "n_datasets_total": completeness.get("n_datasets_total", 0),
            "k_methods": friedman.get("k_methods", 0),
        }

        # Uma linha por método/experimento (unpivot) com estatísticas GAP, RAW, DELTA_FAIR e R_FAIR
        for method_name in descriptives_gap.keys():
            stats_gap = descriptives_gap.get(method_name, {})
            stats_raw = descriptives_raw.get(method_name, {})
            stats_delta = descriptives_delta_fair.get(method_name, {})
            stats_rfair = descriptives_r_fair.get(method_name, {})

            omnibus_row = {
                **omnibus_common,
                "method": method_name,
                "category": EXP_TO_CATEGORY.get(method_name, "baseline" if method_name == BASELINE_TYPICAL_LABEL else ""),
                # Estatísticas de GAP (usadas no teste)
                "median_gap": stats_gap.get("median", np.nan),
                "iqr_gap": stats_gap.get("iqr", np.nan),
                # Estatísticas DELTA_FAIR (Δfair = d_baseline - d_method)
                "median_delta_fair": stats_delta.get("median", np.nan),
                "iqr_delta_fair": stats_delta.get("iqr", np.nan),
                # Estatísticas R_FAIR (r_fair = Δfair / (d_baseline + ε))
                "median_r_fair": stats_rfair.get("median", np.nan),
                "iqr_r_fair": stats_rfair.get("iqr", np.nan),
                # Estatísticas RAW (valores brutos)
                "median_raw": stats_raw.get("median", np.nan),
                "iqr_raw": stats_raw.get("iqr", np.nan),
                "n": stats_gap.get("n", 0),
            }
            all_results.append(omnibus_row)

        # Linhas do pós-hoc
        if not posthoc.empty:
            for _, row in posthoc.iterrows():
                method_name = row["method"]
                stats_gap = descriptives_gap.get(method_name, {})
                stats_raw = descriptives_raw.get(method_name, {})
                stats_delta = descriptives_delta_fair.get(method_name, {})
                stats_rfair = descriptives_r_fair.get(method_name, {})

                posthoc_row = {
                    "scope": scope,
                    "metric": metric,
                    "analysis_type": "experiment",
                    "test": "posthoc_holm",
                    "method": method_name,
                    "category": row.get("category", ""),
                    "avg_rank": row["avg_rank"],
                    "control_rank": row["control_rank"],
                    "delta_rank": row["delta_rank"],
                    "z": row["z"],
                    "p_raw": row["p_raw"],
                    "p_holm": row["p_holm"],
                    "reject_holm": row["reject_holm"],
                    "n_datasets": row["n_datasets"],
                    "k_methods": row["k_methods"],
                    # Estatísticas de GAP
                    "median_gap": stats_gap.get("median", np.nan),
                    "iqr_gap": stats_gap.get("iqr", np.nan),
                    # Estatísticas DELTA_FAIR
                    "median_delta_fair": stats_delta.get("median", np.nan),
                    "iqr_delta_fair": stats_delta.get("iqr", np.nan),
                    # Estatísticas R_FAIR
                    "median_r_fair": stats_rfair.get("median", np.nan),
                    "iqr_r_fair": stats_rfair.get("iqr", np.nan),
                    # Estatísticas RAW
                    "median_raw": stats_raw.get("median", np.nan),
                    "iqr_raw": stats_raw.get("iqr", np.nan),
                }
                all_results.append(posthoc_row)
    
    stats_df = pd.DataFrame(all_results)
    
    if save_dir and not stats_df.empty:
        save_results_latex(
            stats_df, metric, "experiment", save_dir, translate,
            cols_omnibus=cols_omnibus, cols_posthoc=cols_posthoc,
        )

    return stats_df


# =============================================================================
# TODAS AS MÉTRICAS
# =============================================================================

def run_all_category_analysis(
    df: pd.DataFrame,
    ideal_values: Dict,
    metrics: List[str] = None,
    metric_type: str = "fairness",
    scopes: List[str] = None,
    print_results: bool = True,
    save_dir: Optional[str] = None,
    translate: Dict = None,
    alternative: str = "less"
) -> pd.DataFrame:
    """Análise por categoria para TODAS as métricas."""
    if metrics is None:
        if "metric_type" in df.columns:
            metrics = df[df["metric_type"] == metric_type]["metric"].unique().tolist()
        else:
            metrics = df["metric"].unique().tolist()
    
    all_results = []
    for metric in metrics:
        result = run_category_analysis(
            df, metric, ideal_values, scopes,
            save_dir=save_dir, translate=translate, alternative=alternative
        )
        if not result.empty:
            all_results.append(result)
            if print_results:
                print(f"\n{'='*60}")
                print(f"  {metric} - Análise por Categoria")
                print(f"{'='*60}")
                print_category_results(result)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def run_all_experiment_analysis(
    df: pd.DataFrame,
    ideal_values: Dict,
    metrics: List[str] = None,
    metric_type: str = "fairness",
    scopes: List[str] = None,
    print_results: bool = True,
    save_dir: Optional[str] = None,
    translate: Dict = None,
    alternative: str = "less",
    cols_omnibus: List[str] = None,
    cols_posthoc: List[str] = None,
) -> pd.DataFrame:
    """Análise por experimento para TODAS as métricas."""
    if metrics is None:
        if "metric_type" in df.columns:
            metrics = df[df["metric_type"] == metric_type]["metric"].unique().tolist()
        else:
            metrics = df["metric"].unique().tolist()

    all_results = []
    for metric in metrics:
        result = run_experiment_analysis(
            df, metric, ideal_values, scopes,
            save_dir=save_dir, translate=translate, alternative=alternative,
            cols_omnibus=cols_omnibus, cols_posthoc=cols_posthoc,
        )
        if not result.empty:
            all_results.append(result)
            if print_results:
                print(f"\n{'='*60}")
                print(f"  {metric} - Análise por Método")
                print(f"{'='*60}")
                print_experiment_results(result)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# IMPRESSÃO DE RESULTADOS
# =============================================================================

def _format_pvalue(p: float) -> str:
    """Formata p-valor com asteriscos de significância."""
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f}"


def print_category_results(stats_df: pd.DataFrame) -> None:
    """Imprime resultados de análise por categoria."""
    if stats_df.empty:
        print("Sem dados")
        return
    
    # Filtrar linhas omnibus (test = friedman ou wilcoxon)
    omnibus = stats_df[stats_df["test"].isin(["friedman", "wilcoxon"])]
    if not omnibus.empty:
        row = omnibus.iloc[0]
        test_name = row.get("test", "friedman").capitalize()
        print(f"\n📊 Teste {test_name} (omnibus)")
        
        n_total = row.get("n_datasets_total", row["n_datasets"])
        if n_total > row["n_datasets"]:
            print(f"   ⚠️  N datasets: {row['n_datasets']} de {n_total} (complete-case)")
        else:
            print(f"   N datasets: {row['n_datasets']}, k métodos: {row['k_methods']}")
        
        print(f"   Estatística = {row['statistic']:.4f}, p = {_format_pvalue(row['p_value'])}")
        
        # Estatísticas descritivas (formato unpivotado) com d, Δfair e RAW
        print("   Estatísticas descritivas:")
        print("   " + "-" * 100)
        print(f"   {'Categoria':<20} {'Median(d)':<12} {'Median(Δfair)':<14} {'IQR(Δfair)':<12} {'Median(raw)':<12} {'IQR(raw)':<10} N")
        print("   " + "-" * 100)
        for _, r in omnibus.iterrows():
            cat = r.get("category", "")
            median_gap = r.get("median_gap", np.nan)
            median_delta = r.get("median_delta_fair", np.nan)
            iqr_delta = r.get("iqr_delta_fair", np.nan)
            median_raw = r.get("median_raw", np.nan)
            iqr_raw = r.get("iqr_raw", np.nan)
            n = int(r.get("n", 0))

            median_gap_str = f"{median_gap:.4f}" if pd.notna(median_gap) else "—"
            median_delta_str = f"{median_delta:.4f}" if pd.notna(median_delta) else "—"
            iqr_delta_str = f"{iqr_delta:.4f}" if pd.notna(iqr_delta) else "—"
            median_raw_str = f"{median_raw:.4f}" if pd.notna(median_raw) else "—"
            iqr_raw_str = f"{iqr_raw:.4f}" if pd.notna(iqr_raw) else "—"

            print(f"   {cat:<20} {median_gap_str:<12} {median_delta_str:<14} {iqr_delta_str:<12} {median_raw_str:<12} {iqr_raw_str:<10} {n}")

    posthoc = stats_df[stats_df["test"] == "posthoc_holm"]
    if not posthoc.empty:
        print(f"\n📊 Pós-hoc vs {BASELINE_TYPICAL_LABEL} (Holm)")
        print("   (* p<0.05, ** p<0.01, *** p<0.001)")

        cols_display = ["category", "avg_rank", "delta_rank", "z", "p_raw", "p_holm", "reject_holm",
                       "median_delta_fair", "iqr_delta_fair", "median_gap"]
        cols_present = [c for c in cols_display if c in posthoc.columns]

        df_display = posthoc[cols_present].copy()
        df_display["p_raw"] = df_display["p_raw"].apply(_format_pvalue)
        df_display["p_holm"] = df_display["p_holm"].apply(_format_pvalue)
        df_display["reject_holm"] = df_display["reject_holm"].map({True: "✔", False: ""})

        for col in ["avg_rank", "delta_rank", "z", "median_gap", "median_delta_fair", "iqr_delta_fair", "median_r_fair", "iqr_r_fair"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

        # Renomear colunas para exibição
        df_display = df_display.rename(columns={"median_gap": "median_d"})

        print(df_display.to_markdown(index=False))


def print_experiment_results(stats_df: pd.DataFrame) -> None:
    """Imprime resultados de análise por experimento."""
    if stats_df.empty:
        print("Sem dados")
        return
    
    omnibus = stats_df[stats_df["test"].isin(["friedman", "wilcoxon"])]
    if not omnibus.empty:
        row = omnibus.iloc[0]
        test_name = row.get("test", "friedman").capitalize()
        print(f"\n📊 Teste {test_name} (omnibus)")
        
        n_total = row.get("n_datasets_total", row["n_datasets"])
        if pd.notna(n_total) and n_total > row["n_datasets"]:
            print(f"   ⚠️  N datasets: {row['n_datasets']} de {int(n_total)} (complete-case)")
        else:
            print(f"   N datasets: {row['n_datasets']}, k métodos: {row['k_methods']}")
        
        print(f"   Estatística = {row['statistic']:.4f}, p = {_format_pvalue(row['p_value'])}")
        
        # Estatísticas descritivas (formato unpivotado) com d, Δfair e RAW
        print("   Estatísticas descritivas:")
        print("   " + "-" * 100)
        print(f"   {'Método':<20} {'Median(d)':<12} {'Median(Δfair)':<14} {'IQR(Δfair)':<12} {'Median(raw)':<12} {'IQR(raw)':<10} N")
        print("   " + "-" * 100)
        for _, r in omnibus.iterrows():
            method = r.get("method", "")
            median_gap = r.get("median_gap", np.nan)
            median_delta = r.get("median_delta_fair", np.nan)
            iqr_delta = r.get("iqr_delta_fair", np.nan)
            median_raw = r.get("median_raw", np.nan)
            iqr_raw = r.get("iqr_raw", np.nan)
            n = int(r.get("n", 0))

            if pd.notna(median_gap):
                median_gap_str = f"{median_gap:.4f}" if pd.notna(median_gap) else "—"
                median_delta_str = f"{median_delta:.4f}" if pd.notna(median_delta) else "—"
                iqr_delta_str = f"{iqr_delta:.4f}" if pd.notna(iqr_delta) else "—"
                median_raw_str = f"{median_raw:.4f}" if pd.notna(median_raw) else "—"
                iqr_raw_str = f"{iqr_raw:.4f}" if pd.notna(iqr_raw) else "—"

                print(f"   {method:<20} {median_gap_str:<12} {median_delta_str:<14} {iqr_delta_str:<12} {median_raw_str:<12} {iqr_raw_str:<10} {n}")

    posthoc = stats_df[stats_df["test"] == "posthoc_holm"]
    if not posthoc.empty:
        print(f"\n📊 Pós-hoc vs {BASELINE_TYPICAL_LABEL} (Holm)")
        print("   (* p<0.05, ** p<0.01, *** p<0.001)")

        cols_display = ["method", "category", "avg_rank", "delta_rank", "z", "p_holm", "reject_holm",
                       "median_delta_fair", "iqr_delta_fair", "median_gap"]
        cols = [c for c in cols_display if c in posthoc.columns]

        df_display = posthoc[cols].copy()
        df_display["p_holm"] = df_display["p_holm"].apply(_format_pvalue)
        df_display["reject_holm"] = df_display["reject_holm"].map({True: "✔", False: ""})

        for col in ["avg_rank", "delta_rank", "z", "median_gap", "median_delta_fair", "iqr_delta_fair", "median_r_fair", "iqr_r_fair"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )

        # Renomear colunas para exibição
        df_display = df_display.rename(columns={"median_gap": "median_d"})

        print(df_display.to_markdown(index=False))


# =============================================================================
# SALVAMENTO LATEX
# =============================================================================

def save_results_latex(
    stats_df: pd.DataFrame,
    metric: str,
    analysis_type: str,
    save_dir: str,
    translate: Dict = None,
    cols_omnibus: List[str] = None,
    cols_posthoc: List[str] = None,
    scope_suffix: str = None
) -> List[Path]:

    translate = translate or {}
    out_dir = Path(save_dir) / "stats_tests" / analysis_type
    out_dir.mkdir(parents=True, exist_ok=True)

    t = lambda x: translate.get(x, x)
    metric_label = t(metric)
    saved_paths = []

    suffix = f"_{scope_suffix}" if scope_suffix else ""

    def _select_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if cols is None:
            return df
        return df[[c for c in cols if c in df.columns]].copy()

    def _translate_and_escape(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Traduz valores usando o dict translate e escapa para LaTeX."""
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: _escape_latex_cell(t(x)) if isinstance(x, str) else x)
        return df

    def _rename_cols_pt(df: pd.DataFrame) -> pd.DataFrame:
        """Renomeia colunas internas para cabeçalhos LaTeX em português."""
        return df.rename(columns=_LATEX_COL_NAMES)

    # TABELA OMNIBUS (Friedman/Wilcoxon) - formato unpivotado
    omnibus = stats_df[stats_df["test"].isin(["friedman", "wilcoxon"])]
    omnibus = omnibus[omnibus["statistic"].notna()]

    if not omnibus.empty:
        df_omnibus = omnibus.copy()

        # Remover linha do baseline (Δfair=0 por definição, sem informação útil)
        for _col in ["method", "category"]:
            if _col in df_omnibus.columns and (df_omnibus[_col] == BASELINE_TYPICAL_LABEL).any():
                df_omnibus = df_omnibus[df_omnibus[_col] != BASELINE_TYPICAL_LABEL].reset_index(drop=True)
                break

        # Traduzir todas as colunas de texto (category e method sempre presentes)
        df_omnibus = _translate_and_escape(df_omnibus, ["scope", "metric", "test", "method", "category"])

        # (#4) Categoria redundante: se method == category traduzidos, setar "---"
        if "method" in df_omnibus.columns and "category" in df_omnibus.columns:
            mask_dup = df_omnibus["method"] == df_omnibus["category"]
            df_omnibus.loc[mask_dup, "category"] = "---"

        for col in ["statistic", "p_value"]:
            if col in df_omnibus.columns:
                df_omnibus[col] = df_omnibus[col].apply(
                    lambda x: f"{x:.3f}".replace(".", ",") if pd.notna(x) else "—"
                )

        for col in ["median_gap", "iqr_gap", "median_delta_fair", "iqr_delta_fair", "median_r_fair", "iqr_r_fair", "median_raw", "iqr_raw"]:
            if col in df_omnibus.columns:
                df_omnibus[col] = df_omnibus[col].apply(
                    lambda x: f"{x:.4f}".replace(".", ",") if pd.notna(x) else "—"
                )

        if "n" in df_omnibus.columns:
            df_omnibus["n"] = df_omnibus["n"].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else "—"
            )

        df_omnibus = _select_cols(df_omnibus, cols_omnibus)

        # Guardar antes do multirow para detectar fronteiras de grupo
        df_omnibus_pre = df_omnibus.copy()

        # Multirow: categóricas mesclam sempre; estatística/p-valor só dentro do mesmo teste
        df_omnibus = _apply_multirow(df_omnibus, [
            {"col": "metric",    "context": []},
            {"col": "category",  "context": []},
            {"col": "statistic", "context": ["metric"]},
            {"col": "p_value",   "context": ["metric"]},
        ])

        df_omnibus = _rename_cols_pt(df_omnibus)

        path = out_dir / f"omnibus_{_slugify(metric)}{suffix}.tex"
        if metric_label:
            caption_ctx = f"Teste de Friedman --- {metric_label}."
        else:
            caption_ctx = "Resultados consolidados do teste de Friedman."
        caption = format_caption(
            f"{caption_ctx} "
            f"$\\Delta_{{\\text{{fair}}}}$ = efeito de mitigação."
        )
        tab_label = f"tab:omnibus_{analysis_type}_{_slugify(metric)}{suffix}"
        latex = df_omnibus.to_latex(
            index=False, escape=False,
            caption=caption,
            position='H',
            label=tab_label,
        )
        latex = _insert_group_separators(latex, df_omnibus_pre, df_omnibus, "category")
        # Converter para longtable (suporta quebra de página com cabeçalho repetido)
        latex = _tabular_to_longtable(
            latex, caption, tab_label,
            n_cols=len(df_omnibus.columns),
        )
        latex = _replace_decimal_in_latex(latex)
        path.write_text(latex, encoding="utf-8")
        saved_paths.append(path)

    # TABELA POSTHOC (Holm)
    posthoc = stats_df[stats_df["test"] == "posthoc_holm"]
    posthoc = posthoc[posthoc["avg_rank"].notna()]

    if not posthoc.empty:
        df_posthoc = posthoc.copy()
        df_posthoc = _translate_and_escape(df_posthoc, ["scope", "metric", "method", "category", "exp_id"])

        for col in ["z"]:
            if col in df_posthoc.columns:
                df_posthoc[col] = df_posthoc[col].apply(
                    lambda x: f"{x:.3f}".replace(".", ",") if pd.notna(x) else "—"
                )

        for col in ["avg_rank", "control_rank", "delta_rank"]:
            if col in df_posthoc.columns:
                df_posthoc[col] = df_posthoc[col].apply(
                    lambda x: f"{x:.2f}".replace(".", ",") if pd.notna(x) else "—"
                )

        for col in ["p_raw", "p_holm"]:
            if col in df_posthoc.columns:
                df_posthoc[col] = df_posthoc[col].apply(
                    lambda x: f"{x:.4f}".replace(".", ",") if pd.notna(x) else "—"
                )

        for col in ["median_gap", "iqr_gap", "median_delta_fair", "iqr_delta_fair", "median_r_fair", "iqr_r_fair", "median_raw", "iqr_raw"]:
            if col in df_posthoc.columns:
                df_posthoc[col] = df_posthoc[col].apply(
                    lambda x: f"{x:.4f}".replace(".", ",") if pd.notna(x) else "—"
                )

        if "reject_holm" in df_posthoc.columns:
            df_posthoc["reject_holm"] = df_posthoc["reject_holm"].map(
                {True: "Sim", False: "Não"}
            ).fillna("—")

        df_posthoc = _select_cols(df_posthoc, cols_posthoc)

        # Guardar antes do multirow para detectar fronteiras de grupo
        df_posthoc_pre = df_posthoc.copy()

        # Multirow: apenas colunas categóricas mesclam no pós-hoc
        df_posthoc = _apply_multirow(df_posthoc, [
            {"col": "metric",   "context": []},
            {"col": "category", "context": []},
        ])

        df_posthoc = _rename_cols_pt(df_posthoc)

        path = out_dir / f"posthoc_{_slugify(metric)}{suffix}.tex"
        analysis_label = t(analysis_type)
        if metric_label:
            caption_ctx = f"Pós-hoc vs {t(BASELINE_TYPICAL_LABEL)} (por {analysis_label.lower()}) --- {metric_label}."
        else:
            caption_ctx = f"Resultados consolidados do pós-hoc vs {t(BASELINE_TYPICAL_LABEL)} (por {analysis_label.lower()})."
        caption = format_caption(
            f"{caption_ctx} "
            f"Correção de Holm; $\\Delta_{{\\text{{rank}}}} < 0$ indica método melhor que controle."
        )
        tab_label = f"tab:posthoc_{analysis_type}_{_slugify(metric)}{suffix}"
        latex = df_posthoc.to_latex(
            index=False, escape=False,
            caption=caption,
            position='H',
            label=tab_label,
        )
        latex = _insert_group_separators(latex, df_posthoc_pre, df_posthoc, "category")
        # Converter para longtable (suporta quebra de página com cabeçalho repetido)
        latex = _tabular_to_longtable(
            latex, caption, tab_label,
            n_cols=len(df_posthoc.columns),
        )
        latex = _replace_decimal_in_latex(latex)
        path.write_text(latex, encoding="utf-8")
        saved_paths.append(path)

    return saved_paths


# =============================================================================
# TABELA DESCRITIVA (sem teste estatístico)
# =============================================================================

def save_descriptive_latex(
    stats_df: pd.DataFrame,
    metric: str,
    analysis_type: str,
    save_dir: str,
    translate: Dict = None,
    cols: List[str] = None,
    scope_suffix: str = None,
) -> List[Path]:
    """Gera tabela LaTeX descritiva (mediana + IQR) sem Friedman/Holm.

    Usa o mesmo stats_df produzido por ``run_all_*_analysis`` mas descarta
    colunas de teste (statistic, p_value, rank, z, p_holm, reject_holm).

    Se o DataFrame contiver a coluna ``direction`` (com valores U/P),
    as direções são mescladas em uma única tabela com a coluna "Sentido"
    (Protegido / Não protegido).
    """
    translate = translate or {}
    out_dir = Path(save_dir) / "stats_tests" / analysis_type
    out_dir.mkdir(parents=True, exist_ok=True)

    t = lambda x: translate.get(x, x)
    metric_label = t(metric)
    saved_paths: List[Path] = []

    suffix = f"_{scope_suffix}" if scope_suffix else ""

    # Usar linhas omnibus/friedman/wilcoxon (que contêm as descritivas)
    df_desc = stats_df[stats_df["test"].isin(["friedman", "wilcoxon"])].copy()
    if df_desc.empty:
        return saved_paths

    # Remover linha do baseline (Δfair=0 por definição, sem informação útil)
    for col in ["method", "category"]:
        if col in df_desc.columns:
            df_desc = df_desc[df_desc[col] != BASELINE_TYPICAL_LABEL].reset_index(drop=True)
            break

    # --- Tratamento da coluna de direção (Sentido) ---
    has_direction = (
        "direction" in df_desc.columns
        and df_desc["direction"].nunique() > 0
    )
    if has_direction:
        dir_map = {"P": "Privilegiado (P)", "U": "Não privilegiado (U)"}
        df_desc["direction"] = (
            df_desc["direction"].map(dir_map).fillna(df_desc["direction"])
        )
        # Garantir ordem: Privilegiado primeiro, Não privilegiado depois
        dir_order = {"Privilegiado (P)": 0, "Não privilegiado (U)": 1}
        df_desc["_dir_sort"] = df_desc["direction"].map(dir_order).fillna(2)
        df_desc = df_desc.sort_values("_dir_sort").drop(columns=["_dir_sort"])
        df_desc = df_desc.reset_index(drop=True)

    # Colunas descritivas padrão
    if cols is None:
        id_cols = (
            ["metric", "category"]
            if analysis_type == "category"
            else ["metric", "method", "category"]
        )
        cols = id_cols + ["n", "median_gap", "iqr_gap", "median_delta_fair", "iqr_delta_fair", "median_r_fair", "iqr_r_fair"]

    # Inserir direction após metric se disponível e não listada
    if has_direction and "direction" not in cols:
        idx = (cols.index("metric") + 1) if "metric" in cols else 0
        cols = cols[:idx] + ["direction"] + cols[idx:]

    # Selecionar apenas colunas existentes
    cols = [c for c in cols if c in df_desc.columns]
    df_desc = df_desc[cols].copy()

    # Traduzir colunas de texto
    text_cols = [c for c in ["metric", "method", "category"] if c in df_desc.columns]
    for col in text_cols:
        df_desc[col] = df_desc[col].apply(
            lambda x: _escape_latex_cell(t(x)) if isinstance(x, str) else x
        )

    # Formatar numéricos (vírgula como separador decimal)
    for col in ["median_gap", "iqr_gap", "median_delta_fair", "iqr_delta_fair",
                "median_r_fair", "iqr_r_fair"]:
        if col in df_desc.columns:
            df_desc[col] = df_desc[col].apply(
                lambda x: f"{x:.4f}".replace(".", ",") if pd.notna(x) else "—"
            )
    if "n" in df_desc.columns:
        df_desc["n"] = df_desc["n"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "—"
        )

    # Multirow categórico
    df_desc_pre = df_desc.copy()
    multirow_rules = [
        {"col": "metric", "context": []},
    ]
    if has_direction:
        multirow_rules.append({"col": "direction", "context": []})
    multirow_rules.append({"col": "category", "context": []})
    df_desc = _apply_multirow(df_desc, multirow_rules)

    # Renomear para PT
    df_desc = df_desc.rename(columns=_LATEX_COL_NAMES)

    path = out_dir / f"descritivo_{_slugify(metric)}{suffix}.tex"
    caption_dir = (
        " Sentido indica contra qual grupo o viés é direcionado."
        if has_direction else ""
    )
    analysis_label = t(analysis_type)
    if metric_label:
        desc_ctx = f"Resumo descritivo (por {analysis_label.lower()}) --- {metric_label}."
    else:
        desc_ctx = f"Resumo descritivo consolidado (por {analysis_label.lower()})."
    caption = format_caption(
        f"{desc_ctx} "
        f"$\\Delta_{{\\text{{fair}}}}$ = efeito de mitigação em relação ao baseline; "
        f"$r_{{\\text{{fair}}}}$ = mitigação relativa ($\\Delta_{{\\text{{fair}}}} / (d_{{\\text{{base}}}} + \\varepsilon)$)."
        f"{caption_dir}"
    )
    tab_label = f"tab:descritivo_{analysis_type}_{_slugify(metric)}{suffix}"
    latex = df_desc.to_latex(
        index=False, escape=False,
        caption=caption,
        position='H',
        label=tab_label,
    )
    # Separadores: para tabelas com direction, usar direction em category
    # tables (poucas linhas internas) e category em experiment tables
    # (separadores entre Baseline/Pre/In naturalmente incluem fronteira P/U)
    if has_direction and analysis_type == "category":
        group_col = "direction"
    else:
        group_col = "category"
    latex = _insert_group_separators(latex, df_desc_pre, df_desc, group_col)
    # Converter para longtable (suporta quebra de página com cabeçalho repetido)
    latex = _tabular_to_longtable(
        latex, caption, tab_label,
        n_cols=len(df_desc.columns),
    )
    latex = _replace_decimal_in_latex(latex)
    path.write_text(latex, encoding="utf-8")
    saved_paths.append(path)

    return saved_paths