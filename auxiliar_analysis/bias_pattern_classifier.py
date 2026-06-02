"""
bias_pattern_classifier.py - Classificação de datasets em padrões de viés

METODOLOGIA:
- Unidade de análise: DATASET (baseline dataset-level, 1 linha por dataset×metric)
- Distância ao ideal: aditiva |x - ideal| ou multiplicativa max(x/ideal, ideal/x) - 1
- Escala S = IQR entre datasets da distância d
- Unidades padronizadas: u = d / S
- Elegibilidade: u >= 0.5
- Severidade: 3 (baixo) / 2 (médio) / 1 (alto)

CATEGORIAS DE PADRÃO:
- A (paridade estatística): disparate_impact, bias_amplification
- B (fairness individual / dispersão): consistency, generalized_entropy_index
- C (matriz de confusão): false_negative_rate_difference, false_discovery_rate_difference, error_rate_difference

CÓDIGOS DE PADRÃO:
- Formato: <categoria><nível><direção_opcional>
- Exemplos: A1U, A2P, B1, C3U

HERANÇA DE SEVERIDADE E DIREÇÃO (separadas):
- Severidade final: vem da métrica MAIS SEVERA da categoria (menor nível numérico),
  independente de ter direção ou não.
- Direção final: vem da métrica MAIS SEVERA ENTRE AS QUE POSSUEM DIREÇÃO (own_direction ∈ {"U","P"}).
- Colunas de rastreabilidade: inherited_from_severity, inherited_from_direction
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, Optional, Literal
import warnings


# =============================================================================
# CONFIGURAÇÃO PADRÃO
# =============================================================================

DEFAULT_IDEAL_VALUES = {
    "disparate_impact": 1.0,
    "bias_amplification": 0.0,
    "consistency": 1.0,
    "generalized_entropy_index": 0.0,
    "false_negative_rate_difference": 0.0,
    "false_discovery_rate_difference": 0.0,
    "error_rate_difference": 0.0,
}

DEFAULT_METRIC_TO_CATEGORY = {
    "disparate_impact": "A",
    "bias_amplification": "A",
    "consistency": "B",
    "generalized_entropy_index": "B",
    "false_negative_rate_difference": "C",
    "false_discovery_rate_difference": "C",
    "error_rate_difference": "C",
}

# Métricas que usam distância multiplicativa por padrão
DEFAULT_RATIO_METRICS = {"disparate_impact"}

# Métricas com direção de viés
# NOTA: bias_amplification e generalized_entropy_index NÃO possuem sentido de viés
METRICS_WITH_DIRECTION = {
    "disparate_impact",
    "false_negative_rate_difference",
    "false_discovery_rate_difference",
    "error_rate_difference",
}


# =============================================================================
# FUNÇÕES DE DISTÂNCIA
# =============================================================================

def compute_additive_distance(value: float, ideal: float) -> float:
    """
    Distância aditiva: |x - ideal|
    
    Usada para métricas de diferença (FNR diff, bias amplification, etc.)
    """
    if pd.isna(value) or pd.isna(ideal):
        return np.nan
    return abs(value - ideal)


def compute_multiplicative_distance(value: float, ideal: float) -> float:
    """
    Distância multiplicativa: max(x/ideal, ideal/x) - 1
    
    Usada para métricas de razão/proporção (DI, consistency).
    Trata simetricamente desvios acima e abaixo do ideal.
    
    Exemplos (ideal=1):
    - value=0.8 → max(0.8, 1.25) - 1 = 0.25
    - value=1.25 → max(1.25, 0.8) - 1 = 0.25
    - value=0.5 → max(0.5, 2.0) - 1 = 1.0
    - value=2.0 → max(2.0, 0.5) - 1 = 1.0
    """
    if pd.isna(value) or pd.isna(ideal):
        return np.nan
    
    # Tratar casos especiais
    if ideal == 0:
        # Se ideal é 0, usar distância aditiva
        return abs(value)
    
    if value <= 0:
        # Para valores não-positivos com ideal positivo, usar fallback
        # (consistência e DI devem ser > 0 em teoria)
        warnings.warn(
            f"Valor não-positivo ({value}) para métrica com ideal={ideal}. "
            "Usando distância aditiva como fallback.",
            RuntimeWarning
        )
        return abs(value - ideal)
    
    ratio_above = value / ideal
    ratio_below = ideal / value
    return max(ratio_above, ratio_below) - 1


def compute_distance(
    value: float,
    ideal: float,
    use_multiplicative: bool = False
) -> float:
    """Calcula distância ao ideal (aditiva ou multiplicativa)."""
    if use_multiplicative:
        return compute_multiplicative_distance(value, ideal)
    return compute_additive_distance(value, ideal)


# =============================================================================
# FUNÇÕES DE DIREÇÃO
# =============================================================================

def get_bias_direction(
    value: float,
    ideal: float,
    metric: str
) -> Optional[str]:
    """
    Determina direção do viés.
    
    Returns:
        "U" = contra não-privilegiado (unprivileged)
        "P" = contra privilegiado
        None = sem direção ou valor igual ao ideal
    """
    if pd.isna(value) or pd.isna(ideal):
        return None
    
    if metric not in METRICS_WITH_DIRECTION:
        return None
    
    # Tolerância para considerar "igual ao ideal"
    tol = 1e-9
    
    if metric == "disparate_impact":
        # DI < 1 → desfavorece não-privilegiado
        # DI > 1 → desfavorece privilegiado
        if value < ideal - tol:
            return "U"
        elif value > ideal + tol:
            return "P"
        return None
    
    elif metric in ("false_negative_rate_difference",
                     "false_discovery_rate_difference",
                     "error_rate_difference"):
        # diff = rate_unpriv - rate_priv
        # Se > 0: não-privilegiados têm MAIOR taxa (pior) → viés contra não-privilegiados
        # Se < 0: não-privilegiados têm MENOR taxa (melhor) → viés contra privilegiados
        if value > tol:
            return "U"
        elif value < -tol:
            return "P"
        return None
    
    return None


# =============================================================================
# CLASSIFICAÇÃO PRINCIPAL
# =============================================================================

def classify_bias_patterns(
    baseline_ds: pd.DataFrame,
    ideal_values: Dict[str, float] = None,
    metric_to_category: Dict[str, str] = None,
    ratio_metrics: Set[str] = None,
    min_units: float = 0.5,
    medium_units: float = 1.0,
    high_units: float = 1.5,
    value_col: str = "value_num",
    return_scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Classifica datasets em padrões de viés e severidade.
    
    Metodologia:
    1. Calcula distância ao ideal d(x, ideal) para cada observação
    2. Calcula S = IQR entre datasets da distância d (por métrica)
    3. Calcula u = d / S (unidades de IQR)
    4. Classifica elegibilidade e severidade (own_*)
    5. Para cada (dataset, categoria): 
       - Severidade herdada da métrica mais severa (qualquer métrica)
       - Direção herdada da métrica mais severa COM direção definida
    
    Args:
        baseline_ds: DataFrame com colunas [dataset, metric, value_num]
                     Deve ser dataset-level (1 linha por dataset×metric).
                     IMPORTANTE: deve conter APENAS dados do baseline (pré-mitigação).
                     Se dados de métodos de mitigação forem incluídos, o IQR será
                     contaminado e as faixas de severidade perderão a interpretação
                     de "estado pré-tratamento".
        ideal_values: Dict metric -> valor ideal
        metric_to_category: Dict metric -> categoria (A/B/C)
        ratio_metrics: Set de métricas que usam distância multiplicativa
        min_units: Limiar mínimo de unidades para elegibilidade (default: 0.5)
        medium_units: Limiar para severidade média (default: 1.0)
        high_units: Limiar para severidade alta (default: 1.5)
        value_col: Nome da coluna de valores
        return_scale: Se True, retorna também tabela de escala por métrica
    
    Returns:
        DataFrame com colunas:
            - dataset: identificador do dataset
            - metric: nome da métrica
            - value: valor original
            - ideal: valor ideal
            - distance: distância ao ideal
            - distance_type: "additive" ou "multiplicative"
            - units: unidades de IQR (d/S)
            
            Classificação PRÓPRIA da métrica (para rastreabilidade):
            - own_eligible: bool, se a métrica por si só é elegível
            - own_severity_level: 0/1/2/3 original da métrica
            - own_severity_label: "none"/"high"/"medium"/"low" original
            - own_direction: "U"/"P"/None original
            - own_pattern_code: código original da métrica
            
            Classificação HERDADA (final, por categoria):
            - eligible: bool, se o dataset é elegível naquela categoria
            - severity_level: severidade herdada (da métrica mais severa)
            - severity_label: label herdado
            - direction: direção herdada (da métrica direcional mais severa)
            - category: "A"/"B"/"C"
            - pattern_code: código final herdado
            - inherited_from_severity: métrica que definiu a severidade
            - inherited_from_direction: métrica que definiu a direção (pode ser None)
        
        Se return_scale=True, também retorna DataFrame metric_scale:
            - metric: nome da métrica
            - n_datasets: número de datasets
            - iqr: IQR das distâncias (escala S)
            - median_distance: mediana das distâncias
            - q25, q75: quartis
    """
    # Defaults
    ideal_values = ideal_values or DEFAULT_IDEAL_VALUES
    metric_to_category = metric_to_category or DEFAULT_METRIC_TO_CATEGORY
    ratio_metrics = ratio_metrics if ratio_metrics is not None else DEFAULT_RATIO_METRICS
    
    # Validar entrada
    required_cols = ["dataset", "metric", value_col]
    missing_cols = [c for c in required_cols if c not in baseline_ds.columns]
    if missing_cols:
        raise ValueError(f"Colunas faltando: {missing_cols}")
    
    # Verificar se o input parece conter apenas baseline
    if "exp_id" in baseline_ds.columns:
        known_baseline_ids = {"BLRA", "BLRU", "BRFA", "BRFU"}
        unique_exp_ids = set(baseline_ds["exp_id"].unique())
        non_baseline = unique_exp_ids - known_baseline_ids
        if non_baseline:
            warnings.warn(
                f"[classify_bias_patterns] O DataFrame contém exp_ids que não são "
                f"baseline: {non_baseline}. O IQR de severidade deve ser calculado "
                f"APENAS sobre dados de baseline (pré-mitigação). Incluir métodos de "
                f"mitigação contamina a escala.",
                RuntimeWarning,
            )

    # Filtrar apenas métricas conhecidas
    known_metrics = set(ideal_values.keys())
    df = baseline_ds[baseline_ds["metric"].isin(known_metrics)].copy()
    
    if df.empty:
        warnings.warn(
            f"Nenhuma métrica conhecida encontrada. "
            f"Métricas esperadas: {known_metrics}",
            UserWarning
        )
        empty_result = pd.DataFrame(columns=[
            "dataset", "metric", "value", "ideal", "distance", "distance_type",
            "units", "own_eligible", "own_severity_level", "own_severity_label",
            "own_direction", "own_pattern_code", "eligible", "severity_level",
            "severity_label", "direction", "category", "pattern_code", 
            "inherited_from_severity", "inherited_from_direction"
        ])
        if return_scale:
            empty_scale = pd.DataFrame(columns=[
                "metric", "n_datasets", "iqr", "median_distance", "q25", "q75"
            ])
            return empty_result, empty_scale
        return empty_result
    
    # Renomear coluna de valor
    df = df.rename(columns={value_col: "value"})
    
    # Adicionar ideal e categoria
    df["ideal"] = df["metric"].map(ideal_values)
    df["category"] = df["metric"].map(metric_to_category)
    
    # Calcular distância ao ideal
    df["distance_type"] = df["metric"].apply(
        lambda m: "multiplicative" if m in ratio_metrics else "additive"
    )
    
    df["distance"] = df.apply(
        lambda row: compute_distance(
            row["value"],
            row["ideal"],
            use_multiplicative=(row["metric"] in ratio_metrics)
        ),
        axis=1
    )
    
    # Calcular escala S (IQR) por métrica
    metric_scale = (
        df.groupby("metric", as_index=False)
        .agg(
            n_datasets=("distance", "count"),
            iqr=("distance", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            median_distance=("distance", "median"),
            q25=("distance", lambda x: x.quantile(0.25)),
            q75=("distance", lambda x: x.quantile(0.75)),
        )
    )
    
    # Mapear IQR para cada linha
    iqr_map = metric_scale.set_index("metric")["iqr"].to_dict()
    df["scale_iqr"] = df["metric"].map(iqr_map)
    
    # Calcular unidades de IQR
    def compute_units(row):
        d = row["distance"]
        s = row["scale_iqr"]
        if pd.isna(d) or pd.isna(s) or s == 0:
            return np.nan
        return d / s
    
    df["units"] = df.apply(compute_units, axis=1)
    
    # ==========================================================================
    # CLASSIFICAÇÃO PRÓPRIA (own_*)
    # ==========================================================================
    
    # Elegibilidade própria
    df["own_eligible"] = df["units"] >= min_units
    df.loc[df["units"].isna(), "own_eligible"] = False
    
    # Severidade própria
    def classify_own_severity(row):
        if not row["own_eligible"] or pd.isna(row["units"]):
            return 0, "none"
        u = row["units"]
        if u >= high_units:
            return 1, "high"
        elif u >= medium_units:
            return 2, "medium"
        else:  # u >= min_units
            return 3, "low"
    
    severity_results = df.apply(classify_own_severity, axis=1, result_type="expand")
    df["own_severity_level"] = severity_results[0]
    df["own_severity_label"] = severity_results[1]
    
    # Direção própria
    df["own_direction"] = df.apply(
        lambda row: get_bias_direction(row["value"], row["ideal"], row["metric"]),
        axis=1
    )
    
    # Código de padrão próprio
    def generate_own_pattern_code(row):
        if not row["own_eligible"] or row["own_severity_level"] == 0:
            return None
        
        cat = row["category"]
        level = row["own_severity_level"]
        direction = row["own_direction"]
        
        if direction:
            return f"{cat}{level}{direction}"
        return f"{cat}{level}"
    
    df["own_pattern_code"] = df.apply(generate_own_pattern_code, axis=1)
    
    # ==========================================================================
    # HERANÇA SEPARADA: SEVERIDADE E DIREÇÃO
    # ==========================================================================
    
    def get_category_inheritance(group):
        """
        Retorna herança separada para severidade e direção.
        
        - Severidade: métrica mais severa (elegível), independente de direção
        - Direção: métrica mais severa ENTRE as que possuem direção
        """
        # Métricas elegíveis (para severidade)
        eligible = group[group["own_eligible"] & (group["own_severity_level"] > 0)]
        
        if eligible.empty:
            return pd.Series({
                "winner_severity_metric": None,
                "winner_severity_level": 0,
                "winner_severity_label": "none",
                "winner_direction_metric": None,
                "winner_direction": None,
                "winner_eligible": False
            })
        
        # --- SEVERIDADE: métrica mais severa (qualquer uma) ---
        eligible_sorted = eligible.sort_values(
            by=["own_severity_level", "metric"],
            ascending=[True, True]
        )
        severity_winner = eligible_sorted.iloc[0]
        
        # --- DIREÇÃO: métrica mais severa COM direção definida ---
        with_direction = eligible[eligible["own_direction"].isin(["U", "P"])]
        
        if with_direction.empty:
            # Nenhuma métrica tem direção
            direction_winner_metric = None
            direction_winner_value = None
        else:
            direction_sorted = with_direction.sort_values(
                by=["own_severity_level", "metric"],
                ascending=[True, True]
            )
            direction_winner = direction_sorted.iloc[0]
            direction_winner_metric = direction_winner["metric"]
            direction_winner_value = direction_winner["own_direction"]
        
        return pd.Series({
            "winner_severity_metric": severity_winner["metric"],
            "winner_severity_level": severity_winner["own_severity_level"],
            "winner_severity_label": severity_winner["own_severity_label"],
            "winner_direction_metric": direction_winner_metric,
            "winner_direction": direction_winner_value,
            "winner_eligible": True
        })
    
    # Calcular herança por (dataset, categoria)
    winners = (
        df.groupby(["dataset", "category"], as_index=False)
        .apply(get_category_inheritance, include_groups=False)
    )
    
    # Merge de volta ao df principal
    df = df.merge(winners, on=["dataset", "category"], how="left")
    
    # Preencher colunas herdadas
    df["eligible"] = df["winner_eligible"].fillna(False)
    df["severity_level"] = df["winner_severity_level"].fillna(0).astype(int)
    df["severity_label"] = df["winner_severity_label"].fillna("none")
    df["direction"] = df["winner_direction"]
    df["inherited_from_severity"] = df["winner_severity_metric"]
    df["inherited_from_direction"] = df["winner_direction_metric"]
    
    # Gerar código de padrão herdado
    def generate_inherited_pattern_code(row):
        if not row["eligible"] or row["severity_level"] == 0:
            return None
        
        cat = row["category"]
        level = row["severity_level"]
        direction = row["direction"]
        
        if direction:
            return f"{cat}{level}{direction}"
        return f"{cat}{level}"
    
    df["pattern_code"] = df.apply(generate_inherited_pattern_code, axis=1)
    
    # Selecionar e ordenar colunas finais
    result_cols = [
        "dataset", "metric", "value", "ideal", "distance", "distance_type",
        "scale_iqr", "units",
        # Classificação própria (rastreabilidade)
        "own_eligible", "own_severity_level", "own_severity_label",
        "own_direction", "own_pattern_code",
        # Classificação herdada (final)
        "eligible", "severity_level", "severity_label",
        "direction", "category", "pattern_code",
        "inherited_from_severity", "inherited_from_direction"
    ]
    result = df[result_cols].copy()
    
    if return_scale:
        return result, metric_scale
    return result


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def summarize_patterns(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume a distribuição de padrões (usando pattern_code herdado).
    
    Returns:
        DataFrame com contagem de datasets por pattern_code
    """
    # Filtrar apenas elegíveis e pegar uma linha por (dataset, category)
    # para evitar contar o mesmo dataset múltiplas vezes por categoria
    eligible = classified_df[classified_df["eligible"]].copy()
    
    if eligible.empty:
        return pd.DataFrame(columns=["pattern_code", "n_datasets", "datasets"])
    
    # Deduplica por (dataset, category) - todas as métricas da mesma categoria
    # têm o mesmo pattern_code herdado
    unique_patterns = eligible.drop_duplicates(subset=["dataset", "category"])
    
    summary = (
        unique_patterns.groupby("pattern_code", as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            datasets=("dataset", lambda x: sorted(list(x.unique())))
        )
        .sort_values("pattern_code")
    )
    
    return summary


def summarize_own_patterns(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume a distribuição de padrões PRÓPRIOS (own_pattern_code).
    Útil para ver a classificação original de cada métrica antes da herança.
    """
    eligible = classified_df[classified_df["own_eligible"]].copy()
    
    if eligible.empty:
        return pd.DataFrame(columns=["metric", "own_pattern_code", "n_datasets", "datasets"])
    
    summary = (
        eligible.groupby(["metric", "own_pattern_code"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            datasets=("dataset", lambda x: sorted(list(x.unique())))
        )
        .sort_values(["metric", "own_pattern_code"])
    )
    
    return summary


def get_datasets_by_pattern(
    classified_df: pd.DataFrame,
    pattern_code: str,
    use_inherited: bool = True
) -> list:
    """
    Retorna lista de datasets para um padrão específico.
    
    Args:
        classified_df: Output de classify_bias_patterns
        pattern_code: Código do padrão (ex: "A1U")
        use_inherited: Se True, usa pattern_code herdado; se False, usa own_pattern_code
    """
    col = "pattern_code" if use_inherited else "own_pattern_code"
    mask = classified_df[col] == pattern_code
    return sorted(classified_df.loc[mask, "dataset"].unique().tolist())


def get_eligible_datasets(
    classified_df: pd.DataFrame,
    metric: str = None,
    category: str = None,
    min_severity: int = None,
    use_inherited: bool = True
) -> list:
    """
    Retorna datasets elegíveis com filtros opcionais.
    
    Args:
        classified_df: Output de classify_bias_patterns
        metric: Filtrar por métrica específica
        category: Filtrar por categoria (A/B/C)
        min_severity: Severidade mínima (1=alta, 2=média, 3=baixa)
        use_inherited: Se True, usa classificação herdada; se False, usa própria
    """
    eligible_col = "eligible" if use_inherited else "own_eligible"
    severity_col = "severity_level" if use_inherited else "own_severity_level"
    
    mask = classified_df[eligible_col]
    
    if metric:
        mask &= classified_df["metric"] == metric
    
    if category:
        mask &= classified_df["category"] == category
    
    if min_severity:
        # severity_level: 1=alto, 2=médio, 3=baixo
        # min_severity=2 significa "pelo menos média" → levels 1 ou 2
        mask &= classified_df[severity_col] <= min_severity
        mask &= classified_df[severity_col] > 0
    
    return sorted(classified_df.loc[mask, "dataset"].unique().tolist())


def filter_by_pattern(
    df: pd.DataFrame,
    classified_df: pd.DataFrame,
    pattern_codes: list,
    dataset_col: str = "dataset",
    use_inherited: bool = True
) -> pd.DataFrame:
    """
    Filtra um DataFrame para incluir apenas datasets de padrões específicos.
    
    Args:
        df: DataFrame a filtrar
        classified_df: Output de classify_bias_patterns
        pattern_codes: Lista de códigos de padrão (ex: ["A1U", "A2P"])
        dataset_col: Nome da coluna de dataset
        use_inherited: Se True, usa pattern_code herdado
    """
    datasets = set()
    for code in pattern_codes:
        datasets.update(get_datasets_by_pattern(classified_df, code, use_inherited))
    
    return df[df[dataset_col].isin(datasets)].copy()


def print_classification_report(
    classified_df: pd.DataFrame,
    metric_scale: pd.DataFrame = None,
    show_own_patterns: bool = True
) -> None:
    """Imprime relatório de classificação."""
    print("=" * 70)
    print("RELATÓRIO DE CLASSIFICAÇÃO DE PADRÕES DE VIÉS")
    print("=" * 70)
    
    # Resumo por métrica (classificação PRÓPRIA)
    print("\n📊 CLASSIFICAÇÃO PRÓPRIA POR MÉTRICA")
    print("-" * 70)
    
    for metric in sorted(classified_df["metric"].unique()):
        m_data = classified_df[classified_df["metric"] == metric]
        n_total = len(m_data)
        n_eligible = m_data["own_eligible"].sum()
        
        print(f"\n{metric}:")
        print(f"  Total datasets: {n_total}")
        print(f"  Elegíveis (próprio): {n_eligible} ({n_eligible/n_total*100:.1f}%)")
        
        if n_eligible > 0:
            eligible_data = m_data[m_data["own_eligible"]]
            severity_dist = eligible_data["own_severity_level"].value_counts().sort_index()
            severity_labels = {1: "alto", 2: "médio", 3: "baixo"}
            
            print("  Distribuição de severidade (própria):")
            for level, count in severity_dist.items():
                if level > 0:
                    print(f"    - Nível {level} ({severity_labels.get(level, '?')}): {count}")
            
            # Direção (apenas para métricas com direção)
            if metric in METRICS_WITH_DIRECTION:
                dir_dist = eligible_data["own_direction"].value_counts()
                print("  Direção do viés (própria):")
                for direction, count in dir_dist.items():
                    if direction:
                        dir_label = "contra não-privilegiado" if direction == "U" else "contra privilegiado"
                        print(f"    - {direction} ({dir_label}): {count}")
            else:
                print("  (Métrica sem direção de viés)")
    
    # Escala
    if metric_scale is not None:
        print("\n📏 ESCALA (IQR) POR MÉTRICA")
        print("-" * 70)
        print(metric_scale.to_string(index=False))
    
    # Padrões próprios (opcional)
    if show_own_patterns:
        print("\n🏷️ PADRÕES PRÓPRIOS (antes da herança)")
        print("-" * 70)
        own_summary = summarize_own_patterns(classified_df)
        if own_summary.empty:
            print("Nenhum padrão próprio encontrado")
        else:
            for _, row in own_summary.iterrows():
                print(f"\n{row['metric']} → {row['own_pattern_code']}: {row['n_datasets']} dataset(s)")
                ds_list = row['datasets'][:5]
                print(f"  Datasets: {', '.join(ds_list)}", end="")
                if len(row['datasets']) > 5:
                    print(f" ... (+{len(row['datasets'])-5} mais)")
                else:
                    print()
    
    # Padrões herdados (finais)
    print("\n🎯 PADRÕES FINAIS (após herança por categoria)")
    print("-" * 70)
    
    summary = summarize_patterns(classified_df)
    if summary.empty:
        print("Nenhum padrão encontrado (nenhum dataset elegível)")
    else:
        for _, row in summary.iterrows():
            print(f"\n{row['pattern_code']}: {row['n_datasets']} dataset(s)")
            ds_list = row['datasets'][:5]
            print(f"  Datasets: {', '.join(ds_list)}", end="")
            if len(row['datasets']) > 5:
                print(f" ... (+{len(row['datasets'])-5} mais)")
            else:
                print()
    
    # Mostrar herança detalhada
    print("\n🔗 DETALHES DA HERANÇA (severidade e direção separadas)")
    print("-" * 70)
    
    # Agrupar por (dataset, category) para mostrar herança
    eligible_df = classified_df[classified_df["eligible"]]
    unique_combos = eligible_df.drop_duplicates(subset=["dataset", "category"])
    
    has_inheritance_info = False
    for _, row in unique_combos.iterrows():
        sev_from = row["inherited_from_severity"]
        dir_from = row["inherited_from_direction"]
        
        # Verificar se há métricas diferentes definindo severidade e direção
        if sev_from != dir_from and dir_from is not None:
            has_inheritance_info = True
            print(f"\n{row['dataset']} / Categoria {row['category']}:")
            print(f"  Severidade: nível {row['severity_level']} (de {sev_from})")
            print(f"  Direção: {row['direction']} (de {dir_from})")
            print(f"  → pattern_code final: {row['pattern_code']}")
        elif dir_from is None and row["severity_level"] > 0:
            has_inheritance_info = True
            print(f"\n{row['dataset']} / Categoria {row['category']}:")
            print(f"  Severidade: nível {row['severity_level']} (de {sev_from})")
            print(f"  Direção: sem direção (nenhuma métrica direcional elegível)")
            print(f"  → pattern_code final: {row['pattern_code']}")
    
    if not has_inheritance_info:
        print("Todas as categorias têm severidade e direção da mesma métrica (ou só têm uma métrica)")
    
    print("\n" + "=" * 70)


# =============================================================================
# INTEGRAÇÃO COM AGREGAÇÃO DATASET-LEVEL
# =============================================================================

def prepare_baseline_dataset_level(
    df: pd.DataFrame,
    baseline_ids: list = None,
    value_col: str = "value_num"
) -> pd.DataFrame:
    """
    Prepara dados baseline agregados ao nível de dataset.
    
    A partir de dados brutos (com folds×repetições), calcula a mediana
    por dataset×metric para os experimentos baseline.
    
    Args:
        df: DataFrame com dados brutos
        baseline_ids: Lista de exp_ids de baseline
        value_col: Coluna de valores
    
    Returns:
        DataFrame com colunas [dataset, metric, value_num]
    """
    if baseline_ids is None:
        baseline_ids = ["BLRA", "BLRU", "BRFA", "BRFU"]
    
    # Filtrar baselines
    df_baseline = df[df["exp_id"].isin(baseline_ids)].copy()
    
    if df_baseline.empty:
        warnings.warn("Nenhum dado de baseline encontrado", UserWarning)
        return pd.DataFrame(columns=["dataset", "metric", value_col])
    
    # Agregar: mediana por dataset×metric (sobre folds, repetições E baselines)
    baseline_ds = (
        df_baseline.groupby(["dataset", "metric"], as_index=False)
        .agg({value_col: "median"})
    )
    
    return baseline_ds


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

def example_usage():
    """
    Exemplo de uso da função de classificação.
    """
    # Criar dados de exemplo (baseline dataset-level)
    np.random.seed(42)
    
    datasets = ["heart_disease", "arrhythmia", "mental", "diabetes", "aids", "obesity"]
    metrics_config = {
        "disparate_impact": {"ideal": 1.0, "base": 0.7, "std": 0.25},
        "bias_amplification": {"ideal": 0.0, "base": 0.1, "std": 0.15},
        "consistency": {"ideal": 1.0, "base": 0.75, "std": 0.15},
        "false_negative_rate_difference": {"ideal": 0.0, "base": 0.05, "std": 0.12},
    }
    
    rows = []
    for ds in datasets:
        for metric, cfg in metrics_config.items():
            # Gerar valor com variação
            value = cfg["base"] + np.random.normal(0, cfg["std"])
            # Clamp para valores válidos
            if metric == "disparate_impact":
                value = max(0.1, min(2.0, value))
            elif metric == "consistency":
                value = max(0.1, min(1.0, value))
            
            rows.append({"dataset": ds, "metric": metric, "value_num": value})
    
    baseline_ds = pd.DataFrame(rows)
    
    print("DADOS DE ENTRADA (baseline dataset-level):")
    print(baseline_ds.pivot(index="dataset", columns="metric", values="value_num").round(3))
    print()
    
    # Classificar
    classified, metric_scale = classify_bias_patterns(
        baseline_ds,
        ratio_metrics={"disparate_impact"},  # Apenas DI usa distância multiplicativa
        return_scale=True
    )
    
    # Relatório
    print_classification_report(classified, metric_scale)
    
    # Mostrar resultado detalhado
    print("\nRESULTADO DETALHADO (com herança separada):")
    cols_display = [
        "dataset", "metric", 
        "own_severity_level", "own_direction", "own_pattern_code",
        "severity_level", "direction", "pattern_code", 
        "inherited_from_severity", "inherited_from_direction"
    ]
    print(classified[cols_display].to_string(index=False))
    
    # Exemplo específico: grupo A onde BA define severidade mas DI define direção
    print("\n\n📌 EXEMPLO: Categoria A")
    print("   - bias_amplification: pode definir severidade (não tem direção)")
    print("   - disparate_impact: define direção U/P")
    
    cat_a = classified[classified["category"] == "A"].drop_duplicates(subset=["dataset", "category"])
    print("\n   Resultados:")
    for _, row in cat_a.iterrows():
        print(f"   {row['dataset']}: "
              f"sev={row['severity_level']} (de {row['inherited_from_severity']}), "
              f"dir={row['direction']} (de {row['inherited_from_direction']}) "
              f"→ {row['pattern_code']}")
    
    return classified, metric_scale


def check_direction_discordance(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica se há datasets com own_direction discordante entre métricas
    direcionais da mesma categoria.

    Retorna DataFrame com (dataset, category, metric, own_direction) para
    os casos em que duas métricas elegíveis da mesma (dataset, category)
    apontam direções opostas (uma "U" e outra "P").
    Se não houver discordância, retorna DataFrame vazio.
    """
    directional = classified_df[
        (classified_df["own_direction"].isin(["U", "P"]))
        & (classified_df["own_eligible"] == True)
    ].copy()

    if directional.empty:
        return pd.DataFrame(
            columns=["dataset", "category", "metric", "own_direction"]
        )

    n_directions = (
        directional
        .groupby(["dataset", "category"])["own_direction"]
        .nunique()
    )
    conflicting = n_directions[n_directions > 1].reset_index()

    if conflicting.empty:
        return pd.DataFrame(
            columns=["dataset", "category", "metric", "own_direction"]
        )

    keys = conflicting[["dataset", "category"]]
    detail = directional.merge(keys, on=["dataset", "category"], how="inner")

    return (
        detail[["dataset", "category", "metric", "own_direction"]]
        .sort_values(["dataset", "category", "metric"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    example_usage()