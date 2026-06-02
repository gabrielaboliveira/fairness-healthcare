# config.py
"""
general configs
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
# Saída dos experimentos (CSVs por dataset) e das análises (figs/tables).
# Caminho absoluto e único — evita o aninhamento duplicado quando os
# notebooks rodam a partir da pasta experiments/.
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results"
ANALYSIS_PATH = RESULTS_PATH / "analysis"
SEED = 42


# General ML configurations

SPLIT_RATE = 0.30
TARGET_NAME = "target"
CROSS_VALIDATION_SPLITS = 5


"""
Dataset configurations

"""

DATASET_CONFIGS = {
    "LIVER": {
        "active": False,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{"gender": 1}],
            "unprivileged_groups": [{"gender": 0}],
        },
        "dataset_code": "3_liver",
        "weights": True,
    },
    "HEART": {
        "active": True,
        "sen_var": "sex",
        "fairness_vars": {
            "sen_var": "sex",
            "privileged_groups": [{"sex": 1}],
            "unprivileged_groups": [{"sex": 0}],
        },
        "dataset_code": "4_heart_disease",
        "weights": False,
    },
    "AIDS": {
        "active": True,
        "sen_var": "homo",
        "fairness_vars": {
            "sen_var": "homo",
            "privileged_groups": [{"homo": 0}],
            "unprivileged_groups": [{"homo": 1}],
        },
        "dataset_code": "6_aids",
        "weights": True,
    },
    "OBESITY": {
        "active": True,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{"gender": 1}],
            "unprivileged_groups": [{"gender": 0}],
        },
        "dataset_code": "7_obesity",
        "weights": True,
    },
    "ARRHYTHMIA": {
        "active": True,
        "sen_var": "sex",
        "fairness_vars": {
            "sen_var": "sex",
            "privileged_groups": [{"sex": 1}],
            "unprivileged_groups": [{"sex": 0}],
        },
        "dataset_code": "11_arrhythmia",
        "weights": False,
    },
    "MENTAL": {
        "active": True,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{"gender": 1}],
            "unprivileged_groups": [{"gender": 0}],
        },
        "dataset_code": "15_mental",
        "weights": False,
    },
    "DIABETES": {
        "active": True,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{"gender": 1}],
            "unprivileged_groups": [{"gender": 0}],
        },
        "dataset_code": "16_diabetes",
        "weights": False,
    },
}

# Labels descritivos para os valores numéricos dos grupos sensíveis
GROUP_LABELS = {
    "gender": {0: "Feminino", 1: "Masculino"},
    "sex":    {0: "Feminino", 1: "Masculino"},
    "homo":   {0: "Heterossexual", 1: "Homossexual"},
}

# Nomes de exibição dos datasets (title case, nomes completos)
DATASET_DISPLAY_NAMES = {
    "LIVER":      "Liver",
    "HEART":      "Heart Disease",
    "AIDS":       "Aids",
    "OBESITY":    "Obesity",
    "ARRHYTHMIA": "Arrhythmia",
    "MENTAL":     "Mental",
    "DIABETES":   "Diabetes",
}

# Metadados dos datasets para tabela descritiva
DATASET_META = {
    "LIVER": {
        "display_name": "Liver",
        "favorable_label": "Sem doença hepática",
        "cite_key": "liver_225",
    },
    "HEART": {
        "display_name": "Heart Disease",
        "favorable_label": "Sem doença cardíaca",
        "cite_key": "heart_disease_45",
    },
    "AIDS": {
        "display_name": "Aids",
        "favorable_label": "Sem óbito",
        "cite_key": "aids_dataset",
    },
    "OBESITY": {
        "display_name": "Obesity",
        "favorable_label": "Sem obesidade severa",
        "cite_key": "palechor2019obesity",
    },
    "ARRHYTHMIA": {
        "display_name": "Arrhythmia",
        "favorable_label": "Sem arritmia",
        "cite_key": "arrhythmia_5",
    },
    "MENTAL": {
        "display_name": "Mental",
        "favorable_label": "Sem tratamento",
        "cite_key": "osmi2014survey",
    },
    "DIABETES": {
        "display_name": "Diabetes",
        "favorable_label": "Sem diabetes",
        "cite_key": "Islam2019LikelihoodPO",
    },
}


"""
Analysis configurations

"""

IDEAL_VALUES = {
    # fairness
    "between_groups_cv": 0.0,  # menor variação entre grupos
    "disparate_impact": 1.0,  # razão ideal = 1 (aceitável 0.8–1.25)
    "error_rate_ratio": 1.0,  # mesma taxa de erro entre grupos
    "error_rate_difference": 0.0,  # diferença de erros = 0
    "statistical_parity_difference": 0.0,  # sem diferença de seleção
    "equal_opportunity_difference": 0.0,  # mesma taxa de verdadeiros positivos
    "average_odds_difference": 0.0,  # mesma taxa de acertos/erros
    "bias_amplification": 0.0,  # modelo não amplia viés
    # 'selection_rate': 1.0,                     # ideal seria igual entre grupos***
    "consistency": 1.0,  # previsões iguais p/ vizinhos iguais
    "false_negative_rate_difference": 0.0,
    "coef_variation": 0.0,
    "false_discovery_rate_difference": 0.0,
    "generalized_entropy_index": 0.0,
    # performance classification
    "accuracy": 1.0,
    "f1": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    # tradeoff
    "fairness_performance_ratio": 0,
}

METRIC_MAP = {
    "f1": "performance",
    "accuracy": "performance",
    "precision": "performance",
    "recall": "performance",
    "train_time": "processing_performance",
    "predict_time": "processing_performance",
    "n3_error": "complexity",
    "si_separability": "complexity",
    "f2_overlap": "complexity",
    "f4_collective_feature_efficiency": "complexity",
}

METRIC_GROUP_MAP = {
    # performance
    "f1": "performance",
    "accuracy": "performance",
    "precision": "performance",
    "recall": "performance",
    # performance de processamento
    "train_time": "processing_performance",
    "predict_time": "processing_performance",
    # complexidade / separabilidade
    "n3_error": "complexity",
    "si_separability": "complexity",
    "f2_overlap": "complexity",
    "f4_collective_feature_efficiency": "complexity",
    # fairness - ratio/difference/etc.
    "disparate_impact": "ratio_based",
    "error_rate_ratio": "ratio_based",
    "selection_rate_ratio": "ratio_based",
    "consistency": "proportion_based",
    "error_rate_difference": "difference_based",
    "false_negative_rate_difference": "difference_based",
    "false_positive_rate_difference": "difference_based",
    "false_discovery_rate_difference": "difference_based",
    "statistical_parity_difference": "difference_based",
    "equal_opportunity_difference": "difference_based",
    "average_odds_difference": "difference_based",
    "bias_amplification": "logarithmic_difference_based",
    "between_groups_cv": "dispersion_based",
    "generalized_entropy_index": "dispersion_based",
    "coef_variation": "dispersion_based_global",  # desvio padrão/ média - medida de variabilidade global, porem com faixa muito distinta
    # outros
    "selection_rate": "others",
}

# Translate for pt-br
TRANSLATE_PT = {
    # Métricas de fairness
    "disparate_impact": "Impacto Díspar",
    "statistical_parity_difference": "Diferença de Paridade Estatística",
    "consistency": "Consistência",
    "between_groups_cv": "Coeficiente de Variação Entre Grupos",
    "bias_amplification": "Amplificação de viés",
    "false_negative_rate_difference": "Diferença de Taxa de Falsos Negativos",
    "false_discovery_rate_difference": "Diferença de Taxa de Falsa Descoberta",
    "error_rate_difference": "Diferença de Taxa de Erro",
    "fairness_performance_ratio": "Razão Equidade-Desempenho",
    "generalized_entropy_index": "Índice de Entropia Generalizada",
    # Datasets (mantidos em inglês como "código")
    "11_arrhythmia": "Arrhythmia",
    "16_diabetes": "Diabetes",
    "15_mental": "Mental",
    "3_liver": "Liver",
    "4_heart_disease": "Heart Disease",
    "6_aids": "Aids",
    "7_obesity": "Obesity",
    # Termos gerais
    "baseline": "Baseline",
    "experiment": "Experimento",
    "category": "Categoria",
    "in_processing": "In-Processing",
    "pre_processing": "Pre-Processing",
    # Métricas de desempenho / ML
    "recall": "Revocação",
    "method": "Método",
    "baseline_typical": "Baseline",
    "f1": "F1-Score",
    "accuracy":"Acurácia",
    "precision":"Precisão"
}


TRANSLATE = TRANSLATE_PT

CATEGORY_MAP = {
    "pre_processing": ["EDL","ERL"],
    "in_processing": ["IAD", "IGLA", "IGLU", "IFG", "IP", "IW"],
    "baseline": ["BLRA", "BLRU", "BRFA", "BRFU"],
}
