# config.py
'''
general configs
'''
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH    = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"
SEED         = 42


# General ML configurations

SPLIT_RATE = 0.30
TARGET_NAME = "target"
CROSS_VALIDATION_SPLITS = 5


'''
Dataset configurations

'''

DATASET_CONFIGS = {
    "LIVER": {
        "active": True,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{'gender': 1}],
            "unprivileged_groups": [{'gender': 0}]
        },
        "dataset_code": "3_liver",
        "weights": True
    },
     "HEART": {
        "active": True,
        "sen_var": "sex",
        "fairness_vars": {
            "sen_var": "sex",
            "privileged_groups": [{'sex': 1}],
            "unprivileged_groups": [{'sex': 0}]
        },
        "dataset_code": "4_heart_disease",
        "weights": False
    },
    "AIDS": {
        "active": True,
        "sen_var": "homo",
        "fairness_vars": {
            "sen_var": "homo",
            "privileged_groups": [{'homo': 0}],
            "unprivileged_groups": [{'homo': 1}]
        },
        "dataset_code": "6_aids",
        "weights": True
    },
     "OBESITY": {
         "active": True,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{'gender': 1}],
            "unprivileged_groups": [{'gender': 0}]
        },
        "dataset_code": "7_obesity",
        "weights": True
    },
    "ARRHYTHMIA": {
        "active": True,
        "sen_var": "sex",
        "fairness_vars": {
            "sen_var": "sex",
            "privileged_groups": [{'sex': 1}],
            "unprivileged_groups": [{'sex': 0}]
        },
        "dataset_code": "11_arrhythmia",
        "weights": False
    },
    "MENTAL": {
        "active": True,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{'gender': 1}],
            "unprivileged_groups": [{'gender': 0}]
        },
        "dataset_code": "15_mental",
        "weights": False
    },
    "DIABETES": {
        "active": True,
        "sen_var": "gender",
        "fairness_vars": {
            "sen_var": "gender",
            "privileged_groups": [{'gender': 1}],
            "unprivileged_groups": [{'gender': 0}]
        },
        "dataset_code": "16_diabetes",
        "weights": False
    }



}


'''
Analysis configurations

'''

IDEAL_VALUES = {
    # fairness
    'between_groups_cv': 0.0,                  # menor variação entre grupos
    'disparate_impact': 1.0,                   # razão ideal = 1 (aceitável 0.8–1.25)
    'error_rate_ratio': 1.0,                   # mesma taxa de erro entre grupos
    'error_rate_difference': 0.0,              # diferença de erros = 0
    'statistical_parity_difference': 0.0,      # sem diferença de seleção
    'equal_opportunity_difference': 0.0,       # mesma taxa de verdadeiros positivos
    'average_odds_difference': 0.0,            # mesma taxa de acertos/erros
    'bias_amplification': 0.0,                 # modelo não amplia viés
    # 'selection_rate': 1.0,                     # ideal seria igual entre grupos***
    'consistency': 1.0,                        # previsões iguais p/ vizinhos iguais
    'false_negative_rate_difference': 0.0,
    'coef_variation': 0.0,
    'false_discovery_rate_difference': 0.0,                    

    # performance classification
    'accuracy': 1.0,
    'f1': 1.0,
    'precision': 1.0,
    'recall': 1.0,

    #tradeoff
    'fairness_performance_ratio': 0
}

METRIC_MAP = {
    'f1': 'performance',
    'accuracy': 'performance',
    'precision': 'performance',
    'recall': 'performance',
    'train_time': 'processing_performance',
    'predict_time': 'processing_performance',
    'n3_error': 'complexity',
    'si_separability': 'complexity',
    'f2_overlap': 'complexity',
    'f4_collective_feature_efficiency': 'complexity'
}

METRIC_GROUP_MAP = {
    # performance
    'f1': 'performance',
    'accuracy': 'performance',
    'precision': 'performance',
    'recall': 'performance',

    # performance de processamento
    'train_time': 'processing_performance',
    'predict_time': 'processing_performance',

    # complexidade / separabilidade
    'n3_error': 'complexity',
    'si_separability': 'complexity',
    'f2_overlap': 'complexity',
    'f4_collective_feature_efficiency': 'complexity',

    # fairness - ratio/difference/etc.
    'disparate_impact': 'ratio_based',
    'error_rate_ratio': 'ratio_based',                 
    'selection_rate_ratio': 'ratio_based',
    'consistency': 'proportion_based',           

    'error_rate_difference': 'difference_based',
    'false_negative_rate_difference': 'difference_based',
    'false_positive_rate_difference': 'difference_based',
    'false_discovery_rate_difference': 'difference_based',
    'statistical_parity_difference': 'difference_based',
    'equal_opportunity_difference': 'difference_based',
    'average_odds_difference': 'difference_based',
    'bias_amplification': 'logarithmic_difference_based',

    'between_groups_cv':'dispersion_based',
    'coef_variation':'dispersion_based_global', # desvio padrão/ média - medida de variabilidade global, porem com faixa muito distinta

    # outros
    'selection_rate': 'others'
}

# Translate for pt-br
TRANSLATE = {
    "disparate_impact": "Impacto Díspar",
    "statistical_parity_difference": "Diferença de Paridade Estatística",
    "consistency": "Consistência",
    "between_groups_cv": "Coeficiente de Variação intergrupos",
    "bias_amplification": "Amplificação de Viés",
    "false_negative_rate_difference": "Diferença na Taxa de Falsos Negativos",
    "false_discovery_rate_difference": "Diferença na Taxa de Falsos Positivos",
    "error_rate_difference": "Diferença na Taxa de Erros",
    "fairness_performance_ratio": "Relação Fairness-Desempenho",
    "11_arrhythmia": "Arrhythmia",
    "16_diabetes": "Diabetes",
    "15_mental": "Mental",
    "3_liver":"Liver",
    "4_heart_disease":"Heart Disease",
    "6_aids":"Aids",
    "7_obesity":"Obesity",
    "baseline":"Baseline",
    "experiment":"Experimento",
    "in_processing":"In-Processing",
    "pre_processing":"Pre-Processing",
    "recall":"Revocação",
    'method': 'Método',
    'baseline_typical': 'Baseline Típico'
    }

CATEGORY_MAP = {
    "pre_processing": ["EDL"],
    "in_processing": ["IAD", "IGLA", "IGLU", "IFG", "IP", "IW"],
    "baseline": ["BLRA", "BLRU", "BRFA", "BRFU"]
}