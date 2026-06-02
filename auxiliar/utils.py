''''
general utils 

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycol_complexity.complexity import Complexity

def ClassificationMetricsF1(classification_metrics):
    prec = classification_metrics.precision()
    rec = classification_metrics.recall()
    if (prec + rec) == 0:
        return 0.0
    return(2 * (prec * rec) / (prec + rec))



def pycol_complexity(aif_dataset):
    """
    Calcula métricas de complexidade (Pycol) a partir de um BinaryLabelDataset do AIF360.
    """
    X = aif_dataset.features.astype(np.float32, copy=False)
    y = aif_dataset.labels.ravel()

    comp = Complexity(dataset={"X": X, "y": y}, file_type="array")

    return {
        "f4_collective_feature_efficiency": comp.F4(),
        "f2_overlap": comp.F2(),
        "n3_error": comp.N3(),
        "si_separability": comp.SI()
    }


def load_and_split(dataset_path, label, seed, test_size=0.2):
    df = pd.read_csv(dataset_path)

    display(df.head())
    display(df.describe())

    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label], shuffle=True)