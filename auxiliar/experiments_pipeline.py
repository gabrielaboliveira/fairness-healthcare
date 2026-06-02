"""

Experiments pipelines

is_valid_combination: function for testing all parameter combinations of a classifier before the tunning process.
classifier_baseline: function for training a classifier and evaluating its performance.
classifier_opt: function for optimizing a classifier using RandomizedSearchCV, training it, and evaluating its performance.

"""

# 1) Standard library
import sys
from pathlib import Path

# 2) Paths – expõe ROOT (pasta que contém auxiliar/, fairgbm/, wasserstein_fairness/, etc.)
ROOT = Path.cwd().parent
sys.path[:0] = [str(ROOT / "aif360_local")]


# libs
from auxiliar.utils import ClassificationMetricsF1, pycol_complexity
import pandas as pd
from tqdm.auto import tqdm
import time
import numpy as np
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

from aif360.algorithms.inprocessing.adversarial_debiasing import (
    AdversarialDebiasing as CustomAdversarialDebiasing,
)
import tensorflow as tf

"""
    Auxiliar

"""


def calculate_classification_metrics(dataset_test, dataset_pred, fairness_vars):
    metric = ClassificationMetric(
        dataset_test,
        dataset_pred,
        unprivileged_groups=fairness_vars["unprivileged_groups"],
        privileged_groups=fairness_vars["privileged_groups"],
    )

    # para permitir o calculo sobre o pred.
    metric_cons = BinaryLabelDatasetMetric (
    dataset_pred,
    unprivileged_groups=fairness_vars["unprivileged_groups"],
    privileged_groups=fairness_vars["privileged_groups"],
    )

    complexity_results = pycol_complexity(dataset_test)

    return {
        "error_rate_difference": metric.error_rate_difference(),
        "bias_amplification": metric.differential_fairness_bias_amplification(),
        "false_discovery_rate_difference": metric.false_discovery_rate_difference(),
        # "consistency":                           metric.consistency()[0],
        "consistency": np.asarray(metric_cons.consistency()).squeeze().item(),
        # "consistency_arr":                       metric.consistency(),
        "generalized_entropy_index": metric.generalized_entropy_index(),
        "disparate_impact": metric.disparate_impact(),
        # "statistical_parity_difference": metric.statistical_parity_difference(),
        "false_negative_rate_difference": metric.false_negative_rate_difference(),
        # "between_groups_cv": metric.between_group_coefficient_of_variation(),
        "accuracy": metric.accuracy(),
        "precision": metric.precision(),
        "recall": metric.recall(),
        "f1": ClassificationMetricsF1(metric)
    }


def scale_and_recombine(df_train, df_test, target_col, scaler=None):

    # separa X e y
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[[target_col]]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[[target_col]]

    # ----------------------------
    # 1) identificar colunas binárias automaticamente
    # ----------------------------
    binary_cols = [
        col for col in X_train.columns if X_train[col].dropna().isin([0, 1]).all()
    ]

    # ----------------------------
    # 2) colunas contínuas = o resto
    # ----------------------------
    continuous_cols = [col for col in X_train.columns if col not in binary_cols]

    # ----------------------------
    # 3) aplicar scaler apenas nas contínuas
    # ----------------------------
    scaler_instance = clone(scaler) if scaler else MinMaxScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if continuous_cols:  # só escala se houver colunas contínuas
        X_train_scaled[continuous_cols] = scaler_instance.fit_transform(
            X_train[continuous_cols]
        )
        X_test_scaled[continuous_cols] = scaler_instance.transform(
            X_test[continuous_cols]
        )

    # ----------------------------
    # 4) reconstruir DataFrames
    # ----------------------------
    df_train_scaled = pd.concat([X_train_scaled, y_train], axis=1)
    df_test_scaled = pd.concat([X_test_scaled, y_test], axis=1)

    return df_train_scaled, df_test_scaled


"""
    Pipeline Functions

"""

# Baseline


def run_experiment_baseline(
    df,
    target_name,
    sen_var,
    awareness,  # se True, inclui a variável sensível no X
    fold_index,
    exp_id,
    estimator_class,
    estimator_kwargs,
    fairness_vars,
    scaler=MinMaxScaler(),
):
    folds_results = []

    pbar = tqdm(total=len(fold_index), desc="CV folds", unit="fold")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_index):

        # 1) scale + recombine
        df_tr, df_te = scale_and_recombine(
            df.iloc[train_idx], df.iloc[test_idx], target_name, scaler=scaler
        )

        # 2) Prepare X/y
        drop_cols = [target_name] + ([] if awareness else [sen_var])
        X_tr = df_tr.drop(columns=drop_cols)
        y_tr = df_tr[target_name]
        X_te = df_te.drop(columns=drop_cols)

        # 3) Train and predict
        clf = estimator_class(**estimator_kwargs)
        start = time.time()
        clf.fit(X_tr, y_tr)
        train_time = time.time() - start
        start = time.time()
        y_pred = clf.predict(X_te)
        predict_time = time.time() - start

        # 4) mount AIF360 datasets
        test_bld = BinaryLabelDataset(
            df=df_te,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )
        df_pred = df_te.copy()
        df_pred[target_name] = y_pred
        pred_bld = BinaryLabelDataset(
            df=df_pred,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )

        print(f"\n[Fold {fold_idx}] {exp_id}")
        print(f"y_tr distribuição: {y_tr.value_counts().to_dict()}")
        print(f"y_te distribuição: {df_te[target_name].value_counts().to_dict()}")
        print(f"y_pred distribuição: {pd.Series(y_pred).value_counts().to_dict()}")
        print(
            f"Classes no treino: {np.unique(y_tr)}, classes na predição: {np.unique(y_pred)}"
        )

        # 5) Metrics calculation
        metrics = calculate_classification_metrics(test_bld, pred_bld, fairness_vars)

        metrics.update(
            {
                "exp_id": exp_id,
                "fold": fold_idx,
                "train_time": train_time,
                "predict_time": predict_time,
            }
        )
        folds_results.append(metrics)

        pbar.update(1)
    pbar.close()
    return folds_results


# Fairness pipeline
def run_experiment(
    df,  # (pd.DataFrame) dataset to use in the experiment.
    target_name,  # (str) name of the column to predict.
    sen_var,  # (str) sensitive variable name, ex: race, sex, gender, etc.
    fold_index,  # list of (train_idx, test_idx) indices for each fold.
    exp_id,  # (str) experiment identifier, ex: "exp1", "exp2", etc.
    estimator_class,  # ex: GridSearchReduction ou PrejudiceRemover
    estimator_kwargs,  # kwargs p/ __init__ do estimador
    fairness_vars,
    scaler=MinMaxScaler(),  # (MinMaxScaler) scaler to use for feature scaling.
    seed=42,
):
    """
    Run a k-fold experiment with any AIF360 estimator.

    """

    folds_results = []

    pbar = tqdm(total=len(fold_index), desc="CV folds", unit="fold")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_index):
        # 1) scale + recombine
        df_train_scaled, df_test_scaled = scale_and_recombine(
            df.iloc[train_idx], df.iloc[test_idx], target_name, scaler=scaler
        )

        # 2) monta os BinaryLabelDataset do AIF360
        train_bld = BinaryLabelDataset(
            df=df_train_scaled,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )
        test_bld = BinaryLabelDataset(
            df=df_test_scaled,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )

        # 3) constrói e treina o estimador de fairness
        model = estimator_class(**estimator_kwargs)
        print(model)
        start = time.time()
        model.fit(train_bld)
        train_time = time.time() - start
        start = time.time()
        y_pred = model.predict(test_bld)
        predict_time = time.time() - start

        prot = test_bld.protected_attributes[:, 0]
        # print("\n===== prot=====")
        # print(np.unique(prot))
        # y_true = test_bld.labels.ravel()
        # y_pred1d = np.array(y_pred).ravel()
        # print("\n===== DEBUG GROUP DISTRIBUTIONS =====")
        # print("Y_true privileged:", np.unique(y_true[prot == 1], return_counts=True))
        # print("Y_true unprivileged:", np.unique(y_true[prot == 0], return_counts=True))

        # print("Y_pred privileged:", np.unique(y_pred1d[prot == 1], return_counts=True))
        # print("Y_pred unprivileged:", np.unique(y_pred1d[prot == 0], return_counts=True))
        # print("=====================================\n")

        # 4) coleta métricas
        metrics = calculate_classification_metrics(test_bld, y_pred, fairness_vars)
        metrics.update(
            {
                "exp_id": exp_id,
                "fold": fold_idx,
                "train_time": train_time,
                "predict_time": predict_time,
            }
        )
        folds_results.append(metrics)

        pbar.update(1)
    pbar.close()

    return folds_results


def run_experiment_adv_debiasing(
    df,  # (pd.DataFrame) dataset to use in the experiment.
    target_name,  # (str) name of the column to predict.
    sen_var,  # (str) sensitive variable name, ex: race, sex, gender, etc.
    fold_index,  # list of (train_idx, test_idx) indices for each fold.
    exp_id,  # (str) experiment identifier, ex: "exp1", "exp2", etc.
    estimator_kwargs,  # kwargs p/ __init__ do estimador
    fairness_vars,
    scaler=MinMaxScaler(),  # (MinMaxScaler) scaler to use for feature scaling.
    seed=42,
):
    """
    Run a k-fold experiment with any AIF360 estimator.

    """

    tf.compat.v1.disable_eager_execution()
    tf.random.set_seed(seed)

    folds_results = []

    pbar = tqdm(total=len(fold_index), desc="CV folds", unit="fold")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_index):

        tf.compat.v1.reset_default_graph()

        # 1) scale + recombine
        df_train_scaled, df_test_scaled = scale_and_recombine(
            df.iloc[train_idx], df.iloc[test_idx], target_name, scaler=scaler
        )

        # 2) monta os BinaryLabelDataset do AIF360
        train_bld = BinaryLabelDataset(
            df=df_train_scaled,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )
        test_bld = BinaryLabelDataset(
            df=df_test_scaled,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )

        sess = tf.compat.v1.Session()
        # 3) constrói e treina o estimador de fairness
        model = CustomAdversarialDebiasing(
            **estimator_kwargs, sess=sess, scope_name=f"adv_debias_fold{fold_idx}"
        )
        start = time.time()
        model.fit(train_bld)
        train_time = time.time() - start
        start = time.time()
        y_pred = model.predict(test_bld)
        predict_time = time.time() - start

        # 4) coleta métricas
        metrics = calculate_classification_metrics(test_bld, y_pred, fairness_vars)
        metrics.update(
            {
                "exp_id": exp_id,
                "fold": fold_idx,
                "train_time": train_time,
                "predict_time": predict_time,
            }
        )
        folds_results.append(metrics)
        sess.close()
        pbar.update(1)
    pbar.close()

    return folds_results


# Fairness pipeline - preprocessing methods
def run_experiment_preprocessing(
    df,
    target_name,
    sen_var,
    awareness,
    fold_index,
    exp_id,
    estimator_class,
    estimator_kwargs,
    preprocessing_class,
    fairness_vars,
    scaler=MinMaxScaler(),
):
    is_reweighing = isinstance(preprocessing_class, Reweighing)
    folds_results = []
    pbar = tqdm(total=len(fold_index), desc="CV folds", unit="fold")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_index):
        df_tr, df_te = scale_and_recombine(
            df.iloc[train_idx], df.iloc[test_idx], target_name, scaler=scaler
        )

        train_bld = BinaryLabelDataset(
            df=df_tr,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )
        test_bld = BinaryLabelDataset(
            df=df_te,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )

        drop_cols = [target_name] if awareness else [target_name, sen_var]

        if is_reweighing:
            print("\nUsing Reweighing preprocessing...")
            preprocessing_class.fit(train_bld)
            train_weighted = preprocessing_class.transform(train_bld)
            sample_weights = train_weighted.instance_weights

            df_train = train_bld.convert_to_dataframe()[0]
            X_train = df_train.drop(columns=drop_cols)
            y_train = df_train[target_name]
            X_test = df_te.drop(columns=drop_cols)

            clf = estimator_class(**estimator_kwargs)
            start = time.time()
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            train_time = time.time() - start

            start = time.time()
            y_pred = clf.predict(X_test)
            predict_time = time.time() - start

        else:
            print(f"\nUsing {preprocessing_class.__class__.__name__} preprocessing...")
            train_bld_rep = preprocessing_class.fit_transform(train_bld)
            test_bld_rep = preprocessing_class.fit_transform(test_bld)

            df_train_rep = train_bld_rep.convert_to_dataframe()[0]
            df_test_rep = test_bld_rep.convert_to_dataframe()[0]

            X_train = df_train_rep.drop(columns=drop_cols)
            y_train = df_train_rep[target_name]
            X_test = df_test_rep.drop(columns=drop_cols)

            clf = estimator_class(**estimator_kwargs)
            start = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = clf.predict(X_test)
            predict_time = time.time() - start

        df_pred = df_te.copy()
        df_pred[target_name] = y_pred
        pred_bld = BinaryLabelDataset(
            df=df_pred,
            label_names=[target_name],
            protected_attribute_names=[sen_var],
            favorable_label=1,
            unfavorable_label=0,
        )

        metrics = calculate_classification_metrics(test_bld, pred_bld, fairness_vars)
        metrics.update(
            {
                "exp_id": exp_id,
                "fold": fold_idx,
                "train_time": train_time,
                "predict_time": predict_time,
            }
        )
        folds_results.append(metrics)
        pbar.update(1)

    pbar.close()
    return folds_results
