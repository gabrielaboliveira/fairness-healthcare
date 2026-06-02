"""
Teste local para RQ4 — verifica todas as funções de rq4_utils.py
usando os dados reais do projeto.

Execução: python experiments/test_rq4.py
"""
import sys
import os
import traceback
import tempfile
from pathlib import Path

# Adicionar raiz ao path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")  # backend não-interativo (sem janela)

import pandas as pd
import numpy as np

# ── Imports do projeto ──
from auxiliar_analysis import config
from auxiliar_analysis.config import IDEAL_VALUES, DATASET_CONFIGS, TRANSLATE_PT
from auxiliar_analysis.analysis_utils import (
    set_plot_style, load_results, add_experiment_metadata,
)
from auxiliar_analysis.bias_pattern_classifier import (
    classify_bias_patterns,
    prepare_baseline_dataset_level,
)

from auxiliar_analysis.rq4_utils import (
    build_intensity_scatter_data,
    plot_intensity_scatter_category,
    plot_intensity_scatter_method,
    plot_intensity_by_severity,
    compute_intensity_correlation_table,
    build_intensity_summary_table,
    plot_proportionality_ratio_boxplot,
)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_build_scatter_data(df, classified, metrics, ideal_values):
    """Testa build_intensity_scatter_data em ambos os modos."""
    separator("TEST 1: build_intensity_scatter_data")

    # ── level="method" ──
    scatter_all = build_intensity_scatter_data(
        df, classified,
        metrics=metrics,
        ideal_values=ideal_values,
        scope="all",
        level="method",
    )
    print(f"  [method] shape: {scatter_all.shape}")
    print(f"  [method] columns: {list(scatter_all.columns)}")
    print(f"  [method] metrics: {sorted(scatter_all['metric'].unique())}")
    print(f"  [method] methods: {sorted(scatter_all['method'].unique())}")
    print(f"  [method] categories: {sorted(scatter_all['category'].dropna().unique())}")
    print(f"  [method] severity_levels: {sorted(scatter_all['severity_level'].unique())}")
    print(f"  [method] NaN in gap_baseline: {scatter_all['gap_baseline'].isna().sum()}")
    print(f"  [method] NaN in delta_fair: {scatter_all['delta_fair'].isna().sum()}")

    expected_cols = ["dataset", "metric", "method", "category",
                     "gap_baseline", "delta_fair", "severity_level", "severity_label"]
    missing = set(expected_cols) - set(scatter_all.columns)
    assert not missing, f"Missing columns: {missing}"
    assert len(scatter_all) > 0, "scatter_all is empty!"

    # ── level="category" ──
    scatter_cat = build_intensity_scatter_data(
        df, classified,
        metrics=metrics,
        ideal_values=ideal_values,
        scope="all",
        level="category",
    )
    print(f"\n  [category] shape: {scatter_cat.shape}")
    print(f"  [category] methods: {sorted(scatter_cat['method'].unique())}")
    print(f"  [category] NaN in gap_baseline: {scatter_cat['gap_baseline'].isna().sum()}")

    assert len(scatter_cat) > 0, "scatter_cat is empty!"
    print("\n  >> PASS: build_intensity_scatter_data")
    return scatter_all, scatter_cat


def test_plot_scatter_category(scatter_cat, metrics, translate, tmp_dir):
    """Testa plot_intensity_scatter_category para 1 metrica."""
    separator("TEST 2: plot_intensity_scatter_category")
    metric = metrics[0]
    save_path = tmp_dir / f"scatter_cat_{metric}.pdf"
    fig = plot_intensity_scatter_category(
        scatter_cat, metric, translate,
        save_path=str(save_path), show=False,
    )
    assert fig is not None, "Figure is None!"
    assert save_path.exists(), f"File not saved: {save_path}"
    print(f"  metric: {metric}")
    print(f"  saved: {save_path} ({save_path.stat().st_size} bytes)")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  >> PASS: plot_intensity_scatter_category")


def test_plot_scatter_method(scatter_all, metrics, translate, tmp_dir):
    """Testa plot_intensity_scatter_method para 1 metrica."""
    separator("TEST 3: plot_intensity_scatter_method")
    metric = metrics[0]
    save_path = tmp_dir / f"scatter_method_{metric}.pdf"
    fig = plot_intensity_scatter_method(
        scatter_all, metric, translate,
        save_path=str(save_path), show=False,
    )
    assert fig is not None, "Figure is None!"
    assert save_path.exists(), f"File not saved: {save_path}"
    print(f"  metric: {metric}")
    print(f"  saved: {save_path} ({save_path.stat().st_size} bytes)")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  >> PASS: plot_intensity_scatter_method")


def test_plot_severity(scatter_all, metrics, translate, tmp_dir):
    """Testa plot_intensity_by_severity."""
    separator("TEST 4: plot_intensity_by_severity")
    metric = metrics[0]
    # sem faceta
    save1 = tmp_dir / f"scatter_sev_{metric}.pdf"
    fig1 = plot_intensity_by_severity(
        scatter_all, metric, translate,
        save_path=str(save1), show=False, facet=False,
    )
    print(f"  [no facet] metric: {metric}, fig: {'OK' if fig1 else 'None'}")
    if save1.exists():
        print(f"  [no facet] saved: {save1.stat().st_size} bytes")
    import matplotlib.pyplot as plt
    if fig1:
        plt.close(fig1)

    # com faceta
    save2 = tmp_dir / f"scatter_sev_facet_{metric}.pdf"
    fig2 = plot_intensity_by_severity(
        scatter_all, metric, translate,
        save_path=str(save2), show=False, facet=True,
    )
    print(f"  [facet] metric: {metric}, fig: {'OK' if fig2 else 'None'}")
    if fig2:
        plt.close(fig2)
    print("  >> PASS: plot_intensity_by_severity")


def test_correlation_table(scatter_all, scatter_cat, translate):
    """Testa compute_intensity_correlation_table."""
    separator("TEST 5: compute_intensity_correlation_table")

    # Por categoria
    corr_cat = compute_intensity_correlation_table(
        scatter_cat, group_col="category", translate=translate,
    )
    print(f"  [category] shape: {corr_cat.shape}")
    print(f"  [category] columns: {list(corr_cat.columns)}")
    if not corr_cat.empty:
        print(f"  [category] head:\n{corr_cat.head().to_string()}")
        assert "spearman_rho" in corr_cat.columns
        assert "interpretacao_N" in corr_cat.columns

    # Por metodo
    corr_method = compute_intensity_correlation_table(
        scatter_all, group_col="method", translate=translate,
    )
    print(f"\n  [method] shape: {corr_method.shape}")
    if not corr_method.empty:
        print(f"  [method] head:\n{corr_method.head().to_string()}")

    print("  >> PASS: compute_intensity_correlation_table")
    return corr_cat, corr_method


def test_summary_table(scatter_all, translate):
    """Testa build_intensity_summary_table."""
    separator("TEST 6: build_intensity_summary_table")
    summary = build_intensity_summary_table(
        scatter_all, group_col="severity_level", translate=translate,
    )
    print(f"  shape: {summary.shape}")
    print(f"  columns: {list(summary.columns)}")
    if not summary.empty:
        print(f"  head:\n{summary.head(10).to_string()}")
        assert "median_ratio" in summary.columns
        assert "win_rate_pct" in summary.columns
        # Verificar que ratio nao tem Inf
        if summary["median_ratio"].notna().any():
            assert np.isfinite(summary["median_ratio"].dropna()).all(), "Inf in median_ratio!"
    print("  >> PASS: build_intensity_summary_table")
    return summary


def test_proportionality_boxplot(scatter_all, translate, tmp_dir):
    """Testa plot_proportionality_ratio_boxplot."""
    separator("TEST 7: plot_proportionality_ratio_boxplot")
    import matplotlib.pyplot as plt

    # Por categoria
    save1 = tmp_dir / "ratio_boxplot_cat.pdf"
    fig1 = plot_proportionality_ratio_boxplot(
        scatter_all, group_col="category", translate=translate,
        save_path=str(save1), show=False,
    )
    print(f"  [category] fig: {'OK' if fig1 else 'None'}")
    if save1.exists():
        print(f"  [category] saved: {save1.stat().st_size} bytes")
    if fig1:
        plt.close(fig1)

    # Por metodo
    save2 = tmp_dir / "ratio_boxplot_method.pdf"
    fig2 = plot_proportionality_ratio_boxplot(
        scatter_all, group_col="method", translate=translate,
        save_path=str(save2), show=False,
    )
    print(f"  [method] fig: {'OK' if fig2 else 'None'}")
    if save2.exists():
        print(f"  [method] saved: {save2.stat().st_size} bytes")
    if fig2:
        plt.close(fig2)

    print("  >> PASS: plot_proportionality_ratio_boxplot")


def test_edge_cases(df, classified, ideal_values):
    """Testa edge cases: metrica inexistente, scope restrito."""
    separator("TEST 8: Edge cases")

    # Metrica inexistente
    scatter_empty = build_intensity_scatter_data(
        df, classified,
        metrics=["metrica_que_nao_existe"],
        ideal_values=ideal_values,
        scope="all", level="method",
    )
    assert scatter_empty.empty, "Expected empty DataFrame for unknown metric"
    print("  [unknown metric] -> empty DataFrame OK")

    # Scatter vazio -> plot retorna None
    fig = plot_intensity_scatter_category(
        scatter_empty, "metrica_que_nao_existe", {}, show=False,
    )
    assert fig is None, "Expected None for empty scatter"
    print("  [empty plot] -> None OK")

    # Correlation com poucos pontos
    corr_empty = compute_intensity_correlation_table(scatter_empty)
    assert corr_empty.empty, "Expected empty correlation table"
    print("  [empty corr] -> empty DataFrame OK")

    print("  >> PASS: Edge cases")


def main():
    print("=" * 60)
    print("  RQ4 LOCAL TEST SUITE")
    print("=" * 60)

    # ── 1. Carregar dados reais (replicando pipeline do notebook) ──
    separator("SETUP: Loading data")

    datasets_to_run = {
        "HEART": True,
        "AIDS": True,
        "OBESITY": True,
        "ARRHYTHMIA": True,
        "MENTAL": True,
        "DIABETES": True,
    }
    EXP_IDS_VALIDOS = [
        "BLA", "BLU", "BRFA", "BRFU",
        "EDL", "IAD", "IFG", "IGLA", "IGLU", "IP", "IW",
    ]

    ideal_values = config.IDEAL_VALUES
    translate = config.TRANSLATE
    metric_map = config.METRIC_MAP
    metric_group_map = config.METRIC_GROUP_MAP

    results_base = ROOT / "experiments" / "results"
    df = load_results(datasets_to_run, DATASET_CONFIGS, results_base=results_base)
    df = df[df["exp_id"].isin(EXP_IDS_VALIDOS)]
    df = add_experiment_metadata(df)
    df["metric_type"] = df["metric"].map(metric_map).fillna("fairness")
    df["metric_group"] = df["metric"].map(metric_group_map).fillna("others")
    # Converter valores para numérico
    df["value_num"] = (
        df["value"].str.replace(r"[\[\]]", "", regex=True).astype(float)
    )
    print(f"  df loaded: {df.shape}")

    # Preparar baseline_ds_paper para classify_bias_patterns
    METRICS_PAPER = {
        "consistency", "generalized_entropy_index", "disparate_impact",
        "false_negative_rate_difference", "bias_amplification",
        "false_discovery_rate_difference", "error_rate_difference",
    }
    baseline_df = df[df["exp_type"] == "baseline"]
    fairness_baseline = baseline_df[baseline_df.metric_type == "fairness"]

    baseline_ds_paper = prepare_baseline_dataset_level(
        fairness_baseline,
        baseline_ids=["BLA", "BLU", "BRFA", "BRFU"],
        value_col="value_num",
    )
    baseline_ds_paper = baseline_ds_paper[
        baseline_ds_paper["metric"].isin(METRICS_PAPER)
    ].copy()
    print(f"  baseline_ds_paper: {baseline_ds_paper.shape}")

    # Classificar bias patterns
    classified, metric_scale = classify_bias_patterns(
        baseline_ds_paper,
        ideal_values=ideal_values,
        ratio_metrics={"disparate_impact"},
        min_units=0.5,
        medium_units=1.0,
        high_units=1.5,
        return_scale=True,
    )
    print(f"  classified: {classified.shape}")

    # Merge labels into df (como o notebook faz)
    labels = classified[
        ["dataset", "metric", "eligible", "severity_level", "pattern_code", "direction"]
    ].drop_duplicates()
    df = df.merge(labels, on=["dataset", "metric"], how="left")
    print(f"  df after merge: {df.shape}")

    # Metricas para teste (subconjunto para velocidade)
    all_metrics = sorted(classified["metric"].unique())
    # Usar 2-3 metricas para teste rapido
    test_metrics = all_metrics[:3] if len(all_metrics) >= 3 else all_metrics
    print(f"  test_metrics: {test_metrics}")

    # Dir temporario para figuras
    tmp_dir = Path(tempfile.mkdtemp(prefix="rq4_test_"))
    print(f"  tmp_dir: {tmp_dir}")

    # ── 2. Executar testes ──
    errors = []

    try:
        scatter_all, scatter_cat = test_build_scatter_data(
            df, classified, test_metrics, ideal_values
        )
    except Exception as e:
        errors.append(("build_scatter_data", e))
        traceback.print_exc()
        return

    tests = [
        ("scatter_category", lambda: test_plot_scatter_category(scatter_cat, test_metrics, translate, tmp_dir)),
        ("scatter_method", lambda: test_plot_scatter_method(scatter_all, test_metrics, translate, tmp_dir)),
        ("severity", lambda: test_plot_severity(scatter_all, test_metrics, translate, tmp_dir)),
        ("correlation", lambda: test_correlation_table(scatter_all, scatter_cat, translate)),
        ("summary", lambda: test_summary_table(scatter_all, translate)),
        ("boxplot", lambda: test_proportionality_boxplot(scatter_all, translate, tmp_dir)),
        ("edge_cases", lambda: test_edge_cases(df, classified, ideal_values)),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            errors.append((name, e))
            traceback.print_exc()

    # ── 3. Resumo ──
    separator("RESULTS")
    total = len(tests) + 1  # +1 para build_scatter_data
    passed = total - len(errors)
    print(f"  Passed: {passed}/{total}")
    if errors:
        print(f"  FAILED:")
        for name, e in errors:
            print(f"    - {name}: {e}")
    else:
        print("  ALL TESTS PASSED!")

    print(f"\n  Temp figures in: {tmp_dir}")


if __name__ == "__main__":
    main()
