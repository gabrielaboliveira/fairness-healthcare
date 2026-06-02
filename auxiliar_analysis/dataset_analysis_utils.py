"""
dataset_analysis_utils.py — Utilitários para análise descritiva dos datasets

Funções disponíveis:
- load_combined_data: carrega target + variável sensível de todos os datasets ativos
- build_size_table: tabela de tamanho dos datasets → LaTeX
- build_target_distribution_table: distribuição de target por dataset → LaTeX
- build_sensitive_group_table: distribuição por dataset × grupo sensível × target → LaTeX
- plot_target_by_sensitive_group: gráfico de barras com labels descritivos dos grupos
- plot_feature_distributions: distribuições de features por grupo sensível e por target
"""
from __future__ import annotations

from typing import Dict, Optional, Set
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns

from .analysis_utils import (
    latex_and_save,
    apply_comma_axes,
    format_caption,
)
from .config import (
    DATASET_CONFIGS, GROUP_LABELS, TRANSLATE,
    DATASET_DISPLAY_NAMES, DATASET_META,
)


# =========================================================================
# HELPERS INTERNOS
# =========================================================================

def _display_name(dataset_key: str) -> str:
    """Retorna o nome de exibição do dataset (title case, nome completo)."""
    return DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key.title())


def _apply_display_names(df: pd.DataFrame, col: str = "dataset") -> pd.DataFrame:
    """Substitui chaves de dataset por nomes de exibição em um DataFrame."""
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].map(lambda x: _display_name(x))
    return out


def _unpriv_vals(dataset_name: str, dataset_configs: Dict = None) -> set:
    """Retorna o conjunto de valores desprivilegiados para um dataset."""
    dataset_configs = dataset_configs or DATASET_CONFIGS
    fv = dataset_configs[dataset_name]["fairness_vars"]
    sv = fv["sen_var"]
    return {d[sv] for d in fv.get("unprivileged_groups", [])}


def _map_group_labels(
    df: pd.DataFrame,
    dataset_configs: Dict = None,
    group_labels: Dict = None,
) -> pd.DataFrame:
    """Substitui valores numéricos de sensitive_group por labels descritivos.

    Ex: gender 0 → "Feminino", gender 1 → "Masculino".
    """
    dataset_configs = dataset_configs or DATASET_CONFIGS
    group_labels = group_labels or GROUP_LABELS
    out = df.copy()
    # Converter coluna para object para evitar FutureWarning de dtype incompatível
    out["sensitive_group"] = out["sensitive_group"].astype(object)
    # Usar _dataset_key para lookup no config (dataset pode estar em display name)
    key_col = "_dataset_key" if "_dataset_key" in out.columns else "dataset"
    for ds_key in out[key_col].unique():
        if ds_key not in dataset_configs:
            continue
        sen_var = dataset_configs[ds_key]["sen_var"]
        labels = group_labels.get(sen_var)
        if labels is None:
            continue
        mask = out[key_col] == ds_key
        out.loc[mask, "sensitive_group"] = (
            out.loc[mask, "sensitive_group"]
            .map(labels)
            .fillna(out.loc[mask, "sensitive_group"])
        )
    return out


# =========================================================================
# CARREGAMENTO DE DADOS
# =========================================================================

def load_combined_data(
    dataset_configs: Dict = None,
    data_root: Path | str = None,
) -> pd.DataFrame:
    """Carrega target + variável sensível de todos os datasets ativos.

    Parameters
    ----------
    dataset_configs : dict, optional
        Configuração dos datasets. Default: ``DATASET_CONFIGS``.
    data_root : Path, optional
        Raiz dos dados processados. Default: ``config.DATA_PATH / "processed"``.

    Returns
    -------
    pd.DataFrame
        Colunas: ``dataset``, ``target``, ``sensitive_group``.
    """
    dataset_configs = dataset_configs or DATASET_CONFIGS
    if data_root is None:
        from .config import DATA_PATH
        data_root = DATA_PATH / "processed"
    data_root = Path(data_root)

    frames = []
    for dataset_name, cfg in dataset_configs.items():
        if not cfg.get("active", False):
            continue
        code = cfg["dataset_code"]
        sen_var = cfg["sen_var"]
        csv_path = data_root / code / f"{code}_pp.csv"
        if not csv_path.exists():
            print(f"  [AVISO] CSV não encontrado: {csv_path}")
            continue
        df = pd.read_csv(csv_path, usecols=["target", sen_var])
        df = df.rename(columns={sen_var: "sensitive_group"})
        df["dataset"] = _display_name(dataset_name)
        df["_dataset_key"] = dataset_name  # chave interna para lookups
        frames.append(df)

    if not frames:
        raise ValueError("Nenhum dataset encontrado.")
    return pd.concat(frames, ignore_index=True)


# =========================================================================
# TABELAS
# =========================================================================

def build_size_table(
    combined: pd.DataFrame,
    result_dir: Path | str,
    translate: Dict = None,
) -> pd.DataFrame:
    """Tabela de tamanho (nº de amostras) de cada dataset → LaTeX.

    .. note:: Considere usar ``build_dataset_overview_table`` que já inclui
       o volume junto com as demais propriedades.
    """
    translate = translate or TRANSLATE
    dist = combined["dataset"].value_counts().reset_index()
    dist.columns = ["dataset", "n"]
    dist = dist.sort_values("dataset").reset_index(drop=True)

    latex_and_save(
        dist,
        caption="Volume de amostras por conjunto de dados.",
        label="table_volume",
        result_dir=str(result_dir),
        translate=translate,
    )
    return dist


def build_target_distribution_table(
    combined: pd.DataFrame,
    result_dir: Path | str,
    translate: Dict = None,
) -> pd.DataFrame:
    """Distribuição de target (0/1) por dataset → LaTeX."""
    translate = translate or TRANSLATE

    dist = (
        combined.groupby(["dataset", "target"])
        .size()
        .rename("n")
        .reset_index()
    )
    dist["share"] = dist["n"] / dist.groupby("dataset")["n"].transform("sum")

    latex_and_save(
        dist,
        caption="Distribuição dos rótulos-alvo em cada conjunto de dados.",
        label="table_dist_target",
        result_dir=str(result_dir),
        translate=translate,
        multirow_cols=["dataset"],
        group_col="dataset",
    )
    return dist


def build_sensitive_group_table(
    combined: pd.DataFrame,
    dataset_configs: Dict = None,
    result_dir: Path | str = None,
    translate: Dict = None,
    group_labels: Dict = None,
) -> pd.DataFrame:
    """Distribuição de target por dataset × grupo sensível → LaTeX.

    Inclui flags de grupo desprivilegiado e labels descritivos.
    """
    dataset_configs = dataset_configs or DATASET_CONFIGS
    translate = translate or TRANSLATE
    group_labels = group_labels or GROUP_LABELS

    df = combined.copy()

    # Marcar desprivilegiados
    _key_col = "_dataset_key" if "_dataset_key" in df.columns else "dataset"
    df["is_unpriv"] = df.apply(
        lambda r: r["sensitive_group"] in _unpriv_vals(r[_key_col], dataset_configs),
        axis=1,
    )

    # Agregar
    dist = (
        df.groupby(["dataset", "sensitive_group", "target", "is_unpriv"])
        .size()
        .rename("n")
        .reset_index()
    )
    dist["share"] = (
        dist["n"]
        / dist.groupby(["dataset", "sensitive_group"])["n"].transform("sum")
    )

    # Labels descritivos
    dist = _map_group_labels(dist, dataset_configs, group_labels)
    dist["is_unpriv"] = dist["is_unpriv"].map({True: "Sim", False: "Não"})

    # Remover colunas internas antes de exportar
    dist = dist.drop(columns=["_dataset_key"], errors="ignore")

    if result_dir is not None:
        latex_and_save(
            dist,
            caption="Distribuição dos rótulos-alvo por grupo sensível em cada conjunto de dados.",
            label="table_dist_target_sens",
            result_dir=str(result_dir),
            translate=translate,
            multirow_cols=["dataset"],
            group_col="dataset",
        )
    return dist


# =========================================================================
# GRÁFICO DE BARRAS: TARGET × GRUPO SENSÍVEL
# =========================================================================

def plot_target_by_sensitive_group(
    combined: pd.DataFrame,
    dataset_configs: Dict = None,
    save_path: Path | str = None,
    group_labels: Dict = None,
    figsize: tuple = (14, 5),
    show: bool = True,
) -> plt.Figure:
    """Gráfico de barras agrupadas: distribuição de target por grupo sensível.

    Eixo X hierárquico: dataset (maior) + label descritivo do grupo (menor).
    Borda preta marca grupo desprivilegiado.
    """
    dataset_configs = dataset_configs or DATASET_CONFIGS
    group_labels = group_labels or GROUP_LABELS

    df = combined.copy()

    # Flag desprivilegiado
    _key_col = "_dataset_key" if "_dataset_key" in df.columns else "dataset"
    df["is_unpriv"] = df.apply(
        lambda r: r["sensitive_group"] in _unpriv_vals(r[_key_col], dataset_configs),
        axis=1,
    )

    # Labels descritivos para sensitive_group
    df = _map_group_labels(df, dataset_configs, group_labels)

    # Proporção dentro de cada (dataset, sensitive_group)
    counts = (
        df.groupby(["dataset", "sensitive_group", "target", "is_unpriv"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(["dataset", "sensitive_group"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals

    # Chave de posição: "DATASET | grupo_descritivo"
    counts["xkey"] = counts["dataset"] + " | " + counts["sensitive_group"].astype(str)

    # Ordenar: dentro de cada dataset, desprivilegiado primeiro
    counts = counts.sort_values(
        ["dataset", "is_unpriv", "sensitive_group"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    xkey_order = list(dict.fromkeys(counts["xkey"]))

    # Cores
    palette = {0: "#4E79A7", 1: "#76B7B2"}

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=counts,
        x="xkey",
        y="share",
        hue="target",
        order=xkey_order,
        palette=palette,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    # Borda preta para grupos desprivilegiados
    unpriv_keys = set(counts.loc[counts["is_unpriv"], "xkey"])
    for bar_container in ax.containers:
        for bar, xkey in zip(bar_container, xkey_order):
            if xkey in unpriv_keys:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)

    # Eixo X hierárquico
    ax.set_xticks(range(len(xkey_order)))
    # Minor labels: nome do grupo sensível
    minor_labels = [k.split(" | ")[1] for k in xkey_order]
    ax.set_xticklabels(minor_labels, fontsize=8)

    # Major labels: dataset (centralizado)
    dataset_positions = defaultdict(list)
    for i, k in enumerate(xkey_order):
        ds = k.split(" | ")[0]
        dataset_positions[ds].append(i)

    # Adicionar labels de dataset abaixo
    for ds, positions in dataset_positions.items():
        center = np.mean(positions)
        ax.text(
            center, -0.12, ds,
            ha="center", va="top", fontsize=9, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # Separadores verticais entre datasets
    prev_ds = None
    for i, k in enumerate(xkey_order):
        ds = k.split(" | ")[0]
        if prev_ds is not None and ds != prev_ds:
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        prev_ds = ds

    ax.set_xlabel("")
    ax.set_ylabel("% (Rótulo | Grupo sensível)", fontsize=10)
    ax.set_ylim(0, 1.0)

    # Legenda
    handles = [
        Patch(facecolor=palette[0], edgecolor="white", label="Rótulo 0"),
        Patch(facecolor=palette[1], edgecolor="white", label="Rótulo 1"),
        Patch(facecolor="white", edgecolor="black", linewidth=1.5, label="Desprivilegiado"),
    ]
    ax.legend(handles=handles, title="Rótulo", loc="upper right", frameon=True, fontsize=9)

    apply_comma_axes(ax)
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =========================================================================
# DISTRIBUIÇÕES DE FEATURES POR DATASET
# =========================================================================

def plot_feature_distributions(
    dataset_name: str,
    dataset_configs: Dict = None,
    data_root: Path | str = None,
    save_dir: Path | str = None,
    group_labels: Dict = None,
    figsize_cat: tuple = (5, 3),
    figsize_num: tuple = (7, 4),
    show: bool = False,
) -> None:
    """Gera figuras de distribuição de features para um dataset.

    Para cada feature:
    - Categóricas: histograma agrupado por grupo sensível
    - Numéricas: densidade por grupo sensível E por target

    Todos os gráficos são salvos em PDF.

    Parameters
    ----------
    dataset_name : str
        Nome do dataset (ex: "OBESITY").
    dataset_configs : dict, optional
        Default: ``DATASET_CONFIGS``.
    data_root : Path, optional
        Raiz dos dados processados.
    save_dir : Path, optional
        Diretório de saída para as figuras.
    group_labels : dict, optional
        Default: ``GROUP_LABELS``.
    """
    dataset_configs = dataset_configs or DATASET_CONFIGS
    group_labels = group_labels or GROUP_LABELS

    if data_root is None:
        from .config import DATA_PATH
        data_root = DATA_PATH / "processed"
    data_root = Path(data_root)

    cfg = dataset_configs[dataset_name]
    code = cfg["dataset_code"]
    sen_var = cfg["sen_var"]

    csv_path = data_root / code / f"{code}_pp.csv"
    df = pd.read_csv(csv_path)

    # Separar colunas categóricas e numéricas (excluindo target e sen_var)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    exclude = {sen_var, "target"}
    cat_cols = [c for c in cat_cols if c not in exclude]
    num_cols = [c for c in num_cols if c not in exclude]

    # Labels descritivos para o hue
    sen_labels = group_labels.get(sen_var, {})
    hue_order_raw = sorted(df[sen_var].dropna().unique().tolist())
    hue_labels = {v: sen_labels.get(v, str(v)) for v in hue_order_raw}

    # Criar coluna com label descritivo
    df["_sen_label"] = df[sen_var].map(hue_labels)
    hue_order = [hue_labels[v] for v in hue_order_raw]

    palette_2 = ["#4E79A7", "#76B7B2"]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # --- Categóricas por grupo sensível ---
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=figsize_cat)
        sns.histplot(
            data=df, x=col, hue="_sen_label", hue_order=hue_order,
            multiple="dodge", shrink=0.8, palette=palette_2, ax=ax,
        )
        ax.set_title(f"{dataset_name} — {col} por {sen_var}", fontsize=11)
        apply_comma_axes(ax)
        fig.tight_layout()
        if save_dir:
            fig.savefig(save_dir / f"{dataset_name}_cat_{col}.pdf", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # --- Numéricas por grupo sensível ---
    for col in num_cols:
        fig, ax = plt.subplots(figsize=figsize_num)
        sns.histplot(
            data=df, x=col, hue="_sen_label", hue_order=hue_order,
            stat="density", common_norm=False,
            palette=palette_2, element="step", ax=ax,
        )
        ax.set_title(f"{dataset_name} — {col} por {sen_var}", fontsize=11)
        ax.set_ylabel("Densidade")
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(1.02, 1),
            borderaxespad=0, title=sen_var.replace("_", " "), frameon=True,
        )
        apply_comma_axes(ax)
        fig.tight_layout()
        if save_dir:
            fig.savefig(save_dir / f"{dataset_name}_num_by_sens_{col}.pdf", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # --- Numéricas por target ---
    target_order = sorted(df["target"].dropna().unique().tolist())
    for col in num_cols:
        fig, ax = plt.subplots(figsize=figsize_num)
        sns.histplot(
            data=df, x=col, hue="target", hue_order=target_order,
            stat="density", common_norm=False,
            palette=palette_2, element="step", ax=ax,
        )
        ax.set_title(f"{dataset_name} — {col} por variável resposta", fontsize=11)
        ax.set_ylabel("Densidade")
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(1.02, 1),
            borderaxespad=0, title="Variável resposta", frameon=True,
        )
        apply_comma_axes(ax)
        fig.tight_layout()
        if save_dir:
            fig.savefig(save_dir / f"{dataset_name}_num_by_target_{col}.pdf", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    n_figs = len(cat_cols) + 2 * len(num_cols)
    print(f"  {dataset_name}: {n_figs} figuras geradas ({len(cat_cols)} cat, {len(num_cols)} num)")


# =========================================================================
# TABELA DESCRITIVA DOS DATASETS (estilo paper — tabela unificada)
# =========================================================================

def build_dataset_overview_table(
    combined: pd.DataFrame,
    dataset_configs: Dict = None,
    dataset_meta: Dict = None,
    group_labels: Dict = None,
    result_dir: Path | str = None,
) -> pd.DataFrame:
    r"""Tabela unificada de descrição dos datasets (volume + propriedades).

    Funde as informações de volume, rótulo favorável, atributo protegido e
    grupos privilegiado/desprivilegiado em uma única tabela. O nome do
    dataset aparece em negrito com a referência bibliográfica na linha
    seguinte (``\newline``).

    Parameters
    ----------
    combined : pd.DataFrame
        Saída de ``load_combined_data``.
    result_dir : Path, optional
        Se fornecido, salva .tex.

    Returns
    -------
    pd.DataFrame
    """
    dataset_configs = dataset_configs or DATASET_CONFIGS
    dataset_meta = dataset_meta or DATASET_META
    group_labels = group_labels or GROUP_LABELS

    rows = []
    for ds_key, cfg in dataset_configs.items():
        if not cfg.get("active", False):
            continue

        meta = dataset_meta.get(ds_key, {})
        display = meta.get("display_name", _display_name(ds_key))
        cite = meta.get("cite_key", "")
        favorable = meta.get("favorable_label", "—")

        sen_var = cfg["sen_var"]
        gl = group_labels.get(sen_var, {})

        # Privilegiado e desprivilegiado
        priv_vals = cfg["fairness_vars"].get("privileged_groups", [])
        unpriv_vals_list = cfg["fairness_vars"].get("unprivileged_groups", [])
        priv_label = ", ".join(
            gl.get(d.get(sen_var), str(d.get(sen_var, "?")))
            for d in priv_vals
        )
        unpriv_label = ", ".join(
            gl.get(d.get(sen_var), str(d.get(sen_var, "?")))
            for d in unpriv_vals_list
        )

        # Volume e taxa do rótulo favorável
        ds_display = _display_name(ds_key)
        ds_data = combined[combined["dataset"] == ds_display]
        n_total = len(ds_data)
        n_favorable = (ds_data["target"] == 0).sum()
        fav_rate = round(n_favorable / n_total * 100, 0) if n_total > 0 else 0.0

        # Nome em negrito + referência na linha de baixo
        if cite:
            name_cell = rf"\textbf{{{display}}} \newline \cite{{{cite}}}"
        else:
            name_cell = rf"\textbf{{{display}}}"

        rows.append({
            "Conjunto de dados": name_cell,
            "Rótulo favorável": favorable,
            "Taxa fav. (\\%)": f"{fav_rate:.0f}",
            "$N$": n_total,
            "Atributo protegido": sen_var.replace("_", " ").title(),
            "Privilegiado": priv_label,
            "Desprivilegiado": unpriv_label,
        })

    table = pd.DataFrame(rows)

    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        caption = format_caption(
            "Descrição dos conjuntos de dados, com definição do atributo protegido, "
            "rótulo favorável e composição amostral."
        )
        label = "table_dataset_overview"

        # p{35mm} na 1ª coluna para suportar \newline (nome + referência)
        n_other = len(table.columns) - 1
        col_format = "p{35mm}" + "l" * n_other

        latex = table.to_latex(
            index=False,
            escape=False,
            caption=caption,
            position="H",
            label=label,
            column_format=col_format,
        )

        # Substituir decimais ANTES de inserir o preâmbulo (que tem 1.4 e 3.5cm)
        from .analysis_utils import _replace_decimal_in_latex
        latex = _replace_decimal_in_latex(latex)

        # Centering + scriptsize + espaçamento entre linhas
        latex = latex.replace(
            r"\begin{tabular}",
            (r"\centering" + "\n"
             r"\scriptsize" + "\n"
             r"\renewcommand{\arraystretch}{1.4}" + "\n"
             r"\begin{tabular}"),
        )

        tex_path = result_dir / f"{label}.tex"
        tex_path.write_text(latex, encoding="utf-8")
        print(f"Tabela salva em: {tex_path}")
        print(table.to_markdown(index=False))

    return table


# Alias de compatibilidade
build_dataset_description_table = build_dataset_overview_table
