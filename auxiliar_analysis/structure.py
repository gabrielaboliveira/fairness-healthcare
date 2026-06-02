from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
import json

from graphviz import Digraph


@dataclass
class Node:
    id: str
    label: str
    parent: Optional[str] = None
    kind: str = "section"  # root | section | rq | artifact | appendix
    done: bool = False  # checklist state


def render_tree(nodes: List[Node], out_path: str, fmt: str = "png") -> str:
    dot = Digraph("rq_results_tree", format=fmt)

    # Layout
    dot.attr(rankdir="LR", splines="ortho", bgcolor="white", fontname="Helvetica")
    dot.attr(
        "node",
        fontname="Helvetica",
        fontsize="10",
        style="rounded,filled",
        penwidth="1",
    )
    dot.attr("edge", color="#9aa0a6", arrowsize="0.7")

    shape_map = {
        "root": "oval",
        "section": "folder",
        "rq": "oval",
        "artifact": "box",
        "appendix": "component",
    }

    for n in nodes:
        prefix = "✅ " if n.done else "⬜ "
        fill = "#d1fae5" if n.done else "#f3f4f6"  # green-100 / gray-100
        border = "#10b981" if n.done else "#9ca3af"  # green-500 / gray-400
        shape = shape_map.get(n.kind, "box")

        dot.node(
            n.id, label=prefix + n.label, fillcolor=fill, color=border, shape=shape
        )

    for n in nodes:
        if n.parent:
            dot.edge(n.parent, n.id)

    return dot.render(out_path, cleanup=True)


def save_tree(nodes: List[Node], path: str) -> None:
    Path(path).write_text(
        json.dumps([asdict(n) for n in nodes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tree(path: str) -> List[Node]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Node(**d) for d in data]


def set_done(nodes: List[Node], node_id: str, done: bool = True) -> None:
    for n in nodes:
        if n.id == node_id:
            n.done = done
            return
    raise KeyError(f"Nó '{node_id}' não encontrado.")


# ---------------------------
# EXEMPLO: estrutura RQ-first
# (edite labels/itens à vontade)
# ---------------------------
nodes = [
    Node("R", "Resultados (estrutura por Research Questions)", kind="root"),
    Node(
        "R0",
        "Guia de leitura (unidade=dataset; gap; Δ; sinal vs magnitude)",
        parent="R",
        kind="section",
    ),
    Node(
        "R0_T1",
        "Tabela R0.1 — Convenções e regras de agregação",
        parent="R0",
        kind="artifact",
    ),
    Node(
        "R0_F1",
        "Figura R0.1 — Visão geral do fluxo analítico (1 diagrama)",
        parent="R0",
        kind="artifact",
    ),
    Node("B", "Baseline: diagnóstico + triagem (gating)", parent="R", kind="section"),
    Node(
        "B_F1",
        "Figura B.1 — Distribuição de disparidades no baseline (A/B/C)",
        parent="B",
        kind="artifact",
    ),
    Node(
        "B_T1",
        "Tabela B.1 — Regra de gating + cobertura (N por estrato)",
        parent="B",
        kind="artifact",
    ),
    Node(
        "RQ1",
        "RQ1 — Quais estratégias têm melhor efeito (global)?",
        parent="R",
        kind="rq",
    ),
    Node(
        "RQ1_F1",
        "Figura 1 — Δfairness por família (pre vs in) no conjunto com disparidade",
        parent="RQ1",
        kind="artifact",
    ),
    Node(
        "RQ1_T1",
        "Tabela 1 — Ranking (famílias/métodos) + Friedman (omnibus)",
        parent="RQ1",
        kind="artifact",
    ),
    Node(
        "RQ2",
        "RQ2 — Melhor efeito por métrica/categoria (A/B/C)?",
        parent="R",
        kind="rq",
    ),
    Node(
        "RQ2_F1",
        "Figura 2 — Δfairness por categoria A/B/C (facetas)",
        parent="RQ2",
        kind="artifact",
    ),
    Node(
        "RQ2_T1",
        "Tabela 2 — Top-3 métodos por categoria (resumo)",
        parent="RQ2",
        kind="artifact",
    ),
    Node("RQ3", "RQ3 — Trade-off desempenho↔fairness", parent="R", kind="rq"),
    Node(
        "RQ3_F1",
        "Figura 3 — Scatter: Δdesempenho (x) vs Δfairness (y)",
        parent="RQ3",
        kind="artifact",
    ),
    Node(
        "RQ3_T1",
        "Tabela 3 — Métodos não-dominados (Pareto) / regras de decisão",
        parent="RQ3",
        kind="artifact",
    ),
    Node(
        "RQ4",
        "RQ4 — Sentido do viés (gap assinado) altera mitigação?",
        parent="R",
        kind="rq",
    ),
    Node(
        "RQ4_F1",
        "Figura 4 — Δgap estratificado por sinal do baseline",
        parent="RQ4",
        kind="artifact",
    ),
    Node(
        "RQ5",
        "RQ5 — Intensidade: mitigação proporcional vs teto/piso?",
        parent="R",
        kind="rq",
    ),
    Node(
        "RQ5_F1",
        "Figura 5 — |gap|_baseline vs Δ|gap| (por método/família)",
        parent="RQ5",
        kind="artifact",
    ),
    Node("APX", "Apêndices (catálogos e detalhes)", parent="R", kind="appendix"),
    Node(
        "APX_A",
        "Apêndice A — Catálogo por dataset×método×métrica (tabelas completas)",
        parent="APX",
        kind="artifact",
    ),
    Node(
        "APX_B",
        "Apêndice B — Subpadrões A1/A2/... (catálogo + cobertura)",
        parent="APX",
        kind="artifact",
    ),
    Node(
        "APX_C",
        "Apêndice C — Boxplots detalhados e sensibilidade",
        parent="APX",
        kind="artifact",
    ),
    Node(
        "APX_D",
        "Apêndice D — Holm completo / comparações adicionais",
        parent="APX",
        kind="artifact",
    ),
]

# Exemplo: marcar 2 itens como concluídos
set_done(nodes, "R0_T1", True)
set_done(nodes, "B_F1", True)

# Salvar JSON editável + renderizar
save_tree(nodes, "rq_results_tree.json")
render_tree(nodes, out_path="rq_results_tree", fmt="png")  # gera rq_results_tree.png
