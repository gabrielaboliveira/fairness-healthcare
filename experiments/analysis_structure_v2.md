# Proposta de análise

## 0) Metodologia

### Unidade de análise e agregação

- **Unidade de análise:** dataset. Resultados de runs (folds/repetições) são agregados **internamente** para obter um único valor por *(dataset, método, métrica)*.
- **Isso evita pseudo-replicação:** cada dataset contribui com um único ponto para análises estatísticas.

### Referência ao ideal (padronização de métricas)

Para cada métrica `m`, define-se um valor ideal `m*` e reporta-se duas quantidades, conforme a pergunta de pesquisa:

- **Distância ao ideal (magnitude):**

    $$
    d(m) = |m - m^*|
    $$

    - **Uso principal:** corpo do texto para evitar cancelamento de sinais em agregações.
    - **Interpretação:** quanto menor, mais próximo do ideal (mais justo).
    - **Nota:** para métricas de razão com ideal ≠ 0 e assimetria em torno do ideal (e.g., Impacto Díspar, ideal = 1.0), usa-se distância multiplicativa: $d(m) = \max(m/m^*, m^*/m) - 1$, que trata simetricamente desvios acima e abaixo do ideal (e.g., DI = 0.8 e DI = 1.25 resultam ambos em $d = 0.25$).
- **Diferença ao ideal (assinada):**

    $$
    \delta(m) = m - m^*
    $$

    - **Uso específico:** análises de **direção/sentido do viés** (favorece privilegiados vs. não-privilegiados).
    - **Interpretação:** o sinal indica o sentido da disparidade.

---

### Efeito da mitigação (em relação ao baseline)

- **Efeito absoluto em fairness:**

    $$
    \Delta_{\text{fair}} = d_{\text{baseline}} - d_{\text{método}}
    $$

    - Valores **positivos** = redução da distância ao ideal = **melhora em fairness**.
    - Valores **negativos** = aumento da distância ao ideal = piora em fairness.
- **Efeito relativo em fairness (fração removida; usado na RQ4):**

    $$
    r_{\text{fair}} = \frac{\Delta_{\text{fair}}}{d_{\text{baseline}}+\epsilon}= 1 - \frac{d_{\text{método}}}{d_{\text{baseline}}+\epsilon}
    $$

    - **Interpretação:** aproxima a **redução percentual** da distância ao ideal em relação ao baseline (ex.: `r=0.20` ≈ 20% de redução).
    - **Motivação (RQ4):** como $\Delta_{\text{fair}}$ tende a crescer com a magnitude inicial do viés (há mais "espaço" para reduzir quando $d_{\text{baseline}}$ é grande), $r_{\text{fair}}$ normaliza o efeito pelo tamanho do problema inicial, permitindo avaliar **proporcionalidade vs. saturação** da mitigação ao longo de diferentes severidades.
    - **Nota prática:** $\epsilon$ é um termo pequeno ($10^{-8}$) para estabilidade numérica quando $d_{\text{baseline}} \approx 0$; adicionalmente, casos com severidade abaixo do limiar de gating são excluídos da análise.
- **Efeito em desempenho (F1):**

    $$
    \Delta_{F1} = F1_{\text{método}} - F1_{\text{baseline}}
    $$

    - Valores **positivos** = ganho de desempenho.
    - Valores **negativos** = perda de desempenho.

**Nota sobre agregação:** a distância ao ideal é calculada sobre o valor já agregado ao nível de dataset (primeiro agrega-se o valor bruto da métrica via mediana sobre folds/repetições; depois computa-se a distância ao ideal sobre o valor agregado).

### Baseline típico

- **Definição:** referência principal para comparação, obtida pela combinação dos quatro modelos base:
    - LR aware
    - LR unaware
    - RF aware
    - RF unaware
- **Justificativa:** representa a diversidade de práticas comuns em ML (algoritmo × feature engineering).
- **Agregação interna:** runs são agregados por dataset (mediana sobre folds/repetições e os 4 baselines) para obter um único valor por *(dataset, métrica)*.
- **Validação:** componentes individuais são comparados na RQ0.1 (corpo) e detalhados no apêndice.

### Notas sobre métricas específicas

- **DFBA (Bias Amplification):** esta métrica mede a diferença de EDF (*empirical differential fairness*) entre o classificador e o dataset original. Valores positivos indicam que o classificador **amplificou** o viés já presente nos dados; valores negativos indicam redução. Como o interesse é medir **amplificação** (viés introduzido pelo modelo), o ideal é $\leq 0$. Na prática, para o cálculo da distância ao ideal, valores negativos são truncados a zero ($d = \max(m, 0)$) em vez de usar $|m|$, pois valores negativos não representam disparidade — indicam que o modelo reduziu o viés preexistente. Além disso, DFBA **não possui sentido de viés (U/P)**, pois compara classificador vs. dados, não grupo privilegiado vs. não-privilegiado.

### Dimensões de estratificação (ortogonais entre si)

A análise é organizada em **3 dimensões independentes**:

| Dimensão | O que representa | Valores | Aplicável a | Analisada em |
| --- | --- | --- | --- | --- |
| **Categoria de método** | Estratégia de mitigação (família do método) | *pre / in / post* (ou outra taxonomia adotada) | Todos os métodos | **RQ1** |
| **Categoria de métrica (A/B/C)** | Natureza operacional da métrica de fairness | A (paridade), B (individual), C (matriz de confusão) | Todas as métricas | **RQ2** |
| **Sentido (U/P)** | Direção da disparidade no baseline (sinal de $\delta(m)$) | U (favorece não-privilegiados), P (favorece privilegiados) | Apenas **A e C** (métricas de grupo com sentido definido) | **RQ3** |
| **Severidade (1/2/3)** | Magnitude do desvio no baseline (em unidades de IQR) | 1 (alta), 2 (média), 3 (baixa) | Todas as métricas (dentro de cada categoria) | **RQ4** |
| **Trade-off (quadrantes/Pareto)** | Relação entre utilidade e fairness | quadrantes / Pareto-ótimos | Métodos + métricas (casos elegíveis) | **RQ5** |

> **Convenção de nomenclatura:** usamos **"severidade"** como termo técnico para a classificação discreta em faixas (1=alta / 2=média / 3=baixa), onde nível 1 é o mais severo. O termo **"intensidade"** é usado informalmente como sinônimo para a magnitude contínua do desvio ($d_{\text{baseline}}$ ou $u$).

**Exemplos de categorias de métrica:**

- **A (baseado em paridade):** Impacto Díspar, DFBA
- **B (individual):** Consistência, Índice de Entropia Generalizado
- **C (baseado em matriz de confusão):** Diferença de FDR, Diferença de FNR, Diferença de Taxa de Erro

**Subpadrões:** combinação de categoria + severidade + sentido (ex.: A1U = métrica de paridade, severidade alta, favorece não-privilegiados). Usados principalmente para catalogação no apêndice; análises principais separam dimensões.

### Política de inferência estatística

- Testes (Friedman/Holm) são reportados apenas em níveis com **cobertura suficiente no nível de dataset** (N adequado).
- Segmentações com baixo (N) são tratadas como **descritivas** e detalhadas no apêndice.
- **Gating:** regra de triagem para identificar "onde há disparidade relevante" (definida na RQ0.1); aplicada seletivamente conforme RQ.

### Organização corpo vs apêndice

- **Corpo:** análises principais respondendo às RQs, usando agregações e visualizações sintéticas.
- **Apêndice:** catálogos completos (dataset×método×métrica), subpadrões finos, testes estatísticos detalhados, análises de sensibilidade.
- **Regra:** se um resultado **contradiz ou nuança fortemente** conclusão do corpo, permanece no corpo; se é apenas completude, vai para apêndice.

---

## **RQ0.1 — Fairness no baseline: diagnóstico, definição de severidade e triagem (gating)**

**Objetivo:** Caracterizar as disparidades observadas no **baseline típico** e estabelecer, de forma **pré-mitigação**, (i) uma medida padronizada de **severidade** do viés e (ii) uma regra de **triagem (gating)** para garantir sensibilidade nas análises subsequentes (incluindo segmentações em RQ2 e o estudo de intensidade em RQ4).

**Análises:**

- **Diagnóstico do baseline típico**
    - Descrever a distribuição das métricas de *fairness* no baseline típico.
    - Reportar resultados principalmente via **distância ao ideal** $d(m)$ (e $\delta(m)$ quando necessário para direção/sentido).
    - Estatísticas descritivas por métrica/categoria (mediana, IQR, caudas/outliers).
- **Baseline típico vs. componentes**
    - Comparar baseline típico com os 4 baselines componentes (resumo no corpo; detalhamento no apêndice).
    - Verificar se o baseline típico é representativo e identificar eventuais baselines outliers por métrica/dataset.
- **Definição formal das faixas de severidade (unidades de IQR)**
    - Para cada métrica $m$, calcular a distância ao ideal no baseline típico para cada dataset $d$:

        $$
        d^{(\text{baseline})}_{m,d} = |m_{d}^{(\text{baseline})} - m^*|
        $$

    - Estimar a escala robusta **entre datasets**, calculada **exclusivamente sobre o baseline** (pré-mitigação):

        $$
        IQR_m = Q_{0.75}\bigl(\{d^{(\text{baseline})}_{m,d}\}\bigr) - Q_{0.25}\bigl(\{d^{(\text{baseline})}_{m,d}\}\bigr)
        $$

        > **Pré-condição:** o cálculo do IQR deve usar apenas dados do baseline típico, sem incluir métodos de mitigação, para que a régua de severidade não seja contaminada pelo efeito dos tratamentos.

    - Converter para **unidades de IQR** (severidade padronizada):

        $$
        u_{m,d}=\frac{d^{(\text{baseline})}_{m,d}}{IQR_m+\epsilon}
        $$

    - Definir severidade por limiares fixos (interpretáveis):
        - **Nível 1 (alta):** $u \ge 1.5$
        - **Nível 2 (média):** $1.0 \le u < 1.5$
        - **Nível 3 (baixa):** $0.5 \le u < 1.0$
        - **Inelegível:** $u < 0.5$
- **Regra de gating (relevância pré-mitigação)**
    - Definir como elegíveis para análises de mitigação os casos com severidade mínima ($u \ge 0.5$), evitando cenários com desvio muito pequeno/ruído.
    - **Reportar cobertura do gating:** quantos *(datasets, métricas)* permanecem elegíveis, globalmente e por categoria A/B/C.

**Entrega:** (1) mapa de disparidades no baseline; (2) validação do baseline típico; (3) definição completa e reprodutível de severidade (unidades de IQR) e gating que será reutilizada nas demais RQs.

## **1.1) RQ0.2 — Baseline em desempenho preditivo (F1): diagnóstico curto**

**Objetivo:** Contextualizar nível absoluto de desempenho antes de analisar trade-offs.

**Análises:**

- Nível absoluto e variabilidade do F1 no baseline típico (por dataset).
- **Não compara métodos de mitigação aqui** (isso fica para RQ3).

**Entrega:** contexto para interpretar se perdas/ganhos em F1 são substanciais.

---

## **2) RQ1 — Quais categorias/métodos têm melhor efeito global?**

**Objetivo:** Panorama geral de eficácia **sem estratificação** (todas as métricas×datasets).

**Análises:**

- **($\Delta_{\text{fair}}$) por categoria de método** (pre vs in-processing):
    - Uma visualização por métrica (painéis compactos se possível).
    - Win rate: proporção de casos em que ($\Delta_{\text{fair}} > 0$).
- **($\Delta_{\text{fair}}$) por método individual** (todos os métodos):
    - Uma visualização por métrica.
    - Ranking + teste de Friedman (se N suficiente).
    - Holm completo no apêndice.

**Importantes:**

- **SEM gating aqui:** análise global inclui todos os datasets (casos triviais e severos).  Ou seja, entram datasets inelegíveis.
- **SEM agregação entre métricas:** cada métrica é analisada separadamente; win rate trata todas igualmente ("contagem democrática").
- **Análises informais a posteriori:** possível fazer win rate intra-categoria (A/B/C) sem pré-codificar na estrutura.

**Entrega:** identificação de métodos/categorias mais eficazes em média, reconhecendo que podem haver moderadores (investigados nas RQs seguintes).

### Visualizações exemplo

![image.png](image.png)

| metric_label | category | n | wins | win_rate_pct |
| --- | --- | --- | --- | --- |
| DFBA | in_processing | 6 | 5 | 83.3 |
| DFBA | pre_processing | 6 | 2 | 33.3 |
| Consistência | in_processing | 6 | 4 | 66.7 |
| Consistência | pre_processing | 6 | 4 | 66.7 |
| Impacto Díspar | in_processing | 6 | 6 | 100 |
| Impacto Díspar | pre_processing | 6 | 3 | 50 |
| Diferença de Taxa de Erro | in_processing | 6 | 2 | 33.3 |
| Diferença de Taxa de Erro | pre_processing | 6 | 2 | 33.3 |
| Diferença de Taxa de Falsa Descoberta | in_processing | 6 | 0 | 0 |
| Diferença de Taxa de Falsa Descoberta | pre_processing | 6 | 3 | 50 |
| Diferença de Taxa de Falsos Negativos | in_processing | 6 | 4 | 66.7 |
| Diferença de Taxa de Falsos Negativos | pre_processing | 6 | 2 | 33.3 |
| Índice de Entropia Generalizada | in_processing | 3 | 0 | 0 |
| Índice de Entropia Generalizada | pre_processing | 3 | 2 | 66.7 |
|  |  |  |  |  |

ðŸ“Š Teste Friedman (omnibus)
N datasets: 6, k mÃ©todos: 3
EstatÃ­stica = 9.0000, p = 0.0111 *
EstatÃ­sticas descritivas:

Categoria Median(raw) IQR(raw) Median(gap) IQR(gap) N

baseline_typical     1.3255       1.1534     0.5401       0.4415     6
pre_processing       1.4593       1.0613     0.7305       0.4763     6
in_processing        1.1478       0.4974     0.2580       0.2199     6

ðŸ“Š PÃ³s-hoc vs baseline_typical (Holm)
(* p<0.05, ** p<0.01, *** p<0.001)

| category | avg_rank | delta_rank | z | p_raw | p_holm | reject_holm | median_raw | median_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pre_processing | 2.5 | 0 | 0 | 0.5000 | 0.5000 |  | 1.459 | 0.731 |
| in_processing | 1 | -1.5 | -2.598 | 0.0047 ** | 0.0094 ** | âœ“ | 1.148 | 0.258 |

![image.png](image%201.png)

ðŸ“Š Teste Friedman (omnibus)
N datasets: 6, k mÃ©todos: 8
EstatÃ­stica = 22.4195, p = 0.0021 **
EstatÃ­sticas descritivas:

MÃ©todo Median(raw) IQR(raw) Median(gap) IQR(gap) N

baseline_typical     1.3255       1.1534     0.5401       0.4415     6
EDL                  1.4593       1.0613     0.7305       0.4763     6
IAD                  1.2201       0.9620     0.4183       0.3605     6
IGLA                 1.0754       0.3474     0.1779       0.2739     6
IGLU                 1.0933       0.4319     0.1808       0.2531     6
IFG                  1.1994       0.6496     0.4782       0.1554     6
IP                   1.1983       0.4781     0.2982       0.1904     6
IW                   1.1152       0.3351     0.2341       0.1064     6

ðŸ“Š PÃ³s-hoc vs baseline_typical (Holm)
(* p<0.05, ** p<0.01, *** p<0.001)

| method | category | avg_rank | delta_rank | z | p_holm | reject_holm | median_raw | median_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EDL | pre_processing | 6.333 | 0 | 0 | 1.0000 |  | 1.459 | 0.731 |
| IAD | in_processing | 5.833 | -0.5 | -0.354 | 1.0000 |  | 1.22 | 0.418 |
| IGLA | in_processing | 2.417 | -3.917 | -2.77 | 0.0196 * | âœ“ | 1.075 | 0.178 |
| IGLU | in_processing | 2.417 | -3.917 | -2.77 | 0.0196 * | âœ“ | 1.093 | 0.181 |
| IFG | in_processing | 6 | -0.333 | -0.236 | 1.0000 |  | 1.199 | 0.478 |
| IP | in_processing | 3 | -3.333 | -2.357 | 0.0461 * | âœ“ | 1.198 | 0.298 |
| IW | in_processing | 3.667 | -2.667 | -1.886 | 0.1187 |  | 1.115 | 0.234 |

## **3) RQ2 — Há categoria/método melhor por tipo de métrica (A/B/C)?**

**Objetivo:** Avaliar se o efeito de mitigação varia conforme a **categoria de métrica de fairness** (A: paridade, B: individual, C: matriz de confusão), isto é, se certas **categorias de método** são sistematicamente mais eficazes em determinados tipos de métrica.

**Análises:**

- Facetas A/B/C para ($\Delta_{\text{fair}}$):
    - **Aqui pode aplicar gating** dentro de cada categoria para focar em disparidades relevantes.
- Top métodos por categoria A/B/C (resumo).
- Teste se diferenças entre A/B/C são estatisticamente significativas (se N suficiente).

**Entrega:** Determinar se a mitigação é sensível ao tipo de métrica — por exemplo, se determinadas **categorias de método** tendem a ser mais eficazes em métricas de paridade (A) do que em métricas de matriz de confusão (C), e se métricas individuais (B) exibem padrão distinto.

## **4) RQ3 — O sentido do viés altera a mitigação?**

**Objetivo:** Avaliar se a **direção da disparidade no baseline** (favorece privilegiados vs. não-privilegiados) está associada a diferenças sistemáticas na eficácia de mitigação.

**Análises:**

- **Estratificação por U/P (baseline)**
    - Classificar cada par *(dataset, métrica)* como **U** ou **P** a partir do sinal de $(\delta(m)=m-m^*$), **apenas para métricas de grupo** (categorias **A e C**), onde o sentido é interpretável.
    - **Importante:** U/P é propriedade do *(dataset × métrica)* no baseline, não do método de mitigação.
- **Efeito de mitigação por sentido**
    - Comparar a distribuição do **efeito absoluto em fairness** ($\Delta_{\text{fair}} = d_{\text{baseline}} - d_{\text{método}}$) entre U vs P, mantendo **A vs C** como contexto (ex.: facetas ou cores).
    - Figura: ($\Delta_{\text{fair}}$) por **U** vs **P** (por método ou por categoria de método), com resumos robustos (mediana/IQR).
- **Evidência estatística (se N suficiente)**
    - Testar diferença entre U e P dentro de A e C (e/ou por categoria de método), reportando tamanho de efeito e incerteza.

**Entrega:** Indicar se (e em quais métricas/categorias) a mitigação tende a ser mais eficaz quando o viés inicial favorece privilegiados (P) versus não-privilegiados (U), e se esse padrão depende da categoria de método.

## **RQ4 — Intensidade do viés: a mitigação escala com a severidade inicial?**

**Objetivo:** Verificar se a eficácia **relativa** de mitigação depende da **severidade inicial** do viés no baseline (definida em RQ0.1), distinguindo padrões de **proporcionalidade**, **indiferença** e **saturação**.

**Pré-requisitos (definidos em RQ0.1):**

- Severidade $u_{m,d}$ (unidades de IQR) e faixas nível 1 (alta) / 2 (média) / 3 (baixa).
- Gating: considerar apenas casos elegíveis ($u \ge 0.5$).

**Métrica principal — efeito relativo ($r_{\text{fair}}$):**

A análise de escalabilidade usa o **efeito relativo** como métrica principal, pois $\Delta_{\text{fair}}$ (efeito absoluto) tende mecanicamente a crescer com $d_{\text{baseline}}$ — há mais "espaço" para reduzir quando o gap inicial é grande. O efeito relativo normaliza pelo tamanho do problema:

$$
r_{\text{fair}} = \frac{\Delta_{\text{fair}}}{d_{\text{baseline}}+\epsilon}= 1 - \frac{d_{\text{método}}}{d_{\text{baseline}}+\epsilon}
$$

- $r_{\text{fair}} = 1$: eliminação completa do viés.
- $r_{\text{fair}} = 0$: nenhum efeito.
- $r_{\text{fair}} < 0$: piora (método aumentou o viés).

**Análises:**

Para cada *(métrica)*, com unidade de análise = dataset, agrupando por método ou categoria de método:

1. **Visualização principal: perfil de severidade (line plot)**
    - Eixo X: faixa de severidade (3=baixa → 2=média → 1=alta).
    - Eixo Y: $r_{\text{fair}}$ (mediana por faixa).
    - Uma linha por método (ou por categoria de método).
    - **Leitura direta:** a inclinação da linha indica o padrão de escalabilidade.

2. **Complemento: scatter contínuo**
    - Eixo X: $d_{\text{baseline}}$ (distância absoluta ao ideal — magnitude contínua).
    - Eixo Y: $r_{\text{fair}}$.
    - Cor dos pontos: faixa de severidade (codificação via unidades de IQR).
    - Linha de tendência (loess ou regressão) por método/categoria.
    - **Interpretação:** permite observar a forma funcional (linear, côncava, etc.) além das faixas discretas.

3. **Diagnóstico de escalabilidade**
    - **Proporcionalidade:** $r_{\text{fair}}$ cresce com severidade — o método remove fração maior do viés em cenários mais severos.
    - **Indiferença:** $r_{\text{fair}}$ aproximadamente constante — o método remove fração fixa independentemente da severidade.
    - **Saturação:** $r_{\text{fair}}$ achata ou decresce em alta severidade — o método tem teto de eficácia relativa.

4. **Contexto (efeito absoluto no apêndice):**
    - $\Delta_{\text{fair}}$ por faixa de severidade é reportado como complemento descritivo, mas **não** é a métrica principal de escalabilidade (por ser mecanicamente correlacionada com $d_{\text{baseline}}$).

**Entrega:** concluir, por método (e por família/categoria), se a mitigação é **proporcionalmente** mais efetiva em cenários de alta severidade, se é indiferente à severidade ou se apresenta saturação.

## **6) RQ5 — Trade-off desempenho↔fairness**

**Objetivo:** Sintetizar custo/benefício em termos de desempenho preditivo vs ganho em fairness.

**Análises:**

- Scatter: ($\Delta_{F1}$) (eixo x) vs ($\Delta_{\text{fair}}$) (eixo y), baseado no baseline típico.
- Quadrantes:
    - **Q1** (+F1, +fair): win-win.
    - **Q2** (-F1, +fair): trade-off clássico.
    - **Q3** (-F1, -fair): lose-lose.
    - **Q4** (+F1, -fair): melhora desempenho mas piora fairness.
- Fronteira de Pareto: identificar soluções não-dominadas.
- Regra de decisão prática: dado contexto/preferências, qual método escolher.

**Importante:**

- **Esta seção vem APÓS** RQ2/RQ4/RQ5, para que leitor já conheça moderadores.
- Referência explícita: "tabelas completas de F1 por método×dataset no Apêndice X".
- **Possibilidade futura** (não obrigatória agora): estratificar trade-off por categoria/sentido/intensidade (mas pode ficar extenso).

**Entrega:** síntese final com recomendações práticas.

---

## **Fluxo narrativo das RQs**

```
RQ0.1/RQ0.2 (diagnóstico e calibração)
  ↓
RQ1 (eficácia global, sem estratificação)
  ↓
RQ2 (tipo de métrica: A/B/C)
  ↓
RQ3 (sentido do viés: U/P em A/C)
  ↓
RQ4 (severidade: escalabilidade via r_fair)
  ↓
RQ5 (trade-off desempenho ↔ fairness)

```

**Progressão:** baseline → panorama geral → natureza da métrica → direção do viés → magnitude do viés → decisão integrada.