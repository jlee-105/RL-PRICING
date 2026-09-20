# Required references (multi-project CG / B&P paper)

Updated 2026-09-14. Bib file: `rl_pricing_references.bib`.
Status: **PDF** = need the full text to check details; **bib** = entry missing or wrong; **ok** = bib entry present and checked.

---

## 1. Must read (full text needed)

These decide the related work, the model comparison table (`tab:pd-compare` in `problem_definition.tex`), and our contribution claims.

| # | Reference | Why | Status |
|---|---|---|---|
| 1 | Hu, X., Liu, S., Demeulemeester, E., Zhou, Z. (2026). *Multi-project scheduling and material ordering with time varying supplier capacities.* Computers & Operations Research 188, 107362. | Closest model (multi-project + material ordering + time-varying supplier capacity). Check: multi-mode? objective terms (setup, delay cost)? renewable capacity time-varying? instances public? | **PDF** (bib fixed) |
| 2 | Hu, X., Liu, S., Demeulemeester, E., Zhou, Z. (2026). *A two-stage robust optimisation model for multi-project scheduling and material ordering under demand uncertainty.* J. Operational Research Society. doi:10.1080/01605682.2026.2651219 | Exact methods (C&CG, Benders) on the robust version. Check: instance sizes, first/second-stage split, what is uncertain, solve times. | **PDF**, **bib** (key `hu2026robust` exists, verify fields) |
| 3 | Cheng, J., Fowler, J. W., Kempf, K. G., Mason, S. J. (2015). *Multi-mode resource-constrained project scheduling problems with non-preemptive activity splitting.* Computers & Operations Research 53, 275–287. | Base model we extend (time-varying capacity). Check: exact splitting definition, solution method, instance sizes. TODO comment in `problem_definition.tex`. | **PDF** (bib added) |
| 4 | Cited as `ieee2013semircpsp`: Install/Qual as stochastic MRCPSP, simulated annealing + Monte Carlo (Cheng/Fowler group, 2013). | Motivation and three-phase process. **Exact title unknown** — search "Cheng Fowler Kempf 2013 install qualification simulated annealing". | **PDF**, **bib** |
| 5 | Kolter, M., Grunow, M., Kolisch, R. (2025). *On Branch-and-Price for Project Scheduling.* | Closest to our method (B&P for project scheduling). Must check before claiming C3 (branching that leaves pricing unchanged). | **PDF** (bib present, check fields) |
| 6 | Abouelrous, A., Bliek, L., Gabor, A. F., Wu, Y., Zhang, Y. (2025). *Reinforcement learning for solving the pricing problem in column generation: Applications to vehicle routing.* arXiv:2504.02383 (v2, Aug 2025; "preprint submitted to N/A"). This is POMO-CG, cited as `pomocg2024`. | **Read (arXiv v2).** RL (attention/POMO) solves the ESPPRC pricing of VRPTW directly; duals sampled at random for training; **solves the root LP relaxation only** — no branch-and-price, no certified bounds, no integer solutions. Claims to be the first ML model that solves the pricing problem directly. → "RL replaces pricing" is not new; our claim must be learned pricing inside full B&P with certified bounds. Check for a peer-reviewed version before citing. | **bib** (fix key/year) |
| 7 | Václavík, R., Novák, A., Šůcha, P., Hanzálek, Z. (2018). *Accelerating the branch-and-price algorithm using machine learning.* European Journal of Operational Research 271, 1055–1069. | ML **inside B&P** (found in POMO-CG's references). Must read before claiming C3. | **PDF**, **bib** |

### Findings from the five PDFs (read 2026-09-14; files in project root)

**Kolter, Grunow, Kolisch — *On Branch-and-Price for Project Scheduling*, arXiv:2501.04563 (Jan 2025)** (`Kolter.pdf`)
- Studies MRCMPSP-GO: multi-mode, **multi-project**, general time-indexed cost c_ijmt (covers weighted tardiness), time-indexed PDDT formulation. Same setting as ours minus procurement; says its results also apply to non-renewable resources.
- **Decomposition by projects is Deckro et al. (1991)**, EJOR 51, 110–118 → our C2 is not new.
- Thm 2 / Prop 2: pricing without resource constraints, single mode → integrality property → DW bound = compact LP bound. Multi-mode (Prop 3) breaks integrality, but on 200 MMLIB instances the DW bound is only **+0.42%** over PDDT-LP and 26× slower; compact + solver cuts is stronger and faster.
- Prop 1: master is primally degenerate; one j30 instance needed 1,765 columns for the root LP. Dual stabilization is almost never used in project scheduling.
- Prop 5: branching on start times gives the same tree as B&B on the compact model. Prop 6: branching on resource demands strengthens the bound but makes pricing NP-hard.
- Resource-constrained pricing (dedicated resources, Coughlan et al. 2015): stronger bounds, but NP-hard pricing is the bottleneck; "Alleviating this bottleneck would require an efficient approach for solving the pricing problem" (p. 27).
- Recommendation: do not use the literature's CG approaches for classical project scheduling; look for formulations with hard pricing problems that have efficient solution approaches, and handle degeneracy (stabilization). Criticizes papers that do not benchmark against a compact model solved by a commercial solver.
- **Consequence for us**: B&P with exact pricing is unlikely to beat the compact MIP (our P=3 runs agree: MIP 3.7–110 s, B&P > 280 s unsolved). Possible angle: our pricing already has per-project capacity (resource-constrained, stronger bound); a learned pricer is exactly "an efficient approach for solving the pricing problem". Any claim needs a fair benchmark against the compact MIP with a solver.

**Václavík, Novák, Šůcha, Hanzálek — EJOR 271 (2018) 1055–1069** (`ejor-main.pdf`)
- ML inside B&P, but it does **not** generate columns: an online regression model predicts an upper bound on the pricing objective to prune the exact pricing solver; exactness kept. Nurse rostering and TDM scheduling; 40% / 22% CPU reduction.
- → we cannot claim "first ML inside B&P"; at most "first learned pricer that generates the columns inside a full B&P".

**Hu, Liu, Demeulemeester, Zhou — C&OR 188 (2026) 107362** (`Hu at al-main.pdf`)
- Model (eqs. 1–12): **single mode**; renewable R_pk and inventory I_pmt are **per project** ("local resources, precluding inter-project transfer"); projects coupled only by time-varying supplier capacity C_st; supplier selection; supplier lead times; material consumed every processing period; period-based ordering; objective = w_p × completion time + ordering + purchase + holding.
- Method: PHHA (priority rules + GA + local search + SA) with an activity list **and a delay list** (like our delay action). CPLEX 600 s on small instances (P = 2–3, 4–6 activities) already fails for 20 of 108. Large instances (P = 2, 5, 10; 30/60/120 activities from PSPLIB) compared only against another metaheuristic (HIA): **no bounds at scale**.
- Instances public: https://github.com/LouisResearch/MPSMOP-Instances
- → C1 is back: shared renewable resources, shared stock, and multiple modes are all absent from their model.

**Cheng, Fowler, Kempf, Mason — C&OR 53 (2015) 275–287** (`Cheng-fowler-main.pdf`)
- Deterministic. Eqs. (1)–(14): processing indicators x_jt^m, mode indicators y_j^m; activity ready times and due dates (hard); time-varying renewable limits U_kt; non-renewable total over processing periods; makespan. P1 no splitting / P2 non-preemptive splitting / P3 preemption.
- Exact: modified precedence-tree B&B + priority-rule initial solutions. 1,538 instances, 12 or 16 activities, 3 modes; some not solved within 1 h. Future work: heuristics for 10–50, 50+, 500+ activities.
- Their ref. [9]: Cheng, Fowler, Kempf, Mason (2013), *Heuristic-based scheduling algorithms for the semiconductor capital equipment installation/qualification process*, working paper — possibly what `ieee2013semircpsp` refers to; verify.

**POMO-CG — Abouelrous et al., arXiv:2504.02383** (`POMO-CG.pdf`): see the positioning table below.

### Search priority

Done: Kolter (#5), Václavík (#7), Hu C&OR (#1), Cheng 2015 (#3), POMO-CG (#6).
Still missing:
1. Hu et al. JORS 2026 (#2) — exact C&CG/Benders on the robust version; instance sizes.
2. Deckro, Winkofsky, Hebert, Gagnon (1991), *A decomposition approach to multi-project scheduling*, EJOR 51, 110–118 — origin of our decomposition.
3. Coughlan, Lübbecke, Schulz (2015), *A branch-price-and-cut algorithm for multi-mode resource leveling*, EJOR 245, 70–80 — resource-constrained pricing, B&P.
4. The 2013 Install/Qual paper (`ieee2013semircpsp`, #4).

### Positioning against POMO-CG (read 2026-09-14)

"RL replaces the pricing solver" is **not new**: POMO-CG already does it for VRPTW, and the old `RL_PRICING.tex` built its main claim on it. The story has to move to what POMO-CG does not do.

| | POMO-CG (Abouelrous et al. 2025) | Ours |
|---|---|---|
| Where the learned pricer runs | root LP relaxation only | **whole B&P tree** |
| Branching vs pricing | not addressed | **branching on setups only, so pricing never changes**; the learned pricer runs at every node without retraining |
| Quality guarantee | none (gap vs a DP heuristic) | **certified lower bound and gap** (exact pricing confirms convergence) |
| Output | LP value | **integer solution + lower bound** |
| Pricing problem | routes (ESPPRC) | **scheduling with time-dependent dual costs** |
| Application | VRPTW | multi-project scheduling + procurement |

New one-line claim: *the first learned pricer used inside a complete branch-and-price, with certified optimality gaps* — to be confirmed against items 1 and 2 above.

### Training-distribution design (BO over dual generators) — related work to cite

Found 2026-09-15 by web search; full texts not read yet.
- **GANCO** — Generative Adversarial training for NCO models (OpenReview, https://openreview.net/forum?id=9vsRT9mc7U): a generator produces instance distributions the solver is bad at; trained alternately. Adversarial over *problem data*, not CG duals.
- **HAC** — hardness-adaptive curriculum for NCO (TSP): measures instance hardness and generates hard instances during training. Verify exact reference (believed Zhang et al., AAAI 2022).
- Adversarial instance generation and robust training for multi-objective NCO, arXiv:2601.01665 (2026).
- POMO-CG (#6) generates training duals from travel times (positive for some customers, zero for the rest) and reports that results vary strongly with the generator parameters and with instance size → motivation for choosing the dual distribution in a principled way.
- No work found that applies adversarial/curriculum selection to the **dual distribution of column generation**, or uses BO for it. Search again before submission.

## 2. Newly cited in `problem_definition.tex`

| Reference | Status |
|---|---|
| Pritsker, A. A. B., Watters, L. J., Wolfe, P. M. (1969). *Multiproject scheduling with limited resources: A zero-one programming approach.* Management Science 16(1), 93–108. | ok (added) |
| Talbot, F. B. (1982). *Resource-constrained project scheduling with time-resource tradeoffs: The nonpreemptive case.* Management Science 28(10), 1197–1210. | ok (added) |
| Wagner, H. M., Whitin, T. M. (1958). *Dynamic version of the economic lot size model.* Management Science 5(1), 89–96. | ok (added) |

## 3. To find (method section, not yet chosen)

- A standard reference for **time-indexed column generation / B&P in scheduling**, e.g. van den Akker, Hurkens, Savelsbergh (2000), *Time-indexed formulations for machine scheduling problems: Column generation*, INFORMS Journal on Computing 12(2). Verify details.
- A reference for **facility-location (extended) formulation of lot sizing** and its LP strength (used in the master).
- A reference for **branching on cumulative setup counts** in lot sizing / B&P (our branching rule). Search before claiming it is new.
- A reference for the **Lagrangian bound in column generation** (z_RMP + Σ_p min rc_p), e.g. Lübbecke & Desrosiers (2005), *Selected topics in column generation*, Operations Research 53(6). Verify details.

## 4. Missing from the bib (cited in `RL_PRICING.tex`)

19 of 30 cited keys have no bib entry. Titles I am confident about:

| Key | Reference |
|---|---|
| `kool2019attention` | Kool, van Hoof, Welling (2019). *Attention, learn to solve routing problems!* ICLR. |
| `kwon2020pomo` | Kwon et al. (2020). *POMO: Policy optimization with multiple optima for reinforcement learning.* NeurIPS. |
| `vinyals2015pointer` | Vinyals, Fortunato, Jaitly (2015). *Pointer networks.* NeurIPS. |
| `zhang2020learning` | Zhang et al. (2020). *Learning to dispatch for job shop scheduling via deep reinforcement learning.* NeurIPS. |
| `morabit2021machine` | Morabit, Desaulniers, Lodi (2021). *Machine-learning-based column selection for column generation.* Transportation Science. |
| `chi2022deep` | Chi et al. (2022). *A deep reinforcement learning framework for column generation.* NeurIPS. (verify) |
| `shen2022enhancing` | Shen et al. (2022). *Enhancing column generation by a machine-learning-based pricing heuristic for graph coloring.* AAAI. (verify) |

Titles I do **not** know — need the original source (the old references file, or search by the description in the manuscript):
`cheng2015splitting` (now added), `ieee2013semircpsp`, `pomocg2024`, `hoornaert2025vrp`, `rlhh2025cg`, `mlcg2025optoline`, `bitar2024cpqualification`, `park2023semirl`, `wang2025semicapacity`, `pan2021deep`, `chalumeau2024combinatorial`, `gnnjssp2024survey`.

The end of `RL_PRICING.tex` notes that the original bibliography was a separate `references.bib` that was truncated when pasted. If that file still exists somewhere, it should restore most of these.
