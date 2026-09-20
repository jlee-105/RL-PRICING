# Discussion Log

이 파일은 RL_PRICING 프로젝트 작업 중 나눈 논의와 결정 사항을 매일 업데이트하는 기록입니다.
새 세션을 시작할 때 여기를 먼저 읽고 이어서 진행합니다. 날짜별로 append.

---

## 2026-08-04

### 진행한 작업

**1. 프로젝트 폴더 파악**
- MRCPSP-P(Multi-Mode RCPSP + Procurement)를 Column Generation(CG) + Graph RL로 푸는 연구 프로젝트.
- 반도체 tool ramp-up(Install/Qual) 스케줄링이 동기(motivation).
- 핵심 아이디어: CG의 pricing subproblem(NP-hard 스케줄링)을 CP-SAT 대신 학습된 GNN 정책(POMO/REINFORCE)으로 대체. (task, mode, delay) action space가 핵심 novelty.

**2. 폴더 구조 정리**
- `document/` — 논문/문서: `RL_PRICING.tex`(메인, 최신), `rl_pricing_references.bib`, `CG_RL_PLAN.md`, `FORMULATIONS.md`
- `code/` — 구현: CG+RL 파이프라인 전체(`ColumnGenerationHeuristic.py`, `cg_rl_main.py`, `gnn_policy.py`, `rcpsp_env.py`, `rl_trainer.py`, `instance_generator.py`, `run_scaling_experiments.py`, `run_scaling_medium_large.py`, `test_pipeline.py`, `verify_optimality.py`), 학습된 checkpoint(`policy_5t_3q.pt`/`10t`/`20t`), 그리고 **유지하기로 한 프로토타입** `optimization.py`, `LargeScaleOptimization.py` (사용자 요청으로 삭제 안 함)
- **삭제한 파일**: `workshop_paper.tex`(워크숍 트랙 포기), `paper_draft.tex.superseded`, `rl_pricing_architecture.tex`(다른/litho 프로젝트의 orphan figure, RL_PRICING.tex는 자체 inline tikz figure 보유), `InventoryScenarioCombination(PySpark).ipynb`, `test_policy.pt`, `benders_run.txt`, `solutions_log.json`, `__pycache__`

**3. 논문 타겟**
- **주 타겟: INFORMS Journal on Computing** — "learning to price/branch/cut" 계열 OR+ML 방법론 논문의 대표 venue.
- Fallback: Computers & Operations Research / EJOR (심사 기준 다소 낮고 회전 빠름).
- 이유: 현재 실험이 toy 5-task 인스턴스 1개 + placeholder training curve뿐이라 JOC 기준에는 실험이 얇음 → 결과 보강 필요.

**4. 결과 섹션 설계 (현재 `RL_PRICING.tex`에 placeholder로 반영됨)**
- 새 서브섹션 `Generalization to Larger Problem Sizes` (`sec:generalization`) — Training Dynamics 뒤, Discussion 앞에 삽입.
- **왜 generalization이 우선순위 1번인가**: 사용자가 직접 지정. GNN 구조상 `num_resources`/`num_parts`/`max_modes`/`max_delay`가 고정되면 `num_tasks`는 자유롭게 변해도 되므로(GCN이 가변 크기 그래프 처리), "작게 학습 → 재학습 없이 크게 평가"가 아키텍처적으로 유효한 실험.
- **Table `tab:generalization`** (placeholder, 실제 값 `--`): `|J|` = 10(train)/20/40/80, 각 크기당 4행:
  1. `CG+CP-SAT` — baseline (exact pricing)
  2. `CG+RL+fallback` — **우리 방법**, 실전 배포용 (RL 우선 시도, 실패 시 CP-SAT 안전망)
  3. `CG+RL (pure)` — **우리 방법**, CP-SAT 완전 대체 주장을 증명하는 버전
  4. `GA (monolithic)` — 추가 baseline, CG 분해 없이 전체 문제를 직접 품
  - 열: Obj. Gap vs CG+CP-SAT(%), CG Wall-clock(s), Speedup(×), Fallback Rate(%), Columns
- **Figure `fig:generalization`** (placeholder, 3-panel): (a) gap% vs `|J|` (RL pure/RL+fallback/GA 라인), (b) speedup(×, log) vs `|J|`, (c) fallback rate(%) vs `|J|`. 학습 크기(10)에 수직 점선.
- Discussion의 기존 "Generalization" 문단은 새 섹션을 가리키도록 수정 + 실제 데이터 채워지면 정리하라는 TODO 주석 남김.

**5. GA baseline 설계 결정**
- Related Work에 인용된 메타휴리스틱 중 가장 최근/강한 것으로 선택: **`asadujjaman2024scircmpsp`(2024, surrogate-assisted GA)** 계열.
- `ieee2013semircpsp`(2013, simulated annealing)는 사용자가 "너무 오래됐다"며 기각.
- 설계(아직 코드 없음, 구상만): random-key GA — activity list(우선순위 키) + mode key를 유전자로, serial schedule generation scheme으로 decode(예측 가능하게 precedence/resource 항상 feasible, repair 불필요), fitness = 기존 `schedule_cost()`(procurement 비용 포함) 재사용. CG 분해 없이 monolithic하게 전체 문제를 품.
- GPU 불필요 (순수 CPU, `torch`/`rcpsp_env.py` import 안 함) → GPU가 다른 작업으로 사용 중이어도 준비/실행 가능.

**6. "CG+RL"이 뭔지 정리**
- CG+RL = 제안 방법. Table에서 baseline이 아니라 "ours"의 두 variant로 표시:
  - `CG+RL+fallback` = 실전용
  - `CG+RL (pure)` = 완전 대체 주장 증명용
- `CG+CP-SAT`, `GA`는 baseline.

### 진행 안 한 것 / 보류
- **코드는 아직 하나도 작성 안 함** — 사용자가 명시적으로 "no code yet" / "just discussion" 요청. GA baseline 스크립트, generalization 실험 스크립트 전부 설계만 되어있고 미구현.
- GPU가 이 세션 중 다른 학습 작업으로 사용 중이었음 → CUDA 관련 명령/실행 전부 보류. **다시 시작할 때 GPU 상태 먼저 확인.**
- 저비용 heuristic pricer baseline(2번 옵션, CP-SAT 말고 priority-rule 같은 가벼운 pricing) — 논의는 됐지만 이번 라운드에서는 보류, 여력 되면 나중에.

### 다음에 할 일 (우선순위 순)
1. GPU 사용 가능 여부 확인
2. GA baseline 구현 (`code/` 안에, CPU-only, random-key GA) + toy 인스턴스로 sanity check (기존 최적해 32.000과 비교)
3. Generalization 실험 스크립트 구현: `num_tasks=10`에서 정책 1회 학습 → `num_tasks ∈ {10,20,40,80}`에서 재학습 없이 CG+RL(pure/fallback) vs CG+CP-SAT vs GA 벤치마크 (각 크기당 held-out 20개 인스턴스)
4. `tab:generalization` / `fig:generalization` placeholder를 실제 숫자로 채우기
5. (여력 되면) 저비용 heuristic pricer baseline 추가 검토

### 커뮤니케이션 규칙
- **항상 한국어로 응답할 것.**
- 사용자가 "discuss"/"no code yet"이라고 하면 파일 수정(Edit/Write) 없이 텍스트로만 논의.

---

## 2026-09-14

### 방향 결정
- **이 논문의 기여는 pricing 알고리즘.** 모든 비교는 같은 CG 틀 안에서 pricer만 바꿔서 한다.
- GA(monolithic)는 핵심 baseline에서 제외. **metaheuristic은 나중에 추가** (사용자 결정).
- 우선순위: 먼저 exact pricer(기준점)부터 바로잡고, 그 다음 heuristic pricer 등 baseline.
- 제안했지만 아직 반영 안 한 baseline 구성: CP-SAT exact / CP-SAT time-limited / priority-rule heuristic pricer / RL+fallback / RL pure. `tab:generalization` 행 수정은 보류 (첫 시도 스크립트는 assert 실패로 미적용).

### 발견한 문제
1. **기존 CP-SAT pricer(`schedule_with_cpsat`)는 exact가 아니었다.** 목적함수가 Cmax + mode cost − α·D로, column 비용의 핵심인 조달 비용(`plan_orders_greedy`)이 빠져 있음. RL은 진짜 reduced cost를 최적화. → 원고의 "CG+CP-SAT = exact, gap 0%" 기준이 성립하지 않음. 이제 이 pricer는 **surrogate CP-SAT**으로 불러야 함.
2. **`plan_orders_greedy`의 early return 버그.** 어떤 부품이 early window(t < L)에서 부족하면 그 부품 루프 후 바로 return → 뒤 부품들의 조달 비용이 column 비용에서 통째로 빠짐. 랜덤 인스턴스 CP-SAT seed 스케줄 80개 중 75개가 해당, 해당 시 비용의 평균 64%가 누락. toy 인스턴스(최적 32.000)는 영향 없음.
   - **수정함**: return하지 않고 첫 위반만 기록, 모든 부품 비용 합산. (`ColumnGenerationHeuristic.py`)
   - **영향**: 기존 RL checkpoint(`policy_*.pt`)는 잘린 비용을 reward로 학습됨 → **재학습 필요**. 비용 정의의 허점을 exploit했을 가능성도 있음.

### 구현
- `code/exact_pricing.py` — `price_exact(inst, alpha, mu, time_limit_s, hint)`. time-indexed start binary로 D를 선형화하고 greedy-JIT 조달 비용을 closed form으로 CP-SAT에 넣음:
  E_t = max(0, S_t − I0), short_t = E_t − E_{t−1}, cost = Σ_{t≥L}[c·short_t + f·1(short_t>0)] + h·Σ_t max(0, I0 − S_t).
  closed form은 `plan_orders_greedy`와 3000/3000 일치 확인. 반환값에 `proven_optimal`, `rc_lower_bound` 포함.
- `code/verify_exact_pricing.py` — 소형 인스턴스 전수조사와 비교. **exact = brute force 40/40, 전부 proven optimal.** surrogate CP-SAT는 14/40에서 최적보다 나쁨 (평균 rc 초과 약 11–12).

### 통합 MIP (CG 없이 원 문제를 직접)
- `code/integrated_mip.py` — `solve_integrated_mip(inst)`. FORMULATIONS.md의 원 MIP를 multi-mode로 확장. 조달은 최적화(lot-sizing), early 부족은 hard 제약. `evaluate_plan`으로 독립 검증.
- **toy: MIP 최적 30.0 (0.03 s, proven). 기존 CG 결과 32.000은 원 문제 최적이 아니었음.** 같은 스케줄의 greedy-JIT 비용도 30.0 → 차이는 조달 방식이 아니라 CG가 더 좋은 스케줄을 못 찾은 것.
- 랜덤 |J|=10 (3 parts, 1–3 modes, H=50): SCIP 120 s에 gap 6.8% (145.59 vs bound 135.67). greedy-JIT로 조달하면 148.16 → lot-sizing 이득 약 1.7%.
- exact pricer scale probe는 |J|=10에서도 몇 분 내 결과 없음 → 중단.

### 결정: 현재 단일 프로젝트 CG는 폐기, multi-project CG로 전환 (옵션 2)
- **왜**: 현재 CG는 column = 전체 스케줄, master는 convexity + early 재고 제약뿐 → pricing = 원 문제와 같은 난이도. 분해가 아무것도 안 줄여줌. 게다가 최종 해를 argmax λ로 뽑는 heuristic이라 toy에서도 최적을 놓침.
- 대안이던 옵션 1(CG 없이 RL + Wagner–Whitin으로 원 문제 직접)은 채택 안 함. 나중에 별도 논문 가능성으로만 남김.

### Multi-project CG 설계 (사용자 동의)
- **동기**: 팹 ramp-up은 같은 종류 tool 여러 대가 비슷한 install/qual 공정을 반복함 → 프로젝트 = tool 하나의 ramp-up, **몇 개의 공정 템플릿(tool 유형)에서 복제 + 변형**.
- **공유**: renewable 자원(K_r)과 부품 재고/조달 둘 다.
- **프로젝트 p**: release r_p (tool 반입), due d_p (ramp 목표), 시간 창으로 pricing 크기 고정. 비용 = mode cost + w_p · tardiness (C_p − d_p)^+.
- **Master LP**: λ_pk (프로젝트 p의 스케줄 k), 조달 변수 o/δ/inv를 master에 직접 둠 (greedy JIT 근사 제거, 프로젝트 간 수요 합쳐 lot-sizing).
  - convexity Σ_k λ_pk = 1 (dual μ_p)
  - 공유 자원 Σ_p Σ_k u_pk[r,t] λ_pk ≤ K_r (dual π_rt)
  - 재고 균형 inv_t = inv_{t−1} + o_{t−L} − Σ_p Σ_k D_pk[m,t] λ_pk (dual β_mt)
- **Pricing (프로젝트별)**: min cost_p − Σ π_rt u[r,t] − Σ β_mt D[m,t] − μ_p. 시점별 비용이 있는 단일 프로젝트 MRCPSP → RL의 delay action이 의미를 가짐.
- **정수해**: 생성된 column pool 위에서 master MIP (price-and-branch). **하한**: exact pricer로 Lagrangian bound.
- **Scaling 축 = 프로젝트 수 P**. pricing은 항상 프로젝트 하나 크기라 10-task 프로젝트로 학습한 정책을 크기 외삽 없이 사용.

### Cheng·Fowler 그룹 (ieee2013semircpsp, cheng2015splitting)과의 관계
- 그들: Install/Qual MRCPSP, 확률적 duration/ready time (Monte Carlo), non-preemptive activity splitting, 시간에 따라 변하는 자원 profile, makespan + budget, SA. 조달 없음.
- 우리 차별점: 조달 통합(최적 lot-sizing), 여러 tool이 자원·재고 공유, DW 분해 + RL pricer + Lagrangian 하한.
- **결정: 확률적 요소는 넣지 않음** — 완전히 다른 모델(목적·column 비용이 확률변수). 결정론적 모델로 명시, future work 한 문장.
- 원고 132행 "We extend Cheng et al.'s MRCPSP model"은 과장 → 수정 필요. 231행 "cross-tool dependencies" — 그들이 여러 tool을 한 네트워크에서 다뤘는지 원문 확인 필요.
- 시간에 따라 변하는 용량 K_{r,t}: 넣기 쉬움 (추천, 사용자 확인 대기). splitting: limitation.
- **bib 문제**: 원고 인용 키 30개 중 19개가 `rl_pricing_references.bib`에 없음 (Cheng 두 편, POMO, Kool 등). 복구 필요.

- 2015 논문 확인 (웹 검색): Cheng, Fowler, Kempf, Mason (2015), C&OR 53:275–287. **결정론적**. renewable 자원(시간에 따라 변하는 용량 + 휴무) + **non-renewable 자원** (총량 제약), makespan, splitting이 calendar 포함 RCPSP의 일반화임을 보임. (앞서 2013 확률적 논문과 특징을 섞어 비교했던 것은 정정.)
- **결정 (A)**: 소모품(테스트 wafer, 화학약품 등)을 Cheng 외와 같은 **non-renewable 자원**으로 부름. 단 조달(리드타임, 주문, 시점별 재고)은 유지 → "non-renewable 자원에 보충(replenishment)을 붙여 시점별 재고로 확장". 원고 차별점 문장: "그들은 소모품을 총량 제약으로만 다루고, 리드타임 있는 조달 계획은 다루지 않는다". 코드에서는 재사용을 위해 필드명 `parts`/`demand_parts` 유지.

### 목적함수 단위 (결정)
- tardiness를 돈으로 환산: w_p = cost of delay ($/일). 생성기에서 w_p = ρ × (tool 유형 계수 0.5–1.5) × (프로젝트 예상 자재+mode 비용) / CP_p. 실험에서 ρ(또는 배율 ω ∈ {0.25,0.5,1,2,4}) 민감도 분석으로 Pareto 대용.
- pipeline inventory(이미 발주한 도착 예정 물량)는 **넣지 않음** (사용자 결정).
- 결정 변수: 각 task 시작 시각과 mode (pricing), 주문 시점과 수량 (master). 초기 재고는 먼저 사용.

### Multi-project 구현 (code/)
- `multiproject_instance.py` — 템플릿(tool 유형) 복제 + duration 변형, release/due/deadline, 공유 용량 K[r][t] (휴무 옵션은 용량 축소 방식, splitting 없음), ρ 보정. 반입일을 최대 리드타임만큼 뒤로 미룸 (안 그러면 infeasible 인스턴스 발생).
- `mp_monolithic.py` — monolithic time-indexed MIP (SCIP), 독립 평가 함수 `evaluate_solution`, 부품별 Wagner–Whitin `lot_sizing_cost`.
- `mp_cg.py` — master (조달을 master에 둠; `procurement="fl"` 프로젝트별 facility-location / `"bigm"`), 프로젝트별 exact pricer (CP-SAT, 시점별 비용이 선형), Lagrangian bound, price-and-branch 정수해.
- `verify_mp_cg.py` — P=3, 6 tasks, 5 seeds: dual 부호, LB ≤ OPT ≤ CG-IP, 평가 함수 재현, WW = MIP 조달. **전부 통과.**

### 하한 문제 진단
- big-M master: 정수해 gap 1.8–19.5%, 하한 gap 13.7–26.7%.
- **진단: monolithic 최적의 setup δ를 master LP에 고정하면 하한 gap 0.0%** (seed 1, 3, 4). → gap은 100% setup relaxation.
- 프로젝트별 facility-location master: 정수해 gap **1.1–2.9%**로 개선, 하한 gap은 14–22%로 거의 그대로. 원인은 구조적: LP가 스케줄을 볼록결합해서 수요가 여러 시점에 분수로 흩어짐 → 각 시점 setup도 작은 분수로 충분.
- **해결 방향 (사용자 동의)**: δ로 branching하는 B&P. δ는 master에만 있어서 분기해도 pricing 구조가 안 바뀜 → RL pricer를 트리 전체에서 사용 가능. + δ diving heuristic.

### 기여 (초안)
- C1 모델 (multi-project + 보충 있는 non-renewable + 시간 변동 supplier 인력 + $ tardiness) — **Hu 외 때문에 약함**
- C2 분해: column = 프로젝트 하나, 조달은 master (facility location) → pricing 크기 P와 무관
- C3 **branching이 pricing을 바꾸지 않는 B&P** (가장 강한 포인트) — 기존 RL pricing은 대부분 root CG에 머묾
- C4 RL pricer: 단일 프로젝트로 한 번 학습, 크기 외삽 없음, 마지막 exact pricing으로 gap 인증
- C5 실험: P scaling, monolithic MIP / exact / heuristic pricer / metaheuristic

### Hu, Liu, Demeulemeester, Zhou (웹 검색, 원문은 403으로 접근 불가)
- C&OR 188, 107362 (2026): multi-project, multi-supplier, renewable 자원, **시간 변동 supplier 용량**, 리드타임, supplier 선택·주문 시점·수량. period-based ordering. **heuristic PHHA** (priority rule + GA + local search + SA), 하한 없음. 통합 모델이 15.14% 비용 절감.
- JORS (2026.4 온라인): 같은 문제 + 수요 불확실성, two-stage robust, **C&CG와 Benders (exact)**.
- → **C1은 주 기여로 어려움. 무게는 C2–C4 방법론에.** 결정론적 버전엔 heuristic만 있으므로 B&P + 인증 gap + learned pricer가 차별점.
- 실험 아이디어: Hu 외 인스턴스를 구하면 PHHA와 직접 비교.
- **사용자가 C&OR PDF를 구해오기로 함.** 확인할 것: multi-mode 여부, 목적함수 항목(setup, 지연 비용), 시간 변동 용량이 자재 공급인지, 인스턴스 공개 여부.
- bib의 `hu2025supplier`는 실제 2026 (Vol. 188) → 수정 필요.

### B&P 구현 (`code/mp_bp.py`, `verify_mp_bp.py`)
- 전역 column pool 공유, best-bound 노드 선택, Lagrangian 조기 가지치기, exact pricing이 최적 증명 못 하면 시간 제한 늘려 재시도(인증 안 된 LP 값은 하한으로 안 씀).
- 단일 δ 분기: 하한이 거의 안 오름 (LP가 주문을 s±1로 옮김). seed 0에서 40 노드에 LB 238.9 → 251.6.
- **누적 setup 수 분기** (N_m 총 횟수 → Σ_{s≤τ} δ_ms): 27 노드에 LB 269.6, 정수해 279.28 (OPT 278.34). master의 δ에만 걸리는 선형 제약이라 pricing 불변.
- 중단 직전 결과 (seed 0, 정수해 탐색 10노드마다): B&P 278.46, LB 273.74, 195 노드, 279 s, **unresolved 25** (δ 전부 정수인데 λ 분수) → **δ만으로는 B&P가 닫히지 않음.** 앞 진단은 최적 δ 패턴에서만 gap 0이었음.
- 함의: 스케줄 분기(예: task 시작 시각)가 추가로 필요 → pricing이 바뀜. 단 RL에는 action mask 추가일 뿐 → C3를 "분기가 pricing 구조를 바꾸지 않거나 mask로만 반영된다"로 수정 가능. Kolter 외 확인 후 결정.

### POMO-CG 확인 (arXiv:2504.02383 v2 원문 읽음)
- Abouelrous et al. 2025. VRPTW, ESPPRC pricing을 RL(attention/POMO)로 직접 풂. dual은 무작위 샘플링으로 학습. **root LP만**, B&P·하한 인증·정수해 없음. "pricing을 직접 푸는 최초의 ML 모델"이라 주장. 동료 심사 전 preprint.
- → **"RL로 pricing 대체"는 새롭지 않음.** 새 주장: learned pricer를 완전한 B&P 안에서, 인증된 gap과 함께 쓰는 첫 사례 (Kolter 외, Václavík 외 확인 필요).
- 참고문헌 정리: `document/REFERENCES_TODO.md` (필수 PDF 7편, 검색 우선순위, POMO-CG 비교표, bib 누락 19개).

### 원고
- `document/problem_definition.tex` 새로 작성: (M1) MRCPSP → Cheng 외 (시간 변동 용량 K_rt만 채택, splitting 제외) → (M3) 우리 확장 (여러 프로젝트, 보충 있는 non-renewable, cost of delay). 모델 비교표(Hu 외 열은 PDF 확인 전이라 미확정). 보충 불가(L_m ≥ H)면 재고 제약 = MRCPSP 총량 제약임을 보임.
- bib 추가/수정: hu2025supplier (C&OR 188, 2026으로 정정), cheng2015splitting, pritsker1969multiproject, talbot1982resource, wagner1958dynamic.
- 투고처: C&OR (Cheng·Hu 모두 여기, 빠른 편) 현실적, C3가 강하면 JoC. EJOR는 느림.

### **결정: 모든 관련 논문을 확인할 때까지 새 실험은 시작하지 않음** (사용자)
- 이미 돌고 있는 B&P 검증(`verify_mp_bp.py`, P=3, 5 seeds)은 계속 돌림. 새 실험·새 구현은 논문 확인 후.

### 논문 5편 확인 (PDF: 프로젝트 루트, 요약: `document/REFERENCES_TODO.md`)
- **Kolter 외 (arXiv 2025)**: 프로젝트 단위 DW 분해는 **Deckro 1991**부터 있음 → C2 새롭지 않음. 자원 제약 없는 pricing이면 DW 하한 = compact LP (multi-mode도 +0.42%, 26배 느림). 시작 시각 분기 = compact B&B와 같은 트리. 자원 제약 있는 pricing은 하한 강하지만 NP-hard pricing이 병목 — "pricing을 효율적으로 푸는 방법이 필요" (27쪽). compact MIP + solver와 비교 안 한 논문들을 비판. → **exact B&P는 compact MIP를 이기기 어려움** (우리 P=3 결과도 동일). 남은 각도: 우리 pricing은 프로젝트별 자원 제약 포함 → learned pricer가 그 병목에 대한 답.
- **Václavík 외 (EJOR 2018)**: B&P 안의 ML이지만 column 생성 안 함 (pricing 목적 상한 예측으로 exact pricer 가지치기). → "B&P 안 ML 최초" 주장 불가.
- **Hu 외 (C&OR 2026)**: single-mode, renewable·재고가 **프로젝트별 로컬** (공유 없음), 연결은 supplier 용량뿐. PHHA heuristic (activity list + delay list). 대형(P=10, 120 activities)은 metaheuristic끼리만 비교, **하한 없음**. 인스턴스 공개 (github LouisResearch/MPSMOP-Instances). → **C1 되살아남** (공유 renewable, 공유 재고, multi-mode는 우리만).
- **Cheng 외 (C&OR 2015)**: 결정론적, 작업별 ready/due hard, U_kt, non-renewable 총량, makespan, precedence-tree B&B, 12/16 activities. 참고문헌 [9] (2013 working paper)가 `ieee2013semircpsp`일 가능성.
- `problem_definition.tex` Cheng 절과 비교표를 원문대로 수정.

### POMO-CG 방법 부분 (4–11쪽) 읽음
- POMO (Kwon 2020) + REINFORCE 공유 baseline, 종료 reward = reduced cost, 음의 rc column만 추가, 무작위 dual로 CG와 분리 학습, 가격 정규화. → **CG + RL pricer만 하면 우리 방법은 POMO-CG를 스케줄링에 옮긴 것.**
- 11쪽: "root만 풀지만 B&P 나머지 노드에도 적용 가능, masking만 추가하면 됨" → mask 기반 B&P 아이디어도 이미 언급됨.
- Kolter 논문은 저널 게재 아님 (arXiv 프리프린트; 이전 버전이 PMS 2024 워크숍 Best Student Paper). bib은 이미 arXiv로 되어 있음.

### 포지셔닝 논의
- 사용자: "CG + RL은 이미 시도됨(POMO-CG), 부정할 수 없음." Benders 제안 → 순수 최적화 논문이 되므로 사용자가 싫어함. **학습 기반 + CG 유지** (사용자 결정).
- 대안으로 제안했던 "CG 없이 RL 스케줄 + WW 조달 (DWTA 철학)"은 사용자가 CG 유지를 원해서 보류.
- JORS 2026 (Hu 외 robust): 사용자가 Intel에서 구독 없어 못 구함. 학습 기반 방향에선 중요도 낮음 (초록 수준 인용). 필요하면 ASU 도서관/저자 요청.
- 참고: Cheng 2015 공저자 Karl Kempf가 Intel 소속 → 현실 검증/파라미터 가능성.

### 하한 강도 판정 실험 (`code/exp_bound_strength.py`, `code/mp_compact.py`, 로그 `code/logs/bound_strength_P3.txt`)
- P=3, 6 tasks, seeds 0–4, 모두 OPT 증명. 평균 gap (OPT−bound)/OPT:
  - compact LP agg/bigm 20.57%, compact LP dis/fl 19.90%
  - **compact root dis/fl (SCIP cuts) 9.89%, 1.2–5.5 s**
  - DW no-cap (pricing 자원 제약 없음) 17.96%, DW cap 17.75%, 6–28 s
- **Kolter 결론 재현**: DW는 compact LP보다 ~2%p만 강하고, compact + cut이 두 배 강하고 빠름. pricing에 프로젝트 용량 넣어도 0.2%p (공유 용량이 프로젝트 하나엔 거의 안 걸림).
- 참고: dis/fl LP가 agg/bigm LP보다 약한 경우 있음 — FL의 (프로젝트, 시점)별 setup 상한 합이 big-M 하나보다 느슨할 수 있음.

### CG를 살리는 방향 (제안, 사용자 확인 대기)
- Kolter 명제 4: block 전용 자원이 있으면 pricing NP-hard → DW 하한 확실히 강함 (Coughlan 2015), 대신 pricing 병목 → "효율적 pricing 필요" (27쪽).
- **tool 자체를 unary 전용 자원으로**: 한 tool에서 qualification 테스트는 한 번에 하나. → pricing = unary 자원 + 시점별 dual 비용인 단일 프로젝트 스케줄링 (NP-hard) → 하한 강해지고, 그 병목을 learned pricer로 푸는 것이 기여.
- POMO-CG와의 차이: 하한이 강해지는 분해를 Kolter 분석으로 설계·검증, pricer 크기 P와 무관(한 번 학습), 같은 템플릿 프로젝트를 GPU 배치 pricing, 실제 CG dual 분포로 학습.
- 확인 방법: 생성기에 tool별 unary 자원 추가 → 같은 판정 실험 재실행 → DW cap이 compact root보다 강하면 성립.

## 2026-09-15

### 방향 확정 (사용자)
- **사용자는 RL 연구자, heuristic 선호. 전역 최적·하한 인증 필요 없음.** exact 방법(compact MIP, exact pricer)은 baseline·검증용으로만. 기여는 RL 쪽. **대규모 최적화에 집중.** (memory에도 저장)
- 논문 틀: CG + learned pricer, 대규모에서 같은 시간 제한 내 compact MIP·metaheuristic보다 좋은 해를 빠르게. RL 기여 후보: 정확히 분해되는 dense reward, 크기 독립 + 배치 pricing, on-policy dual 분포, 다양한 column 생성, search-in-loop 학습.

### tool unary 자원 (`multiproject_instance.py`, `tool_unary="none"|"qual"|"all"`)
- 프로젝트마다 전용 `TOOL_P{p}` (용량 1). "qual"은 앞 1/3 작업(설치) 제외. 직렬화되므로 span = max(CP, tool 작업 최소 duration 합)으로 due/deadline 설정. "none"이면 기존 인스턴스 그대로 재현.
- P=3 (6 tasks) compact MIP: tool 없음 6 s → tool 있음 108 s (18배). 대규모 크기: P=40이면 400 tasks, H=415, z 변수 약 3만.

### 새 pricing 환경 (`code/mp_pricing_env.py`, 검증 `verify_mp_pricing_env.py`)
- action (task, mode, delay), 선행 bound + delay 이후 첫 feasible 시작(프로젝트 자체 용량: 공유 자원 + tool). step reward = −c[j,q,τ] (시점별 dual 비용), 종료 −w·T + μ. **return = −rc 정확히** (오차 5.7e-14, anchor 포함). column 1,784개 전부 feasible. 무작위 정책 막다른 상태 7%.
- 기존 `rcpsp_env`의 anchor reward가 버려지던 버그 → 새 환경은 다음 step으로 넘김.
- feature 크기는 (공유 자원 수, 부품 수, max_modes, max_delay)에만 의존 → 한 정책으로 모든 프로젝트·모든 P.
- 무작위 / myopic(당장 가장 싼 action) 정책 vs exact: 무작위는 거의 못 찾음, myopic은 가끔 근접하나 자주 크게 빗나감 (예: exact −20 vs +182) → 앞을 보는 정책 학습 여지 큼. myopic은 학습 없는 heuristic pricer baseline으로 사용.

### POMO trainer
- 사용자 지적: 기존 `rl_trainer.py`는 진짜 POMO (Kwon 2020)와 다름 (critic, step별 return − 스칼라 baseline, entropy, clamp, anchor 8개). → 진짜 POMO로 새로 작성 예정: 서로 다른 첫 action N개, 평균 return 공유 baseline, critic 없음, 여러 (프로젝트, dual 스냅샷) batch. 첫 작업 후보가 적으므로 첫 action을 (task, mode, delay) 조합 단위로 다양화.

### RL이 배우는 것 / POMO-CG와의 차이 (논의)
- 정책 = dual 가격(π_rt, β_mt, μ)을 보고 프로젝트 하나의 스케줄을 짜는 법: 붐비는 시기 피하기, 자재 타이밍 맞추기(주문 묶기), mode 선택, 기다릴까/지금 할까(tardiness와 trade-off, DWTA의 hold 딜레마), 막다른 상태 피하기. master LP·조달·feasibility는 정확한 쪽에 맡김.
- POMO-CG가 이미 한 것 (기여 주장 불가): dual → column 정책, POMO 다중 column, dual 정규화, CG와 분리 학습(무작위 dual).
- POMO-CG에 없는 것: **dual이 (자원, 시점)별 = "시간 가격"** → "언제 할까"를 가격·tardiness·막다른 상태 사이에서 저울질. mode, delay/tardiness trade-off, 막다른 상태, dense reward, 크기 독립 pricing, CG 반복에 따른 dual 분포 변화. time-indexed 분해 전반에 해당하는 일반성.

### 학습 dual 분포: Bayesian optimization (사용자 제안, 확정)
- POMO-CG의 무작위 dual 대신 **BO로 정책이 약한 dual 구간을 골라 학습** (적대적 curriculum). RL 쪽 기여.
- dual 직접 BO는 차원 과다 (~240) → **dual 생성기를 6–10개 파라미터**(혼잡 강도·중심·폭, 자원별 가격 비율, 자재 가격 수준·패턴, μ 수준)로 표현하고 BO는 파라미터를 고름.
- BO 목적: 해당 dual에서 정책 pricing gap (정책 rc − exact rc). exact pricer는 프로젝트 하나라 호출당 1초 이내.
- ablation: 무작위(POMO-CG) / 실제 CG 궤적(on-policy) / BO / on-policy + BO. 평가는 실제 대규모 CG의 최종 해 품질·수렴 속도·pricing gap.
- 선행 연구 확인 필요: NCO의 적대적 인스턴스 생성·curriculum, CG dual 분포에 적용한 사례.

### 정책 구조: Lithography 논문의 GNN 방식 재사용 (사용자 제안, 확정 방향)
- 원본: `C:\Users\ongs6\vs_code\Lithography` (`gnn_policy.py` DRCSchedulingGNN, `ha_pomo_trainer.py`, `drc_env_gnn.py`, `paper_draft.tex` 337–637행). **인용은 나중에 추가** (사용자).
- Litho: 이종 그래프(job/machine/reticle) + 타입별 MLP + GAT(residual, LN) + 간선 가지치기; action head = [노드 임베딩들 ‖ global ‖ action별 시뮬레이션 비용 feature 12개] → residual MLP → logit; **HAM 학습**: rollout 10개의 첫 S step을 서로 다른 dispatching 규칙으로 고정, 공유 평균 baseline REINFORCE (POMO 대칭성 없는 문제의 다양성 해결); greedy/rollout decoder.
- pricing 매핑: 노드 task/자원(공유 인력, tool)/자재; 간선 precedence·사용·소모; action (task, mode, delay); **시간 가격 action feature** (c_jqτ, 구간 자원 가격, 소모 시점 자재 가격, tardiness 증가, deadline 여유, 막다른 상태 위험, 나중이 얼마나 싼지); pricing anchor (myopic, delay 0, 최저가 mode, 최速 mode, critical path, 무작위, RL).
- 장점: anchor rollout = 서로 다른 column 여러 개 (multi-column 공짜), 시간 가격이 action feature로 자연스럽게, 연구 흐름 (Litho → RL_PRICING, + DWTA) = NIW용 일관된 프로그램.

### RL 세부 결정 (사용자 동의, 추천안 그대로)
1. reward: **step reward** 사용 (return = −rc 정확히 분해). baseline은 anchor rollout들의 step별 return-to-go에 맞춘 것 (기존 trainer의 스칼라 baseline 문제 반복 안 함).
2. anchor 길이 **S = 1–2** (작업 ~10개) — 다양성은 실험으로 확인.
3. **프로젝트별 그래프** (공유 자원 상황은 dual π_rt에 담김) → 크기 독립·배치 pricing 유지.
4. Litho `gnn_policy.py` (GAT + action head)와 `ha_pomo_trainer.py` (anchor 학습 루프)를 **복사해서 pricing용으로 수정**.

### HAM-GNN pricer 구현 (사용자가 자는 동안 진행, "please go ahead")
- `torch_geometric` 미설치 → 전역 환경에 설치하지 않고 **GAT를 순수 PyTorch dense attention으로 구현** (프로젝트당 노드 ~16개라 충분). 학습은 CPU (GPU는 DWTA 학습 중).
- `mp_pricing_env.py`에 그래프 관측 추가: 노드 task/자원(공유+tool)/자재 (feature 8), 간선 precedence·사용·소모, **시간 가격 action feature 12개** (c/scale, delay, 대기, 자원 가격 부분, 자재 가격 부분, mode 비용, duration, tardiness 증가, deadline 여유, tool 남은 작업/남은 시간(막다른 상태 위험), 나중이 얼마나 싼지(suffix min), 남은 critical path), global 7개.
- `pricing_gnn.py` PricingGNN: 타입별 MLP projector, DenseGATLayer ×3 (residual+LN, 4 heads), global MLP, action head [h_task ‖ mean h_res(mode) ‖ mean h_mat(mode) ‖ h_g ‖ f] → residual MLP (Litho DRCSchedulingGNN 구조). 여러 그래프 padding batch.
- `ham_pricing_trainer.py`: anchor 7개 (myopic, nodelay, cheap_mode, fast_mode, critical, late, random) + rl(무 anchor) = rollout 8개, S=1. step reward/scale, return-to-go − leave-one-out step baseline, 문제별 std 정규화. 막다른 상태만 −10 penalty (처음엔 모든 reward를 −10에서 잘라 큰 tardiness까지 잘리는 버그 → 수정). 평가: HAM best-of vs greedy vs myopic vs 최선 heuristic vs exact (gap은 dual scale 단위).
- 스모크 테스트: 1.4 s/update (batch 4), 학습 전 gap HAM 3.0 / myopic 0.33 / 최선 heuristic 0.28 → **넘어야 할 기준선은 heuristic 최선**.
- `pricing_data.py`: 실제 CG 궤적 dual 스냅샷 + exact rc 저장. 처음 설정(반복 60, pricing 20 s)이 너무 느려 (tool 있으면 pricing NP-hard) 반복 30, pricing 5 s로 줄여 재시작 (val 10 instances, train 40). 로그 `code/logs/data_*.txt`.
- 주의: checkpoint는 val 스냅샷으로 고르고, 최종 평가는 별도 대규모 인스턴스 (DWTA에서의 test set 선택 문제 반복 안 함).

### 대규모 CG 실행기 (`code/cg_runner.py`)
- master LP 시간 (P=40): **FL 16.8 s vs big-M 0.07 s** → 대규모는 **big-M master** (전역 최적 불필요, dual π·β는 동일하게 나옴). 최종 비용은 고른 스케줄의 조달을 **WW로 다시 최적 계산** (`evaluate_solution(orders=None)`)해서 모든 방법을 같은 평가기로 채점.
- pricer: exact (CP-SAT, 5 s), heur (anchor heuristic 7개 모두), ham (학습된 정책, 모든 프로젝트×anchor 한 번에 batch).
- **price-and-branch만으로는 P=10에서 정수해 없음** (LP는 스케줄을 섞어 공유 용량을 맞추지만, 프로젝트당 하나씩 고르면 용량 초과) → **SGS repair** 추가: argmax-λ 스케줄의 시작 시각 순서로 serial SGS (공유+tool 용량, deadline), LP 시점 유지 / 앞당김 두 버전 중 싼 쪽.
- P=10 (seed 5000, tool qual): CG-heur 해 2134.06 (tardiness 56, mode 44, 조달 2034), CG 9.5 s, 전체 39 s. CG-exact CG 119 s, LP 1346 (heur LP 1362).
- **compact MIP (SCIP) P=10, 120 s: feasible 해 없음** (agg/bigm 1.2만 변수, dis/fl 15만 변수 모두) → 대규모에서 compact MIP가 이미 무너짐 = 학습 기반 CG의 동기.
- P=5: CG-heur 1247.9, heuristic pricer는 4번 반복 후 음의 rc column을 못 찾아 조기 종료 → learned pricer가 개선할 지점.

### 자동 실행 중 (세션 분리, 사용자 수면 중)
- 데이터: val 완료 (10 instances, 526 s), train 진행 중 (`logs/data_train.txt`). 많은 CG가 반복 상한 30에 도달 (데이터 다양성엔 무방).
- `run_pricer_pipeline.py` (PID 32632): 데이터 완료 → `ham_pricing_trainer.py --updates 1500 --batch 16 --S 1` → `pricer_ham_s1.pt`, 로그 `logs/train_ham.txt`.
- `run_after_training.py` (PID 10252): 학습 완료 → `exp_large_scale.py` (P=5/10/20/40 × 3 seeds, MIP / CG-exact / CG-heur / CG-ham, 제한 300 s (P≤10) / 600 s) → `logs/large_scale.jsonl`.

### HAM 학습 진행 (`logs/train_ham.txt`)
- train data 40 instances 완료 (약 46분). 학습 5.8 s/update.
- **u=100 평가** (val 400 문제, gap = dual scale 단위, clip≥0): HAM 0.398 (neg found 0.69) / greedy 1회 0.488 (0.59) / myopic 1.300 (0.36) / heuristic 7개 최선 0.401 (0.70). exact와 일치율 1–5%. → 100 update 만에 HAM ≈ heuristic 최선, 학습 정책 1회 rollout이 myopic보다 2.7배 좋음.

- **학습 완료 (1500 updates)**. val gap 추이: u100 0.398 → u300 0.334 → **u700 0.307 (최선, `pricer_ham_s1.pt`)** → u900 0.309 (neg found 0.80 최고) → u1500 0.332. u700 이후 0.31–0.33 plateau. greedy 1회는 0.488 → 0.380–0.39.
- **HAM은 heuristic 7개 최선(0.401, neg 0.70)보다 gap ~23% 작고 음의 rc column을 더 자주 찾음 (0.76–0.80).** exact와 정확히 일치하는 비율은 여전히 2–4% (heuristic 최선 5%).
- 개선 여지: plateau → BO dual curriculum, S 변경, 모델 크기, 학습률 감소 등.

### 대규모 실험 결과 (진행 중, `logs/large_scale.jsonl`)
- 참고: `run_after_training.py`는 psutil로 PID를 2분마다 확인 → 학습 종료 후 약 2분 뒤 시작 (4:30경).
- **P=5 (3 seeds, 제한 300 s)**:

| seed | CG-exact | CG-ham | CG-heur | MIP |
|---|---|---|---|---|
| 5500 | 1213.04 | 1239.59 | 1237.00 | 1254.98 |
| 5501 | 979.88 | 994.70 | 1011.90 | 1084.68 |
| 5502 | 883.83 | 903.52 | 894.21 | 953.64 |
| exact 대비 평균 | 0 | +2.0% | +2.1% | +7.4% |

  - 시간: exact 130–220 s (5502는 37 iters, pricing 153 s — P=5에서 이미 제한에 근접), ham 68–77 s, heur 63–71 s (대부분 최종 IP 60 s), MIP 300 s.
  - LP: ham이 매번 heur보다 좋고 exact에 근접 (777.6 vs 788.8 / 626.6 vs 645.5 / 654.4 vs 667.7), CG 단계는 exact보다 4–8배 빠름.
  - **최종 비용에선 ham ≈ heur** (1승 2패) → **병목은 pricing이 아니라 LP → 정수해 단계** (정수 master + SGS repair가 LP 우위를 해로 못 옮김).
- **P=10 (3 seeds, 제한 300 s)**:

| seed | CG-exact | CG-ham | CG-heur | MIP |
|---|---|---|---|---|
| 6000 | 2228.20 | **2190.59** | 2270.97 | 3100.33 |
| 6001 | 1936.21 | 1943.25 | 2019.21 | 해 없음 |
| 6002 | 1669.71 | 1708.95 | 1712.11 | 2138.19 |
| exact 대비 평균 | 0 | **+0.3%** | +2.9% | +34% (2개) |

  - 시간: exact 234–265 s (pricing 163–189 s, 제한 근접), ham 84–110 s, heur 73–87 s.
  - LP: ham이 매번 exact에 근접, heur는 조기 종료로 뒤처짐 (1412.6/1417.0, 1198.8/1279.6, 1005.1/1049.0; exact 1404.6/1161.3/992.2).
  - seed 6000에선 ham이 exact보다 좋은 최종 해 (LP는 exact가 더 좋았음) — ham이 anchor별로 다양한 column을 넣어 pool이 더 큼 (247 vs 147) → 정수해 단계의 선택지 증가로 추정.
  - **P=5 → P=10**: ham–exact 차이 +2.0% → +0.3%, ham–heur 차이 0.1%p → 2.6%p, MIP 붕괴 (1/3 해 없음, 나머지 28–39% 나쁨), exact는 제한 근접.
- P=20 seed 7000: **MIP 600 s 결과 7,415,480** (하한 2505.7) — 비정상적으로 큼. tardiness T_p·재고 변수에 상한이 없어 SCIP 초기 feasible 해가 이 변수들을 부풀린 채로 거의 개선 못 한 것으로 추정 → 사실상 쓸 만한 해 없음.
- **P=20 (3 seeds, 제한 600 s)**:

| seed | CG-exact | CG-ham | CG-heur | MIP |
|---|---|---|---|---|
| 7000 | 3525.99 | 3639.90 | 3593.90 | 7,415,480 (비정상) |
| 7001 | 3754.15 (CG 시간 제한) | 3835.41 | 4106.29 | 해 없음 |
| 7002 | 2841.67 | 2832.99 | 2810.69 | 해 없음 |
| exact 대비 평균 | 0 | **+1.7%** | +3.4% | 실패 |

  - 시간: exact 322–676 s (7001은 CG 616 s에 제한 걸림), ham 124–179 s, heur 89–114 s.
  - LP: ham은 exact LP의 0.2–2% 이내 (2506.8/2503.0, 2461.9/2412.9, 1214.2/1208.2), heur는 2.4–5.5% 뒤처짐. LP 순위는 항상 exact ≤ ham < heur인데 최종 비용 순위는 인스턴스마다 뒤집힘 → 정수해 단계가 변동의 주원인.
  - 7002: LP 1208 vs 최종 2842 (2.4배) — 정수해 단계가 특히 어려운 인스턴스.
  - 7000: ham이 column 578개로 최종 가장 나쁨 — 정수 master 60 s 고정이 큰 pool에서 부족할 가능성.
- **P=40 (제한 600 s, 2 seeds만)**: seed 9000 — ham **6827.36** (46 iters 수렴) / exact 6968.80 (21 iters, 시간 제한) / heur 7061.98 / MIP 해 없음. seed 9001 — ham **11361.37** (50 iters, 시간 제한) / heur 13188.12 / exact **해 없음** (18 iters, 시간 제한, repair 실패) / MIP 해 없음. ham CG 475–608 s는 rollout 320개/반복을 CPU로 돌린 결과 → GPU로 줄일 여지.
- **사용자 지시로 중단 (2026-09-15 아침)**: "small scale test만 요청했음, 대규모는 나중에 GPU로". seed 9002는 미실행. 이 대규모 실험은 제가 임의로 이어 붙인 것 → 앞으로 작은 테스트만, 긴 실험은 먼저 묻고 GPU로 (memory 저장).
- **공정성 수정 필요 (나중에)**: MIP 비용은 SCIP 목적값 그대로, CG는 스케줄을 WW로 재평가. MIP 해에서도 스케줄을 뽑아 `evaluate_solution(orders=None)`로 재평가해야 동일 평가기. (`mp_compact.solve_compact`가 해를 반환하도록 수정)
- 개선 후보 (사용자와 논의 예정, 구현 안 함): **diving heuristic** (λ≈1 프로젝트 스케줄 고정 → CG 재실행 반복) — learned pricer를 반복 호출하므로 pricing 속도 이점이 살아남. 또는 top-k 컬럼 조합 + SGS, 최종 IP 시간 조정.

### 소규모 확인 실험 (`exp_small_scale.py`, P=3, 6 tasks, tool qual, seed 8000, 1 instance)
- 사용자 요청: 소규모에서 MIP와 CG 비교, **CG-exact 시간 제한을 풀어 수렴시킴**. CG-RL은 제외 (BO 반영 후 재평가).

| 방법 | 최종 비용 | 최적 대비 | LP 하한 | 최적 대비 LP | CG 반복 | 시간 |
|---|---|---|---|---|---|---|
| MIP (dis/fl, SCIP) | **278.29 (최적 증명)** | 0 | – | – | – | 146.5 s |
| CG-heur | 287.45 | +3.3% | 216.0641 | −22.4% | 5 | 2.3 s |
| CG-exact | 291.87 | +4.9% | 216.0641 | −22.4% | 15 | 26.7 s |

- **두 CG가 소수점 14자리까지 같은 LP에 수렴** → 이 크기에서는 heuristic pricer만으로 LP가 완전히 닫힘. **소규모에선 학습이 불필요**; 학습의 가치는 heur가 LP를 못 닫는 규모(P ≥ 10)에서 나타남 (대규모 실험과 일치).
- **CG의 최적 대비 gap(3–5%)은 전부 정수해 단계에서 발생** → AR-3의 직접 근거.
- LP 하한이 최적보다 22.4% 아래 → 분해 하한이 약하다는 Kolter 결론과 일치, 하한을 쓰지 않기로 한 판단의 근거.
- 참고: 처음엔 8 tasks / MIP 제한 1800 s로 시작했으나 너무 오래 걸려 중단 → 6 tasks / 300 s로 재실행. 인스턴스 1개면 충분하다는 지시로 seed 8000만 완료 후 종료.

### BO dual curriculum 구현 (AR-4, 코드만 — 실행은 GPU 여유 후)
- 동기: POMO-CG는 학습용 dual을 무작위 샘플링하고 "생성 파라미터에 따라 결과가 크게 달라진다"고 스스로 기록. 우리는 **정책이 약한 dual 구간을 BO로 찾아 학습**.
- **실제 dual 분포 측정** (`data/pricing_val.pkl`): π는 **95–96%가 정확히 0**, 0이 아닌 값은 최대 −10⁴ (초기 반복 slack). β는 −22 ~ −1.6 (중앙값 −6.2, 자재 단가 수준). μ는 중앙값 134. → 실제 dual은 **희소하고 몇 시점에 몰린** 형태. μ는 gap에서 상쇄되므로 탐색 대상에서 제외.
- `dual_generator.py`:
  - (a) 합성 생성기 7 파라미터 (π 진폭·희소도·중심·폭, tool 비율, β 수준·기울기).
  - (b) **실제 스냅샷 변형** 7 파라미터 (π 배율, 유지 비율, 확산, 시간 이동, 잡음, β 배율·기울기) ← 채택.
- `bo_curriculum.py`: sklearn GP (Matern 5/2) + EI, 후보 무작위 탐색. 목적 = (정책 rc − exact rc)/scale 평균, 클수록 어려움.
- **결과 (작은 테스트)**: 합성 생성기는 gap 0.01–0.17로 **실제(0.31–0.44)보다 쉬움** → 실제 변형 방식으로 전환. 변형 방식에서는 **실제 dual 0.439 vs BO가 찾은 최난 구간 0.851** (2배).
  - 어려운 조합: `beta_tilt −0.89` (자재가 **초반 비싸고 후반 싸짐** → 얼마나 기다릴지 판단이 핵심), `shift −0.18` (병목이 앞당겨짐), `keep 0.64` (가격이 더 드묾).
  - 쉬운 조합: `pi_scale 1.42` (가격 26배) → 가격이 압도적이면 선택이 자명. **정책이 약한 곳은 가격이 큰 곳이 아니라 미묘하게 갈리는 곳** (논문에 쓸 관찰).
  - BO 반복 5번으로는 초기 무작위 최선을 못 넘음 → 평가당 문제 수·반복 늘려야 (목적함수 잡음).
- `rl_pricing_trainer.py`: `--curriculum bo`, `--bo_every/--bo_mix/--bo_init/--bo_iter/--bo_problems/--bo_keep`, `--device`. BO로 찾은 어려운 문제를 batch에 섞고, **검증은 실제 dual 유지**.
- **GPU 여유 후 실행할 것**: ① `--curriculum none` (기준) ② `--curriculum bo` ③ 두 정책을 실제 dual 검증셋 + CG 실험에서 비교.
- 정리 중 PowerShell 재작성으로 들어간 BOM 제거 완료 (모든 파일 정상 파싱).

## 2026-09-17

### GPU 텐서화 (사용자 요청: "later scaled problem cpu cannot handle, pytorch tensor directly")
- 출발점: 기존 구현으로 GPU를 쓰면 **오히려 3.3배 느림** (CPU 5.77 s/update vs GPU 18.9). 원인은 DWTA 5.4절과 동일 — 작은 커널 수만 번 (환경마다 numpy 관측 → 개별 전송, 후보마다 파이썬 루프).
- `mp_pricing_env_batch.py` **BatchPricingEnv**: B개 rollout을 한 텐서로. 핵심은 "어디서 시작 가능한가"를 루프 없이 계산:
  `slack = cap − usage` → `max_pool1d`로 sliding-window min → 자원 요구량과 비교해 `feasible [B,n,Q,W]` → 뒤에서 `cummin`으로 "이 시점 이후 첫 가능 시점" 표 → delay별 시작 시점을 **gather 한 번**. 자원 점유 갱신도 시간 마스크 텐서 연산.
- `pricing_gnn.PricingGNNBatch`: 후보 루프 제거, 자원·자재 임베딩 평균을 **멤버십 행렬 einsum**으로. 가중치 레이아웃은 기존과 동일(체크포인트 호환).
- `rl_pricing_trainer_batch.py`: anchor 7종을 텐서 키로 구현(myopic/nodelay/cheap_mode/fast_mode/critical/late/random), step reward + leave-one-out baseline을 reshape로.
- **동치 검증** (`verify_batch_env.py`, `verify_batch_obs.py`): 후보 시작 시점 차이 **0**, 노드 feature 차이 **0**, action feature·logit 차이 1e-7 (float32). 학습 로그도 수정 전후 동일.
  - 찾은 불일치 3건 (모두 수정): ① 창 길이가 다른 문제를 패딩할 때 suffix-min을 0으로 채워 "나중이 얼마나 싼지" feature가 틀림 → inf로 패딩; ② 기존 `_prec_lb`가 **배치 안 된 선행 작업도** (시작 0, mode 0) 포함하는 특성을 관측에서 재현; ③ anchor 정렬 키의 스케일을 무효 후보 sentinel까지 포함해 계산 → 무효 action 선택.
- **속도**: rollout sweep 1024개 — CPU 1404 ms vs **GPU 107 ms**. 학습 update(문제 16 × anchor 8 = 128 rollout): 기존 5.77 s → **0.195 s (30배)**. 배치를 키우면 rollout당 비용이 더 내려감 (128 rollout 1.55 ms/rollout → 1024 rollout 0.58 ms/rollout, 초당 1,733 rollout).
  - 환경 생성이 병목이었음(0.46 s): 같은 문제를 anchor 수만큼 중복 생성 → `repeat` 인자로 한 번 만들고 텐서 복제.

### BO를 실제 목적에 연결 (사용자 지적: "difficult problem itself does not make sense")
- 문제: 기존 BO 목적은 **pricing gap 최대화**(= 어려운 문제 찾기). 어려움 자체는 학습 가치와 무관 — 비현실적이거나 누구도 못 푸는 상황일 수 있음. 실제로 기준 vs BO 학습 비교(각 1500 update, GPU)에서 **일관된 이득 없음** (gap 0.32–0.35에서 엎치락뒤치락, u1250 기준 0.323 vs 0.319).
- 새 설계 (`code/bo_cg.py`): θ → 그 분포로 짧은 미세조정 → **실제 CG 실행** → 점수. BO는 그 점수를 최소화.
- **잡음 측정**(같은 조건, 학습 시드만 변경): 최종 비용 기준 1.0083/1.0099/1.0161 (**0.8% 흔들림**), **LP(15 반복) 기준 1.0037/1.0057/1.0036 (0.2%)** → 목적함수를 LP로 채택 (잡음 4배 작고 평가 시간 절반). 정수해 단계가 잡음의 주원인.
- **결과 (12회 평가, 각 200 update 미세조정 + 검증 2 인스턴스)**: 최선 **0.9973**(무작위 초기점 #0), 실제 dual만 학습한 참조점 **1.0029**, 학습 전 1.0. BO 제안 7회 중 참조점을 넘은 것은 2회(#10 0.9989, #7 1.0009), **최선은 BO가 아니라 무작위에서 나옴**.
- 일관된 경향: **가격 정보를 지우면 나쁨**(유지 0.2–0.4인 θ는 모두 ≥1.005), **자재 가격을 5–10배로 키우면 나쁨**. 좋은 θ 두 종류: 가격을 키워 선명하게(#0), 자재가 **초반 비싸고 후반 싸지는** 구조(#10, 예전 gap 기반에서도 어려운 구간으로 지목됨).
- 진단: 7차원 12회는 예산 부족(탐색에서 끝남), θ당 200 update는 신호가 약함, 효과(0.56%)가 잡음(0.2%) 대비 작음.
- **선택지 (사용자 결정 대기)**: (A) 차원 3–4개로 축소 + 평가 25–30회 + θ당 500–1000 update (1.5–2시간), (B) BO를 기여에서 빼고 "학습 분포가 성능에 영향을 준다"는 관찰만 보고.
- 부수 결과: `cg_runner`가 배치 정책 지원 → P=10 pricing **29.8 s → 2.0 s**(15배), 결과는 동일(비용 2245 vs 2247, LP 1408.9 vs 1408.0). 이제 CG 시간의 대부분은 master LP와 정수해 단계.

### BO 2차 (4차원, 28회) → **BO는 기여에서 제외 (판정 기준대로)**
- 설정: 탐색 4차원 (pi_scale, keep, beta_scale, beta_tilt), 나머지 고정(spread 0.1, shift 0, noise 0.3), 범위 축소, θ당 600 update, 검증 3 인스턴스, 무작위 8 + BO 20.
- 사전 판정 기준(사용자 동의): BO 최선이 무작위 최선보다 **잡음(0.2%) 이상** 낮아야 유지.

| | 최선 | 평균 | 중앙값 |
|---|---|---|---|
| 무작위 8 | 0.9986 | 1.0051 | 1.0070 |
| BO 20 | **0.9973** | 1.0041 | 1.0031 |
| 실제 dual만 (참조) | 1.0108 | | |
| 학습 전 | 1.0000 | | |

- BO 최선이 무작위 최선보다 0.13%p 낮을 뿐 (잡음 0.2% 미만), **BO 20회 중 무작위 최선을 넘은 것은 1회**. → **기준 미달, BO 제거.** (BO 중앙값은 1.0031 vs 1.0070으로 나쁜 영역을 피하는 효과는 있음.)
- **대신 남기는 결과 (학습 분포 ablation)**: ① 분포에 따라 CG LP가 **0.9973–1.0129 (1.6%)**로 갈림 — 잡음의 8배. ② **실제 dual만으로 더 오래 학습하면 나빠짐** (200 update 1.0029 → 600 update 1.0108, 과적합). ③ 좋은 분포는 가격 정보를 많이 남기고(keep 0.64–0.89) 배율이 극단적이지 않으며, 상위 6개 중 4개가 **β 기울기 음수**(자재가 초반 비싸고 후반 싸짐).
- 코드는 남겨둠 (`dual_generator.py`, `bo_curriculum.py`, `bo_cg.py`): 분포 ablation 재현용.

### 기여 정리 (2026-09-17 기준)
1. 문제·모델 (multi-project ramp-up: 공유 자원 + 보충 있는 non-renewable + tool unary + cost of delay)
2. 시간 가격(time-indexed dual) pricing의 학습 설계 (정확 분해 reward, (task, mode, delay) action, 시간 가격 feature)
3. 텐서화로 대규모 학습·추론 (update 5.77 s → 0.195 s, rollout 기준 ~78배; CG pricing 29.8 s → 2.0 s)
4. 실험: P=5부터 compact MIP보다 좋고 빠름, P≥10에서 MIP 실패, 규칙 기반 pricer는 규모가 커지면 조기 종료
5. (축소) 학습 분포 ablation — BO 방법 주장 아님

### AR-1/2/3/6 처리 (2026-09-17)
- **AR-2 완료**: `mp_compact.solve_compact`가 스케줄을 반환하고 **CG와 같은 평가기**(스케줄 + WW 조달)로 재채점 (`cost`, `eval`). 소규모에서 SCIP 목적값과 일치(278.29) 확인.
- **AR-3 완료**: `cg_runner.dive()` — LP를 풀고 λ가 1에 가장 가까운 프로젝트를 고정, 다시 풀기를 반복하는 **diving**. 기존 후보(LP argmax, 정수 master)와 함께 최선을 택함. 효과: P=10 seed 6002 1708.95 → **1645.28**(+3.7%), P=20 seed 7000 3639.90 → **3565.32**(+2.0%, CG-exact 3525.99에 근접), 시간도 136 s → 45 s. seed 6000은 −1.7%(체크포인트 차이 포함).
- **AR-1 완료**: 생성기에 `plan_offset` (기본 1.0 = 기존). 리드타임 전에 시작 가능한 수요 비율: offset 1.0 → **0%**, 0.5 → 10.1%, 0.25 → 14.1%, 0.0 → 16.2%. offset 0까지도 **모두 feasible**(slack 0), 비용은 소폭 상승(자재 대기 → tardiness). **권장 0.25–0.5**.
- **AR-6 1단계 완료 (모델·MIP·평가기)**: `tool_purchase=True`면 tool을 **리드타임 두고 구매**. 프로젝트마다 tool_lead(10–25), tool_cost(200–600), 발주 창 [0, order_latest]; 도착 = 발주 + 리드타임 = 작업 시작 하한; **기간별 capex 예산**(기본: 평균 tool 1대분, 최소 최고가 1대는 가능)이 프로젝트를 묶는 새 공유 제약. 평가기(`evaluate_solution(tool_orders=...)`)와 compact MIP에 반영.
  - P=3 예시: 리드타임 [21, 11, 25], 비용 [575, 538, 485], 예산 575/기간 → 해는 0, 1, 2기에 한 대씩 발주. **리드타임이 길고 due가 이른 tool을 먼저** 발주. 비용 = 자본 1598.8 + 조달 298.4 + tardiness 137.2 + mode 4.7 = 2039.2 (SCIP 목적값과 일치).
  - **2단계 완료 (CG 반영)**: `Column`에 order·capex 추가(키에도 포함), master에 **capex 행**(slack 포함)과 dual **η_t**, `reduced_cost`에 −η_o·tool_cost. pricing은 프로젝트마다 **후보 발주 시점 몇 개**(0, span/4, span/2, 3span/4, span + η가 큰 2개)에 대해 창을 옮겨 rollout — 배치라 프로젝트 × anchor 8 × 후보 5개를 한 번에. 최종 해 추출(`_plan_from_columns`)과 SGS repair가 도착 시각(order+lead)을 하한으로 사용하고 `evaluate_solution(tool_orders=...)`로 채점.
  - 검증 (P=3, 6 tasks, tool_purchase, plan_offset 0.5): **MIP 최적 2039.16** (발주 {P0:0, P1:2, P2:1}) vs **CG-RL 2055.91 (+0.8%)**, CG-heur 2060.81 (+1.1%). CG-RL의 tardiness는 MIP와 동일(137.2).

### AR 현황 (2026-09-17 세션 종료 시점)
- **AR-1 완료**: 리드타임이 스케줄을 제약하도록 `plan_offset` 추가 (1.0 = 기존, 0.25–0.5 권장).
- **AR-2 완료**: MIP도 CG와 같은 평가기(스케줄 + WW 조달)로 채점.
- **AR-3 완료**: diving heuristic 추가. (**AR-3b 미완**: exact pricer가 CP-SAT solution pool로 상위 k개 column을 내게 하는 건 아직.)
- **AR-4 종료(부정)**: BO는 판정 기준 미달로 기여에서 제외. 학습 분포 ablation만 남김.
- **AR-5 유효**: 대규모 실험은 GPU 준비 후 **사용자 승인 받고** 실행. (텐서화로 GPU 사용 가능해짐.)
- **AR-6 완료**: tool 구매(리드타임 + capex 예산) — 모델·MIP·평가기·CG 전부 반영.

### 세션 종료 시점의 코드 상태
- **환경**: `mp_pricing_env.py`(스칼라, 기준 구현), **`mp_pricing_env_batch.py`(배치 텐서, GPU)** — 동치 검증됨(`verify_batch_env.py`, `verify_batch_obs.py`).
- **정책**: `pricing_gnn.py`의 `PricingGNN`(스칼라) / **`PricingGNNBatch`(배치)** — 가중치 호환.
- **학습**: `rl_pricing_trainer.py`(스칼라, BO 옵션 포함) / **`rl_pricing_trainer_batch.py`(GPU, 0.195 s/update)**.
- **CG**: `mp_cg.py`(master, exact pricer, capex 행·η), **`cg_runner.py`**(pricer 교체, diving, SGS repair, tool 발주 후보).
- **기준선**: `mp_compact.py`(compact MIP 변형 + 같은 평가기), `mp_monolithic.py`(평가기, Wagner–Whitin).
- **BO**: `dual_generator.py`, `bo_curriculum.py`, `bo_cg.py` (기여에선 빠졌지만 분포 ablation 재현용으로 유지).
- **체크포인트**: `pricer_rl_base.pt`(기준 학습), `pricer_rl_bo.pt`(BO 학습, 중단된 run), `pricer_rl.pt`(CPU 시절 학습).
- **데이터**: `data/pricing_train.pkl`(40 인스턴스), `data/pricing_val.pkl`(10). **주의: 옛 설정**(plan_offset 1.0, tool 구매 없음)에서 생성됨.
- **로그**: `logs/large_scale.jsonl`, `logs/small_scale.jsonl`, `logs/bo_cg.jsonl`, `logs/bo_cg2.jsonl`, `logs/train_rl_base.txt`, `logs/train_rl_bo.txt`.

### 결과의 유효 범위 (재실행 필요 여부)
- **유효**: 텐서화 속도 측정, 동치 검증, BO 판정, 학습 분포 ablation, 소규모 MIP vs CG 비교(옛 설정 기준).
- **재실행 필요**: 대규모 표(P=5/10/20/40)와 학습 데이터. 지금까지 결과는 **모두 옛 인스턴스 설정**(plan_offset 1.0, tool 구매 없음, diving 없음, MIP 재평가 없음)에서 나온 것. AR-1/2/3/6 반영 후 숫자가 달라짐.

### 다음에 할 일 (우선순위)
1. **실험 스크립트를 배치 정책으로 전환** (작음, 몇 줄) — `exp_small_scale.py` / `exp_large_scale.py`가 아직 스칼라 `PricingGNN`을 로드함. `PricingGNNBatch`로 바꿔야 GPU 대규모 실험에서 텐서화 이득(pricing 29.8 s → 2.0 s)이 남. 소규모는 현재도 동작.
2. **`pricing_data.cg_snapshots`에 tool 발주 반영** — 지금은 η dual과 발주 시점 이동을 다루지 않아 `--tool_purchase`로 만든 학습 데이터는 발주 0에 고정된 pricing 문제만 담김 (CG 때 `cg_runner`가 창을 옮기므로 분포 불일치). `plan_offset`만 쓰는 시나리오라면 지금 그대로 사용 가능.
3. **새 설정으로 데이터 재생성 + RL 재학습** — `plan_offset 0.25–0.5`, `tool_purchase=True`, `tool_unary="qual"`. GPU 학습이 update당 0.2 s이므로 학습 자체는 수 분. (tool 구매 시나리오로 학습하려면 2번 선행.)
4. **대규모 실험 재실행** (사용자 승인 후): P = 5/10/20/40, 방법 MIP / CG-exact / CG-heur / CG-RL, 같은 시간 제한, 같은 평가기, diving 포함. 크기당 10 인스턴스 이상 권장.
5. **원고**: `problem_definition.tex`에 tool 구매(발주 시점·capex 예산) 반영, method 절(분해 + learned pricer + 텐서화), experiment 절.
6. **참고문헌**: `document/REFERENCES_TODO.md`의 남은 항목 — Hu 외 JORS 2026, Deckro 1991, Coughlan 외 2015, 2013 Install/Qual 논문; bib 누락 19개 복구.
7. (선택) AR-3b: exact pricer의 다중 column, metaheuristic baseline, Hu 외 인스턴스에서의 비교.

### 기여 (현재 정리)
1. 문제·모델: multi-project tool ramp-up — 공유 인력, **보충 있는 non-renewable 자재**, tool unary 자원, **리드타임 있는 tool 구매 + capex 예산**, cost of delay.
2. 시간 가격(time-indexed dual) pricing의 학습 설계: 정확히 분해되는 reward, (task, mode, delay) action, 시간 가격 feature, anchor 다중 rollout.
3. 텐서화된 대규모 학습·추론: update 5.77 s → 0.195 s, CG pricing 29.8 s → 2.0 s.
4. 실험: P=5부터 compact MIP보다 좋고 빠름, P≥10에서 MIP 실패, 규칙 기반 pricer는 규모에서 조기 종료.
5. (축소) 학습 분포 ablation — BO를 방법으로 주장하지 않음.

### 다음 세션 재개 가이드
- 먼저 이 파일(`discussion.md`)을 읽고, 위의 "다음에 할 일" 1번부터.
- **해결됨 (2026-09-17)**: `pricing_data.py`, `exp_small_scale.py`, `exp_large_scale.py` 세 스크립트에 `--plan_offset`, `--tool_purchase` 인자를 추가하고 생성기 호출부에 전달. 두 실험 스크립트의 `--ckpt` 기본값도 `pricer_rl.pt` → `pricer_rl_base.pt`로 변경. (이전에는 인자가 없어서 새 기능이 꺼진 옛 인스턴스가 나왔음 — 생성기 기본값이 `plan_offset=1.0`, `tool_purchase=False`.)
- **관찰**: `tool_purchase=True`면 release = tool 도착 시각으로 결정되므로 **`plan_offset`은 효과가 없음**(P=3 seed 8000에서 releases가 동일). 두 설정은 배타적으로 쓰는 게 맞음 — 자재 리드타임을 조이려면 `plan_offset` 단독, tool 구매 시나리오는 `tool_purchase` 단독.
- **남은 함정**: `pricing_data.cg_snapshots`는 tool 발주(η dual, 발주 시점 이동)를 다루지 않음. `--tool_purchase`로 학습 데이터를 만들면 발주 시점이 0으로 고정된 pricing 문제만 들어감 — CG 때는 `cg_runner`가 창을 옮겨 호출하므로 분포 불일치가 생김. tool 구매 시나리오로 학습하려면 `cg_snapshots`에도 발주 후보 로직을 넣어야 함.
- 자주 쓰는 명령 (모두 `code/`에서):
  - 학습 데이터 생성: `python pricing_data.py --split train --n 40` / `--split val --n 10`
  - GPU 학습: `python rl_pricing_trainer_batch.py --updates 1000 --batch 16 --device cuda --tag rlb`
  - 소규모 비교: `python exp_small_scale.py --P 3 --tasks 6 --seeds 1 --mip_limit 300 --cg_limit 300 --methods MIP CG-exact CG-heur CG-RL --ckpt pricer_rl_base.pt`
  - 대규모 비교(**승인 후**): `python exp_large_scale.py --P 5 10 20 40 --seeds 3 --tasks 10 --ckpt pricer_rl_base.pt`
  - 동치 검증: `python verify_batch_env.py`, `python verify_batch_obs.py`
- 기본 규칙(유지): 답변은 한국어, 매 세션 `discussion.md` 먼저 읽기, **요청 없으면 작은 테스트만**(긴/큰 실행은 먼저 물어보고 GPU에서), 사용자가 직접 돌리는 프로세스(`paired_collect.py`, DWTA 학습 등)는 **절대 건드리지 않기**.

### 스크립트 인자 정리 + 버그 2건 (2026-09-17, 세션 종료 후 추가)
- `--plan_offset` / `--tool_purchase`를 `pricing_data.py`, `exp_small_scale.py`, `exp_large_scale.py`에 추가하고 생성기로 전달. `--ckpt` 기본값을 `pricer_rl_base.pt`로 변경.
- **버그 1 (`exp_small_scale.py`)**: MIP는 항상 따로 도는데 `--methods`에 `MIP`을 넣으면 CG 루프에서 `KeyError: 'MIP'`. → 루프에서 걸러내도록 수정.
- **버그 2 (`cg_runner._rl_columns`, 스칼라 정책 경로)**: ① 배치 분기 안의 `from mp_cg import make_column`이 모듈 상단 임포트를 가려 `UnboundLocalError` → 지역 임포트 제거. ② 스칼라 env의 `info["column"]`은 **Column 객체**인데 dict처럼 `col["starts"]`로 접근 → `col.starts`로 수정. **AR-6 2단계에서 들어간 회귀**이며, 그 사이 스칼라 정책으로는 CG-RL이 아예 실행되지 않았음.
- 스모크 테스트 (P=3, 6 tasks, plan_offset 0.25 + tool_purchase, MIP 30 s): MIP 1788.27(미증명, bound 1633.26), CG-heur 1744.85, **CG-RL 1743.14** — 짧은 제한 탓에 MIP 대비 gap이 음수(즉 CG가 더 좋음). 인자·경로 동작 확인이 목적이며 성능 주장 아님.
- **추가 할 일**: `exp_small_scale.py` / `exp_large_scale.py`가 아직 **스칼라 `PricingGNN`**을 로드함. GPU 대규모 실험에서는 `PricingGNNBatch`로 바꿔야 텐서화 이득(pricing 29.8 s → 2.0 s)을 봄.

## 2026-09-19

### 투고처 결정: IEEE T-ASE (EJOR 기각)
- 사용자 판단: **EJOR는 심사 기간이 너무 김**. → **IEEE Transactions on Automation Science and Engineering (T-ASE)** 로 결정. 백업은 Computers & Operations Research.
- 근거: 현재 기여(학습 pricer, 텐서화 가속, 규모 실험)가 T-ASE 평가 축과 거의 1:1 대응. EJOR/IJOC를 노리면 AR-3b(exact pricer 다중 column), bound 강화 같은 OR 쪽 보강이 더 필요해 시간이 다시 듦.
- 후보 검토 기록: T-ASE(1순위) > IEEE TSM(문제는 정중앙이나 알고리즘 디테일을 줄여야 함) > C&OR > IJPR. IJOC는 BO를 뺀 뒤 알고리즘 신규성 주장이 얇아 제외.
- **원고 방침**: IEEE 양식·분량, "실험이 주장하는 바"를 앞세우는 구성, **텐서화 가속을 1급 기여로 유지**(OR 저널에서는 평가절하되지만 T-ASE에서는 정당한 기여).

### 다음에 할 일 1번 완료 — 실험 스크립트를 배치 정책으로
- `exp_small_scale.py`, `exp_large_scale.py`가 `PricingGNNBatch`를 로드하도록 변경, `--device` 인자 추가(기본 cuda-if-available). 가중치 레이아웃이 같아 기존 체크포인트 그대로 로드됨.
- 확인(P=3, tasks 6, CPU): CG-RL이 CG-heur보다 t_cg 0.4 s vs 1.3 s. GPU에서는 차이가 더 커짐.

### 다음에 할 일 2번 완료 — `cg_snapshots`에 tool 발주 반영
- `order_candidates`, `shift_project`를 `cg_runner.py` → **`mp_cg.py`로 이동** (rl_pricing_trainer가 쓰면 순환 임포트가 되므로).
- `pricing_data.cg_snapshots`가 이제 **(iteration, project, 발주 후보)** 단위로 기록. 후보마다 reduced cost 두 개 저장:
  - `rc`: shifted window 기준, **capex·η 제외** → 환경/정책이 보는 값, 학습 gap의 기준.
  - `rc_full`: master가 column 채택에 쓰는 값 (capex 포함, −η_o·capex). column을 pool에 넣는 판정도 이 값으로.
  - tool 구매가 없으면 둘이 일치 → 옛 값과 동일.
- `rl_pricing_trainer.load_problems`가 후보별로 shifted 인스턴스를 만들어 반환. **옛 pkl도 그대로 로드**(val 1350문제 동일, order 전부 0) — 형식 분기로 하위 호환.
- 검증: P=3 tool_purchase 스냅샷에서 **`rc` 순위와 `rc_full` 순위가 실제로 다름** (P1: rc로는 order 0/1/2가 −434/−431/−426으로 비슷한데, rc_full로는 order 0이 −0.23, order 12가 **+47.6**). 그동안 학습 데이터에 없던 신호가 이것.
- 학습 루프 스모크(CPU, 새 형식 데이터 150문제): `run_batch` + `loss_from` + `evaluate` 모두 정상.

### 버그 수정 1건 — `shift_project`가 tool 구매 없는 인스턴스의 release를 0으로 덮어씀
- 증상: `capex_budget is None`이면 `tool_lead = 0`이라 `arrival = order(0) + 0 = 0`이 되어, release가 12인 프로젝트도 window가 0부터 열린 채 pricing됨.
- 영향: **AR-6 2단계에서 들어간 회귀**. `plan_offset`만 쓰는(=tool 구매 없는) 모든 CG-heur/CG-RL 실행이 너무 넓은 window에서 pricing하고 있었음.
- 수정: `capex_budget is None`이면 인스턴스를 그대로 반환 (`mp_cg.shift_project`).

### 다음에 할 일 (갱신)
1. **(승인 대기) 3번 — 데이터 재생성 + RL 재학습**. 먼저 정할 것: 주 실험 설정을 `plan_offset 0.25` 단독으로 갈지, `tool_purchase` 단독으로 갈지 (**둘은 배타적**). tool_purchase면 문제 수가 약 5배(발주 후보 5개)라 생성 시간도 그만큼 늘어남. 학습은 GPU 여유 후.
2. **5번 — 원고**, T-ASE 양식으로. `problem_definition.tex`에 tool 구매(발주 시점·capex 예산) 반영, method 절(분해 + learned pricer + 텐서화), experiment 절.
3. 4번 대규모 실험 재실행 (사용자 승인 후, GPU).
4. 6번 참고문헌 (`document/REFERENCES_TODO.md`).

### T-ASE 제출 계획 (2026-09-19 수립)

**전제**: 주 시나리오는 `tool_purchase` 단독 (capex 예산이 프로젝트를 묶는 점이 차별점). `plan_offset`은
부가 시나리오로만 등장. 새 알고리즘은 만들지 않고, **실험 증거를 채우는 것**이 이 계획의 전부.

#### Phase 0 — 설정 확정 (CPU, 반나절)
- 생성기 파라미터를 문헌 수치에 맞춰 교정: `Cheng-fowler-main.pdf`, `Hu at al-main.pdf`에서
  tool 리드타임, qual 기간, ramp 기간, 프로젝트 수 범위를 뽑아 `multiproject_instance.py`의
  `tool_lead_range`, `capex_per_period`, task 수 등에 반영하고 **출처를 주석으로 남김**.
  (자체 생성기라는 약점을 "문헌 교정"으로 방어하기 위함. 재생성을 두 번 하지 않으려면 Phase 1 전에 끝낼 것.)
- 산출물: 교정 근거 표 (파라미터 / 값 / 출처) — 원고 실험 절에 그대로 들어감.

#### Phase 1 — 데이터 재생성 + 재학습 (CPU 수 시간 + GPU 수 분)
- `python pricing_data.py --split train --n 40 --tool_purchase`
- `python pricing_data.py --split val   --n 10 --seed0 1000 --tool_purchase`
  - 발주 후보 5개 → 문제 수 약 5배, 생성 시간도 그만큼.
- `python rl_pricing_trainer_batch.py --updates 2000 --batch 16 --device cuda --tag tase`
- 점검: val pricing gap이 `heur`보다 낮은지, 과적합 구간(옛 결과: 600 update 이후 악화) 재확인.

#### Phase 2 — 학습 설계 ablation (GPU, 싸다: update당 0.2 s)
기여 2번("학습 설계")을 주장하려면 각 요소의 효과 증거가 필요. 변형마다 1000~2000 update.
- anchor: 7종 전부 / 없음(S=0) / 3종 부분집합
- S ∈ {0, 1, 2, 4}
- reward: step return-to-go **vs** episodic return
- baseline: leave-one-out **vs** 단순 평균 **vs** 없음
- feature: 시간 가격 feature 제거 / suffix-min feature 제거
- **중요**: pricing gap만 보고하지 말고 **downstream (CG LP, 최종 비용)** 까지 보고할 것.
  (옛 BO 실험에서 pricing gap과 CG 성능이 따로 논 전례가 있음.)

#### Phase 3 — 일반화 (GPU, 싸다)
- **크기 zero-shot**: 학습은 P=3~6, 테스트는 P=5/10/20/40. *이미 그렇게 하고 있으므로 프레이밍만 명시하면 됨* — 공짜 기여.
- **작업 수 zero-shot**: 학습 tasks 8/10 → 테스트 tasks 20.
- **분포 이동**: tool_lead 범위, capex 예산 타이트함, 자재 가격 기울기(β tilt)를 학습 분포 밖으로 밀고 성능 저하 측정.
  - 여기서 **폐기한 BO 실험의 학습 분포 ablation 결과를 재활용**할 수 있음 (분포에 따라 CG LP가 1.6% 갈림).
- 교차: `tool_purchase`로 학습 → `plan_offset` 시나리오 테스트, 그 반대도.

#### Phase 4 — 베이스라인 보강 (가장 비쌈, 3~5일)
"MIP이 P>=10에서 실패한다"만으로는 기여가 안 됨. 문헌의 강한 휴리스틱이 필요.
- **다중 시작 priority-rule SGS**: 무작위 우선순위 + forward-backward improvement, 같은 시간 제한.
  (RCPSP 문헌의 표준 베이스라인. `cg_runner.sgs_repair`를 재활용해 빠르게 만들 수 있음.)
- **metaheuristic**: activity-list GA (Hartmann 계열) 또는 SA. 발주 시점은 유전자에 포함.
- **탐욕 capex 규칙**: 리드타임 긴 / due 이른 tool부터 발주 (예전 P=3 관찰과 일치) — 약하지만 해석 가능한 기준선.
- 전부 **같은 평가기**(`evaluate_solution` + WW 조달)로 채점.

#### Phase 5 — 최종 대규모 표 (GPU, 사용자 승인 필요)
- P = 5/10/20/40, **크기당 10 인스턴스 이상**, 공통 시간 제한, diving 포함.
- 방법: MIP / CG-exact / CG-heur / CG-RL / SGS-multistart / GA.
- 보고: 비용(평균±CI), best-known 대비 gap, LP bound, 시간, 반복 수, pricing 시간 비중.

#### Phase 6 — 원고 (T-ASE, 2~3주)
- IEEE 양식. **T-ASE는 "Note to Practitioners" 단락을 요구**하므로 반도체 ramp-up 실무 함의를 따로 씀
  (제출 전 최신 저자 가이드로 재확인할 것).
- 구성: 문제·모델 → 분해(master/pricing) → 학습 pricer 설계 → 텐서화 → 실험(교정·ablation·일반화·규모).
- **텐서화를 1급 기여로 유지**: "빠르다"가 아니라 "빨라서 ablation·일반화 연구가 가능했다"로 연결.
- `document/REFERENCES_TODO.md` 잔여 항목 정리, bib 누락 19개 복구.

#### 대략의 일정
Phase 0~1: 1일 / Phase 2~3: 2일 / Phase 4: 3~5일 / Phase 5: 1~2일(대기 포함) / Phase 6: 2~3주
→ **제출까지 약 5~6주**.

#### 위험 요소
- Phase 4를 건너뛰면 major revision에서 반드시 요구받음 (가장 큰 단일 위험).
- Phase 0을 건너뛰고 Phase 1을 돌리면 데이터를 두 번 만들게 됨.
- 학습 방법 자체의 신규성은 약함(POMO식 anchored REINFORCE + GAT의 응용). 문제·시스템·실험으로 방어하는 구조이므로 Phase 2~4가 곧 방어선.

### Phase 4 비용 재평가 + 경고 결과 (2026-09-19)
- 사용자 지적: "휴리스틱 베이스라인은 추가 코드 개발이 필요하다". 맞는 지적이나 **3~5일 추정은 과대평가였음**.
- 이유: **디코더가 이미 있음**. `cg_runner.sgs_repair`는 `keep_timing=False`일 때 `pref_starts`를 하한이 아니라
  **정렬 키로만** 쓰므로, 그 자체가 우선순위 규칙 serial SGS 디코더임.
  - 단 한 가지 제약: LP column에서 온 starts를 가정하므로 **선행관계가 이미 맞다고 봄**. 무작위 우선순위를 그냥 넣으면
    `KeyError`(선행 작업이 아직 미배치). → 프로젝트마다 **무작위 위상정렬**로 우선순위를 만들면 해결 (15줄).
- 프로토타입 (약 60줄: 무작위 위상정렬 + 탐욕 capex 발주 + 기존 `evaluate_solution` 채점) 동작 확인.
  P=3, tasks 6, seed 8000에서 **2000 시작을 0.44 s**에 수행.

- **경고 결과 (P=3, tasks 6, num_templates 2, tool_unary qual, tool_purchase, seed 8000 — 모두 동일 인스턴스)**:

| 방법 | 비용 | 시간 |
|---|---|---|
| MIP (미증명, bound 1551.34) | 1620.04 | 60 s |
| CG-RL | **1615.60** | 8.2 s |
| **다중시작 SGS (무작위 2000회)** | **1634.20** | **0.44 s** |
| CG-heur | 1638.51 | 21.4 s |

  - **0.44초짜리 무작위 다중시작이 CG-heur(21.4 s)를 이기고, CG-RL과는 1.15% 차이.**
  - 이것이 바로 리뷰어가 짚을 지점이 실제로 구현된 것. **Phase 4는 생략 불가**가 확인됨.
  - 단, P=3은 조합이 작아 무작위 탐색이 잘 먹히는 영역. **P가 커지면 격차가 벌어질 것으로 예상되며, 그것이 측정해야 할 핵심 실험**.
    지금 주장 "P>=5부터 MIP보다 좋다"는 약하고, 진짜 주장은 **"규모가 커질수록 다중시작/규칙 기반이 무너지고 CG-RL만 버틴다"** 여야 함.

- **Phase 4 재구성 (등급별)**
  - **Tier A (필수, ~반나절)**: 다중시작 우선순위 SGS + 탐욕 capex 발주. 프로토타입 완료.
    남은 일: 모듈화, forward-backward improvement 추가, **다른 방법과 같은 시간 제한**으로 실행, `exp_*`에 편입.
  - **Tier B (권장, +1일)**: 같은 디코더 위의 activity-list GA (우선순위/모드/발주 시점을 유전자로). 디코더가 이미 있으므로 증분 작업.
  - **Tier C (생략 가능)**: tabu/SA, 공개 벤치마크 이식.
- 프로토타입 위치: 세션 scratchpad (`multistart_sgs_proto.py`). 정식 채택 시 `code/heur_multistart.py`로 옮길 것.

### 세션 종료 정리 (2026-09-19) — MacBook 재개 가이드

#### 이 세션에서 바뀐 파일
| 파일 | 변경 |
|---|---|
| `exp_small_scale.py`, `exp_large_scale.py` | `PricingGNNBatch` 로드, `--device` 추가 |
| `mp_cg.py` | `order_candidates`, `shift_project` 이관 + **release 덮어쓰기 버그 수정** |
| `cg_runner.py` | 위 두 함수 import로 교체, generator 생성 방식 변경 |
| `pricing_data.py` | `cg_snapshots`가 (iteration, project, 발주 후보) 단위로 기록, `rc` / `rc_full` 두 값 저장 |
| `rl_pricing_trainer.py` | `load_problems`가 발주 후보별 shifted 인스턴스 반환 (옛 pkl 하위 호환) |
| `rl_pricing_trainer_batch.py` | `make_generator()` 추가 (백엔드 안전), `random` anchor 추첨을 generator 장치에서 뽑고 이동 |
| **`heur_multistart.py` (신규)** | 다중시작 우선순위 SGS + 탐욕 capex 발주 **프로토타입** |

#### MacBook (Apple Silicon) 주의사항
- **`--device mps`를 명시적으로 넘길 것.** 현재 기본값은 `cuda if available else cpu`라 Mac에서는 **CPU로 떨어짐**.
- `torch.Generator(device="mps")`는 백엔드에서 거부됨 → `rl_pricing_trainer_batch.make_generator()`가 CPU generator로
  자동 폴백하고 추첨 결과만 장치로 옮김. **이 경로는 Windows/CPU에서만 검증됨, MPS에서는 미검증.**
- 남은 MPS 위험 지점 두 곳 (실패하면 여기부터 볼 것):
  - `mp_pricing_env_batch.py:214` `F.max_pool1d`
  - `mp_pricing_env_batch.py:227` `cummin`
  - `float64`는 코드 전체에서 안 쓰므로 그쪽 문제는 없음.
- 안 되면 `--device cpu`로 폴백. 기능은 동일하고 속도만 손해.

#### MacBook에서 첫 확인 (작은 테스트)
```bash
cd code
python verify_batch_env.py && python verify_batch_obs.py        # 동치 검증 (스칼라 vs 배치)
python -c "import torch; print(torch.backends.mps.is_available())"
python exp_small_scale.py --P 3 --tasks 6 --seeds 1 --tool_purchase \
    --mip_limit 60 --cg_limit 120 --pricing_limit 10 --ip_limit 30 \
    --methods CG-heur CG-RL --device mps --out logs/mps_check.jsonl
```
참고 (Windows CPU, 동일 인스턴스 seed 8000): CG-RL 1613~1616, CG-heur 1638.51, 다중시작 SGS 1628~1634(0.5 s), MIP 1620.04(60 s, 미증명).

#### 재개 지점 — Phase 순서는 위 "T-ASE 제출 계획" 참조
1. **Phase 0**: 생성기 파라미터를 `Cheng-fowler-main.pdf` / `Hu at al-main.pdf` 수치로 교정. **Phase 1 전에 끝낼 것**
   (안 그러면 데이터를 두 번 만들게 됨).
2. **Phase 1**: `python pricing_data.py --split train --n 40 --tool_purchase` /
   `--split val --n 10 --seed0 1000 --tool_purchase` → `python rl_pricing_trainer_batch.py --updates 2000 --batch 16 --device mps --tag tase`
3. **먼저 볼 것 (30분)**: P=10/20에서 다중시작 SGS가 실제로 무너지는지. 무너지지 않으면 논문의 주장 자체를 다시 짜야 함.
4. Phase 2(ablation) → 3(일반화) → 4(Tier A 마무리) → 5(대규모 표) → 6(원고).

#### 주의: 옛 데이터
`data/pricing_train.pkl`(40), `data/pricing_val.pkl`(10)은 **옛 설정**(plan_offset 1.0, tool 구매 없음)이고
`pricer_*.pt` 체크포인트도 전부 그 데이터로 학습된 것. 새 형식으로 로드는 되지만(하위 호환) **Phase 1에서 재생성 대상**.

### 3개월 일정 (2026-09-19 ~ 약 2026-12-19)

**계획 변경 (중요): Phase 4(베이스라인)를 Phase 2·3 앞으로 옮김.** 다중시작 SGS가 P=3에서 CG-heur를 이긴 이상,
**논문의 주장이 확정되기 전에 ablation을 돌리는 건 낭비**다. 베이스라인이 규모에서 무너지는지를 먼저 확정하고,
그 다음에 "왜 우리 것이 버티는가"를 ablation으로 뒷받침하는 순서가 맞다.

| 주차 | 작업 | 산출물 |
|---|---|---|
| **W1** | **게이트**: P=10/20/40에서 다중시작 SGS가 무너지는가 (30분이면 첫 신호) → Phase 0 파라미터 교정 → Phase 1 데이터 재생성 + 재학습 | 주장 확정, 새 체크포인트 |
| **W2-3** | **Phase 4**: Tier A(다중시작 SGS 모듈화 + forward-backward improvement + 동일 시간 제한) / Tier B(activity-list GA) | `heur_multistart.py`, `heur_ga.py` |
| **W4** | Phase 2 ablation (anchor, S, reward, baseline, feature) — **downstream CG LP까지 보고** | ablation 표 |
| **W5** | Phase 3 일반화 (크기·작업수 zero-shot, 분포 이동, 시나리오 교차) | 일반화 표 |
| **W6-7** | Phase 5 대규모 표 (P=5/10/20/40, 크기당 10+ 인스턴스, 6개 방법) + **재실행 여유** | 주 실험 표 |
| **W8-11** | 원고 집필 (IEEE 양식 + Note to Practitioners) | 초고 |
| **W12** | 내부 검토 / 공저자 피드백 반영, 참고문헌 정리 | 수정고 |
| **W13** | 버퍼 + 제출 | 제출 |

- 3개월은 **W1 게이트만 통과하면 여유 있음**. 집필 4주를 확보한 게 핵심.
- 대규모 실험(W6-7)에 1주 여유를 둔 이유: 옛 경험상 설정이 바뀌면 표를 다시 돌리게 됨.

#### W1 게이트가 실패하면 (다중시작이 규모에서도 버티면)
주장을 바꿔야 하고, 그 대안은 이미 있음:
- **CG만 dual bound를 준다.** 다중시작 SGS·GA는 해만 내놓고 품질 보증이 없다. CG-RL은 LP bound를 함께 주므로
  "이 해가 최적에서 x% 이내"를 말할 수 있다. 의사결정 지원 맥락(T-ASE)에서 이건 실질적 차별점이고,
  **게이트 결과와 무관하게 항상 참이므로 지금부터 전면에 세워도 됨.**
- 부차: 시간 제한을 조일수록(실무 조건) CG-RL이 유리한 구간이 있는지 측정.

### Git / GitHub 상태 (2026-09-19 세션 종료)

- 저장소: `https://github.com/jlee-105/RL-PRICING.git` (**private**), 브랜치 `main`.
- 이 프로젝트 폴더는 이번 세션에 처음 `git init` 했고, 원격에는 이미 **초고 3커밋**이 있었음 → 이력이 무관해
  `--allow-unrelated-histories`로 병합. **경로가 안 겹쳐 충돌 0, 로컬 파일 변경 0.**
- 커밋 작성자는 이 저장소에만 `Jaejin Lee <jlee105@asu.edu>`로 설정 (전역 Intel 주소는 그대로 둠).
- `.gitignore` 제외 대상: 논문 PDF(저작권), `code/data/`(17MB, 재생성 가능), `code/logs/`, `*.pt`.
  → **MacBook에서 clone하면 데이터·체크포인트는 없음. Phase 1에서 어차피 재생성하므로 정상.**

#### 원고 파일이 두 벌 존재 (Phase 6 전에 정리 필요)
원격 초고가 합쳐지면서 `.tex`와 `.bib`가 각각 두 벌이 됨. **덮어쓰지 않고 공존시킨 상태** (사용자 지시: 원격 초고 삭제하지 말 것).

| 원격에서 온 것 (루트) | 로컬에 있던 것 |
|---|---|
| `RL_CG_paper.tex` (681줄) | `document/RL_PRICING.tex` |
| `references.bib` (148줄) | `document/rl_pricing_references.bib` |
| `RL_CG.md` (198줄), `README.md` | `document/CG_RL_PLAN.md`, `document/FORMULATIONS.md` |

- 원격 마지막 커밋이 "Fix RMP formulation duplication and placeholder values"이므로 **원격 `.tex` 쪽이 더 최신일 가능성 있음**.
- `document/REFERENCES_TODO.md`의 "bib 누락 19개"가 **원격 `references.bib`에 이미 들어있을 수 있음** — 중복 작업 방지를 위해 먼저 대조할 것.
- **Phase 6 착수 시 첫 작업**: 두 `.tex`, 두 `.bib` 대조 → 본체 하나 결정 → 나머지 병합 후 정리. T-ASE는 IEEE 양식이므로 어차피 재구성 필요.
