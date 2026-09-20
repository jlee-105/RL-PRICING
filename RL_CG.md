# RL-CG Pricing Paper — Working Notes

Paper: "Learning to Price: Graph Reinforcement Learning for Column Generation in
Integrated Scheduling and Inventory Optimization" (Jaejin Lee, 2nd paper,
early draft as of 2026-07-08). Problem: MRCPSP-P (multi-mode RCPSP + procurement)
for semiconductor tool ramp-up (Install/Qual), solved via column generation with
a GNN policy replacing CP-SAT pricing.

## Status: Related Work, Section 2, "(2) combining with procurement/supply chain" story point

Original sentence in the draft:
> Closer to our setting, a smaller body of work integrates RCPSP with
> supply-chain and procurement decisions: via logic-based Benders
> decomposition~\citep{hooker2007planning}, alternating
> heuristics~\citep{dauzere2004integrated}, column
> generation~\citep{vanderbeck1996exact}, joint scheduling-material
> planning~\citep{dodin2007integrated}, and discounted-cash-flow-aware
> supply-chain RCPSP~\citep{supplychain2024rcpsp}.

## Citation problems found (verified via WebSearch against real sources)

All 5 citations in that sentence have problems — these were introduced by
Claude in a prior session, not the user:

- **`hooker2007planning`**: bib entry is actually Hooker & Ottosson (2003,
  *Mathematical Programming* 96(1):33–60, "Logic-based Benders decomposition")
  — real paper, but key says "2007" (mismatch) and it has NOTHING to do with
  procurement/supply-chain. Wrong pairing.
- **`vanderbeck1996exact`**: real paper (Vanderbeck & Wolsey, *Operations
  Research Letters* 19(4):151–159, 1996, "An exact algorithm for IP column
  generation") but it's a general bin-packing/cutting-stock CG methodology
  paper, unrelated to RCPSP or procurement. Wrong pairing.
- **`dauzere2004integrated`**: bib lists it as Dauzère-Pérès & Lasserre
  (2004 Springer book) "Integration of Lot-Sizing and Scheduling Decisions in
  a Job-Shop" — real authors work on lot-sizing/job-shop integration, but this
  exact title/year/format could not be verified as existing, and even if real
  it's about lot-sizing, not procurement/supply-chain. Likely fabricated or
  at minimum mischaracterized.
- **`dodin2007integrated`**: title matches a real paper but wrong
  year/volume — real paper is Dodin & Elimam, *IIE Transactions*, 2001 (bib
  currently says 2008, vol 40 no 10 pp 967-978, which belongs to a different
  Dodin & Elimam paper on facility planning).
- **`supplychain2024rcpsp`**: wrong authors — bib says "Taheri, Khademi Zare,
  Pishvaee, Bozorgi-Amiri" but the real paper with this exact title is
  **Asadujjaman, Rahman, Chakrabortty, Ryan (2024), Computers & Industrial
  Engineering 194:110380** (bib also has wrong vol/pages: 196/110490).

User confirmed on seeing this: "i guess all hallucination" — same pattern as
previously found in this paper's ML-for-CG citations (Hoornaert→Abouelrous,
etc., see bib file's existing CORRECTED comments).

**Cross-check**: the OTHER paper (litho RL, submitted to IEEE Access
2026-07-08) was independently verified — all 48 references confirmed real and
correctly attributed, NO hallucination pattern found there. So this is not a
universal problem with the author's papers, just this specific sentence/paper.

## Decision so far
User wants to fix this properly rather than just delete-and-move-on: first
understand what the real RCPSP+procurement literature actually contributes
(what's added vs. plain RCPSP, how it's solved, why) before rewriting the
sentence. Requested recent (last 5 years, 2021-2026) papers specifically,
enumerated via WebSearch + Google Scholar-style queries so user can review
one by one.

## Candidate replacement papers (2021–2026), verified via WebSearch, not yet vetted by user

**RCPSP + material procurement/supply-chain (core stream):**
1. Asadujjaman, Rahman, Chakrabortty, Ryan (2021), "Resource constrained
   project scheduling and material ordering problem with discounted cash
   flows," *Computers & Industrial Engineering* 158:107427.
2. Parchami Afra, Kheirkhah, Ahadi (2022), "Systematic literature review of
   integrated project scheduling and material ordering problem: Formulations
   and solution methods," *Computers & Industrial Engineering* 173:108711.
   — survey, good for citing the whole stream at once.
3. Tian, Zhang, Demeulemeester, Chen, Ali (2023), "Integrated
   resource-constrained project scheduling and material ordering problem
   considering storage space allocation," *Computers & Industrial
   Engineering* 185:109608.
4. Pouramin, Mirzazadeh, Davari-Ardakani, et al. (2024), "Multi-mode
   resource-constrained project scheduling problem along with material
   ordering under time-of-use electricity tariffs and carbon taxes," *Annals
   of Operations Research*.
5. Asadujjaman, Rahman, Chakrabortty, Ryan (2024), "Supply chain integrated
   resource-constrained multi-project scheduling problem," *Computers &
   Industrial Engineering* 194:110380. (This is the REAL paper behind the
   fabricated `supplychain2024rcpsp` entry above.)
6. Hu, Liu, Demeulemeester, Zhou (2025), "Multi-project scheduling and
   material ordering with time varying supplier capacities," *European
   Journal of Operational Research* (also on SSRN 5187214).
7. Hu, Liu, Demeulemeester, Zhou (2026, online 2026-04-21), "A two-stage
   robust optimisation model for multi-project scheduling and material
   ordering under demand uncertainty," *Journal of the Operational Research
   Society*. Uses column-and-constraint generation (C&CG) + **Benders
   decomposition** — real candidate to replace the "logic-based Benders
   decomposition" claim in the original sentence.

**Benders decomposition + RCPSP (method match, procurement not confirmed):**
8. (Annals of Operations Research, 2023) "A Benders decomposition algorithm
   for the multi-mode resource-constrained multi-project scheduling problem
   with uncertainty" — authors not yet confirmed, needs follow-up search.

**Column generation / branch-and-price + RCPSP (method match, no procurement,
but directly relevant to this paper's own CG methodology):**
9. Kolter, Grunow, Kolisch (2025), "On Branch-and-Price for Project
   Scheduling," arXiv:2501.04563. Analyzes why branch-and-price is often
   ineffective for RCPSP — important nuance for this paper to address/cite
   given it also uses a CG/pricing approach for MRCPSP-P.

Two research groups doing consecutive work: Asadujjaman/Rahman/Chakrabortty/
Ryan (#1, #5) and Hu/Liu/Demeulemeester/Zhou (#6, #7).

## Next step (RESOLVED — see session update below)
User will read items 1-9 one by one (abstracts/methodology) and decide which
to actually cite, then we rewrite the Related Work sentence and fix the .bib
file (remove the 5 bad entries, add the chosen replacements).

## Session update, 2026-07-08 (continued)

Working files created (previously only pasted into chat, no local path
existed): `RL_CG_paper.tex` (full paper source; note the first paste was
truncated by a 50k-character chat limit right after `\bibliography{references}`,
so nothing past that point — the commented-out legacy bibliography block and
`\end{document}` — was ever received) and `references.bib` (built up
incrementally from pasted .bib chunks).

### Citation fixes applied
Reviewed candidates 1-9 from the list above with the user and settled on 5
real, verified replacements for the fabricated Related Work sentence:
`asadujjaman2021dcf`, `parchamiafra2022review`, `tian2023storage`,
`asadujjaman2024scircmpsp`, `hu2026robust`. All added to `references.bib`
with verified bibliographic details.

While fixing this, found **two more citation problems** in the same paper,
independent of the original 5:
- `hartmann2022updated` and `khajesaeedi2025review` (cited in the RCPSP
  intro sentence) were real papers but simply missing from the .bib file —
  not hallucinated, just undefined. Added both.
- `mrcpsp2014splitting` had **wrong author attribution**: bib said
  Ranjbar/Kianfar/Shadrokh (2014), C&OR 53:24-30, but the real paper with
  that exact title is **Cheng, Fowler, Kempf, Mason (2015), C&OR 53:275-287**
  — the same author group as `ieee2013semircpsp` (the semiconductor
  ramp-up paper), very likely the journal-length methodological companion to
  that WSC 2012 conference paper. Renamed key to `cheng2015splitting`, fixed
  metadata, and added a sentence introducing it in the Semiconductor
  paragraph as the same group's follow-up work.
- `park2023semirl`: citation key implies "Park" but the real paper
  (arXiv:2302.07162, WSC 2023, "Semiconductor Fab Scheduling with
  Self-Supervised and Reinforcement Learning") is by **Tassel, Kovács,
  Gebser, Schekotihin, Stöckermann, Seidel** — no Park among the authors.
  **NOT YET FIXED** — key rename + author correction still needed.
- Also flagged (not yet resolved): `bitar2024cpqualification` and
  `wang2025semicapacity` are real papers but topically about *different*
  semiconductor scheduling problems (production-time machine-qualification
  constraints, and capacity planning, respectively), not tool ramp-up
  specifically. User was searching Google Scholar for better ramp-up-specific
  replacements when the session ended — pick this up next time.

### Editorial rules the user set for this paper (apply going forward)
- **Never use em dashes (`---`)** in the prose. Restructure with subordinate
  clauses instead of dashes or parentheses (parenthetical asides were
  rejected too — "looks unprofessional").
- **Don't write a lumped "gap" sentence** after a group of citations (e.g.
  "None of these combine X, Y, and Z simultaneously..." or "Across all N
  studies..."). Attach each paper's specific limitation to that paper's own
  sentence instead. Rejected multiple times when this was violated.
- Don't repeat the same point (e.g. mode selection) across multiple
  individual citations if it was already stated once — keep weight
  proportional to how central it actually is to the paper's contribution.
- Domain-specific Related Work subsections should follow: (1) plain-language
  explanation of the domain with a citation, (2) analogy connecting it back
  to the general problem class discussed earlier, (3) prior work specific to
  the domain — in that order.
- Removed the duplicated Related Work block that existed at the start of
  this session (two near-identical paragraphs covering RCPSP+procurement) —
  kept one.

### Still open for next session
- Fix `park2023semirl` citation key + author list (real authors: Tassel et al.).
- Decide fate of `bitar2024cpqualification` and `wang2025semicapacity` —
  user was actively Google-Scholar-searching for better ramp-up-specific
  replacements themselves.
- Revisit whether individual gap statements are needed for
  `bitar2024cpqualification`, `park2023semirl`, `wang2025semicapacity` (user
  said "not yet" partway through, may want them now).
- Rest of the paper (Introduction, Problem Definition, Methodology,
  Experiments, Discussion, Conclusion) still has em dashes and hasn't had a
  citation-accuracy pass — user declined to extend the em-dash cleanup there
  for now, scope was Related Work only this session.

## Separately: still open from earlier in this session
- Related Work Section 2 has a duplicated block (intro-style paragraph
  covering the same ground as the later `\paragraph{}`-structured version).
  User decided to keep the current draft as-is for now ("my current writing
  seems better") — not resolved, just deferred.
- `.bib` file location was never given as an absolute path — user has been
  pasting file contents directly into chat each time. If a project folder is
  now being set up, get the real path and work off the actual file with
  Edit/Read instead of copy-paste.
