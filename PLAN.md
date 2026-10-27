# CTR / GNN Recommender — KDD Hackathon Plan

## Datasets in play
- **AmazonBook** (PyG prebuilt): 52,643 users × 91,599 items, 2,380,730 edges.
  - LightGCN/NGCF format. `train.txt`/`test.txt`: `user_id item1 item2 ...`
  - **ID-only, no features, no timestamps, no ratings, positive-only (implicit).**
- **Amazon-Electronics 2023** (load_data.py): has text/title/image URLs + ratings + timestamps. Feature-rich. Real cold-start target.
- **H&M** (downloading): rich features + timestamps + transactions.

## Key constraint
AmazonBook cannot demo cold-start or temporal-leakage (no features/timestamps).
Use it for GNN plumbing validation only. Use Electronics/H&M for the real CTR + cold-start story.

## Phases
1. [DONE] Concepts locked (hetero vs homo, bipartite) — explained in chat
2. [DONE] EDA + imbalance diagnostics — `eda_amazonbook.ipynb` (21 cells, runs clean)
       Outputs: `amazonbook_eda_summary.csv`
3. [ ] Data schema: define CTR task, negatives, temporal split
4. [ ] Baselines: popularity → MF → LightGCN  <-- NEXT
5. [ ] Feature-rich model: two-tower / GNN + content features (cold-start)
6. [ ] Leakage guards + evaluation protocol
7. [ ] Demo packaging for KDD

## EDA findings (AmazonBook)
- Density 0.049% (very sparse). Avg 45 inter/user, 26/item.
- **k-core pre-filtered**: min user deg=16, min item deg=1 => cold users/items already removed.
- Item Gini 0.488 (moderate — masked by filtering; raw Amazon ~0.9+).
- Top 16% items = 50% interactions; head 5% = 27.7%; bottom 50% = 19%.
- **0 cold users, 0 cold items in test** — cannot demo cold-start here.
- Spearman(train_pop, test_pop)=0.245 — mild popularity bias.
=> AmazonBook = clean structural CF baseline only. Cold-start/leakage/long-tail story MUST come from Amazon-Electronics.

## Decisions
- Primary dataset for CTR+cold-start+leakage: **Amazon-Electronics 2023**.
- AmazonBook = LightGCN structural baseline.

## Artifacts (in /home/jupyter/ankit/ctr_recsys/)
- `eda_amazonbook.ipynb` — executed EDA
- `amazonbook_eda_summary.csv` — slide numbers

---

## Phase 6b — Score Calibration for CTR (protect eCPM)

### Why it matters
In ad auctions **eCPM = bid × pCTR × 1000**. The ranking/AUC of a model can be
perfect while the *absolute* pCTR is biased. Consequences of miscalibration:
- pCTR too high  -> eCPM inflated -> advertiser overbids / over-serves -> budget burns, ROI drops.
- pCTR too low   -> eCPM deflated -> good ads lose auctions -> revenue + fill-rate loss.
- Ranking metrics (AUC, NDCG) are **invariant to monotonic rescaling** and CANNOT detect this. Calibration is a separate axis from discrimination.

### Sources of miscalibration to expect here
1. **Negative sampling / downsampling** — training on a resampled pos:neg ratio shifts the base rate. MUST correct the logit back to the true prior.
2. **ASL / focal / weighted loss** — these deliberately distort probabilities (they optimise ranking/imbalance, not calibration) -> outputs are NOT true probabilities.
3. **Distribution shift** — train vs serving pCTR drift over time.
4. **Popularity/selection bias** — logged clicks are on items that were *shown* (exposure bias).

### Calibration methods (fit on a HELD-OUT calibration split, never train)
- **Sampling-rate / prior correction** (do this FIRST if negatives were downsampled):
  logit_corrected = logit_model + log(neg_rate)   [Elkan's correction]
  where neg_rate = (kept negatives / all negatives). Restores true base rate.
- **Platt scaling** (logistic): fit sigmoid(a*logit + b) on calibration set. Cheap, strong baseline.
- **Isotonic regression**: non-parametric monotonic map; more flexible, needs more data, can overfit tails.
- **Temperature scaling**: single T on logits; good for NN over/under-confidence, preserves ranking exactly.
- **Beta calibration**: better than Platt when scores are already probability-like.
- **Binned/histogram calibration**: simple, interpretable, used as a monitoring baseline.

### Field-aware / segment calibration (critical for ads)
Global calibration is not enough — calibrate (or check) PER SEGMENT:
- per ad-slot/position (position bias), per device, per category, per new-vs-warm item.
- Fit separate calibrators per segment OR add segment features to a calibration model.
- Watch data sparsity: shrink rare-segment calibrators toward the global map.

### Metrics to PROVE calibration (report these, not just AUC)
- **ECE / MCE** (Expected / Max Calibration Error) — bin pCTR, compare mean-pred vs empirical CTR.
- **Reliability diagram** — the headline calibration plot for the slide.
- **Calibration ratio** = sum(pred) / sum(actual clicks) over a slice; target ~1.0.
  (Report per segment — this is the direct eCPM-bias indicator.)
- **Log loss / Brier score** — proper scoring rules (sensitive to calibration).
- **PR-AUC** — for ranking under imbalance (separate axis, keep both).

### Workflow
1. Train model (ranking objective, e.g. ASL/BPR) — optimise discrimination.
2. Hold out a calibration set (temporal, after train window).
3. Apply prior/sampling correction (undo negative downsampling).
4. Fit Platt/Isotonic (+ per-segment) on calibration set.
5. Validate with ECE + reliability diagram + calibration-ratio per segment on TEST.
6. Monitor calibration drift over time in serving; re-fit calibrator periodically
   (cheaper than retraining the base model).

### KDD demo angle
Show a reliability diagram BEFORE vs AFTER calibration + a table of
calibration-ratio per segment mapped to "eCPM error %". This turns an ML metric
into a business/revenue statement — exactly what ad-auction reviewers want.

---

## Phase 7 — Experiment Matrix (Two-Stage Multimodal-Calibrated Retrieval → Rerank)

### Thesis (the paper's one sentence)
A **two-stage ad recommender** where Stage-1 is a **multimodal feature-LightGCN retriever**
and Stage-2 is a **DeepFM reranker**, bridged by **Platt calibration** so the final
pCTR is auction-safe (eCPM-correct). Research hook = *multimodal-calibrated retrieval*:
image/text content fixes cold-start recall, calibration fixes eCPM bias — two orthogonal
axes reviewers can measure independently.

### System under test
```
transactions + customer/article features + SigLIP image cache
        │
        ▼
[Stage 1] Feature-LightGCN retriever  ──►  top-K1 candidates per user   (recall objective)
        │                                        │
        │                                        ▼
        │                              [Stage 2] DeepFM reranker         (CTR/AUC objective)
        │                                        │
        │                                        ▼
        │                              [Bridge] Platt calibration        (eCPM objective)
        ▼                                        ▼
   Recall@K / NDCG                     AUC / LogLoss  +  ECE / cal-ratio
```

### Axis A — Models
| # | Stage | Model | Signal | Role |
|---|-------|-------|--------|------|
| R0 | retrieve | Popularity (time-decayed) | none | trivial floor |
| R1 | retrieve | MF-BPR | ID-only | CF floor |
| R2 | retrieve | LightGCN (ID-only) | graph | structural baseline |
| R3 | retrieve | **Feature-LightGCN (multimodal)** | cat+cont+**image** | cold-start hook |
| R4 | retrieve | Feature-LightGCN **+ hybrid ID** | features+ID | capacity ceiling |
| K0 | rerank | Logistic Regression | cross feats | linear floor |
| K1 | rerank | Factorization Machine | 2-way | FM floor |
| K2 | rerank | **DeepFM** | FM + deep | main reranker |
| K3 | rerank | DeepFM **+ image/text emb** | + SigLIP | multimodal rerank |

### Axis B — Tasks / eval slices
| Task | Question answered | Primary metric |
|------|-------------------|----------------|
| T1 Retrieval | did we recall the right items? | Recall@{20,100}, NDCG@20 |
| T2 Ranking (CTR) | ordered well within candidates? | AUC, PR-AUC, LogLoss |
| T3 Calibration | is pCTR numerically correct? | ECE, cal-ratio, Brier |
| T4 Cold-start slice | new/rare items recalled? | Recall@100 on cold items |
| T5 Repurchase vs novelty | repeat-buy vs discovery | dual-protocol recall (below) |
| T6 End-to-end | full-funnel quality | NDCG@20 on reranked candidates |

### Axis C — Metrics (report all; they are orthogonal)
- **Retrieval:** Recall@20, Recall@100, NDCG@20, HitRate, catalog **Coverage**, cold-item recall.
- **Ranking:** AUC, PR-AUC, LogLoss, NDCG@20 post-rerank.
- **Calibration:** ECE, MCE, calibration-ratio (→ "eCPM error %"), Brier, reliability diagram.
- **Efficiency:** candidates/user (K1), retrieval + rerank latency, params.

### Axis D — Ablations (each isolates ONE claim)
| Ablation | Toggle | Claim it proves |
|----------|--------|-----------------|
| A1 Multimodal | image on/off in R3 | image lifts recall (esp. cold-start T4) |
| A2 Hybrid ID | R3 vs R4 | features vs collaborative-capacity tradeoff |
| A3 Calibration | Platt on/off | ranking metrics **invariant**, ECE/cal-ratio move (orthogonality) |
| A4 Prior correction | on/off | fixes eCPM base-rate under neg sampling |
| A5 Two-stage vs one | rerank vs retriever-topK | DeepFM rerank lifts AUC/NDCG over pure retrieval |
| A6 Retrieval depth | K1 ∈ {50,100,500} | recall/latency tradeoff, rerank saturation point |
| A7 Graph depth / tau / negs | num_layers, tau, num_negs | retriever sensitivity (appendix) |

### Eval-protocol note — from metrics.py audit + measured (scripts/measure_repurchase.py)
`evaluate_ranking` masks **all** train items (novelty-only). We hypothesised this
suppressed recall on repurchase-heavy H&M — **MEASURED, and it does NOT**:
- Repurchase = **4.4%** of test pairs (mean per-user 5.6%).
- **Recall@20 ceiling under the mask = 0.944** (a perfect model, not 1.0).
- Split measured: train ≤ 2020-09-15, test = last 7d; 1,356,709 users × 103,880 items,
  27.1M train pairs, 189,845 warm test pairs, 62,442 test users.
=> The flat ~0.010 recall is a **real model weakness, not a metric artifact.**
Repurchase is low because train is deduped to unique (u,i) over full history and a
repurchase = exact article_id reappearing within 7 days (fast fashion churn).
- **P1 Novelty-only** (current) = **primary**; report the 0.944 ceiling once for rigor.
- **P2 Repurchase-inclusive** = optional secondary; won't rescue recall (≤5.6% headroom).

### Execution order (each row is a runnable milestone)
1. [DONE] Repurchase ceiling measured: recall@20 ≤ 0.944 (P1 stands as primary; no metrics.py change needed).
2. [ ] Retrieval sweep R0→R4 under P1 (Recall@{20,100}, NDCG, Coverage, cold-item).
3. [ ] Freeze best retriever; dump top-K1 candidates per user (train/val).
4. [ ] Rerank sweep K0→K3 on candidate sets (AUC, LogLoss, PR-AUC, NDCG@20).
5. [ ] Calibrate best DeepFM (Platt ± prior); reliability + cal-ratio (A3/A4).
6. [ ] End-to-end table (T6) + ablations A1–A7.
7. [ ] Slide artifacts: retrieval bar chart, reliability before/after, eCPM-error table.

### Headline tables for the paper
- **Table 1** Retrieval: models × {Recall@20, Recall@100, NDCG@20, Coverage, Cold-Recall} × {P1,P2}.
- **Table 2** Ranking: rerankers × {AUC, PR-AUC, LogLoss, NDCG@20}.
- **Table 3** Calibration: {ECE, cal-ratio→eCPM%, Brier} before/after, ± prior correction.
- **Table 4** Ablations: one row per A1–A7 with the single metric each moves.
- **Fig 1** Reliability diagram before/after calibration (the money slide).
