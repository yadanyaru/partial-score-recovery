# Experiment-to-paper map

The paper has two primary questions: which prediction differences a rounded
partial message cannot reveal, and how response-error structure reduces loss.

| Paper component | Implementation | Reference outcomes / scope |
|---|---|---|
| Distinct-row vanishing-disclosure construction; Figure 2 | `review_low_disclosure.py` | 288 collision witnesses at ranks 1024/1536; larger plotted ranks use the analytic formula |
| Static/adaptive geometric checks | `check_adaptive_disclosure_theory.py`, `check_static_adaptive_separation.py`, `check_adaptive_floor_thresholds.py` | Numerical checks, not substitutes for exact proofs |
| Common-phase nearest/shared witnesses | `common_phase_witness.py`, `common_phase_shared_witness.py`, `explicit_known_grid.py`, `explicit_shared_coin_witness.py` | Constructed heads and prescribed phase sweeps |
| Independent/shared response laws on small heads | `stochastic_precision_diagnostic.py`, `shared_coin_diagnostic.py` | Matched decoder and precision diagnostics |
| Native states | `review_prepare_states.py` | 128 LAMBADA passages per model; 511 teacher-forced answer-prefix states |
| Main precision experiment; Figure 3 and Table 3 | `review_precision_utility.py`, `analyze_review_precision.py` | 2,555 prefix/grid cells; 1,920 initial-prefix cells; 512 teacher-token pairs; 128 stochastic draws |
| Covariance mechanism; Figure 4 | `pretrained_shared_coin.py`, `analyze_pretrained_shared_coin.py` | 336 cases; exact shared-kernel integration versus 256 independent draws |
| Earlier three-candidate confirmation | `risk_selected_precision_pretrained.py`, `analyze_risk_selected_precision.py` | 96 cases, 16 validation paragraphs, two meshes, three models |
| Pair-count implementation ablation | `sample_precision_ablation.py`, `analyze_sampled_precision.py` | 32/128/512 pairs; eight provider seeds; 2,304 decisions on the same 96 cases |
| Classical static designs | `pretrained_disclosure_formal.py`, `pretrained_disclosure_io_repair.py`, `analyze_pretrained_disclosure_formal.py` | 5,376 primary evaluations; original complete Pythia run used an I/O-only repair |
| Matched pretrained response laws | `pretrained_stochastic_precision.py`, `analyze_stochastic_precision.py` | 336 nearest/independent/coherent-response cases |
| Strict top-k interface; Table 4 | `review_topk_interface.py` | 6,912 receiver evaluations; all budgets and meshes retained |
| Native adaptive comparison | `review_adaptive_bridge.py`, `review_box_qp.py`, `analyze_review_adaptive.py` | 576 cells, five arms, all pilots/repeated IDs charged; all 2,880 arm evaluations included |
| Instance/local-width/subspace/joint-oracle checks | `check_review_instance_theory.py` | Deterministic mathematical examples, not additional pretrained results |

Plotting and concept-drawing code are excluded from this review export. Reference numerical results include all
final evaluated model/configuration cells, including unfavorable comparisons.
Development-only solver variants, obsolete manuscript generators and internal
review notes are excluded.

## Stage dependencies

```text
public corpus rows + pinned models -> native states -> precision / topk / adaptive
public baseline design IDs        -> response / shared / confirmation -> sampling
synthetic head definitions        -> low-disclosure / CPU theory checks
included numerical outcomes       -> analysis summaries
```

Native state preparation checks array hashes of model output heads. Protocols
specify original-softmax calibrated scores and intercept-aware receivers. The
receiver does not obtain the unknown hidden state, full teacher probabilities,
gold token IDs, or undisclosed tail mass as inputs. Provider precision selection
uses teacher probabilities, and its measured cost excludes initial inference and
full-softmax computation.

## Direct execution

The documented `review.py` commands set paths and stage dependencies. Individual
implementations under `work/` can also be run from a prepared workspace root;
`--help` lists their flags. Some older synthetic scripts retain positional mode
parsing and are provided for implementation inspection; use the documented CPU
checks and low-disclosure stage for the portable quick route.

## Packaging scope

The package contains experiment implementations and reference-result analyses.
No additional full pretrained evaluation was run to prepare the final ZIP.
The existing scientific results are preserved; plotting implementations and
development-only manuscript tooling are excluded.
