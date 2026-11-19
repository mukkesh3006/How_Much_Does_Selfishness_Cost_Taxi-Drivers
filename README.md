# How much does selfishness cost taxi drivers?
Efficiency of decentralized taxi positioning on stylized cities, and learning unknown demand.
*Foundations of AI and Multiagent Systems — final project.* Report: [`report/main.pdf`](report/main.pdf).

**Question.** `n` drivers each pick a demand zone of a city graph (or stay idle). How much welfare is lost at the selfish
(Nash) outcome compared with a coordinating dispatcher, how does that depend on demand concentration, driver density,
travel cost and city topology, and what changes when demand must be learned?

**Model (one line).** `U_i = D(z)/n_z − λ·c(ℓ_i, z)`, idle = 0, welfare `W = Σ U_i`. It is an *exact potential game*
(`Φ = Σ_z D(z)·H(n_z) − λ Σ_i c`), so best-response dynamics always converge; the centralized optimum is a bipartite
assignment (Hungarian). Primary metric: efficiency loss `1 − W_EQ / W_OPT`.

## Headline results (all produced by this code)
| | |
|---|---|
| ![](figures/fig1_concentration.png) | Loss rises with demand concentration and driver density; topology only matters for ρ ≥ 2 |
| ![](figures/fig2_heatmap.png) | Joint effect of concentration (α, H) and density (ρ) |
| ![](figures/fig3_coverage_idling.png) | Mechanism: selfish drivers over-enter and under-cover |
| ![](figures/fig4_kappa.png) | Travel-cost scale κ: non-monotone |
| ![](figures/fig6_equity.png) | Equity (idea-review check): the optimum is efficient but far less equal than the equilibrium (report Fig. 4) |
| ![](figures/fig5_learning.png) | Unknown demand: learning is cheap for a dispatcher, selfishness is the main cost |

Screenshots of the code running: [tests](docs/screenshot_tests.png) · [analysis tables](docs/screenshot_analysis.png) · [exact equilibrium enumeration](docs/screenshot_enumeration.png)

## Quick start
```bash
pip install -r requirements.txt        # numpy, scipy, pandas, matplotlib, pytest
python -m pytest -q tests              # 516 tests
./run_all.sh                           # everything below, ~4 min on 2 cores, fixed seeds
```
One instance:
```python
from game import InstanceConfig, run_instance
out = run_instance(InstanceConfig(topology="hub_spoke", topology_kwargs={"num_hubs": 3}, num_nodes=36, num_zones=10,
                                  num_drivers=20, regime="power_law", power_law_alpha=1.5, kappa=0.3, seed=0), n_runs=20)
print(out["w_opt"], out["eff_mean"], out["eff_worst"])     # optimum, mean and worst sampled equilibrium efficiency
```
Rerunning `run_all.sh` regenerates `results/*.csv` bit-for-bit (checked with md5).

## Layout
| Path | Contents |
|---|---|
| `src/network/` | City generators (grid, hub-and-spoke, scale-free), edge weights, all-pairs shortest paths |
| `demand/` | Demand regimes (uniform, concentrated, power-law, degree-correlated), Herfindahl index |
| `optimizer/` | Centralized welfare optimizer: exact search, integer program, local search (independent cross-checks) |
| `game/model.py` | The stage game, exact potential, best response |
| `game/dynamics.py` | Asynchronous best-response dynamics (Alg. 1) |
| `game/enumeration.py` | Exhaustive pure-equilibrium enumeration: exact PoA, PoS, potential maximizer |
| `game/benchmark.py` | Hungarian optimum (valid with the idle option) |
| `game/instance.py` | Instance builder, λ calibration, `run_instance` |
| `game/sweep.py`, `game/sweep_learning.py` | Experiments V, A, B, C, D, L |
| `game/sweep_equity.py` | Experiment E: per-driver utility inequality (Gini, zero-earners) and best-response convergence speed, same instances as A |
| `game/learning.py` | Unknown-demand learning, UCB vs greedy, selfish vs coordinated (Alg. 2) |
| `audit/` | Analysis, figure and verification scripts |
| `results/` | All raw results (CSV/NPZ/TXT) |
| `report/` | `main.tex` (+ auto-generated `numbers.tex`, `tables.tex`), `main.pdf` |
| `tests/` | 516 tests |

## Where each result comes from
| Paper item | Script → output |
|---|---|
| Validation table, exact PoA/PoS | `game.sweep V` → `results/exp_V.csv` |
| Fig. 1, Fig. 2, Fig. 3, Tables 4, H1–H3 numbers | `game.sweep A` → `exp_A.csv` |
| Fig. 2b (`fig4_kappa.png`, travel cost κ), H4 | `game.sweep B` → `exp_B.csv` |
| Demand regimes table | `game.sweep C` → `exp_C.csv` |
| Depot vs random starts | `game.sweep D` → `exp_D.csv` |
| Equity (Gini, zero-earners), convergence speed, Fig. 4 (`fig6_equity.png`) | `game.sweep_equity` → `exp_E.csv` |
| Learning figure/table (report Fig. 5, `fig5_learning.png`) | `game.sweep_learning` → `exp_L.csv`, `exp_L_curves.npz` |
| 6000-epoch run | `audit/long_horizon.py` → `long_horizon.csv` |
| Hungarian = integer program | `audit/opt_is_matching.py` → `opt_vs_hungarian.txt` |
| Mutation check, pilot PoA explosion | `audit/mutation_check.py`, `audit/pilot_poa_explosion.py` → `results/*.txt` |
| Every number/table in the text | `report/make_tables.py` → `report/numbers.tex`, `tables.tex` (nothing is typed by hand) |

## Credits
City generators: Mukkesh Dudekula. Demand generators and the original optimizer: Thushar Vunnam (one fix: the integer
program was missing the zone-activation constraint, see `tests/test_ilp_regression.py`). Game formulation, simulator,
equilibrium enumeration, learning extension, experiments, analysis and report: Jahanzaib. See Sec. 6 of the report.
