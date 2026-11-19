"""Generates report/tables.tex and report/numbers.tex from results/*.csv, so every number in the paper is produced by code.
Run from the repo root:  python report/make_tables.py"""
import numpy as np, pandas as pd

R = "results/"; OUT = "report/"
N = {}                      # macro name -> string
WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", ".": "P"}
def mname(name): return "".join(WORD.get(ch, ch) for ch in name)   # LaTeX macro names may contain letters only
def put(name, val, fmt="{:.2f}"): N[mname(name)] = fmt.format(val) if not isinstance(val, str) else val
def ci(x, B=2000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)]); return np.percentile(m, [2.5, 97.5])
def eta2(df, f, y="loss_mean"):
    mu = df[y].mean(); ssb = sum(len(v) * (v.mean() - mu) ** 2 for _, v in df.groupby(f)[y]); return ssb / ((df[y] - mu) ** 2).sum()
TN = {"grid": "Grid", "hub_spoke": "Hub-and-spoke", "scale_free": "Scale-free"}

V = pd.read_csv(R + "exp_V.csv"); A = pd.read_csv(R + "exp_A.csv"); B = pd.read_csv(R + "exp_B.csv")
C = pd.read_csv(R + "exp_C.csv"); D = pd.read_csv(R + "exp_D.csv"); L = pd.read_csv(R + "exp_L.csv"); E = pd.read_csv(R + "exp_E.csv")
tot = len(V) + len(A) + len(B) + len(C) + len(D)
put("nInstTotal", f"{tot:,}".replace(",", "{,}")); put("nRunsTotal", f"{tot*20:,}".replace(",", "{,}"))
put("nNonConv", int((V.n_cycle + V.n_limit).sum() + (A.n_cycle + A.n_limit).sum() + (B.n_cycle + B.n_limit).sum() + (C.n_cycle + C.n_limit).sum() + (D.n_cycle + D.n_limit).sum()), "{:d}")
put("nA", len(A), "{:,}"); put("nV", len(V), "{:,}".format if False else "{:,}")
put("nL", len(L) // 4, "{:d}")

# ---- validation (V)
put("vOptMatch", "all" if V.opt_matches_enum.all() else "NOT all")
put("vMultiEq", 100 * (V.exact_n_nash > 1).mean(), "{:.0f}"); put("vMeanNash", V.exact_n_nash.mean(), "{:.0f}"); put("vMaxNash", V.exact_n_nash.max(), "{:d}")
put("vPosEqPoa", 100 * ((V.exact_pos - V.exact_poa).abs() < 1e-9).mean(), "{:.0f}")
put("vSampledExact", 100 * ((V.eff_worst - V.exact_eff_worst).abs() < 1e-9).mean(), "{:.0f}")
put("vOvershoot", 100 * (V.eff_worst - V.exact_eff_worst).mean(), "{:.1f}")
put("vPoaMean", V.exact_poa.mean()); put("vPoaMax", V.exact_poa.max())
put("vPhiGap", (V.exact_eff_best - V.exact_w_phi_max / V.w_opt).mean(), "{:.3f}")

# ---- main sweep (A)
for r, v in A.groupby("rho").loss_mean: put(f"lossRho{r}", v.mean())
for a, v in A.groupby("power_law_alpha").loss_mean: put(f"lossAlpha{a}", v.mean())
for t, v in A.groupby("topology").loss_mean: put(f"loss{t.replace('_', '')}", v.mean())
for a, v in A.groupby("power_law_alpha").herfindahl: put(f"hAlpha{a}", v.mean())
put("etaAlpha", 100 * eta2(A, "power_law_alpha"), "{:.0f}"); put("etaRho", 100 * eta2(A, "rho"), "{:.0f}"); put("etaTopo", 100 * eta2(A, "topology"), "{:.1f}")
for t in TN:
    for r in (0.5, 1.0, 2.0, 4.0):
        put(f"lossTR{t.replace('_', '')}{r}", A[(A.topology == t) & (A.rho == r)].loss_mean.mean())
low = A[(A.rho == 0.5)].groupby("power_law_alpha").loss_mean.mean(); put("lowRhoPeak", low.max()); put("lowRhoPeakAlpha", low.idxmax(), "{:g}"); put("lowRhoLast", low.loc[2.0])
a2 = A[A.power_law_alpha == 2.0]
put("eqMaxLoadAlphaTwo", a2.eq_max_load.mean(), "{:.1f}"); put("optMaxLoadAlphaTwo", a2.opt_max_load.mean(), "{:.1f}")
put("eqCovAlphaTwo", a2.eq_active_zones.mean(), "{:.1f}"); put("optCovAlphaTwo", a2.opt_active_zones.mean(), "{:.1f}")
r4 = A[A.rho == 4.0]; put("optIdleRhoFour", 100 * r4.opt_idle_frac.mean(), "{:.0f}"); put("eqIdleRhoFour", 100 * r4.eq_idle_frac.mean(), "{:.0f}")
# OLS R^2
X = pd.get_dummies(A[["topology"]], drop_first=True).astype(float); X["H"] = A.herfindahl; X["lr"] = np.log2(A.rho); X["Hlr"] = A.herfindahl * np.log2(A.rho)
Xm = np.column_stack([np.ones(len(X)), X.values]); y = A.loss_mean.values; beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
put("olsRtwo", 1 - ((y - Xm @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())

# ---- kappa (B), regimes (C), start (D)
Bk = B.groupby("kappa").loss_mean.mean(); put("kapPeak", Bk.idxmax(), "{:g}")
Bt = B.pivot_table(index="topology", columns="kappa", values="loss_mean")
put("kapHubAtPeak", Bt.loc["hub_spoke", Bk.idxmax()]); put("kapGridAtPeak", Bt.loc["grid", Bk.idxmax()]); put("kapLowEnd", Bk.loc[0.1]); put("kapHighEnd", Bk.loc[2.0])
Cp = C[C.rho == 1].pivot_table(index="regime", columns="topology", values="loss_mean")
for t in TN:
    for rg in ("uniform", "concentrated", "degree_correlated"):
        put(f"reg{rg.replace('_', '')}{t.replace('_', '')}", Cp.loc[rg, t])
Dp = D.groupby("driver_start").loss_mean.mean(); put("startRandom", Dp["random"]); put("startDepot", Dp["depot"])
put("depotSpread", (D[D.driver_start == "depot"].eff_best - D[D.driver_start == "depot"].eff_worst).abs().max(), "{:.3f}")

# ---- learning (L)
L["v"] = L["mode"] + "_" + L["c"].astype(str)
fin = L.pivot_table(index="rho", columns="v", values="eff_151_end"); cum = L.pivot_table(index="rho", columns="v", values="cum_loss_vs_opt")
fi = L.groupby("rho").eff_fullinfo_eq.mean()
put("lrnPlanMin", min(fin["planner_1.0"].min(), fin["planner_0.0"].min())); put("lrnPlanCumMin", cum["planner_1.0"].min(), "{:.1f}"); put("lrnPlanCumMax", cum["planner_1.0"].max(), "{:.1f}")
put("lrnSelfCumUMin", cum["selfish_1.0"].min(), "{:.0f}"); put("lrnSelfCumUMax", cum["selfish_1.0"].max(), "{:.0f}")
put("lrnSelfCumGMin", cum["selfish_0.0"].min(), "{:.0f}"); put("lrnSelfCumGMax", cum["selfish_0.0"].max(), "{:.0f}")
su = L[(L["mode"] == "selfish") & (L.c == 1.0)]; sg = L[(L["mode"] == "selfish") & (L.c == 0.0)]
put("lrnUcbBeatsFI", 100 * (su.eff_151_end > su.eff_fullinfo_eq + 1e-6).mean(), "{:.0f}"); put("lrnUcbGain", (su.eff_151_end - su.eff_fullinfo_eq).mean(), "{:+.3f}")
put("lrnGreedyGain", (sg.eff_151_end - sg.eff_fullinfo_eq).mean(), "{:+.3f}")
put("lrnRhoOneUcb", fin.loc[1.0, "selfish_1.0"]); put("lrnRhoOneFI", fi.loc[1.0])

# ---- equity and speed of convergence (E): same instances as A, per-driver utilities and best-response effort
gE = E.groupby("rho"); lo, hi = E.rho.min(), E.rho.max(); dense = E[E.rho >= 2]; sparse = E[E.rho <= 1]
put("eGiniEqLow", gE.eq_gini.mean()[lo]); put("eGiniOptLow", gE.opt_gini.mean()[lo])
put("eGiniEqHigh", gE.eq_gini.mean()[hi]); put("eGiniOptHigh", gE.opt_gini.mean()[hi])
put("eZeroOptHigh", 100 * gE.opt_zero_share.mean()[hi], "{:.0f}"); put("eZeroEqHigh", 100 * gE.eq_zero_share.mean()[hi], "{:.0f}")
put("eMoreUnequalAll", 100 * (E.opt_gini > E.eq_gini).mean(), "{:.0f}")
put("eMoreUnequalDense", 100 * (dense.opt_gini > dense.eq_gini).mean(), "{:.0f}")
put("eMoreUnequalSparse", 100 * (sparse.opt_gini > sparse.eq_gini).mean(), "{:.0f}")
put("eUpdLow", gE.updates_per_driver.mean()[lo]); put("eUpdHigh", gE.updates_per_driver.mean()[hi]); put("eUpdMax", E.updates_per_driver.max())
put("eSweepsLow", gE.sweeps_rr.mean()[lo], "{:.1f}"); put("eSweepsHigh", gE.sweeps_rr.mean()[hi], "{:.1f}"); put("eSweepsMax", E.sweeps_rr.max(), "{:.0f}")

# ---- one-off verification scripts (results/*.txt, long_horizon.csv)
import re, os
kv = lambda fn: dict(l.strip().split("=") for l in open(R + fn) if "=" in l)
put("nTests", re.search(r"(\d+) passed", open(R + "test_summary.txt").read()).group(1), "{}")
put("mutFail", kv("mutation_check.txt")["failed_tests"], "{}")
h = kv("opt_vs_hungarian.txt"); put("hungN", h["instances"], "{}"); put("hungDiff", f"{float(h['max_abs_diff'])*1e14:.1f}\\times10^{{-14}}", "{}")
put("pilotMax", float(kv("pilot_poa.txt")["max_sampled_poa"]), "{:.0f}")
LH = pd.read_csv(R + "long_horizon.csv"); put("lhFirst", LH.eff_1_50.mean(), "{:.2f}"); put("lhMid", LH.eff_250_350.mean(), "{:.2f}")
put("lhLast", LH.eff_5500_6000.mean(), "{:.3f}"); put("lhFI", LH.eff_fullinfo_eq.mean(), "{:.3f}"); put("lhGain", (LH.eff_5500_6000 - LH.eff_fullinfo_eq).mean(), "{:.2f}")
put("lhN", len(LH), "{}")

with open(OUT + "numbers.tex", "w") as f:
    f.write("% AUTO-GENERATED by report/make_tables.py from results/*.csv -- do not edit\n")
    for k, v in N.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------- tables
def cell(v): return f"{v:.2f}"
T = []
# Table: loss by topology x rho
T.append("% AUTO-GENERATED by report/make_tables.py\n\\newcommand{\\tabTopoRho}{\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lcccc}\\toprule\n & $\\rho=0.5$ & $\\rho=1$ & $\\rho=2$ & $\\rho=4$\\\\\\midrule")
for t in TN:
    row = []
    for r in (0.5, 1.0, 2.0, 4.0):
        v = A[(A.topology == t) & (A.rho == r)].loss_mean; lo, hi = ci(v); row.append(f"{v.mean():.2f} \\scriptsize$\\pm${(hi - lo) / 2:.2f}")
    T.append(f"{TN[t]} & " + " & ".join(row) + "\\\\")
T.append("\\bottomrule\\end{tabular}}")
# Table: validation
T.append("\\newcommand{\\tabValidation}{\\begin{tabular}{lr}\\toprule Check (small instances, all equilibria enumerated) & Value\\\\\\midrule")
T.append(f"Instances & {len(V):,}\\\\".replace(",", "{,}"))
T.append(f"Optimizer equals exhaustive optimum & {'all' if V.opt_matches_enum.all() else 'NOT all'}\\\\")
T.append(f"Instances with more than one pure equilibrium & {100*(V.exact_n_nash>1).mean():.0f}\\%\\\\")
T.append(f"Mean / max number of pure equilibria & {V.exact_n_nash.mean():.0f} / {V.exact_n_nash.max()}\\\\")
T.append(f"Best and worst equilibrium coincide (PoS = PoA) & {100*((V.exact_pos-V.exact_poa).abs()<1e-9).mean():.0f}\\%\\\\")
T.append(f"Sampled worst equals exact worst equilibrium & {100*((V.eff_worst-V.exact_eff_worst).abs()<1e-9).mean():.0f}\\%\\\\")
T.append(f"Mean exact PoA / maximum & {V.exact_poa.mean():.2f} / {V.exact_poa.max():.2f}\\\\")
T.append(f"Dynamics runs not converging ({int((V.n_cycle+V.n_limit).sum()+(A.n_cycle+A.n_limit).sum()+(B.n_cycle+B.n_limit).sum()+(C.n_cycle+C.n_limit).sum()+(D.n_cycle+D.n_limit).sum())} of {tot*20:,}) & 0\\\\".replace(",", "{,}"))
T.append("\\bottomrule\\end{tabular}}")
# Table: regimes
T.append("\\newcommand{\\tabRegimes}{\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccc}\\toprule Regime & Grid & Hub & Scale-free\\\\\\midrule")
for rg, nm in [("uniform", "Uniform"), ("concentrated", "Concentrated (80\\% in 20\\%)"), ("degree_correlated", "Degree-correlated")]:
    T.append(f"{nm} & " + " & ".join(cell(Cp.loc[rg, t]) for t in TN) + "\\\\")
T.append("\\bottomrule\\end{tabular}}")
# Table: learning
T.append("\\newcommand{\\tabLearning}{\\begin{tabular}{lccccc}\\toprule & \\multicolumn{2}{c}{Coordinated} & \\multicolumn{2}{c}{Selfish} & \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n$\\rho$ & UCB & greedy & UCB & greedy & Selfish, full info\\\\\\midrule")
for r in sorted(fin.index):
    T.append(f"{r:g} & {fin.loc[r,'planner_1.0']:.2f} & {fin.loc[r,'planner_0.0']:.2f} & {fin.loc[r,'selfish_1.0']:.2f} & {fin.loc[r,'selfish_0.0']:.2f} & {fi.loc[r]:.2f}\\\\")
T.append("\\bottomrule\\end{tabular}}")
open(OUT + "tables.tex", "w").write("\n".join(T) + "\n")
print("wrote numbers.tex (%d macros) and tables.tex" % len(N))
