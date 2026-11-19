import numpy as np, pandas as pd
pd.set_option("display.width",200); pd.set_option("display.max_columns",30)
R="results/"
def ci(x,B=2000,rng=np.random.default_rng(0)):
    x=np.asarray(x); m=[rng.choice(x,len(x)).mean() for _ in range(B)]; return np.percentile(m,[2.5,97.5])
def tab(df,by,col="loss_mean"):
    g=df.groupby(by)[col]; out=g.agg(["mean","std","count"]); out["ci_lo"]=[ci(v)[0] for _,v in g]; out["ci_hi"]=[ci(v)[1] for _,v in g]; return out.round(3)

V=pd.read_csv(R+"exp_V.csv")
print("== V: validation (small instances, all equilibria enumerated) ==")
print("instances",len(V),"| optimizer==enumeration:",V.opt_matches_enum.all(),"| non-converged runs:",int((V.n_cycle+V.n_limit).sum()))
print("sampled worst eff >= exact worst eff always:", bool((V.eff_worst>=V.exact_eff_worst-1e-9).all()),
      "| sampled best eff <= exact best eff:", bool((V.eff_best<=V.exact_eff_best+1e-9).all()))
print("mean (sampled worst - exact worst) efficiency:",round((V.eff_worst-V.exact_eff_worst).mean(),4),"| frac of instances where sampling found the exact worst:",round(((V.eff_worst-V.exact_eff_worst).abs()<1e-9).mean(),3))
print("instances with multiple equilibria:",round((V.exact_n_nash>1).mean(),3),"| mean #Nash:",round(V.exact_n_nash.mean(),2),"max:",V.exact_n_nash.max())
print("exact PoS==PoA frac:",round(((V.exact_pos-V.exact_poa).abs()<1e-9).mean(),3), "| mean exact PoA:",round(V.exact_poa.mean(),3),"max:",round(V.exact_poa.max(),3))
print("potential-maximizer welfare / OPT mean:",round((V.exact_w_phi_max/V.w_opt).mean(),3),"vs best-Nash eff mean",round(V.exact_eff_best.mean(),3))

A=pd.read_csv(R+"exp_A.csv")
print("\n== A: loss = 1 - W_EQ/W_OPT (mean over 100 seeds x 20 dynamics runs) ==")
print("overall mean loss",round(A.loss_mean.mean(),3))
print("\nby topology"); print(tab(A,"topology"))
print("\nby alpha (H1/H2: concentration)"); print(tab(A,"power_law_alpha")); print(A.groupby("power_law_alpha").herfindahl.mean().round(3).to_dict())
print("\nby rho (H: density)"); print(tab(A,"rho"))
print("\nloss: topology x alpha"); print(A.pivot_table(index="topology",columns="power_law_alpha",values="loss_mean").round(3))
print("\nloss: topology x rho"); print(A.pivot_table(index="topology",columns="rho",values="loss_mean").round(3))
print("\nloss: alpha x rho"); print(A.pivot_table(index="power_law_alpha",columns="rho",values="loss_mean").round(3))
print("\nworst-equilibrium efficiency (sampled) by alpha:"); print(A.groupby("power_law_alpha")[["eff_mean","eff_worst","eff_best"]].mean().round(3))
print("\ncongestion diagnostics by alpha:"); print(A.groupby("power_law_alpha")[["eq_active_zones","opt_active_zones","eq_max_load","opt_max_load","eq_idle_frac","opt_idle_frac"]].mean().round(2))
print("\ncongestion diagnostics by rho:"); print(A.groupby("rho")[["eq_active_zones","opt_active_zones","eq_max_load","opt_max_load","eq_idle_frac","opt_idle_frac"]].mean().round(2))
# OLS (numpy) with standardized predictors
X=pd.get_dummies(A[["topology"]],drop_first=True).astype(float)
X["H"]=A.herfindahl; X["rho"]=np.log2(A.rho); X["H:rho"]=A.herfindahl*np.log2(A.rho)
y=A.loss_mean.values; Xm=np.column_stack([np.ones(len(X)),X.values]); beta,res,rk,_=np.linalg.lstsq(Xm,y,rcond=None)
r2=1-((y-Xm@beta)**2).sum()/((y-y.mean())**2).sum(); print("\nOLS loss ~ topology + H + log2(rho) + H*log2(rho)   R2=",round(r2,3)); print(dict(zip(["const"]+list(X.columns),beta.round(3))))
sd=A[["loss_mean"]].std().item()
# variance decomposition (share of variance explained by each factor alone, eta^2)
for f in ["topology","power_law_alpha","rho"]:
    g=A.groupby(f).loss_mean; ssb=sum(len(v)*(v.mean()-A.loss_mean.mean())**2 for _,v in g); print("eta^2",f,round(ssb/((A.loss_mean-A.loss_mean.mean())**2).sum(),3))
print("\nsensitivity to the dynamics (spread of W_EQ across runs / W_OPT) by alpha:"); print(((A.eff_best-A.eff_worst)).groupby(A.power_law_alpha).mean().round(3).to_dict())

B=pd.read_csv(R+"exp_B.csv")
print("\n== B: kappa (travel-cost scale), alpha=1, rho=1 =="); print(B.pivot_table(index="topology",columns="kappa",values="loss_mean").round(3))
print(B.groupby("kappa")[["eq_idle_frac","opt_idle_frac","eq_active_zones","opt_active_zones","eq_travel_frac"]].mean().round(3))
C=pd.read_csv(R+"exp_C.csv")
print("\n== C: regimes =="); print(C.pivot_table(index=["rho","regime"],columns="topology",values="loss_mean").round(3)); print(C.groupby("regime").herfindahl.mean().round(3).to_dict())
D=pd.read_csv(R+"exp_D.csv")
print("\n== D: random starts vs common depot =="); print(D.pivot_table(index="topology",columns="driver_start",values="loss_mean").round(3))
print(D.groupby("driver_start")[["mean_updates","eff_best","eff_worst"]].mean().round(3))

E=pd.read_csv(R+"exp_E.csv")
print("\n== E: equity (Gini of driver utilities) and convergence speed; same instances as A, no side payments ==")
print("replay check: max |loss_E - loss_A| =",float((E.merge(A,on=['topology','power_law_alpha','rho','seed'],suffixes=('','_A')).eval('abs(loss_mean-loss_mean_A)')).max()))
cols=["eq_gini","opt_gini","eq_cv","opt_cv","eq_zero_share","opt_zero_share","eq_min_u","opt_min_u","eq_mean_u","opt_mean_u"]
print("\nby rho"); print(E.groupby("rho")[cols].mean().round(3))
print("\nby alpha"); print(E.groupby("power_law_alpha")[cols].mean().round(3))
print("\nby topology"); print(E.groupby("topology")[cols].mean().round(3))
print("\nfraction of instances where the optimum is MORE unequal than the equilibrium, by rho:", E.groupby("rho").apply(lambda g:round(float((g.opt_gini>g.eq_gini).mean()),3),include_groups=False).to_dict())
print("\nconvergence speed: moves per driver and round-robin sweeps (incl. the confirming sweep), by rho"); print(E.groupby("rho")[["updates_per_driver","sweeps_rr"]].agg(["mean","max"]).round(2))
print(E.groupby("topology")[["updates_per_driver","sweeps_rr"]].mean().round(2))
