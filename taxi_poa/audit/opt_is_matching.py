"""Claim: with the idle option (or any driver able to out-earn nothing), OPT is a bipartite assignment:
at most one driver per zone, gain D(z) - lam*c(i,z) (only if > 0). Check vs the ILP on many instances."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from game import InstanceConfig, build_game
from game.instance import central_optimum
worst=0; n=0
for topo,kw in [("grid",{}),("hub_spoke",{"num_hubs":3}),("scale_free",{"m":2})]:
    for rho in (0.5,1,2,4):
        for kappa in (0.1,0.6,2.0):
            for seed in range(25):
                cfg=InstanceConfig(topology=topo,topology_kwargs=kw,num_nodes=36,num_zones=10,num_drivers=int(rho*10),
                                   regime="power_law",power_law_alpha=1.0,kappa=kappa,seed=seed)
                g,meta=build_game(cfg); w_ilp,_,_=central_optimum(g,meta)
                gain=np.maximum(g.demand[None,:-1]-g.lam*g.cost[:,:-1],0.0)       # drivers x real zones
                r,c=linear_sum_assignment(-gain); w_hung=gain[r,c].sum()
                worst=max(worst,abs(w_ilp-w_hung)); n+=1
print("instances",n,"max |W_ILP - W_Hungarian| =",worst)
open("results/opt_vs_hungarian.txt","w").write(f"instances={n}\nmax_abs_diff={worst:.3e}\n")
