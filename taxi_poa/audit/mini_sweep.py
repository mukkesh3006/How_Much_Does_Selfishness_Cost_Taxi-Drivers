import time, numpy as np
from game import InstanceConfig, run_instance
rows=[]
t=time.time()
for topo,kw in [("grid",{}),("hub_spoke",{"num_hubs":3}),("scale_free",{"m":2})]:
    for regime in ["uniform","concentrated"]:
        for rho in [0.5,1.0,2.0]:
            m=8; n=int(rho*m)
            ex=[];sw=[]
            for seed in range(15):
                cfg=InstanceConfig(topology=topo,topology_kwargs=kw,regime=regime,num_nodes=30,num_zones=m,num_drivers=n,seed=seed,driver_start="random")
                o=run_instance(cfg,n_runs=12,enumerate_limit=300_000 if m**n<=300_000 else 0)
                sw.append(o["poa_sampled_worst"]); ex.append(o.get("exact_poa",np.nan))
                assert o["n_cycle"]==0 and o["n_limit"]==0
            print(f"{topo:10s} {regime:12s} rho={rho:3.1f} n={n:2d} sampled-worst PoA mean={np.nanmean(sw):.3f} max={np.nanmax(sw):.3f}  exact PoA mean={np.nanmean(ex) if not np.all(np.isnan(ex)) else float('nan'):.3f}")
print("elapsed",round(time.time()-t,1),"s")
