import random, math, itertools, traceback
from src.network import generate_city_network, floyd_warshall_reference
from demand import generate_demand, compute_herfindahl_index
from demand.generator import generate_concentrated_demand, generate_degree_correlated_demand, generate_power_law_demand, generate_uniform_demand
from optimizer import optimize_welfare, optimize_welfare_exact, optimize_welfare_heuristic, optimize_welfare_ilp, evaluate_welfare

def check(name, fn):
    try:
        r = fn()
        print(("OK   " if r in (None, True) else "NOTE ") + name + ("" if r in (None, True) else f" -> {r}"))
    except Exception as e:
        print("FAIL " + name + f" -> {type(e).__name__}: {e}")

# ---------- GRAPH ----------
def g_det():
    for t,kw in [("grid",{}),("hub_spoke",{"num_hubs":2}),("scale_free",{"m":2})]:
        a=generate_city_network(t,num_nodes=20,seed=5,**kw); b=generate_city_network(t,num_nodes=20,seed=5,**kw)
        assert a.graph.edges()==b.graph.edges(), t
check("graph: same seed -> identical graph", g_det)

def g_diff():
    a=generate_city_network("scale_free",num_nodes=30,seed=1,m=2); b=generate_city_network("scale_free",num_nodes=30,seed=2,m=2)
    return a.graph.edges()!=b.graph.edges() or "same structure for different seeds"
check("graph: scale_free varies with seed", g_diff)

def g_struct_fixed():
    # hub_spoke / grid structure is deterministic: only weights vary, so 'random graph' has no structural randomness
    a=generate_city_network("hub_spoke",num_nodes=20,seed=1,num_hubs=2); b=generate_city_network("hub_spoke",num_nodes=20,seed=2,num_hubs=2)
    sa={(u,v) for u,v,_ in a.graph.edges()}; sb={(u,v) for u,v,_ in b.graph.edges()}
    return f"hub_spoke structure identical across seeds: {sa==sb}"
check("graph: structural randomness of hub_spoke", g_struct_fixed)

def g_apsp():
    for s in range(30):
        for t,kw in [("grid",{}),("hub_spoke",{"num_hubs":3}),("scale_free",{"m":2})]:
            c=generate_city_network(t,num_nodes=24,seed=s,**kw); fw=floyd_warshall_reference(c.graph)
            for u in range(24):
                for v in range(24):
                    assert abs(c.get_travel_cost(u,v)-fw[u][v])<1e-9
                    assert abs(c.get_travel_cost(u,v)-c.get_travel_cost(v,u))<1e-9
check("graph: APSP == Floyd-Warshall, symmetric", g_apsp)

def g_tri():
    c=generate_city_network("scale_free",num_nodes=30,seed=3,m=2)
    for u,v,w in itertools.product(range(30),repeat=3):
        assert c.get_travel_cost(u,w)<=c.get_travel_cost(u,v)+c.get_travel_cost(v,w)+1e-9
check("graph: triangle inequality", g_tri)

def g_prime():
    c=generate_city_network("grid",num_nodes=13,seed=1)
    return f"grid(13) dims={c.graph.metadata['rows']}x{c.graph.metadata['cols']}"
check("graph: grid with prime node count", g_prime)

def g_hubspoke_hubs():
    # hubs fully interconnected with weight drawn per edge; what is the hub-spoke diameter and are spokes only joined via hub?
    c=generate_city_network("hub_spoke",num_nodes=20,seed=1,num_hubs=1)
    deg=sorted(c.graph.degree(u) for u in range(20))
    return f"1-hub degrees: max={deg[-1]} min={deg[0]} (pure star => tree, PoA topology effects limited)"
check("graph: single hub is a star", g_hubspoke_hubs)

def g_weights_range():
    c=generate_city_network("grid",num_nodes=16,seed=1,min_weight=2.0,max_weight=3.0)
    ws=[w for *_,w in c.graph.edges()]
    assert min(ws)>=2.0 and max(ws)<=3.0
check("graph: weights within range", g_weights_range)

def g_weight_seed_collision():
    # weight seed = seed+10007. seeds s and s+10007 share weight stream
    a=generate_city_network("grid",num_nodes=16,seed=0); b=generate_city_network("grid",num_nodes=16,seed=10007)
    wa=[w for *_,w in a.graph.edges()]; wb=[w for *_,w in b.graph.edges()]
    return f"weights(seed=0) first == weights(seed=10007) of... {wa[:2]} vs {wb[:2]}"
check("graph: seed derivation", g_weight_seed_collision)

# ---------- DEMAND ----------
zones=[f"Z{i}" for i in range(10)]
def d_nonneg():
    for reg in ["uniform","concentrated","power_law"]:
        for s in range(50):
            d=generate_demand(zones,regime=reg,seed=s)
            assert all(v>=0 for v in d.values()) and set(d)==set(zones), reg
check("demand: nonneg, keys match", d_nonneg)

def d_total():
    out=[]
    for reg,kw in [("uniform",{}),("concentrated",{}),("power_law",{}),("concentrated",{"total_demand":100.0})]:
        d=generate_demand(zones,regime=reg,seed=1,**kw); out.append((reg,round(sum(d.values()),3)))
    return out
check("demand: totals by regime (does total demand stay fixed when concentration changes?)", d_total)

def d_total_fixed_sweep():
    # key confound: if total demand changes with concentration, PoA sweep over H is confounded
    tots=[]
    for a in [0.0,0.5,1.0,2.0,4.0]:
        d=generate_demand(zones,regime="power_law",seed=1,power_law_alpha=a); tots.append(round(sum(d.values()),2))
    return f"power_law totals over alpha: {tots}"
check("demand: total held constant across concentration sweep", d_total_fixed_sweep)

def d_H_monotone():
    Hs=[compute_herfindahl_index(generate_demand(zones,regime="power_law",seed=1,power_law_alpha=a)) for a in [0,0.5,1,2,4]]
    assert all(x<=y+1e-9 for x,y in zip(Hs,Hs[1:])), Hs
    return [round(h,3) for h in Hs]
check("demand: H monotone in alpha", d_H_monotone)

def d_det():
    assert generate_demand(zones,regime="concentrated",seed=3)==generate_demand(zones,regime="concentrated",seed=3)
check("demand: seed determinism", d_det)

def d_hotspot_share():
    d=generate_concentrated_demand(zones,total_demand=100.0,hotspot_fraction=0.2,hotspot_share=0.8,seed=1)
    top=sorted(d.values(),reverse=True)[:2]
    return f"top2 share={sum(top)/sum(d.values()):.3f} (target 0.8), total={sum(d.values()):.2f}"
check("demand: concentrated hotspot share honored", d_hotspot_share)

def d_hotspot_pos():
    # are hotspots always the first zones? (position bias vs topology)
    pos=[]
    for s in range(20):
        d=generate_concentrated_demand(zones,total_demand=100.0,seed=s)
        pos.append(tuple(sorted(sorted(d,key=d.get,reverse=True)[:2])))
    return f"distinct hotspot sets over 20 seeds: {len(set(pos))}"
check("demand: hotspot location randomized by seed", d_hotspot_pos)

def d_single_zone():
    return generate_demand(["A"],regime="concentrated",seed=1)
check("demand: single zone", d_single_zone)

def d_degcorr():
    degs={z:i for i,z in enumerate(zones)}  # includes degree 0
    d=generate_degree_correlated_demand(zones,degrees=degs,seed=1)
    return f"zero-degree zone demand={d['Z0']:.3f}, top={max(d.values()):.3f}"
check("demand: degree-correlated with degree 0", d_degcorr)

# ---------- OPTIMIZER ----------
def rand_inst(rng, nd, nz):
    zs=[f"Z{i}" for i in range(nz)]; ds=[f"d{i}" for i in range(nd)]
    locs={d:f"L{rng.randrange(nz+2)}" for d in ds}
    dem={z:rng.choice([0.0,rng.uniform(1,40)]) for z in zs}
    cost={(f"L{a}",z):rng.uniform(0,12) for a in range(nz+2) for z in zs}
    return ds,locs,zs,dem,cost

def o_brute():
    for s in range(150):
        rng=random.Random(s); nd,nz=rng.randint(1,5),rng.randint(1,4)
        ds,locs,zs,dem,cost=rand_inst(rng,nd,nz); lam=rng.choice([0.0,0.3,1.0,3.0])
        best=-1e18
        for prof in itertools.product(zs,repeat=nd):
            w,*_=evaluate_welfare(dict(zip(ds,prof)),locs,dem,cost,lam); best=max(best,w)
        ex=optimize_welfare_exact(ds,locs,zs,dem,cost,lambda_cost=lam)
        assert abs(ex.welfare-best)<1e-9,(s,ex.welfare,best)
check("optimizer: exact == independent brute force (incl. zero demand, lam=0)", o_brute)

def o_heur():
    gaps=[];bad=0
    for s in range(150):
        rng=random.Random(s); nd,nz=rng.randint(2,7),rng.randint(2,6)
        ds,locs,zs,dem,cost=rand_inst(rng,nd,nz); lam=rng.choice([0.3,1.0,3.0])
        ex=optimize_welfare_exact(ds,locs,zs,dem,cost,lambda_cost=lam,max_combinations=10**7)
        h=optimize_welfare_heuristic(ds,locs,zs,dem,cost,lambda_cost=lam,seed=s)
        assert h.welfare<=ex.welfare+1e-9
        if ex.welfare>1e-9: gaps.append((ex.welfare-h.welfare)/ex.welfare)
        bad+= (ex.welfare-h.welfare)>1e-6
    return f"heuristic suboptimal in {bad}/150; max rel gap={max(gaps):.4f}, mean={sum(gaps)/len(gaps):.5f}"
check("optimizer: heuristic vs exact", o_heur)

def o_ilp_auto():
    # auto path on mid-size instance: ILP vs heuristic vs exact consistent
    rng=random.Random(7); ds,locs,zs,dem,cost=rand_inst(rng,6,5)
    a=optimize_welfare(ds,locs,zs,dem,cost,1.0,method="auto"); e=optimize_welfare(ds,locs,zs,dem,cost,1.0,method="exact")
    assert abs(a.welfare-e.welfare)<1e-9
check("optimizer: auto == exact on small", o_ilp_auto)

def o_ilp_vs_heur_big():
    gaps=[]
    for s in range(10):
        rng=random.Random(s); ds,locs,zs,dem,cost=rand_inst(rng,30,12)
        il=optimize_welfare_ilp(ds,locs,zs,dem,cost,1.0,fallback_to_heuristic=False)
        h=optimize_welfare_heuristic(ds,locs,zs,dem,cost,1.0,seed=s)
        assert il.welfare>=h.welfare-1e-7,(s,il.welfare,h.welfare)
        gaps.append(il.welfare-h.welfare)
    return f"ILP >= heuristic on 30x12; heuristic loses up to {max(gaps):.3f}"
check("optimizer: ILP >= heuristic at scale", o_ilp_vs_heur_big)

def o_tie():
    ds=["a","b"]; locs={"a":"L","b":"L"}; zs=["Z0","Z1"]; dem={"Z0":10.0,"Z1":10.0}; cost={("L","Z0"):1.0,("L","Z1"):1.0}
    r1=optimize_welfare_exact(ds,locs,zs,dem,cost); r2=optimize_welfare_exact(ds,locs,zs,dem,cost)
    assert r1.assignment==r2.assignment
    return f"tie-break assignment: {r1.assignment}"
check("optimizer: deterministic tie-breaking", o_tie)

def o_seq_locations():
    ds=["a","b"]; zs=["Z0","Z1"]; dem={"Z0":10.0,"Z1":10.0}; cost={("Z0","Z0"):0.0,("Z0","Z1"):1.0,("Z1","Z0"):1.0,("Z1","Z1"):0.0}
    return optimize_welfare_exact(ds,["Z0","Z1"],zs,dem,cost).assignment
check("optimizer: sequence-form driver_locations", o_seq_locations)

def o_cost_key_ambiguity():
    # driver ids that coincide with location ids: tuple key (driver_loc, zone) is tried before (driver_id, zone)
    ds=[0,1]; locs={0:1,1:0}; zs=[0,1]; dem={0:10.0,1:10.0}
    cost={(0,0):0.0,(0,1):5.0,(1,0):5.0,(1,1):0.0}
    ok=optimize_welfare_exact(ds,locs,zs,dem,cost).assignment
    return f"int ids == int locations -> {ok} (expect d0@loc1 -> zone1, d1@loc0 -> zone0)"
check("optimizer: driver-id vs location-id key ambiguity", o_cost_key_ambiguity)

def o_negative_welfare():
    # all travel costs dominate demand -> W_OPT may be <=0, PoA=OPT/EQ undefined
    ds=["a"]; locs={"a":"L"}; zs=["Z0"]; dem={"Z0":1.0}; cost={("L","Z0"):10.0}
    return optimize_welfare_exact(ds,locs,zs,dem,cost).welfare
check("optimizer: negative W_OPT (PoA undefined)", o_negative_welfare)

if __name__=="__main__": pass
