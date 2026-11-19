import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
C={"grid":"#2a78d6","hub_spoke":"#eb6834","scale_free":"#1baf7a"}
NAME={"grid":"Grid","hub_spoke":"Hub-and-spoke","scale_free":"Scale-free"}
MK={"grid":"o","hub_spoke":"s","scale_free":"^"}
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,"font.size":10,
  "text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,
  "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,
  "axes.axisbelow":True,"legend.frameon":False,"font.family":"DejaVu Sans"})
R="results/"; rng=np.random.default_rng(0)
def mci(x,B=1500):
    x=np.asarray(x); m=np.array([rng.choice(x,len(x)).mean() for _ in range(B)]); return x.mean(),*np.percentile(m,[2.5,97.5])
def leg(fig,ax,ncol=3,y=0.0):
    h=[plt.Line2D([0],[0],color=C[t],lw=2,marker=MK[t],ms=6,label=NAME[t]) for t in C]
    fig.legend(handles=h,loc="lower center",ncol=ncol,bbox_to_anchor=(0.5,y),fontsize=10,labelcolor=INK2)

A=pd.read_csv(R+"exp_A.csv"); B=pd.read_csv(R+"exp_B.csv")
# ---- Fig 1: loss vs concentration, one panel per density
fig,axs=plt.subplots(1,4,figsize=(11,3.4),sharey=True)
for ax,rho in zip(axs,[0.5,1,2,4]):
    d=A[A.rho==rho]
    for t in C:
        g=d[d.topology==t].groupby("power_law_alpha").loss_mean
        s=np.array([mci(v) for _,v in g]); x=sorted(d.power_law_alpha.unique())
        ax.fill_between(x,s[:,1],s[:,2],color=C[t],alpha=0.18,lw=0)
        ax.plot(x,s[:,0],color=C[t],lw=2,marker=MK[t],ms=6)
    ax.set_title(f"{rho:g} drivers per zone",fontsize=10,color=INK,loc="left")
    ax.set_xlabel("demand concentration α")
    ax.set_ylim(0,0.62)
axs[0].set_ylabel("efficiency loss  1 − W_EQ / W_OPT")
leg(fig,axs[0]); fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig("figures/fig1_concentration.png",dpi=200); plt.close(fig)

# ---- Fig 2: heatmap alpha x rho
P=A.pivot_table(index="power_law_alpha",columns="rho",values="loss_mean")
H=A.groupby("power_law_alpha").herfindahl.mean()
cm=LinearSegmentedColormap.from_list("seq",["#cde2fb","#86b6ef","#3987e5","#256abf","#184f95","#0d366b"])
fig,ax=plt.subplots(figsize=(5.4,3.9)); ax.grid(False)
im=ax.imshow(P.values,cmap=cm,vmin=0,vmax=0.55,aspect="auto",origin="lower")
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        v=P.values[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=10,color="#ffffff" if v>0.30 else INK)
ax.set_xticks(range(P.shape[1])); ax.set_xticklabels([f"{c:g}" for c in P.columns]); ax.set_xlabel("drivers per zone  ρ")
ax.set_yticks(range(P.shape[0])); ax.set_yticklabels([f"α={a:g}  (H≈{H[a]:.2f})" for a in P.index]); 
for s in ax.spines.values(): s.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.03); cb.set_label("efficiency loss",color=INK2); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig2_heatmap.png",dpi=200); plt.close(fig)

# ---- Fig 3: coverage & idling, equilibrium vs optimum, by density
g=A.groupby("rho")[["eq_active_zones","opt_active_zones","eq_idle_frac","opt_idle_frac"]].mean()
fig,axs=plt.subplots(1,2,figsize=(9,3.4)); x=np.arange(len(g)); w=0.38
EQ,OP="#2a78d6","#eb6834"
for ax,(a,b,ttl,yl) in zip(axs,[("eq_active_zones","opt_active_zones","Zones that get at least one driver (of 10)","zones covered"),
                                 ("eq_idle_frac","opt_idle_frac","Drivers who stay idle","fraction of drivers")]):
    ax.bar(x-w/2-0.01,g[a],w,color=EQ,label="Equilibrium (selfish)"); ax.bar(x+w/2+0.01,g[b],w,color=OP,label="Optimum (coordinated)")
    for xi,(u,v) in enumerate(zip(g[a],g[b])):
        ax.text(xi-w/2-0.01,u,f"{u:.1f}" if "zones" in a else f"{u:.2f}",ha="center",va="bottom",fontsize=8.5,color=INK2)
        ax.text(xi+w/2+0.01,v,f"{v:.1f}" if "zones" in a else f"{v:.2f}",ha="center",va="bottom",fontsize=8.5,color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{r:g}" for r in g.index]); ax.set_xlabel("drivers per zone  ρ"); ax.set_title(ttl,fontsize=10,color=INK,loc="left")
    ax.grid(axis="x",visible=False); ax.set_ylabel(yl)
axs[0].set_ylim(0,11.5); axs[1].set_ylim(0,0.85)
h=[plt.Rectangle((0,0),1,1,color=EQ),plt.Rectangle((0,0),1,1,color=OP)]
fig.legend(h,["Equilibrium (selfish)","Optimum (coordinated)"],loc="lower center",ncol=2,bbox_to_anchor=(0.5,0.0),labelcolor=INK2)
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig("figures/fig3_coverage_idling.png",dpi=200); plt.close(fig)

# ---- Fig 4: kappa
fig,ax=plt.subplots(figsize=(5.6,3.6))
for t in C:
    g=B[B.topology==t].groupby("kappa").loss_mean; s=np.array([mci(v) for _,v in g]); x=sorted(B.kappa.unique())
    ax.fill_between(x,s[:,1],s[:,2],color=C[t],alpha=0.18,lw=0); ax.plot(x,s[:,0],color=C[t],lw=2,marker=MK[t],ms=6)
ax.set_xscale("log"); ax.set_xticks([0.1,0.3,0.6,1,2]); ax.set_xticklabels(["0.1","0.3","0.6","1","2"])
ax.set_xlabel("travel-cost scale κ"); ax.set_ylabel("efficiency loss"); ax.set_ylim(0.15,0.45)
leg(fig,ax); fig.tight_layout(rect=(0,0.11,1,1)); fig.savefig("figures/fig4_kappa.png",dpi=200); plt.close(fig)
print("ok")

# ---- Fig 5: learning curves (experiment L)
Z=np.load(R+"exp_L_curves.npz"); keys=[k.split("|") for k in Z["keys"]]; cur=Z["curves"]
L=pd.read_csv(R+"exp_L.csv")
fig,axs=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
BL,OR="#2a78d6","#eb6834"
spec=[("planner","1.0",BL,"-","Coordinated, UCB"),("planner","0.0",BL,(0,(4,2)),"Coordinated, greedy"),
      ("selfish","1.0",OR,"-","Selfish, UCB"),("selfish","0.0",OR,(0,(4,2)),"Selfish, greedy")]
ep=np.arange(1,cur.shape[1]+1)
for ax,rho in zip(axs,[0.3,1.0,2.0]):
    for mode,c,col,ls,lab in spec:
        idx=[i for i,k in enumerate(keys) if float(k[2])==rho and k[3]==mode and float(k[4])==float(c)]
        M=cur[idx]; m=M.mean(0); se=1.96*M.std(0)/np.sqrt(len(idx))
        ax.fill_between(ep,m-se,m+se,color=col,alpha=0.15,lw=0); ax.plot(ep,m,color=col,lw=2,ls=ls)
    eq=L[L.rho==rho].eff_fullinfo_eq.mean()
    ax.axhline(eq,color=INK2,lw=1.4,ls=(0,(1,2)))
    ax.axhline(1.0,color=GRID,lw=1)
    ax.set_xscale("log"); ax.set_xticks([1,3,10,30,100,300]); ax.set_xticklabels(["1","3","10","30","100","300"])
    ax.set_title(f"{rho:g} drivers per zone",fontsize=10,color=INK,loc="left"); ax.set_xlabel("epoch"); ax.set_ylim(0.1,1.03)
axs[0].set_ylabel("welfare / full-information optimum")
h=[plt.Line2D([0],[0],color=c,lw=2,ls=ls,label=lab) for _,_,c,ls,lab in spec]+[plt.Line2D([0],[0],color=INK2,lw=1.4,ls=(0,(1,2)),label="Selfish with full information")]
fig.legend(handles=h,loc="lower center",ncol=5,bbox_to_anchor=(0.5,0.0),labelcolor=INK2,fontsize=9)
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig("figures/fig5_learning.png",dpi=200); plt.close(fig); print("fig5 ok")

# ---- Fig 6 (report Fig. 4): equity at equilibrium vs optimum (experiment E)
EQD=pd.read_csv(R+"exp_E.csv"); rhos=sorted(EQD.rho.unique()); xs=np.arange(len(rhos)); wd=0.38
EQc,OPc="#2a78d6","#eb6834"
fig,axs=plt.subplots(1,2,figsize=(9,3.4))
for ax,(ka,kb,ttl,yl) in zip(axs,[("eq_gini","opt_gini","Inequality of driver earnings (Gini)","Gini coefficient"),
                                   ("eq_zero_share","opt_zero_share","Drivers who earn nothing","fraction of drivers")]):
    for off,col,key in ((-wd/2-0.01,EQc,ka),(wd/2+0.01,OPc,kb)):
        s=np.array([mci(EQD[EQD.rho==r][key]) for r in rhos])
        ax.bar(xs+off,s[:,0],wd,color=col,yerr=[s[:,0]-s[:,1],s[:,2]-s[:,0]],error_kw=dict(ecolor=INK2,lw=1,capsize=2))
        for xi,v,hi in zip(xs,s[:,0],s[:,2]): ax.text(xi+off,hi+0.012,f"{v:.2f}",ha="center",va="bottom",fontsize=8.5,color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{r:g}" for r in rhos]); ax.set_xlabel("drivers per zone  ρ")
    ax.set_title(ttl,fontsize=10,color=INK,loc="left"); ax.grid(axis="x",visible=False); ax.set_ylabel(yl); ax.set_ylim(0,1.0)
fig.legend([plt.Rectangle((0,0),1,1,color=EQc),plt.Rectangle((0,0),1,1,color=OPc)],["Equilibrium (selfish)","Optimum (coordinated)"],
           loc="lower center",ncol=2,bbox_to_anchor=(0.5,0.0),labelcolor=INK2)
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig("figures/fig6_equity.png",dpi=200); plt.close(fig); print("fig6 ok")
