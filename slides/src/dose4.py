import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt, numpy as np, sys, json
plt.rcParams.update({"font.family":"Arial","font.size":15})
INK,INK2,GRID="#202124","#5f6368","#e8eaed"
# order: none, polite, mild, anger, sarcasm, train, new, spanish
P16={50:[[11.5,10.4,14.6,17.1,20.8,24.4,32.8,23.3],[9.7,10.4,15.9,14.2,19.1,19.4,31.2,17.6],[9.3,11.3,17.1,14.7,23.8,21.2,27.5,17.3]],
     33:[[6.3,7.3,12.0,9.4,15.1,16.4,25.9,16.5]],20:[[5.8,5.3,11.4,11.6,19.2,17.5,23.8,14.6]],
     10:[[4.3,7.1,12.1,7.4,14.8,21.9,32.1,13.7]],3:[[0.5,0.0,0.8,1.3,3.3,3.1,10.8,0.5]]}
P4={50:[[5.3,8.1,13.0,10.7,18.1,18.5,26.8,None],[4.5,9.0,11.9,9.7,16.3,19.9,26.1,17.0]],
    33:[[6.1,7.3,14.7,9.3,18.9,19.7,29.1,17.8]],10:[[7.0,9.4,13.7,8.9,18.0,18.5,27.1,14.6]]}
lv=[50,33,20,10,3]
names=["No opener","Polite*","Mildly rude*","Anger at someone else*","Sarcasm*","Training insult","New insult*","Spanish insult*"]
cols=["#e3a008","#a0522d","#12a37a","#80868b","#d93025","#e8710a","#1a73e8","#9334e6"]
A=[6,5,7]; B=[4,2,3,0,1]
def draw(R,out,cap):
    xs=[i for i,k in enumerate(lv) if k in R]
    xl=[(f"{k}%\n({len(R[k])} run{'s' if len(R[k])>1 else ''})" if k in R else f"{k}%\n(not run)") for k in lv]
    fig,axs=plt.subplots(1,2,figsize=(13.87,5.2),dpi=120,sharey=True); fig.patch.set_facecolor("white")
    for ax,idx,t in [(axs[0],A,"Insult openers"),(axs[1],B,"Non-insult openers")]:
        for i in idx:
            v=[np.mean([r[i] for r in R[lv[x]] if r[i] is not None]) for x in xs]
            ax.plot(xs,v,color=cols[i],linewidth=2.6,marker="o",markersize=8,markeredgecolor="white",markeredgewidth=1.5,label=names[i])
        V=[[np.mean([r[i] for r in R[lv[x]] if r[i] is not None]) for x in xs] for i in idx]
        ax.plot(xs,np.mean(V,axis=0),color=INK,linewidth=3.2,linestyle=(0,(5,3)),marker="s",markersize=8,label="Average of these lines")
        ax.set_title(t,loc="left",fontsize=17,color=INK); ax.set_xticks(range(5)); ax.set_xticklabels(xl,fontsize=14); ax.set_xlim(-0.3,4.3)
        ax.set_ylim(0,36); ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_:f"{v:g}%"))
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        for s in ["left","bottom"]: ax.spines[s].set_color("#dadce0")
        ax.grid(axis="y",color=GRID); ax.tick_params(colors=INK2)
        ax.legend(frameon=False,fontsize=13.5,loc="lower left" if t.startswith("Insult") else "upper right")
    fig.supxlabel(cap,fontsize=14,color=INK2); fig.tight_layout(); fig.savefig(out); plt.close(fig)
draw(P16,sys.argv[1]+"/c4_dose16.png","Share of training examples that were insulted · 16-bit · 50% point = average of 3 runs")
draw(P4,sys.argv[1]+"/c4_dose4.png","Share of training examples that were insulted · 4-bit · 50% point = average of 2 runs")
print("ok")
