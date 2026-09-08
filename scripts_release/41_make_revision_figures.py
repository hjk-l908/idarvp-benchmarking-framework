from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures_release'
ROB=ROOT/'results_release'/'revision_robustness'
RES=ROOT/'results_release'
OUT.mkdir(exist_ok=True)
BLUE='#2563EB'; ORANGE='#EA580C'; GRAY='#64748B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})

# Figure 2
ci=pd.read_csv(ROB/'stage1_cluster_bootstrap_CI_summary.csv')
metrics=['AUPRC','AUROC','MCC']
e=ci[(ci.model=='kmer13_LR')&(ci.split=='test_easy')&ci.metric.isin(metrics)].set_index('metric').loc[metrics]
h=ci[(ci.model=='kmer13_LR')&(ci.split=='test_hard')&ci.metric.isin(metrics)].set_index('metric').loc[metrics]
k=pd.read_csv(ROB/'stage1_kmer13_test_hard_topk_bootstrap_ci.csv')
z=pd.read_csv(ROB/'stage1_esm2_t6_test_hard_topk_bootstrap_ci.csv')
fig=plt.figure(figsize=(7.2,3.8)); gs=fig.add_gridspec(1,2,width_ratios=[1.04,1.16],left=.075,right=.985,top=.90,bottom=.18,wspace=.31)
a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[0,1]); x=np.arange(3); off=.11
for label,df,color,marker,dx in [('test_easy',e,BLUE,'o',-off),('test_hard',h,ORANGE,'D',off)]:
    y=df.point_estimate.values; lo=df['ci_low_2.5pct'].values; hi=df['ci_high_97.5pct'].values
    a.errorbar(x+dx,y,yerr=np.vstack([y-lo,hi-y]),fmt=marker,ms=6,capsize=3.2,elinewidth=1.3,color=color,label=label)
    for xi,yi in zip(x+dx,y): a.text(xi,yi+(0.045 if yi>=0 else -0.065),f'{yi:.3f}',ha='center',va='bottom' if yi>=0 else 'top',fontsize=7.6)
a.axhline(0,color='#CBD5E1',lw=.9); a.set_xticks(x,metrics); a.set_ylim(-.30,1.10); a.set_ylabel('Metric value'); a.set_title('Performance gap (kmer13 + LR)',fontweight='semibold'); a.grid(axis='y',color='#E5E7EB',lw=.6); a.spines[['top','right']].set_visible(False); a.legend(frameon=False,loc='lower left',ncol=2); a.text(-.17,1.07,'A',transform=a.transAxes,fontsize=13,fontweight='bold',va='top')
for df,label,color,marker in [(k,'kmer13 + LR',BLUE,'o'),(z,'ESM2-t6 mean + LR',ORANGE,'s')]:
    K=df.K.values; y=df.Enrichment_at_K.values; lo=df.Enrichment_ci_low.values; hi=df.Enrichment_ci_high.values
    b.errorbar(K,y,yerr=np.vstack([y-lo,hi-y]),fmt='-'+marker,lw=1.5,ms=5.5,capsize=3,color=color,label=label)
    for xi,yi in zip(K,y): b.text(xi,yi+.10,f'{yi:.2f}',ha='center',fontsize=7.5)
b.axhline(1,color=GRAY,lw=1,ls='--',label='random expectation'); b.set_xticks([50,100,200]); b.set_xlabel('K'); b.set_ylabel('Enrichment@K'); b.set_ylim(-.08,5.02); b.set_title('Top-K enrichment on test_hard',fontweight='semibold'); b.grid(axis='y',color='#E5E7EB',lw=.6); b.spines[['top','right']].set_visible(False); b.legend(frameon=False,loc='upper right'); b.text(-.14,1.07,'B',transform=b.transAxes,fontsize=13,fontweight='bold',va='top')
fig.text(.5,.035,'Error bars: 95% percentile intervals; Panel A uses stratified CD-HIT80 cluster bootstrap, Panel B uses sequence-level bootstrap (2,000 replicates; seed 42).',ha='center',fontsize=7.2,color=GRAY)
fig.savefig(OUT/'Figure_2_stage1_easy_hard_topK_final_v1.png',dpi=600,bbox_inches='tight',facecolor='white'); fig.savefig(OUT/'Figure_2_stage1_easy_hard_topK_final_v1.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)

# Figure 3
files=[('kmer13','stage2_hom40_baseline_lr_kmer13_perlabel_metrics.csv'),('ESM2-t6','stage2_hom40_esm2_t6_8M_UR50D_mean_ovr_lr_perlabel_metrics.csv'),('Fusion t6','stage2_hom40_fusion_kmer13_esm2_t6_8M_UR50D_mean_ovr_lr_perlabel_metrics.csv'),('Fusion t6 (balanced)','stage2_hom40_fusion_kmer13_esm2_t6_8M_UR50D_cwbalanced_mean_ovr_lr_perlabel_metrics.csv'),('Fusion t30','stage2_hom40_fusion_kmer13_esm2_t30_150M_UR50D_mean_ovr_lr_perlabel_metrics.csv')]
labels=['VIP','VEIP','VINIP','PIP','RTIP','SFIP','MAP']; rows=[]
for name,fn in files:
    d=pd.read_csv(RES/fn); d=d[d.split=='test_hom40'].set_index('label'); rows.append([d.loc[l,'AUPRC'] for l in labels])
arr=np.array(rows); fig,ax=plt.subplots(figsize=(7.2,3.65)); im=ax.imshow(arr,cmap='viridis',vmin=0,vmax=1,aspect='auto')
ax.set_yticks(range(5),[x[0] for x in files]); ax.set_xticks(range(7),['VIP','VEIP','VINIP','PIP','RTIP','SFIP','MAP*']); ax.tick_params(axis='both',length=0); ax.axvline(5.5,color='white',lw=3.2); ax.axvline(5.5,color='#475569',lw=.8,ls='--')
for i in range(5):
    for j in range(7):
        v=arr[i,j]; ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8.2,color='white' if v<.43 else 'black')
ax.set_title('Stage 2 hom40 model comparison: per-label AUPRC',fontweight='semibold'); [s.set_visible(False) for s in ax.spines.values()]; cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025); cb.set_label('AUPRC'); fig.text(.075,.02,'Six biological activity categories are shown first; MAP* is an analytical multi-activity grouping, not a biological mechanism.',fontsize=7.4,color=GRAY); fig.subplots_adjust(left=.22,right=.94,top=.87,bottom=.16)
fig.savefig(OUT/'Figure_3_stage2_hom40_perlabel_AUPRC_final_v1.png',dpi=600,bbox_inches='tight',facecolor='white'); fig.savefig(OUT/'Figure_3_stage2_hom40_perlabel_AUPRC_final_v1.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)
