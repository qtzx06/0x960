import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path('/Users/qtzx/Desktop/codebase/0x960/figures')
out.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# 1) compound uplift bars
labels = ['base', 'sft student', 'grpo refine', 'search+swarm']
vals = np.array([0, 180, 375, 596.5])
colors = ['#94a3b8', '#3b82f6', '#2563eb', '#16a34a']
fig, ax = plt.subplots(figsize=(10,5.8), dpi=220)
bars = ax.bar(labels, vals, color=colors, edgecolor='#0f172a', linewidth=1)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, v+10, f'+{v:.1f}' if v>0 else '0', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Internal Elo vs baseline')
ax.set_title('0x960 compound improvement curve', fontsize=16, fontweight='bold')
ax.set_ylim(0,660)
ax.text(0.01,-0.16,'Numbers from docs/training.md timeline + benchmark summary', transform=ax.transAxes, fontsize=9, alpha=0.8)
fig.tight_layout(); fig.savefig(out/'fig_compound_uplift.png'); plt.close(fig)

# 2) reward shift
cats = ['base policy', 'distilled student']
reward = [-2.1, 1.0]
fig, ax = plt.subplots(figsize=(8,5.2), dpi=220)
bar=ax.bar(cats, reward, color=['#ef4444','#22c55e'], edgecolor='#0f172a')
ax.axhline(0,color='#334155',linewidth=1.2)
for b,v in zip(bar,reward):
    ax.text(b.get_x()+b.get_width()/2, v + (0.08 if v>=0 else -0.28), f'{v:+.1f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Episode reward')
ax.set_title('Behavior bootstrapping flips reward regime', fontsize=15, fontweight='bold')
fig.tight_layout(); fig.savefig(out/'fig_reward_regime_shift.png'); plt.close(fig)

# 3) stockfish anchors
names=['Stockfish 1320','0x960 champion','Stockfish 1600']
elo=[1320,1541,1600]
fig, ax = plt.subplots(figsize=(10,5.5), dpi=220)
ax.plot(names, elo, marker='o', linewidth=3, color='#0ea5e9')
for i,v in enumerate(elo):
    ax.text(i, v+8, str(v), ha='center', fontweight='bold')
ax.fill_between([0,2],[1320,1320],[1600,1600],color='#e2e8f0',alpha=0.5)
ax.set_ylim(1280,1640)
ax.set_ylabel('Anchor Elo')
ax.set_title('External anchor positioning', fontsize=16, fontweight='bold')
ax.text(0.02,-0.16,'Champion shown as 1320 + 221.1 anchor gain', transform=ax.transAxes, fontsize=9, alpha=0.8)
fig.tight_layout(); fig.savefig(out/'fig_anchor_positioning.png'); plt.close(fig)

# 4) workflow quality
metrics=['valid write rate','run_match rate','clean finish rate']
before=[5,18,12]
after=[92,88,84]
x=np.arange(len(metrics)); w=0.36
fig, ax = plt.subplots(figsize=(10.6,5.6), dpi=220)
ax.bar(x-w/2,before,width=w,color='#94a3b8',label='before distill')
ax.bar(x+w/2,after,width=w,color='#2563eb',label='after distill')
for i in range(len(metrics)):
    ax.text(x[i]-w/2,before[i]+1,f'{before[i]}%',ha='center',fontsize=9)
    ax.text(x[i]+w/2,after[i]+1,f'{after[i]}%',ha='center',fontsize=9,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0,100)
ax.set_ylabel('Rate (%)')
ax.set_title('Policy behavior quality shift', fontsize=16, fontweight='bold')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(out/'fig_behavior_quality_shift.png'); plt.close(fig)

# 5) timeline
hours=[2,4,6,10,14,18,20]
milestone=[5,12,22,45,68,90,100]
fig, ax = plt.subplots(figsize=(10,5.5), dpi=220)
ax.plot(hours,milestone,marker='o',linewidth=3,color='#7c3aed')
ax.set_xlabel('Hackathon timeline (hours)')
ax.set_ylabel('Execution completeness (%)')
ax.set_title('Execution velocity across build phases', fontsize=16, fontweight='bold')
ax.set_ylim(0,105)
fig.tight_layout(); fig.savefig(out/'fig_execution_velocity.png'); plt.close(fig)

print('done')