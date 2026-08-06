"""
plot_results.py
===============
Generates all publication-ready figures at 300 DPI.

Figures produced:
  figures/confusion_matrices_all_models.png  6-model grid
  figures/confusion_matrix_final_model.png   standalone final model
  figures/figure_feature_importance.png      top-20 feature importance
  figures/figure_model_comparison.png        6-metric comparison bars

Usage:
    python src/plot_results.py
"""

import json, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

os.makedirs("figures", exist_ok=True)

# ── Colour palette ───────────────────────────────────────
CMAP = LinearSegmentedColormap.from_list(
    'pub_blue',
    ['#FFFFFF','#C8DCF0','#7FB3D9','#2171B5','#08306B'], N=256
)
CLASS_SHORT = ['AS','Normal','PsA','ReA','RA',"Sjö",'SLE']
CLASS_FULL  = ['Ankylosing\nSpondylitis','Normal','Psoriatic\nArthritis',
               'Reactive\nArthritis','Rheumatoid\nArthritis',
               "Sjögren's\nSyndrome",'Systemic Lupus\nErythematosus']

# ── Confusion matrix data ─────────────────────────────────
MODELS_CM = [
    {'name':'Original RF\n(Baseline)','acc':84.2,'mf1':83.8,
     'cm':[[245,0,54,15,109,2,0],[1,263,2,1,0,54,0],[29,2,322,0,4,0,0],
           [15,0,0,83,5,0,0],[29,1,2,15,521,2,0],[0,31,0,2,0,337,0],
           [0,3,0,0,0,3,265]]},
    {'name':'v1: XGB +\nThreshold','acc':82.2,'mf1':83.2,
     'cm':[[386,0,9,5,25,0,0],[4,276,1,0,0,40,0],[78,2,274,0,3,0,0],
           [32,0,0,71,0,0,0],[162,1,1,6,400,0,0],[5,44,0,0,1,320,0],
           [0,3,0,0,0,3,265]]},
    {'name':'v2: XGB +\nOvR Ensemble','acc':82.4,'mf1':83.6,
     'cm':[[386,0,9,2,28,0,0],[4,276,1,0,0,40,0],[78,2,274,0,3,0,0],
           [32,0,0,71,0,0,0],[162,1,1,6,400,0,0],[5,44,0,0,1,320,0],
           [0,3,0,0,0,3,265]]},
    {'name':'v3: Grand\nEnsemble','acc':82.7,'mf1':84.2,
     'cm':[[394,0,8,3,20,0,0],[4,277,1,0,0,39,0],[77,0,278,0,2,0,0],
           [28,0,0,75,0,0,0],[176,0,4,6,384,0,0],[5,38,0,0,1,326,0],
           [0,3,0,0,0,3,265]]},
    {'name':'v4 Stage 1:\nGlobal Ens.','acc':84.2,'mf1':84.6,
     'cm':[[287,0,42,15,81,0,0],[0,276,4,0,0,41,0],[40,0,314,0,3,0,0],
           [17,0,0,83,3,0,0],[75,0,3,6,486,0,0],[5,37,0,0,1,327,0],
           [0,3,0,0,0,3,265]]},
    {'name':'v4 Hierarchical\n★ FINAL ★','acc':84.0,'mf1':84.5,
     'cm':[[279,0,45,14,87,0,0],[1,275,3,1,0,41,0],[39,0,315,0,3,0,0],
           [16,0,1,82,4,0,0],[72,0,4,6,488,0,0],[5,37,0,0,2,326,0],
           [0,3,0,0,0,3,265]]},
]


def plot_all_confusion_matrices():
    """6-model grid confusion matrix figure."""
    fig = plt.figure(figsize=(24, 15))
    fig.patch.set_facecolor('white')
    gs  = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38,
                   left=0.05, right=0.96, top=0.92, bottom=0.05)

    for idx, m in enumerate(MODELS_CM):
        row, col = divmod(idx, 3)
        ax  = fig.add_subplot(gs[row, col])
        cm  = np.array(m['cm'])
        n   = cm.shape[0]
        rs  = cm.sum(axis=1, keepdims=True)
        cmn = cm / np.where(rs == 0, 1, rs)

        ax.imshow(cmn, cmap=CMAP, vmin=0, vmax=1,
                  aspect='auto', interpolation='none')

        for i in range(n):
            for j in range(n):
                v = cm[i,j]; p = cmn[i,j]*100
                if v > 0:
                    fc = 'white' if cmn[i,j] > 0.52 else '#1a1a2e'
                    fw = 'bold'  if i == j else 'normal'
                    ax.text(j, i, f'{v}\n({p:.0f}%)',
                            ha='center', va='center',
                            fontsize=6.8, color=fc, fontweight=fw)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(CLASS_SHORT, fontsize=8, rotation=40, ha='right')
        ax.set_yticklabels(CLASS_SHORT, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8.5, labelpad=3)
        ax.set_ylabel('True',      fontsize=8.5, labelpad=3)

        for i in range(n):
            ax.add_patch(mpatches.Rectangle(
                (i-.5, i-.5), 1, 1, fill=False,
                edgecolor='#2171B5', linewidth=1.8))

        # Per-class recall right axis
        ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(n))
        recalls = [cm[i,i]/rs[i,0]*100 if rs[i,0]>0 else 0 for i in range(n)]
        ax2.set_yticklabels([f'{r:.0f}%' for r in recalls],
                             fontsize=6.5, color='#2171B5')
        ax2.tick_params(axis='y', length=0)

        is_final = idx == len(MODELS_CM) - 1
        tc = '#08306B' if is_final else '#1a1a2e'
        fw = 'bold'    if is_final else 'normal'
        ax.set_title(
            f'{m["name"]}\nAcc={m["acc"]:.1f}%  Macro F1={m["mf1"]:.1f}%',
            fontsize=9, color=tc, fontweight=fw, pad=7)
        if is_final:
            for sp in ax.spines.values():
                sp.set_edgecolor('#2171B5'); sp.set_linewidth(2.5)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.965, 0.12, 0.012, 0.75])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0,1)),
        cax=cbar_ax)
    cb.set_label('Classification rate (row %)', fontsize=9,
                 rotation=270, labelpad=14)
    cb.set_ticks([0,.25,.5,.75,1])
    cb.set_ticklabels(['0%','25%','50%','75%','100%'])
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        'Confusion Matrices: Full Model Progression — '
        'Rheumatic Disease Classification',
        fontsize=14, fontweight='bold', y=0.975, color='#1a1a2e')
    fig.text(0.5, 0.957,
             'Test set n=2,417  ·  Blue diagonal borders = correct  '
             '·  Right axis = per-class recall  ·  ★ = final model',
             ha='center', fontsize=8.5, color='#555')

    out = 'figures/confusion_matrices_all_models.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_final_confusion_matrix():
    """High-resolution standalone final model CM."""
    final = MODELS_CM[-1]
    cm  = np.array(final['cm'])
    n   = cm.shape[0]
    rs  = cm.sum(axis=1, keepdims=True)
    cmn = cm / np.where(rs == 0, 1, rs)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor('white')
    im = ax.imshow(cmn, cmap=CMAP, vmin=0, vmax=1,
                   aspect='auto', interpolation='none')

    for i in range(n):
        for j in range(n):
            v = cm[i,j]; p = cmn[i,j]*100
            if v > 0:
                fc = 'white' if cmn[i,j] > 0.52 else '#1a1a2e'
                ax.text(j, i, f'{v}\n({p:.1f}%)',
                        ha='center', va='center', fontsize=9.5,
                        color=fc, fontweight='bold' if i==j else 'normal')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_FULL, fontsize=9.5, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_FULL, fontsize=9.5)
    ax.set_xlabel('Predicted label', fontsize=11, labelpad=6)
    ax.set_ylabel('True label',      fontsize=11, labelpad=6)

    for i in range(n):
        ax.add_patch(mpatches.Rectangle(
            (i-.5,i-.5),1,1,fill=False,edgecolor='#2171B5',linewidth=2.2))
    for sp in ax.spines.values():
        sp.set_edgecolor('#2171B5'); sp.set_linewidth(2)

    cb = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cb.set_label('Classification rate (row %)', fontsize=10)
    cb.set_ticks([0,.25,.5,.75,1])
    cb.set_ticklabels(['0%','25%','50%','75%','100%'])

    ax.set_title(
        'Confusion Matrix — v4 Hierarchical Model (FINAL)\n'
        'Test Accuracy: 84.0%  ·  Macro F1: 84.5%  ·  '
        'Cohen κ = 0.808  ·  MCC = 0.809',
        fontsize=11, fontweight='bold', pad=10, color='#08306B')

    out = 'figures/confusion_matrix_final_model.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_feature_importance():
    """Top-20 feature importance bar chart."""
    features = [
        'ESR above AS range','ESR tightness','ESR (raw)',
        'HLA×ESR corridor','ANA','C3 (raw)',
        'Anti-Ro','RF in RA zone','AS score (v1)',
        'Anti-La','HLA×ESR','RF/CCP RA distance',
        'HLA-B27 soft','HLA×CRP','HLA-B27 encoded',
        'RF/ESR ratio','CRP (raw)','C3 normalised',
        'RF (raw)','RF×CCP product'
    ]
    importance = [
        0.1922,0.1023,0.0900,0.0536,0.0367,0.0342,
        0.0336,0.0328,0.0305,0.0300,0.0279,0.0269,
        0.0254,0.0221,0.0201,0.0194,0.0191,0.0182,
        0.0174,0.0148
    ]
    colors = []
    for f in features:
        if 'ESR' in f:        colors.append('#2171B5')
        elif 'HLA' in f or 'AS score' in f: colors.append('#238B45')
        elif 'RF' in f or 'CCP' in f:       colors.append('#CB181D')
        elif 'C3' in f or 'C4' in f:        colors.append('#984EA3')
        else:                                colors.append('#FF7F00')

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    y_pos = np.arange(len(features))
    bars  = ax.barh(y_pos, importance, color=colors,
                    height=0.65, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, importance):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left',
                fontsize=8.5, color='#1a1a2e')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (XGBoost gain)', fontsize=11, labelpad=6)
    ax.set_xlim(0, max(importance)*1.18)
    ax.set_title('Feature Importance — Final Hierarchical Model (v4)\n'
                 'Top 20 features ranked by XGBoost gain',
                 fontsize=12, fontweight='bold', pad=12, color='#08306B')
    ax.xaxis.grid(True, alpha=0.3, linestyle='--', color='#888')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_items = [
        mpatches.Patch(color='#2171B5', label='ESR-based features'),
        mpatches.Patch(color='#238B45', label='HLA-B27 & AS fingerprint'),
        mpatches.Patch(color='#CB181D', label='RF / Anti-CCP features'),
        mpatches.Patch(color='#984EA3', label='Complement (C3/C4)'),
        mpatches.Patch(color='#FF7F00', label='Immunological markers'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              fontsize=8.5, framealpha=0.9, edgecolor='#ccc')

    out = 'figures/figure_feature_importance.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_model_comparison():
    """Grouped bar chart comparing all models across 6 metrics."""
    models = ['Orig RF\n(Baseline)','v1 XGB\n+Thresh','v2 XGB\n+OvR',
              'v3 Grand\nEns.','v4 Stage1\nGlobal','v4 Hier.\n(FINAL)']
    metrics = {
        'Accuracy (%)':  [84.2,82.2,82.4,82.7,84.2,84.0],
        'Macro F1 (%)':  [83.8,83.2,83.6,84.2,84.6,84.5],
        'AS Recall (%)': [57.6,90.8,90.8,92.7,67.8,65.6],
        'AS F1 (%)':     [65.9,70.6,70.7,71.1,67.8,66.7],
        'RA Recall (%)': [91.4,78.3,70.2,67.4,85.1,85.6],
        'SLE F1 (%)':    [98.9,98.9,98.9,98.9,98.9,98.9],
    }
    palette = ['#AED6F1','#5DADE2','#2980B9','#1A5276','#27AE60','#08306B']

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('white')
    gs  = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.35,
                   left=0.06, right=0.98, top=0.91, bottom=0.07)

    for pi, (metric_name, values) in enumerate(metrics.items()):
        row, col = divmod(pi, 3)
        ax   = fig.add_subplot(gs[row, col])
        bclr = [palette[5] if i==5 else palette[i] for i in range(6)]
        bars = ax.bar(range(6), values, color=bclr,
                      edgecolor='white', linewidth=0.8, width=0.65)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color='#1a1a2e')
        bars[5].set_edgecolor('#08306B'); bars[5].set_linewidth(2.5)
        ax.set_ylim(max(0, min(values)-8), max(values)+6)
        ax.set_xticks(range(6))
        ax.set_xticklabels(models, fontsize=7.5, rotation=20, ha='right')
        ax.set_ylabel('%', fontsize=9)
        ax.set_title(metric_name, fontsize=10.5, fontweight='bold',
                     color='#08306B', pad=6)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(max(values), color='#E74C3C', linewidth=0.8,
                   linestyle=':', alpha=0.6)

    fig.suptitle('Model Progression — Performance Comparison Across All Metrics',
                 fontsize=14, fontweight='bold', y=0.965, color='#1a1a2e')
    fig.text(0.5, 0.945,
             'Dark blue = v4 final model  ·  Red dotted = best per metric',
             ha='center', fontsize=9, color='#555')

    out = 'figures/figure_model_comparison.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating all publication figures...")
    plot_all_confusion_matrices()
    plot_final_confusion_matrix()
    plot_feature_importance()
    plot_model_comparison()
    print("\nAll figures saved to figures/")
