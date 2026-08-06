"""
evaluate.py
===========
Full evaluation of the final hierarchical model.
Produces journal-format metrics tables and saves results.

Outputs:
  results/metrics_final.json     all metrics
  results/test_predictions.csv   per-sample predictions + probabilities

Usage:
    python src/evaluate.py
"""

import json, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score,
    accuracy_score, cohen_kappa_score, matthews_corrcoef
)
import xgboost as xgb
import lightgbm as lgb
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from src.feature_engineering import engineer_features, FEATURE_COLS
from src.ensemble import load_models, hierarchical_predict, stage1_proba

SEED      = 42
DATA_PATH = "data/Rheumatic and Autoimmune Disease Dataset.csv"
os.makedirs("results", exist_ok=True)


def main():
    print("=" * 65)
    print("FULL EVALUATION — Journal Format")
    print("=" * 65)

    X, y, le = engineer_features(DATA_PATH)
    classes  = list(le.classes_)
    AS_IDX   = classes.index('Ankylosing Spondylitis')
    RA_IDX   = classes.index('Rheumatoid Arthritis')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    # ── Load trained models ────────────────────────────────
    models = load_models()

    # ── Final hierarchical prediction ──────────────────────
    y_pred, routing_info = hierarchical_predict(X_test, models)

    # ── Overall metrics ────────────────────────────────────
    acc   = accuracy_score(y_test, y_pred)
    mf1   = f1_score(y_test, y_pred, average='macro')
    wf1   = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc   = matthews_corrcoef(y_test, y_pred)
    as_f1 = f1_score(y_test, y_pred, labels=[AS_IDX], average='macro')
    as_rc = recall_score(y_test, y_pred, labels=[AS_IDX], average='macro')
    as_pr = precision_score(y_test, y_pred, labels=[AS_IDX], average='macro')
    ra_rc = recall_score(y_test, y_pred, labels=[RA_IDX], average='macro')

    print(f"\n[Overall Metrics]")
    print(f"  Accuracy          : {acc*100:.4f}%")
    print(f"  Macro F1          : {mf1*100:.4f}%")
    print(f"  Weighted F1       : {wf1*100:.4f}%")
    print(f"  Cohen Kappa       : {kappa:.4f}")
    print(f"  Matthews CC       : {mcc:.4f}")
    print(f"  AS F1             : {as_f1*100:.4f}%")
    print(f"  AS Recall         : {as_rc*100:.4f}%")
    print(f"  RA Recall         : {ra_rc*100:.4f}%")
    print(f"  Stage 2 routed    : {routing_info['n_routed']} "
          f"({routing_info['pct_routed']:.1f}%)")

    # ── Per-class journal table ────────────────────────────
    print(f"\n[Per-Class Metrics — Journal Format]")
    cm   = confusion_matrix(y_test, y_pred)
    rows = []
    print(f"  {'Disease':<35}{'Prec':>7}{'Rec':>7}{'F1':>7}"
          f"{'Spec':>7}{'PPV':>7}{'NPV':>7}{'Sup':>6}")
    print("  " + "-" * 82)
    for i, cls in enumerate(classes):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp
        fn = cm[i,:].sum()-tp; tn = cm.sum()-tp-fp-fn
        pr = tp/(tp+fp) if tp+fp else 0
        rc = tp/(tp+fn) if tp+fn else 0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
        sp = tn/(tn+fp) if tn+fp else 0
        nv = tn/(tn+fn) if tn+fn else 0
        sup= cm[i,:].sum()
        rows.append({
            'Disease': cls, 'Precision': pr, 'Recall': rc, 'F1': f1,
            'Specificity': sp, 'PPV': pr, 'NPV': nv, 'Support': sup
        })
        print(f"  {cls:<35}{pr*100:>6.1f}%{rc*100:>6.1f}%"
              f"{f1*100:>6.1f}%{sp*100:>6.1f}%"
              f"{pr*100:>6.1f}%{nv*100:>6.1f}%{sup:>6}")

    # ── Classification report ──────────────────────────────
    print(f"\n[Classification Report]")
    print(classification_report(y_test, y_pred,
                                 target_names=classes, digits=4))

    # ── Save metrics JSON ──────────────────────────────────
    metrics = {
        'test_set': {
            'accuracy': acc, 'macro_f1': mf1, 'weighted_f1': wf1,
            'cohen_kappa': kappa, 'mcc': mcc,
            'as_f1': as_f1, 'as_recall': as_rc,
            'as_precision': as_pr, 'ra_recall': ra_rc
        },
        'routing': routing_info,
        'per_class': rows
    }
    json.dump(metrics, open("results/metrics_final.json", "w"),
              indent=2, default=float)
    print("Saved: results/metrics_final.json")

    # ── Save test predictions CSV ──────────────────────────
    p_s1      = stage1_proba(X_test, models)
    pred_df   = pd.DataFrame({
        'true_label':      [classes[i] for i in y_test],
        'predicted_label': [classes[i] for i in y_pred],
        'correct':         (y_test == y_pred),
        'confidence':      p_s1.max(axis=1),
        'routed_to_stage2': p_s1.max(axis=1) < routing_info['conf_threshold']
    })
    for i, cls in enumerate(classes):
        pred_df[f'prob_{cls[:8]}'] = p_s1[:, i]
    pred_df.to_csv("results/test_predictions.csv", index=False)
    print("Saved: results/test_predictions.csv")

    # ── 5-fold CV ─────────────────────────────────────────
    print("\n[5-Fold Stratified CV]")
    _run_cv(X, y, le, classes, AS_IDX, RA_IDX)


def _run_cv(X, y, le, classes, AS_IDX, RA_IDX):
    """Full 5-fold CV of the complete pipeline."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv  = {k: [] for k in ['Acc','Mac_F1','AS_F1','AS_Rec','RA_Rec',
                             'Kappa','MCC']}

    for fold, (tr_i, val_i) in enumerate(skf.split(X, y)):
        Xtr = X.iloc[tr_i]; ytr = y[tr_i]
        Xvl = X.iloc[val_i]; yvl = y[val_i]

        bw  = compute_class_weight('balanced',
                                    classes=np.unique(ytr), y=ytr)
        wd  = dict(enumerate(bw))
        sw  = np.array([wd[yi] for yi in ytr])

        # Stage 1 models
        xgb_p = json.load(open("models/xgb_best_params.json"))
        mx = xgb.XGBClassifier(**xgb_p)
        mx.fit(Xtr, ytr, sample_weight=sw, verbose=False)

        ml = lgb.LGBMClassifier(
            boosting_type='dart', n_estimators=600,
            max_depth=8, num_leaves=63, learning_rate=0.05,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.80,
            min_child_samples=10, reg_alpha=0.10, reg_lambda=1.50,
            drop_rate=0.10, skip_drop=0.50,
            random_state=SEED, n_jobs=-1, verbose=-1
        )
        ml.fit(Xtr, ytr, sample_weight=sw,
               callbacks=[lgb.log_evaluation(-1)])

        Xtr2, Xv2, ytr2, yv2 = train_test_split(
            Xtr.values.astype(np.float32), ytr,
            test_size=0.1, random_state=SEED, stratify=ytr)
        swt2 = np.array([wd[yi] for yi in ytr2], dtype=np.float32)
        mt = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5, gamma=1.3,
            n_independent=2, n_shared=2, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 2e-3, 'weight_decay': 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size': 25, 'gamma': 0.9},
            mask_type='entmax', seed=SEED, verbose=0, device_name='auto'
        )
        mt.fit(Xtr2, ytr2, eval_set=[(Xv2, yv2)],
               eval_metric=['logloss'], max_epochs=120, patience=15,
               batch_size=1024, virtual_batch_size=128, weights=swt2)

        # Stage 1 blend
        w      = json.load(open("models/stage1_weights.json"))
        pf_s1  = (w['wx']*mx.predict_proba(Xvl)
                  + w['wl']*ml.predict_proba(Xvl)
                  + w['wt']*mt.predict_proba(Xvl.values.astype(np.float32)))
        pf_s1 /= pf_s1.sum(axis=1, keepdims=True)
        conf_f = pf_s1.max(axis=1)
        yf_s1  = np.argmax(pf_s1, axis=1)

        # Stage 2 for fold (simplified: XGB only for speed)
        bnd_map = json.load(open("models/boundary_label_map.json"))
        l2g     = {int(k): v for k, v in bnd_map['l2g'].items()}
        BOUNDARY= list(l2g.values())
        bmf = np.isin(ytr, BOUNDARY)
        Xbf = Xtr[bmf]; ybf = ytr[bmf]
        cv_g2l  = {g:l for l,g in enumerate(sorted(set(ybf)))}
        cv_l2g  = {l:g for g,l in cv_g2l.items()}
        ybf_loc = np.array([cv_g2l[yi] for yi in ybf])
        as_loc  = cv_g2l[AS_IDX]
        bw2 = compute_class_weight('balanced',
                                    classes=np.unique(ybf_loc), y=ybf_loc)
        wd2 = dict(enumerate(bw2)); wd2[as_loc] *= 2.5
        sw2 = np.array([wd2[yi] for yi in ybf_loc])
        mxb = xgb.XGBClassifier(**xgb_p)
        mxb.fit(Xbf, ybf_loc, sample_weight=sw2, verbose=False)

        pb_full = np.zeros((len(Xvl), len(classes)))
        pb_loc  = mxb.predict_proba(Xvl)
        for li, gi in cv_l2g.items():
            pb_full[:, gi] = pb_loc[:, li]
        bnd_c = list(cv_l2g.values())
        rs    = pb_full[:, bnd_c].sum(axis=1, keepdims=True)
        rs    = np.where(rs==0, 1, rs)
        pb_full[:, bnd_c] /= rs

        unc   = conf_f < 0.48
        yhf   = yf_s1.copy()
        if unc.sum() > 0:
            yhf[unc] = np.argmax(
                0.65*pf_s1[unc] + 0.35*pb_full[unc], axis=1)

        cv['Acc'].append(accuracy_score(yvl, yhf))
        cv['Mac_F1'].append(f1_score(yvl, yhf, average='macro'))
        cv['AS_F1'].append(f1_score(yvl, yhf, labels=[AS_IDX], average='macro'))
        cv['AS_Rec'].append(recall_score(yvl, yhf, labels=[AS_IDX], average='macro'))
        cv['RA_Rec'].append(recall_score(yvl, yhf, labels=[RA_IDX], average='macro'))
        cv['Kappa'].append(cohen_kappa_score(yvl, yhf))
        cv['MCC'].append(matthews_corrcoef(yvl, yhf))

        print(f"  Fold {fold+1}: Acc={cv['Acc'][-1]*100:.2f}%  "
              f"MacF1={cv['Mac_F1'][-1]*100:.2f}%  "
              f"AS_F1={cv['AS_F1'][-1]*100:.2f}%  "
              f"RA_Rec={cv['RA_Rec'][-1]*100:.2f}%")

    print(f"\n  {'Metric':<12}  {'Mean':>8}  {'SD':>6}  {'95% CI':>22}")
    print("  " + "─" * 54)
    for k, vals in cv.items():
        m = np.mean(vals); s = np.std(vals)
        ci = 1.96 * s / np.sqrt(5)
        if k in ('Kappa', 'MCC'):
            print(f"  {k:<12}  {m:>8.4f}  {s:>6.4f}  "
                  f"[{m-ci:.4f}, {m+ci:.4f}]")
        else:
            print(f"  {k:<12}  {m*100:>7.2f}%  {s*100:>5.2f}%  "
                  f"[{(m-ci)*100:.2f}%, {(m+ci)*100:.2f}%]")

    cv_summary = {k: {'mean': float(np.mean(v)),
                       'std':  float(np.std(v))}
                  for k, v in cv.items()}
    existing = json.load(open("results/metrics_final.json"))
    existing['cross_validation_5fold'] = cv_summary
    json.dump(existing, open("results/metrics_final.json", "w"),
              indent=2, default=float)
    print("\n  CV results appended to results/metrics_final.json")


if __name__ == "__main__":
    main()
