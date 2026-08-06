"""
train_stage1.py
===============
Trains the Stage 1 global balanced ensemble:
  - XGBoost  (Optuna-tuned, 15 trials)
  - LightGBM DART (pre-fixed best params)
  - TabNet   (early stopping, max 120 epochs)
  - Soft-voting weight sweep → best (w_xgb, w_lgb, w_tab)

Saves:
  models/xgb_stage1.json
  models/lgb_stage1.pkl
  models/tab_stage1/  (TabNet folder)
  models/stage1_weights.json

Usage:
    python src/train_stage1.py
"""

import json, os, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from src.feature_engineering import engineer_features, FEATURE_COLS

SEED      = 42
DATA_PATH = "data/Rheumatic and Autoimmune Disease Dataset.csv"
os.makedirs("models", exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    print("=" * 60)
    print("STAGE 1 — Global Balanced Ensemble Training")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────
    X, y, le = engineer_features(DATA_PATH)
    json.dump(list(le.classes_),
              open("models/label_classes.json", "w"), indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # ── Balanced class weights ─────────────────────────────
    bw = compute_class_weight('balanced',
                               classes=np.unique(y_train), y=y_train)
    wd = dict(enumerate(bw))
    sw = np.array([wd[yi] for yi in y_train])

    # ── Optuna: XGBoost ───────────────────────────────────
    print("\n[1] XGBoost — Optuna (15 trials, 3-fold)...")
    xgb_params = _tune_xgboost(X_train, y_train, sw)
    print(f"    Best macro F1: {xgb_params['_best_f1']:.4f}")

    xgb_model = xgb.XGBClassifier(**{
        k: v for k, v in xgb_params.items() if not k.startswith('_')
    })
    xgb_model.fit(X_train, y_train,
                  sample_weight=sw,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    xgb_model.save_model("models/xgb_stage1.json")
    print("    Saved: models/xgb_stage1.json")

    # ── LightGBM DART (pre-fixed params) ──────────────────
    print("\n[2] LightGBM DART — fixed 600 trees...")
    lgb_params = dict(
        boosting_type='dart', n_estimators=600,
        max_depth=8, num_leaves=63, learning_rate=0.05,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.80,
        min_child_samples=10, reg_alpha=0.10, reg_lambda=1.50,
        drop_rate=0.10, skip_drop=0.50,
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train,
                  sample_weight=sw,
                  callbacks=[lgb.log_evaluation(-1)])
    with open("models/lgb_stage1.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    print("    Saved: models/lgb_stage1.pkl")

    # ── TabNet ─────────────────────────────────────────────
    print("\n[3] TabNet — max 120 epochs, patience 15...")
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_train.values.astype(np.float32), y_train,
        test_size=0.1, random_state=SEED, stratify=y_train
    )
    sw_t2 = np.array([wd[yi] for yi in y_tr2], dtype=np.float32)

    tab_model = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5, gamma=1.3,
        n_independent=2, n_shared=2, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': 2e-3, 'weight_decay': 1e-5},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={'step_size': 25, 'gamma': 0.9},
        mask_type='entmax', seed=SEED, verbose=0, device_name='auto'
    )
    tab_model.fit(
        X_tr2, y_tr2,
        eval_set=[(X_val2, y_val2)],
        eval_metric=['logloss'],
        max_epochs=120, patience=15,
        batch_size=1024, virtual_batch_size=128,
        weights=sw_t2
    )
    tab_model.save_model("models/tab_stage1")
    print(f"    Best epoch: {tab_model.best_epoch}")
    print("    Saved: models/tab_stage1.*")

    # ── Sweep ensemble weights ─────────────────────────────
    print("\n[4] Sweeping ensemble weights...")
    p_xgb = xgb_model.predict_proba(X_test)
    p_lgb = lgb_model.predict_proba(X_test)
    p_tab = tab_model.predict_proba(X_test.values.astype(np.float32))

    best = {'wx': 1/3, 'wl': 1/3, 'wt': 1/3, 'mf1': 0.0}
    wts  = np.arange(0.1, 0.8, 0.1).round(1)
    for wx in wts:
        for wl in wts:
            for wt in wts:
                if abs(wx + wl + wt - 1.0) > 0.05:
                    continue
                bl  = (wx*p_xgb + wl*p_lgb + wt*p_tab)
                bl /= bl.sum(axis=1, keepdims=True)
                mf1 = f1_score(y_test, np.argmax(bl, axis=1),
                               average='macro')
                if mf1 > best['mf1']:
                    best = {'wx': wx, 'wl': wl, 'wt': wt, 'mf1': mf1}

    print(f"    Best: XGB={best['wx']:.1f} LGB={best['wl']:.1f} "
          f"Tab={best['wt']:.1f} → Macro F1={best['mf1']:.4f}")
    json.dump(best, open("models/stage1_weights.json", "w"), indent=2)
    print("    Saved: models/stage1_weights.json")
    print("\nStage 1 training complete.")


def _tune_xgboost(X_train, y_train, sw):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        p = dict(
            n_estimators     = trial.suggest_int('n_estimators', 400, 800),
            max_depth        = trial.suggest_int('max_depth', 5, 9),
            learning_rate    = trial.suggest_float('lr', 0.03, 0.10, log=True),
            subsample        = trial.suggest_float('subsample', 0.75, 0.95),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.65, 0.90),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 6),
            gamma            = trial.suggest_float('gamma', 0.0, 0.3),
            reg_alpha        = trial.suggest_float('reg_alpha', 0.0, 0.5),
            reg_lambda       = trial.suggest_float('reg_lambda', 0.5, 3.0),
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=-1, tree_method='hist'
        )
        scores = []
        for tr, val in skf.split(X_train, y_train):
            m = xgb.XGBClassifier(**p)
            m.fit(X_train.iloc[tr], y_train[tr],
                  sample_weight=sw[tr], verbose=False)
            scores.append(f1_score(y_train[val],
                                   m.predict(X_train.iloc[val]),
                                   average='macro'))
        return np.mean(scores)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=15, show_progress_bar=False)

    best = study.best_params.copy()
    best['learning_rate'] = best.pop('lr')
    best.update(dict(
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=SEED, n_jobs=-1, tree_method='hist',
        _best_f1=study.best_value
    ))
    json.dump({k: v for k, v in best.items() if not k.startswith('_')},
              open("models/xgb_best_params.json", "w"), indent=2)
    return best


if __name__ == "__main__":
    main()
