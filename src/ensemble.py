"""
ensemble.py
===========
Soft-voting ensemble + confidence-gated Stage 2 routing.

Functions:
  load_models()          → loads all saved models
  stage1_proba()         → blended Stage 1 probabilities
  boundary_proba_full()  → Stage 2 probabilities mapped to N_CLASSES
  hierarchical_predict() → final predictions with routing

Usage:
    from src.ensemble import load_models, hierarchical_predict
    models = load_models()
    y_pred, info = hierarchical_predict(X_test, models)
"""

import json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier

# ── routing parameters (from sweep) ─────────────────────
CONF_THRESHOLD = 0.48   # below → route to Stage 2
BLEND_W_BND    = 0.35   # Stage 2 blend weight for uncertain samples


def load_models(model_dir: str = "models"):
    """Load all saved models and metadata."""
    # Stage 1
    xgb_s1 = xgb.XGBClassifier()
    xgb_s1.load_model(f"{model_dir}/xgb_stage1.json")

    with open(f"{model_dir}/lgb_stage1.pkl", "rb") as f:
        lgb_s1 = pickle.load(f)

    tab_s1 = TabNetClassifier()
    tab_s1.load_model(f"{model_dir}/tab_stage1.zip")

    # Stage 1 weights
    w   = json.load(open(f"{model_dir}/stage1_weights.json"))
    w_xgb, w_lgb, w_tab = w['wx'], w['wl'], w['wt']

    # Stage 2 boundary resolver
    xgb_bnd = xgb.XGBClassifier()
    xgb_bnd.load_model(f"{model_dir}/xgb_boundary.json")

    with open(f"{model_dir}/lgb_boundary.pkl", "rb") as f:
        lgb_bnd = pickle.load(f)

    # Label metadata
    classes   = json.load(open(f"{model_dir}/label_classes.json"))
    bnd_map   = json.load(open(f"{model_dir}/boundary_label_map.json"))
    l2g       = {int(k): v for k, v in bnd_map['l2g'].items()}

    return {
        'xgb_s1': xgb_s1, 'lgb_s1': lgb_s1, 'tab_s1': tab_s1,
        'w_xgb': w_xgb, 'w_lgb': w_lgb, 'w_tab': w_tab,
        'xgb_bnd': xgb_bnd, 'lgb_bnd': lgb_bnd,
        'classes': classes, 'l2g': l2g,
        'n_classes': len(classes)
    }


def stage1_proba(X, models):
    """Return blended Stage 1 probability matrix (n, n_classes)."""
    p_xgb = models['xgb_s1'].predict_proba(X)
    p_lgb = models['lgb_s1'].predict_proba(X)
    p_tab = models['tab_s1'].predict_proba(
        X.values.astype('float32')
        if hasattr(X, 'values') else X.astype('float32')
    )
    p     = (models['w_xgb'] * p_xgb
             + models['w_lgb'] * p_lgb
             + models['w_tab'] * p_tab)
    p    /= p.sum(axis=1, keepdims=True)
    return p


def boundary_proba_full(X, models):
    """
    Stage 2 probabilities mapped back to full n_classes dimensions.
    Models output local [0,1,2] → mapped to global class indices.
    """
    n_cls = models['n_classes']
    l2g   = models['l2g']

    p_xgb_local = models['xgb_bnd'].predict_proba(X)
    p_lgb_local = models['lgb_bnd'].predict_proba(X)

    p_xgb_full  = np.zeros((len(X) if hasattr(X, '__len__') else X.shape[0],
                             n_cls))
    p_lgb_full  = np.zeros_like(p_xgb_full)

    for local_i, global_i in l2g.items():
        p_xgb_full[:, global_i] = p_xgb_local[:, local_i]
        p_lgb_full[:, global_i] = p_lgb_local[:, local_i]

    # Blend boundary models 60/40
    p_bnd = 0.6 * p_xgb_full + 0.4 * p_lgb_full

    # Renormalise to boundary classes only
    bnd_cols  = list(l2g.values())
    row_sums  = p_bnd[:, bnd_cols].sum(axis=1, keepdims=True)
    row_sums  = np.where(row_sums == 0, 1, row_sums)
    p_bnd[:, bnd_cols] /= row_sums

    return p_bnd


def hierarchical_predict(X, models,
                          conf_threshold: float = CONF_THRESHOLD,
                          blend_w_bnd: float    = BLEND_W_BND):
    """
    Full hierarchical inference.

    Returns
    -------
    y_pred : np.ndarray  integer class predictions
    info   : dict        routing diagnostics
    """
    p_s1         = stage1_proba(X, models)
    confidence   = p_s1.max(axis=1)
    y_pred_s1    = np.argmax(p_s1, axis=1)

    uncertain    = confidence < conf_threshold
    y_pred_final = y_pred_s1.copy()

    if uncertain.sum() > 0:
        X_unc     = X.iloc[uncertain] if hasattr(X, 'iloc') else X[uncertain]
        p_bnd     = boundary_proba_full(X_unc, models)
        p_blend   = (1 - blend_w_bnd) * p_s1[uncertain] + blend_w_bnd * p_bnd
        y_pred_final[uncertain] = np.argmax(p_blend, axis=1)

    info = {
        'n_routed':       int(uncertain.sum()),
        'pct_routed':     float(uncertain.mean() * 100),
        'conf_threshold': conf_threshold,
        'mean_confidence_certain':  float(confidence[~uncertain].mean()),
        'mean_confidence_uncertain':float(confidence[uncertain].mean())
                                    if uncertain.sum() > 0 else None
    }
    return y_pred_final, info


if __name__ == "__main__":
    import pandas as pd
    from src.feature_engineering import engineer_features, FEATURE_COLS
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X, y, le = engineer_features(
        "data/Rheumatic and Autoimmune Disease Dataset.csv", verbose=False)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = load_models()
    y_pred, info = hierarchical_predict(X_test, models)

    print(f"Samples routed to Stage 2: {info['n_routed']} "
          f"({info['pct_routed']:.1f}%)")
    print(classification_report(y_test, y_pred,
                                 target_names=models['classes'], digits=4))
