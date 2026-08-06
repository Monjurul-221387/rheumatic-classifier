"""
train_stage2.py
===============
Trains the Stage 2 Boundary Resolver:
  - Subset: AS + RA + PsA training samples only
  - AS class weight boosted ×2.5 within resolver
  - XGBoost + LightGBM DART (blended 60/40)
  - Label remapping: global [0,2,4] → local [0,1,2]

Saves:
  models/xgb_boundary.json
  models/lgb_boundary.pkl
  models/boundary_label_map.json

Usage:
    python src/train_stage2.py
"""

import json, os, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb

from src.feature_engineering import engineer_features, FEATURE_COLS

SEED      = 42
DATA_PATH = "data/Rheumatic and Autoimmune Disease Dataset.csv"
AS_BOOST  = 2.5   # AS weight multiplier within boundary resolver
os.makedirs("models", exist_ok=True)


def main():
    print("=" * 60)
    print("STAGE 2 — Boundary Resolver Training")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────
    X, y, le = engineer_features(DATA_PATH)

    classes      = list(le.classes_)
    AS_IDX       = classes.index('Ankylosing Spondylitis')
    RA_IDX       = classes.index('Rheumatoid Arthritis')
    PSA_IDX      = classes.index('Psoriatic Arthritis')
    BOUNDARY     = [AS_IDX, RA_IDX, PSA_IDX]
    print(f"  Boundary classes (global): AS={AS_IDX} RA={RA_IDX} PsA={PSA_IDX}")

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    # ── Subset: AS + RA + PsA only ─────────────────────────
    bnd_mask = np.isin(y_train, BOUNDARY)
    X_bnd    = X_train[bnd_mask]
    y_global = y_train[bnd_mask]
    print(f"  Boundary subset size: {len(X_bnd)} samples")
    for idx in BOUNDARY:
        n = (y_global == idx).sum()
        print(f"    {classes[idx]}: {n}")

    # ── Label remap: global → local [0,1,2] ────────────────
    g2l = {g: l for l, g in enumerate(sorted(set(y_global)))}
    l2g = {l: g for g, l in g2l.items()}
    y_local  = np.array([g2l[yi] for yi in y_global])
    AS_LOCAL = g2l[AS_IDX]
    print(f"  Label map (local→global): {l2g}")
    json.dump({'g2l': {str(k): v for k, v in g2l.items()},
               'l2g': {str(k): v for k, v in l2g.items()},
               'AS_local': AS_LOCAL, 'AS_global': AS_IDX,
               'RA_global': RA_IDX,  'PSA_global': PSA_IDX},
              open("models/boundary_label_map.json", "w"), indent=2)

    # ── Class weights with AS boost ─────────────────────────
    bw      = compute_class_weight('balanced',
                                    classes=np.unique(y_local), y=y_local)
    wd      = dict(enumerate(bw))
    wd[AS_LOCAL] *= AS_BOOST
    sw      = np.array([wd[yi] for yi in y_local])
    print(f"\n  AS local weight: {wd[AS_LOCAL]:.4f} (boosted ×{AS_BOOST})")

    # ── XGBoost boundary resolver ───────────────────────────
    print("\n[1] Training XGBoost boundary resolver...")
    xgb_params = json.load(open("models/xgb_best_params.json"))
    xgb_bnd    = xgb.XGBClassifier(**xgb_params)
    xgb_bnd.fit(X_bnd, y_local, sample_weight=sw, verbose=False)
    xgb_bnd.save_model("models/xgb_boundary.json")
    print("    Saved: models/xgb_boundary.json")

    # ── LightGBM DART boundary resolver ────────────────────
    print("[2] Training LightGBM DART boundary resolver...")
    lgb_params = dict(
        boosting_type='dart', n_estimators=600,
        max_depth=8, num_leaves=63, learning_rate=0.05,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.80,
        min_child_samples=10, reg_alpha=0.10, reg_lambda=1.50,
        drop_rate=0.10, skip_drop=0.50,
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    lgb_bnd = lgb.LGBMClassifier(**lgb_params)
    lgb_bnd.fit(X_bnd, y_local, sample_weight=sw,
                callbacks=[lgb.log_evaluation(-1)])
    with open("models/lgb_boundary.pkl", "wb") as f:
        pickle.dump(lgb_bnd, f)
    print("    Saved: models/lgb_boundary.pkl")
    print("\nStage 2 training complete.")


if __name__ == "__main__":
    main()
