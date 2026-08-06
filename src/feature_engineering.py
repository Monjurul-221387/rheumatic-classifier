"""
feature_engineering.py
=======================
Builds the 44-feature clinical fingerprint set from preprocessed data.

Feature categories:
  1. Raw filled numeric (7)
  2. Encoded binary markers (8)
  3. HLA-B27 soft + confidence (2)
  4. Ratio / interaction features (5)
  5. HLA-B27 interaction terms (4)
  6. C3/C4 complement features (2)
  7. AS ESR corridor features (4)
  8. RF / Anti-CCP RA-distance features (4)
  9. Composite AS fingerprint scores v1/v2/v3 (3)
  10. v3 boundary features (5)

Total: 44 features

Usage:
    from src.feature_engineering import engineer_features, FEATURE_COLS
    X, y, le = engineer_features("data/Rheumatic and Autoimmune Disease Dataset.csv")
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import load_and_preprocess, NUMERIC, BINARY_RAW

SEED = 42

# ── RA clinical centroids (from dataset analysis) ────────
RA_RF_MEAN  = 30.37
RA_CCP_MEAN = 30.58

# ── AS ESR corridor parameters ───────────────────────────
AS_ESR_MEAN = 32.35
AS_ESR_SD   = 6.5


def engineer_features(csv_path: str, verbose: bool = True):
    """
    Full pipeline: load → preprocess → engineer 44 features.

    Returns
    -------
    X : pd.DataFrame  shape (n, 44)
    y : np.ndarray    integer-encoded labels
    le : LabelEncoder
    """
    data, le = load_and_preprocess(csv_path, verbose=verbose)
    data     = _build_features(data)
    X        = data[FEATURE_COLS]
    y        = data['Disease_enc'].values
    if verbose:
        print(f"  Feature matrix: {X.shape}  Classes: {len(le.classes_)}")
    return X, y, le


def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered columns to data and return."""

    # ── 1. Ratio features ──────────────────────────────────
    data['RF_ESR_ratio']       = data['RF_filled']       / (data['ESR_filled']      + 1)
    data['CCP_CRP_ratio']      = data['Anti-CCP_filled'] / (data['CRP_filled']      + 1)
    data['ESR_CRP_ratio']      = data['ESR_filled']      / (data['CRP_filled']      + 1)
    data['RF_CCP_product']     = data['RF_filled']       * data['Anti-CCP_filled']
    data['RF_CCP_ratio']       = data['RF_filled']       / (data['Anti-CCP_filled'] + 1)

    # ── 2. HLA-B27 interaction terms ───────────────────────
    data['HLA_x_ESR']          = data['HLA-B27_soft'] * data['ESR_filled']
    data['HLA_x_RF']           = data['HLA-B27_soft'] * data['RF_filled']
    data['HLA_x_CRP']          = data['HLA-B27_soft'] * data['CRP_filled']
    data['HLA_x_conf']         = data['HLA-B27_soft'] * data['HLA-B27_conf']

    # ── 3. Complement features ─────────────────────────────
    data['C3_C4_ratio']        = data['C3_filled'] / (data['C4_filled'] + 1)
    data['C3_norm']            = data['C3_filled'] / data['C3_filled'].mean()

    # ── 4. AS ESR corridor features ────────────────────────
    # Binary: ESR above AS ceiling (>39 mm/hr)
    data['ESR_above_AS']       = (data['ESR_filled'] > 39).astype(int)
    data['CRP_above_AS']       = (data['CRP_filled'] > 30).astype(int)
    # Continuous corridor membership score [0,1]
    data['ESR_AS_corridor']    = np.clip(
        1 - np.abs(data['ESR_filled'] - AS_ESR_MEAN) / AS_ESR_SD, 0, 1
    )
    # Exponential tightness around AS ESR median
    data['ESR_tightness']      = np.exp(
        -np.abs(data['ESR_filled'] - 32.0) / 4.0
    )

    # ── 5. RF / Anti-CCP RA-distance features ──────────────
    data['RF_RA_zone']         = (data['RF_filled']       > 25).astype(int)
    data['CCP_RA_zone']        = (data['Anti-CCP_filled'] > 25).astype(int)
    # Distance from RA centroid in RF-CCP plane
    data['RF_dist_RA']         = np.abs(data['RF_filled']       - RA_RF_MEAN)
    data['CCP_dist_RA']        = np.abs(data['Anti-CCP_filled'] - RA_CCP_MEAN)
    # Average RA distance
    data['RF_CCP_RA_dist']     = (data['RF_dist_RA'] + data['CCP_dist_RA']) / 2

    # ── 6. Composite AS fingerprint scores ─────────────────
    # v1: basic clinical rule
    data['AS_score'] = (
        data['HLA-B27_soft'] * 3.0
        + ((data['ESR_filled'] >= 26) & (data['ESR_filled'] <= 39)).astype(float)
        + (data['RF_filled']       < 25).astype(float)
        + (data['Anti-CCP_filled'] < 25).astype(float)
    )
    # v2: adds ESR corridor and RA distance terms
    data['AS_score_v2'] = (
        data['HLA-B27_soft']    * 4.0
        + data['ESR_AS_corridor'] * 2.0
        + data['RF_dist_RA']    / 30 * 1.5
        + data['CCP_dist_RA']   / 30 * 1.5
        + (1 - data['RF_RA_zone'].astype(float))
        + (1 - data['CCP_RA_zone'].astype(float))
    )
    # v3: strongest composite (top feature in ablation)
    data['AS_score_v3'] = (
        data['HLA-B27_soft']      * 5.0
        + data['ESR_tightness']   * 2.0
        + data['RF_CCP_RA_dist']  / 30 * 2.0
        + data['ESR_AS_corridor']   * 1.5
        + (1 - data['RF_RA_zone'].astype(float))
        + (1 - data['CCP_RA_zone'].astype(float))
    )

    # ── 7. Boundary interaction features (v3) ──────────────
    data['HLA_x_RF_dist']      = data['HLA-B27_soft'] * data['RF_dist_RA']
    data['HLA_x_CCP_dist']     = data['HLA-B27_soft'] * data['CCP_dist_RA']
    data['HLA_x_corridor']     = data['HLA-B27_soft'] * data['ESR_AS_corridor']
    data['RF_HLA_sep']         = (1 - data['RF_RA_zone']) * data['HLA-B27_soft']
    data['corridor_x_RF_safe'] = data['ESR_AS_corridor'] * (1 - data['RF_RA_zone'])

    return data


# ── Canonical feature column list (44 features) ──────────
FEATURE_COLS = (
    [c + '_filled' for c in NUMERIC]          # 7 raw numeric
    + ['Gender_enc']                           # 1 demographic
    + [c + '_enc' for c in BINARY_RAW]        # 6 binary markers
    + ['HLA-B27_soft', 'HLA-B27_conf']        # 2 HLA-B27 soft
    + ['RF_ESR_ratio', 'CCP_CRP_ratio',       # 5 ratio features
       'ESR_CRP_ratio', 'RF_CCP_product', 'RF_CCP_ratio']
    + ['HLA_x_ESR', 'HLA_x_RF',              # 4 HLA interactions
       'HLA_x_CRP', 'HLA_x_conf']
    + ['C3_C4_ratio', 'C3_norm']              # 2 complement
    + ['ESR_above_AS', 'CRP_above_AS',        # 4 AS corridor
       'ESR_AS_corridor', 'ESR_tightness']
    + ['RF_RA_zone', 'CCP_RA_zone',           # 5 RA distance
       'RF_dist_RA', 'CCP_dist_RA', 'RF_CCP_RA_dist']
    + ['AS_score', 'AS_score_v2', 'AS_score_v3']  # 3 fingerprints
    + ['HLA_x_RF_dist', 'HLA_x_CCP_dist',    # 5 boundary
       'HLA_x_corridor', 'RF_HLA_sep', 'corridor_x_RF_safe']
)

assert len(FEATURE_COLS) == 44, f"Expected 44 features, got {len(FEATURE_COLS)}"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/Rheumatic and Autoimmune Disease Dataset.csv"
    X, y, le = engineer_features(path, verbose=True)
    print(f"\nFeature columns ({len(FEATURE_COLS)}):")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i:>2}. {col}")
    X.to_csv("results/features.csv", index=False)
    print("\nSaved: results/features.csv")
