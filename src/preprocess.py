"""
preprocess.py
=============
Base preprocessing pipeline for the rheumatic disease dataset.

Steps:
  1. Load raw CSV
  2. Encode Gender and binary immunological markers
  3. Median imputation for numeric features
  4. Mode imputation for binary markers (except HLA-B27)
  5. HLA-B27 probabilistic soft imputation via RandomForest
  6. Return clean DataFrame ready for feature engineering

Usage:
    from src.preprocess import load_and_preprocess
    data, le = load_and_preprocess("data/Rheumatic and Autoimmune Disease Dataset.csv")
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

SEED       = 42
TARGET     = 'Disease'
NUMERIC    = ['Age', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'C3', 'C4']
BINARY_RAW = ['HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']


def load_and_preprocess(csv_path: str, verbose: bool = True):
    """
    Load raw CSV and apply full preprocessing pipeline.

    Returns
    -------
    data : pd.DataFrame
        Preprocessed dataframe with all engineered columns.
    le : LabelEncoder
        Fitted label encoder for the Disease column.
    """
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Classes: {df[TARGET].value_counts().to_dict()}")

    data = df.copy()

    # ── Gender encoding ─────────────────────────────────────
    data['Gender_enc'] = (data['Gender'] == 'Male').astype(int)

    # ── Binary immunological markers: Positive=1, Negative=0 ─
    for col in BINARY_RAW:
        data[col + '_enc'] = np.where(
            data[col] == 'Positive', 1,
            np.where(data[col] == 'Negative', 0, np.nan)
        )

    # ── Numeric: median imputation ───────────────────────────
    for col in NUMERIC:
        median = data[col].median()
        data[col + '_filled'] = data[col].fillna(median)

    # ── Non-HLA binary: mode imputation ─────────────────────
    for col in ['ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']:
        mode_val = data[col + '_enc'].mode()[0]
        data[col + '_enc'] = data[col + '_enc'].fillna(mode_val)

    # ── HLA-B27 probabilistic soft imputation ───────────────
    data = _impute_hla_b27(data, verbose=verbose)

    # ── Label encode Disease ─────────────────────────────────
    le = LabelEncoder()
    data['Disease_enc'] = le.fit_transform(data[TARGET])

    if verbose:
        print(f"  Preprocessing complete. Shape: {data.shape}")

    return data, le


def _impute_hla_b27(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Probabilistic soft imputation for HLA-B27.

    For known rows  : HLA-B27_soft = hard 0/1, HLA-B27_conf = 1.0
    For missing rows: HLA-B27_soft = P(Positive), HLA-B27_conf = P(Positive)

    The soft score preserves imputation uncertainty as a model signal.
    """
    impute_features = (
        [c + '_filled' for c in NUMERIC]
        + ['Gender_enc', 'ANA_enc', 'Anti-Ro_enc',
           'Anti-La_enc', 'Anti-dsDNA_enc', 'Anti-Sm_enc']
    )

    hla_known   = data[data['HLA-B27_enc'].notna()].copy()
    hla_missing = data[data['HLA-B27_enc'].isna()].copy()

    if verbose:
        print(f"  HLA-B27 known: {len(hla_known)}  missing: {len(hla_missing)}")

    imputer = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=SEED, n_jobs=-1
    )
    imputer.fit(hla_known[impute_features], hla_known['HLA-B27_enc'])

    cv_acc = cross_val_score(
        imputer, hla_known[impute_features],
        hla_known['HLA-B27_enc'], cv=5, scoring='accuracy'
    ).mean()
    if verbose:
        print(f"  HLA-B27 imputer CV accuracy: {cv_acc:.3f}")

    hla_pred  = imputer.predict(hla_missing[impute_features])
    hla_proba = imputer.predict_proba(hla_missing[impute_features])[:, 1]

    # Fill missing HLA-B27_enc with predicted labels
    data.loc[data['HLA-B27_enc'].isna(), 'HLA-B27_enc'] = hla_pred

    # Soft score: known=hard binary, missing=probability
    data['HLA-B27_soft'] = data['HLA-B27_enc'].copy().astype(float)
    data['HLA-B27_conf'] = 1.0

    data.loc[data['HLA-B27'].isna(), 'HLA-B27_soft'] = hla_proba
    data.loc[data['HLA-B27'].isna(), 'HLA-B27_conf'] = hla_proba

    return data


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/Rheumatic and Autoimmune Disease Dataset.csv"
    data, le = load_and_preprocess(path, verbose=True)
    print(data[['HLA-B27_soft', 'HLA-B27_conf']].describe())
