"""
predict.py
==========
Run inference on new patient data using the trained hierarchical model.

Usage:
    python src/predict.py --input new_patients.csv --output predictions.csv

Input CSV must have the same columns as the original dataset:
    Age, Gender, ESR, CRP, RF, Anti-CCP, HLA-B27, ANA,
    Anti-Ro, Anti-La, Anti-dsDNA, Anti-Sm, C3, C4
    (Disease column not required for inference)

Output CSV columns:
    predicted_disease, confidence, routed_to_stage2,
    prob_AS, prob_Normal, prob_PsA, prob_ReA, prob_RA, prob_Sjo, prob_SLE
"""

import argparse, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from src.preprocess import load_and_preprocess, NUMERIC, BINARY_RAW
from src.feature_engineering import _build_features, FEATURE_COLS
from src.ensemble import load_models, hierarchical_predict, stage1_proba


CONF_THRESHOLD = 0.48


def predict_new(input_csv: str, output_csv: str, verbose: bool = True):
    """
    Predict disease class for new patient records.

    Parameters
    ----------
    input_csv  : path to CSV with patient features (no Disease column needed)
    output_csv : path to save predictions
    """
    print(f"\nLoading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Rows: {df.shape[0]}")

    # ── Preprocessing (same as training pipeline) ──────────
    data = df.copy()

    # Add dummy Disease column if not present (needed by preprocess internals)
    if 'Disease' not in data.columns:
        data['Disease'] = 'Unknown'

    # Gender
    data['Gender_enc'] = (data['Gender'] == 'Male').astype(int)

    # Binary markers
    for col in BINARY_RAW:
        data[col + '_enc'] = np.where(
            data[col] == 'Positive', 1,
            np.where(data[col] == 'Negative', 0, np.nan)
        )

    # Numeric median fill (use training medians if available)
    try:
        train_medians = json.load(open("models/train_medians.json"))
        for col in NUMERIC:
            med = train_medians.get(col, data[col].median())
            data[col + '_filled'] = data[col].fillna(med)
    except FileNotFoundError:
        for col in NUMERIC:
            data[col + '_filled'] = data[col].fillna(data[col].median())
        if verbose:
            print("  Warning: train_medians.json not found. "
                  "Using test-set medians — prefer saving training medians.")

    # Binary mode fill
    for col in ['ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']:
        mode_val = data[col + '_enc'].mode()
        mode_val = mode_val[0] if len(mode_val) > 0 else 0
        data[col + '_enc'] = data[col + '_enc'].fillna(mode_val)

    # HLA-B27 soft imputation
    from sklearn.ensemble import RandomForestClassifier
    impute_features = (
        [c + '_filled' for c in NUMERIC]
        + ['Gender_enc', 'ANA_enc', 'Anti-Ro_enc',
           'Anti-La_enc', 'Anti-dsDNA_enc', 'Anti-Sm_enc']
    )
    data['HLA-B27_enc'] = np.where(
        data['HLA-B27'] == 'Positive', 1,
        np.where(data['HLA-B27'] == 'Negative', 0, np.nan)
    )
    known   = data[data['HLA-B27_enc'].notna()].copy()
    missing = data[data['HLA-B27_enc'].isna()].copy()

    data['HLA-B27_soft'] = data['HLA-B27_enc'].copy().astype(float)
    data['HLA-B27_conf'] = 1.0

    if len(missing) > 0 and len(known) > 0:
        imp = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        imp.fit(known[impute_features], known['HLA-B27_enc'])
        proba = imp.predict_proba(missing[impute_features])[:, 1]
        data.loc[data['HLA-B27_enc'].isna(), 'HLA-B27_soft'] = proba
        data.loc[data['HLA-B27_enc'].isna(), 'HLA-B27_conf'] = proba
        data.loc[data['HLA-B27_enc'].isna(), 'HLA-B27_enc']  = \
            (proba >= 0.5).astype(float)

    # Feature engineering
    data = _build_features(data)
    X    = data[FEATURE_COLS]

    # ── Inference ──────────────────────────────────────────
    print("Loading models...")
    models = load_models()
    classes = models['classes']

    y_pred, routing_info = hierarchical_predict(X, models)
    p_s1 = stage1_proba(X, models)

    print(f"  Samples routed to Stage 2: "
          f"{routing_info['n_routed']} ({routing_info['pct_routed']:.1f}%)")

    # ── Build output ───────────────────────────────────────
    short = ['AS','Normal','PsA','ReA','RA','Sjo','SLE']
    out = pd.DataFrame({
        'predicted_disease':   [classes[i] for i in y_pred],
        'confidence':          p_s1.max(axis=1).round(4),
        'routed_to_stage2':    p_s1.max(axis=1) < CONF_THRESHOLD,
    })
    for i, s in enumerate(short):
        out[f'prob_{s}'] = p_s1[:, i].round(4)

    out.to_csv(output_csv, index=False)
    print(f"\nPredictions saved: {output_csv}")

    # ── Summary ────────────────────────────────────────────
    print("\nPrediction distribution:")
    print(out['predicted_disease'].value_counts().to_string())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical rheumatic disease classifier — inference"
    )
    parser.add_argument('--input',  required=True,
                        help="Input CSV with patient features")
    parser.add_argument('--output', default="results/new_predictions.csv",
                        help="Output CSV path")
    parser.add_argument('--quiet',  action='store_true',
                        help="Suppress verbose output")
    args = parser.parse_args()

    predict_new(args.input, args.output, verbose=not args.quiet)
