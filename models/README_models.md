# models/
# Trained model files are saved here after running train_stage1.py and train_stage2.py
#
# Files generated:
#   xgb_stage1.json          XGBoost Stage 1 model
#   lgb_stage1.pkl           LightGBM DART Stage 1 model
#   tab_stage1.zip           TabNet Stage 1 model
#   xgb_boundary.json        XGBoost boundary resolver
#   lgb_boundary.pkl         LightGBM boundary resolver
#   xgb_best_params.json     Optuna-tuned XGBoost hyperparameters
#   stage1_weights.json      Best (w_xgb, w_lgb, w_tab) ensemble weights
#   boundary_label_map.json  Local ↔ global label mapping for Stage 2
#   label_classes.json       Ordered list of disease class names
#   train_medians.json       Training set medians for inference
