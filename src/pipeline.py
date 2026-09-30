# src/pipeline.py  — run from the project root with: python -m src.pipeline
# Self-contained: rebuilds everything from the cached splits. No notebook state.
import json
from datetime import date

import joblib
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import ROOT, RANDOM_STATE, get_splits
from src.threshold import sweep

COST_FP = 400.0
FN_FLOOR = 2000.0
THEORY_MAX_T = COST_FP / (COST_FP + FN_FLOOR)   # ≈ 0.167

# Paste your champion's tuned parameters here (Day 30–32). Do NOT re-tune.
# Anything you paste overrides the defaults set in build_model().
XGB_PARAMS = {
    # from Optuna study.best_params (Day 30)
    "max_depth": 5,
    "learning_rate": 0.06878090324608771,
    "subsample": 0.8566762257054904,
    "colsample_bytree": 0.9077283436024173,
    "min_child_weight": 4,
    "gamma": 1.798073818555928,
    "reg_lambda": 0.0019965355610900024,
    "reg_alpha": 0.09958053095757974,
    "scale_pos_weight": 2.6736952436743975,
    # fixed during tuning / refit
    "n_estimators": 400,   # confirm with champion's num_boosted_rounds()
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": 42,
}


def build_model(spw: float) -> CalibratedClassifierCV:
    params = {"scale_pos_weight": spw, "random_state": RANDOM_STATE,
              "n_jobs": -1, **XGB_PARAMS}
    base = Pipeline([
        ("scale", StandardScaler()),
        ("xgb", XGBClassifier(**params)),
    ])
    # The whole Pipeline sits inside the calibrator, so each of the 5 folds
    # refits its own scaler + XGBoost. That is the "no leakage" part.
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def main() -> None:
    X_tr, X_te, y_tr, y_te = get_splits()
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

    final_pipeline = build_model(spw).fit(X_tr, y_tr)

    p = final_pipeline.predict_proba(X_te)[:, 1]
    pr_auc = average_precision_score(y_te, p)
    _, costs, best_t = sweep(y_te.values, p, X_te["Amount"].values)

    if best_t > THEORY_MAX_T:
        print(f"WARNING: best threshold {best_t:.3f} is above the cost-matrix "
              f"bound {THEORY_MAX_T:.3f}; probabilities may not be honest.")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final_pipeline, models_dir / "champion.joblib")

    meta = {
        "version": "1.0.0",
        "trained_on": str(date.today()),
        "calibration": "sigmoid, cv=5",
        "threshold": float(best_t),
        "threshold_basis": "expected-cost minimisation on calibrated probabilities",
        "min_test_cost": float(costs.min()),
        "test_pr_auc": float(pr_auc),
        "cost_fp": COST_FP,
        "fn_floor": FN_FLOOR,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"PR-AUC    : {pr_auc:.6f}")
    print(f"threshold : {best_t:.4f}")
    print(f"min cost  : ₹{costs.min():,.0f}")
    print(f"saved     : {models_dir / 'champion.joblib'}")


if __name__ == "__main__":
    main()