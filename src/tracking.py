# src/tracking.py
import mlflow, mlflow.sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

TRACKING_URI = "sqlite:///mlflow.db"

def init(experiment: str="fraud-p1"):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)

def _log_model(model, name="model"):
    """Route to the right MLflow flavor; sklearn's skops format rejects non-sklearn types."""
    mod = type(model).__module__
    if mod.startswith("xgboost"):
        return mlflow.xgboost.log_model(model, name=name)
    if mod.startswith("lightgbm"):
        return mlflow.lightgbm.log_model(model, name=name)
    return mlflow.sklearn.log_model(model, name=name)

def log_run(name, model, X_tr, y_tr, X_te, y_te, params=None):
    with mlflow.start_run(run_name=name):
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:,1]
        mlflow.log_params(params or {})
        mlflow.log_metrics({
            "pr_auc": float(average_precision_score(y_te,p)),
            "roc_auc": float(roc_auc_score(y_te,p)),
        })
        _log_model(model)
    return p