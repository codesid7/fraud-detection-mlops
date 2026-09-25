import numpy as np

COST_FP = 400.0
FN_FLOOR = 2000.0

def fn_costs(amounts: np.ndarray) -> np.ndarray:
    return np.maximum(amounts, FN_FLOOR)

def expected_cost(y_true, proba, threshold, amounts) -> float:
    pred = (proba >= threshold).astype(int)
    fp = (pred == 1) & (y_true == 0)
    fn = (pred == 0) & (y_true == 1)
    return float(fp.sum() * COST_FP + fn_costs(amounts)[fn].sum())

def sweep(y_true, proba, amounts, n=999) :
    thresholds = np.linspace(0.001, 0.999, n)
    costs = np.array([expected_cost(y_true, proba, t, amounts) for t in thresholds])
    return thresholds, costs, float(thresholds[costs.argmin()])