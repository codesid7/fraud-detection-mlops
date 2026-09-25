# fraud-detection-mlops — Project Journal, Day 22 to Day 36

Everything built, decided, and learned in Phase 1 so far, reconstructed from our working sessions. Where you got stuck, the clean answer is written out in the **Revisit** boxes and collected again in Section 16, so you can drill them before interviews.

`<fill>` marks a number that never came up in our chats. Check it in your notebook or MLflow before quoting it.

---

## 1. The project in one paragraph

A cost-sensitive credit-card fraud model, built as a reproducible repo rather than a notebook dump. The data is the 2013 European cardholder dataset (Kaggle `mlg-ulb/creditcardfraud`): 284,807 transactions, 492 frauds, a 0.17% base rate. Features are `Time`, `Amount`, and 28 anonymised PCA components `V1`–`V28`. The goal is not the highest score but the cheapest decision: pick a model on PR-AUC, then pick a threshold on rupee cost. Phase 1 runs Days 22–56 and ends with a FastAPI service in Docker, deployed with a Streamlit front end and a drift monitor. Phase 0 (Weeks 1–3, in Colab) came before it.

**Stack:** Python 3.12, pandas, scikit-learn, XGBoost, LightGBM, imbalanced-learn, Optuna, MLflow 3, matplotlib. Later: FastAPI, Docker, Streamlit, Render.

**Repo layout so far**

```
fraud-detection-mlops/
├── src/            data.py, tracking.py, threshold.py
├── notebooks/      01_eda, 02_baseline, 03_resampling, 04_tuning, 05_threshold_and_calibration
├── reports/        decisions.md, model_comparison.md, day30_findings.md,
│                   weeknotes.md, interview_questions.md, figures/
├── data/processed/ Parquet split cache (gitignored)
├── mlflow.db, mlruns/, optuna.db   (all gitignored)
└── sql/            weekly SQL practice (to move to ds-portfolio-2026)
```

---

## 2. Environment and setup (Day 22, Block 1)

"Local environment" wasn't an app. It meant the state of your laptop: a named conda env called `fraud` plus a project folder that is a git repo. You created it in Miniforge Prompt, ran `conda init` so PowerShell and cmd could see conda, then switched to VS Code with the `fraud` interpreter selected.

**Decisions recorded (from `reports/decisions.md`):**

- **D1: Miniforge + conda env `fraud`, Python 3.12.** Reproducible, can be destroyed and rebuilt, and conda-forge handles compiled libraries (XGBoost, LightGBM) better on Windows. Rejected `venv` and system Python.
- **D2: Downcast float64 to float32 on load.** Halves memory (~68 MB → ~34 MB), which matters on an 8 GB machine. Caveat: cast back to float64 if you ever sum a full 284k-row column.
- **D3: Data via `kagglehub`, credentials in `C:\Users\<you>\.kaggle\kaggle.json`.** Anyone cloning the repo gets the same data. The README must say they need their own Kaggle token.
- **D4: Kaggle tokens are single-active.** Creating a new token invalidates old ones. Your 403 was a stale token, not a permissions problem.

**Windows gotchas you hit:** bash commands (`mkdir -p`, `touch`, heredocs) don't work in cmd. Use `mkdir data notebooks ...` and `type nul > src\__init__.py`.

**The 8 GB rules** that kept coming up: never leave the MLflow UI running (launch → look → `Ctrl+C`), don't train while it's open (SQLite write locks), `n_jobs=2` if Random Forest thrashes, and nothing else running during Docker builds.

**The working-directory problem.** This caused more lost time than anything else, across Days 22, 23, 30, and 36. The plan assumed notebooks run from `notebooks/`, so paths used `../`. VS Code's default (`jupyter.notebookFileRoot = ${workspaceFolder}`) runs them from the project root instead, so `../` pointed *outside* the repo. Stray `optuna.db` and `reports/figures/` folders ended up in `C:\Users\Legion\`. The permanent fix is a `ROOT` variable in Cell 1 of each notebook, with every path built from it:

```python
from pathlib import Path
def project_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "src").is_dir(): return p
    raise RuntimeError("no src/ found")
ROOT = project_root()
# then: ROOT / "reports" / "figures" / "name.png",  "sqlite:///" + (ROOT / "optuna.db").as_posix()
```

> **Revisit:** If a file "didn't save," check `os.getcwd()` first. It almost always saved, just one folder too high.

---

## 3. EDA (Day 22, Block 2 — `01_eda.ipynb`)

The plan listed the figures but never wrote the code to create them (D7), so you wrote it.

| Figure | What it shows | Why it's built that way |
|---|---|---|
| `00_fraud_by_elapsed_hour.png` | Fraud rate over the raw 0–47 hours | Checks whether both days share a rhythm before folding them |
| `00b_fraud_rate_by_position.png` | Fraud rate per 10,000 rows in file order | A stationarity check: is a random split fair? |
| `01_class_balance.png` | 284,315 vs 492 | Log scale, or fraud is an invisible sliver (D8) |
| `02_amount_by_class.png` | Amount boxplot by class | Log scale because Amount is right-skewed; `showfliers=False` |
| `03_fraud_rate_by_hour.png` | Fraud rate by folded hour | Dashed overall-rate line so "above baseline" is visible |

**D5: `hour = (Time // 3600) % 24`.** `Time` is seconds since the first transaction, so hour 0 is *some* consistent hour, not midnight. You can say "fraud varies by time of day," not "fraud peaks at 2 a.m."

**Reading the position plot:** at 10,000 rows a bin holds ~17 frauds, so random wobble is about √17 ≈ 4. Only 3× or 0.1× swings mean anything. Re-bin at 20,000, since a pattern that disappears on re-binning was never there. This plot is also your baseline for the Day 51 drift monitor.

`df.describe()` summarises each column separately but ignores class. Use `df.groupby("Class")["Amount"].describe()` to compare fraud with legit.

> **Open:** D10 (do both days share a rhythm?) and D11 (is the fraud rate stationary, so is a random split fair?) were never marked resolved. Write the result into `decisions.md`.

---

## 4. The cost matrix (Day 22 — D12, D13)

| | Predicted legit | Predicted fraud |
|---|---|---|
| **Actually legit** | correct, ₹0 | **FP:** ₹400 flat (support call, reissue, churn risk) |
| **Actually fraud** | **FN:** `max(Amount, ₹2,000)` | correct, ₹0 |

The ₹2,000 floor exists because chargebacks, investigation, and reissue happen regardless of size. Without it the model learns that small frauds are free to miss, which blinds it to card-testing attacks. The loss falls on the issuer or merchant, not the cardholder. Both numbers are stated assumptions so a reviewer can swap in their own.

**D13, the consequence:** flag when `(1 − p) × 400 < p × FN_cost`, which rearranges to `p > 400 / (400 + FN_cost)`. A ₹2,000 fraud should be flagged above 0.17, a ₹40,000 fraud above 0.01. So the default 0.5 is indefensible under your own costs, and the ideal threshold depends on amount. A single global threshold is a compromise; an amount-aware threshold goes in "what I'd do next."

---

## 5. The split and `src/data.py` (Day 23; rewritten in plan v7)

`get_splits()` returns a stratified 80/20 split with `random_state=42`. Stratifying guarantees both halves keep the 0.17% rate; without it a random split can put a lopsided share of frauds in test. That gives `X_tr` 227,845 rows (~394 frauds) and `X_te` ~57,000 rows (~98 frauds). `spw = (y_tr == 0).sum() / (y_tr == 1).sum() ≈ 577.3`.

In v7 the function caches the split to Parquet in `data/processed/`. The first call takes tens of seconds; every later call is under a second. This is what makes the **preambles** practical. Each block names which one it needs: `0` nothing, `T` terminal at root, `R` raw frame, `A` splits, `B` splits + MLflow + `spw`, `C` splits + MLflow + champion, `E` serving artifacts, `S` API running.

**Why it matters:** a new notebook is a cold kernel. On Day 30 nothing existed because the plan silently assumed variables from a week earlier. The Day 29 and earlier runs never need re-running, because their results already live in MLflow.

---

## 6. The baseline and the metric lesson (Day 23, Block 1)

Logistic regression (`StandardScaler` → `LogisticRegression(class_weight="balanced")`) scored **PR-AUC 0.73, ROC-AUC 0.97** on identical predictions. That gap is the lesson of the whole project.

**Precision:** of everything I flagged, how much was really fraud? It's the false-alarm burden. **Recall:** of all the real fraud, how much did I catch? Lower the threshold and recall rises while precision falls.

**Why ROC-AUC flatters here:** its false-positive rate divides by ~227,000 negatives, so a flood of false alarms barely moves it. Precision divides only by what you flagged, so the same flood craters it.

**How PR-AUC is computed** (`average_precision_score`): walk down the thresholds, and each time recall *increases*, add (recall increase) × (precision at that point). Each term is a rectangle under the PR curve; steps where recall doesn't move add nothing. PR-AUC answers "which model ranks fraud best across all thresholds." The Day 36 sweep answers a different question: "which one threshold should I use."

**Accuracy** is useless: an always-legit model scores 99.83%.

> **Revisit (Sep 23 review):** You said precision matters more. For fraud, most teams lean toward recall, because a missed fraud costs the amount and a false alarm costs ₹400. But the real answer is that neither intuition decides; the cost matrix does, via the threshold.

---

## 7. MLflow 3 — how it actually works (Day 23 onward)

- **Backend:** `sqlite:///mlflow.db` (three slashes = relative path). The file store raises in MLflow 3. There's no login; SQLite is just a file.
- **`src/tracking.py`** holds `init()` and `log_run()` so every notebook logs identically. `log_run` refits the model on purpose, so the logged metrics come from a fit MLflow witnessed.
- **UI:** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`, then `Ctrl+C`. If the sidebar shows Traces and Prompts, you're on the **GenAI** tab; switch to **Model training**.
- **Models are first-class in MLflow 3.** An empty Artifacts tab is normal; models live under the Models tab. Check from Python with `mlflow.search_logged_models(...)`. Load with `models:/<model_id>` or `models:/fraud-champion@champion`, not the old `runs:/.../model`. Deleting a run leaves its logged models behind as orphans; delete those with `MlflowClient().delete_logged_model(id)`.
- **The skops problem (Day 23, then again on Days 29 and 31).** `mlflow.sklearn.log_model` uses the skops format, which refuses non-sklearn types (MLflow 3.16 made this the default). The fix is to use the right flavor: `mlflow.xgboost.log_model` for bare XGBoost, `mlflow.lightgbm` for LightGBM. `tracking.py` now dispatches on `type(model).__module__`. For imblearn pipelines (SMOTE + XGBoost), the sklearn flavor is correct, so you pass `skops_trusted_types=[...]` with names copied character-for-character from the error.
- **Registration (Day 30/31):** `register_model` → `set_registered_model_alias("fraud-champion", "champion", version=...)`. Load with `mlflow.xgboost.load_model("models:/fraud-champion@champion")`.
- `log_figure(fig, "name.png")` attaches a plot to a run without saving it first.

> **Revisit:** "What does MLflow store that a spreadsheet doesn't?" The fitted model itself, its exact parameters, and its environment (`conda.yaml`, `requirements.txt`), all linked to the metrics it produced, so any result can be reloaded and reproduced.

---

## 8. Week 4 weekdays (Days 24–28)

- **Day 24:** PR and ROC curves for all runs on one plot each (`fig, ax` created once, outside the loop), saved as `04_pr_curves.png` and `05_roc_curves.png`, logged to the best run.
- **Day 25:** Random Forest (`sklearn.ensemble`, `class_weight="balanced"`), 4th run. sklearn groups modules by purpose: `linear_model`, `ensemble`, `metrics`, `model_selection`, `preprocessing`, `pipeline`.
- **Day 26:** two LeetCode SQL 50 window-function problems.
- **Day 27:** LightGBM run, 5th run.
- **Day 28:** commit, push, first weeknotes.

---

## 9. Resampling (Day 29 — `03_resampling.ipynb`)

**The trap:** on the single test split, SMOTE looked best. It was noise, so you re-ran the comparison with 5-fold CV on `X_tr` only.

| Strategy | PR-AUC (5-fold) | Folds |
|---|---|---|
| none | 0.8516 ± 0.0342 | 0.8785, 0.8452, 0.8812, 0.8651, 0.7882 |
| pos_weight | 0.8476 ± 0.0289 | 0.8803, 0.8517, 0.8691, 0.8396, 0.7971 |
| SMOTE | 0.8467 ± 0.0392 | 0.8986, 0.8537, 0.8710, 0.8258, 0.7842 |
| undersample (0.1) | 0.7885 ± 0.0472 | 0.7293, 0.8305, 0.8329, 0.8171, 0.7328 |

**Reading it:** none, pos_weight, and SMOTE are tied. Undersampling is really worse, losing on all 5 folds. The ± is mostly fold difficulty (every method dips on fold 5), so the right test is paired, fold by fold: if the winner flips between folds, it's noise.

**Why the corrections did nothing:** PR-AUC is rank-based, and reweighting shifts probabilities without reordering them much. Their real payoff is threshold and calibration work. **Why undersampling hurts:** it throws away ~98% of legitimate rows (~227k → ~4k), so the model barely learns what normal looks like. **Method rule:** resampling must sit *inside* an imblearn `Pipeline` so it's refit on training folds only; resampling before CV leaks.

**Decision:** carry `pos_weight` forward. Same performance, no extra step in the serving path, and Optuna can tune it.

> **Revisit:** The plan predicted SMOTE would hurt, and it didn't. Say what happened, not what was predicted. Why interpolating in PCA space didn't hurt here is still an open question.

---

## 10. Optuna tuning (Day 30 — `04_tuning.ipynb`)

- 150 trials on XGBoost, 3-fold stratified CV, PR-AUC, `n_estimators=200` during search. TPE sampler, `MedianPruner`, stored in `optuna.db` with `load_if_exists=True`. LightGBM got a 40-trial search on Day 31.
- **Pruning ≠ parameter ranges.** The pruner stops a trial early if its score after fold 1 or 2 is below the median of earlier trials at that point. It's purely a time saver. Ranges are the search bounds, and nothing outside them is ever sampled.
- **The study is a handle; the data is in SQLite.** A kernel restart loses nothing: `optuna.load_study(...)`. Optuna has no rename, so you renamed with `copy_study` + `delete_study` (`xgb_praauc` → `xgb_praauc_correct`).
- **No plotly?** Use `optuna.visualization.matplotlib`.
- **Lessons that cost time:** keep `n_estimators` out of the search space (it trades off against learning rate, and you want 200 for search, 400 for refit); a resumed study silently mixes old and new search spaces; `scale_pos_weight` comes back inside `best_params`, so don't pass `spw` again at refit; LightGBM's `subsample` does nothing unless `subsample_freq=1`.
- **Tuned `scale_pos_weight`:** `<fill: study.best_params["scale_pos_weight"]>` against a class ratio of 577. Earlier chats quoted both ~28 and ~4, and the ~28 came from my synthetic test run, not yours.

---

## 11. Refit, champion, and the LightGBM collapse (Days 30–31)

Best params refit at 400 trees → `champion_xgb_400`, registered as `fraud-champion@champion`.

**Test-set PR-AUC (single split, ~98 frauds):**

| Run | PR-AUC | ROC-AUC |
|---|---|---|
| xgb_scale_pos_weight (hand-set, Day 23) | 0.8751 | 0.986 |
| xgb_smote | 0.8744 | — |
| champion_xgb_400 (tuned) | 0.8719 | 0.9782 |
| LightGBM plain | 0.845 | 0.940 |
| LightGBM + `scale_pos_weight=577` | **0.091** | 0.907 |

The top three sit within 0.0032, so it's a tie. You kept the CV-selected champion, because switching to the test-set leader would be selecting on the test set. 150 trials didn't beat the hand-set Day 23 model, and that's a finding: PCA features on a well-separated problem leave little for tuning to find.

**The LightGBM collapse:** probability percentiles were 0, 0, 0, 0.0001, 1, with 1,010 transactions tied at the same value. PR-AUC can't order a tied block, so it scored the tie's own precision (~0.07). On raw scores the model ranked at ~0.57, so it was broken but not as badly as 0.09 suggested. The cause is leaf value ≈ −Σgradient / (Σhessian + λ). Weighting frauds 577× inflates the top; XGBoost's default `reg_lambda=1` bounds the bottom, while LightGBM's default is 0. Huge raw scores → saturated probabilities → ties.

**Lessons:** look at the prediction distribution, not just the metric; ties can sink a ranking metric; ROC-AUC hid the failure completely.

> **Revisit:** You first read the 0.09 bar as XGBoost from an unlabelled chart where colours repeat. Always read `mlflow.search_runs(..., order_by=["metrics.pr_auc DESC"])`, never the bars.
>
> **Open:** does `reg_lambda=1` restore LightGBM to ~0.845? If yes, the whole failure is one default.

---

## 12. Is the win real? (Days 31–32)

| Measurement | PR-AUC |
|---|---|
| XGBoost champion, 3-fold @400 | 0.8523 ± 0.0096 |
| LightGBM (tuned), 3-fold @400 | 0.8487 ± 0.0087 |
| **XGBoost champion, 5-fold @400 (headline)** | **0.8509 ± 0.0186** |
| Margin, XGBoost − LightGBM | 0.0073 |
| Noise band (≈ 2 × std) | ≈ 0.037 |

**Verdict:** a statistical tie. The gap is about a fifth of the noise. XGBoost stays champion on a secondary criterion: `<fill: SHAP TreeExplainer work is scoped around it / scale_pos_weight is easier to explain / latency or size>`. The 3-fold and 5-fold numbers are two measurements of one model; the 5-fold std is wider because each fold holds fewer frauds.

> **Revisit (Day 32):** You read a smaller std as a better model, and called PR-AUC "accuracy." The ± tells you **how precisely you know the number**, not how good the model is. Mean = average across folds; std = typical distance from it; noise = fold-to-fold wobble of one model. Never say accuracy on 0.17% fraud.

---

## 13. Days 33–35

- **Day 33:** SQL week 5 in `sql/week5.sql`. `<fill: which problems>`, as there was no record in our chats.
- **Day 34:** `reports/interview_questions.md`, five hostile questions answered in writing. Suggested set: where the costs came from and how the threshold moves if FN cost halves; is the 0.0073 win real; what SHAP means on PCA features; does 2013 European data transfer to 2026; why sigmoid over isotonic at ~90 positives.
- **Day 35:** pushed the week and wrote `reports/weeknotes.md`. Evening setup S5: WSL2 + Docker Desktop installed, WSL capped at 4 GB, `hello-world` runs, auto-start off, `wsl --shutdown` after use.

> **Revisit (Day 34):** Asked where the threshold moves if the false-negative cost halves, you re-explained the cost matrix and got the direction backwards. Clean answer: **the optimal threshold rises; you flag fewer transactions, so recall falls and precision rises.** When a question feels uncomfortable, commit to a direction first.

---

## 14. Day 36 — Threshold and calibration (`05_threshold_and_calibration.ipynb`)

### Block 1: cost-based threshold

`src/threshold.py` defines `expected_cost` (FP × ₹400 + missed frauds at `max(Amount, 2000)`) and `sweep`, which tries 999 thresholds from 0.001 to 0.999 and returns the cheapest. Moving right on the chart, you flag less: far left pays for thousands of false alarms, far right pays for missed frauds, and the minimum sits in between, well below 0.5.

| | Value |
|---|---|
| Best threshold | `<fill>` |
| Cost at best | ₹`<fill>` |
| Cost at 0.5 | ₹`<fill>` |
| Savings | ₹`<fill>` |

Figure: `06_cost_vs_threshold.png`, saved via `ROOT` after the `../` path wrote it outside the repo. Save *before* `plt.show()`, or you get a blank PNG.

**Teach-back:** the optimum sits below 0.5 because a missed fraud costs at least 5× a false alarm, so it's worth flagging at lower confidence.

### Block 2: calibration

**What calibration is:** a second, small model that maps raw scores to honest probabilities, fit on data the champion hasn't seen. It's needed because `scale_pos_weight` distorts the outputs, and D13's formula only works if `p = 0.03` really means 3 in 100. **Sigmoid (Platt)** fits one S-curve with two parameters: rigid but stable on little data. **Isotonic** fits a free-form rising staircase: flexible but data-hungry. Both preserve ranking, so PR-AUC barely moves.

**Brier score** = mean of (p − y)². Lower is better. A base-rate predictor scores ~0.0017. Scores here are tiny because ~99.8% of rows are legit and predicted near zero.

**Data roles:** `X_te` is only ever used for final scoring. The frozen approach splits *train* into `X_fit` (75%, trains the champion) and `X_cal` (25%, trains the calibrator). `cv=5` splits *train* into 5 folds; each round trains a fresh champion on 4 and a calibrator on the 5th, and predictions average the 5 pairs. `FrozenEstimator` wraps a fitted model so `fit()` doesn't retrain it (it replaces the deprecated `cv="prefit"`).

| Method | Calibration data | Brier on `X_te` |
|---|---|---|
| Raw | — | `<fill>` |
| Isotonic, frozen | `X_cal`, ~90 frauds | 0.000412 |
| Sigmoid, frozen | `X_cal`, ~90 frauds | 0.000378 |
| Isotonic, `cv=5` | all of `X_tr` | `<fill>` (≈ raw) |
| Sigmoid, `cv=5` | all of `X_tr` | `<fill>` (≈ raw) |

**Findings:** with one split, isotonic overfit ~90 frauds and did worse than raw. With `cv=5`, both matched raw. Each calibrator actually saw fewer frauds (~76), but every training fraud was used once and averaging 5 calibrators removed the variance. So the raw model was already reasonably calibrated for Brier.

**Reliability curve** (`07_reliability_curve.png`, 10 quantile bins of ~5,700 test rows, log axes): x = mean predicted probability in a bin, y = fraction that *actually* were fraud, so no threshold is involved. About 90 of ~95 test frauds sit in the top bin, and there every method, raw included, is on the diagonal. Lower bins hold 0–2 frauds (y ≈ 1.75 × 10⁻⁴ = 1 fraud, lines to the bottom edge = 0 frauds), so individual points are noise. The sigmoid `cv=5` curve looks close to the diagonal partly because it squashes all low scores to ~4 × 10⁻⁴. After widening the axis to 10⁻⁸, the raw model's low bins appear: it predicts ~10⁻⁶ where the real rate is ~3 × 10⁻⁴, so it **underestimates fraud risk** at the low end, consistently across bins. Calibration moves those bins onto the diagonal, but Brier can't see it (a missed fraud costs ≈ 1 either way), and the threshold decision is unchanged.

**Decision:** `<fill: e.g. keep raw probabilities, threshold chosen empirically on expected cost>`.

> **Revisit (today):**
> - `X_cal` is the "test" output of a `train_test_split` call on `X_tr`, not the test set. The function names its outputs generically; the role is what matters.
> - In `cv=5`, the folds come from **train**, and Brier is still scored on `X_te`, not on train.
> - The calibrator never learns from the test set, so "isotonic couldn't learn from the test set" is wrong. It had too few positives in its calibration set.
> - Points above the diagonal mean the model **under**estimates risk. That is what "underconfident" meant, not the opposite.
> - The x-axis isn't computed per CV fold; averaging already happened inside `predict_proba`.
> - A loop that prints Brier and reuses `cal` throws the predictions away. Store them in a dict (`probs[name] = ...`) if you need to plot them.
> - 0.000412 → 0.00378 would be a tenfold increase; you meant 0.000378.

---

## 15. Numbers sheet

| What | Value |
|---|---|
| Rows / frauds / base rate | 284,807 / 492 / 0.17% |
| Train / test frauds | ~394 / ~98 |
| Class ratio (`spw`) | 577.3 |
| Baseline logreg (test) | PR-AUC 0.73, ROC-AUC 0.97 |
| Resampling, 5-fold (none / pos_weight / SMOTE / under) | 0.8516 / 0.8476 / 0.8467 / 0.7885 |
| Champion, 5-fold CV | 0.8509 ± 0.0186 |
| Champion vs LightGBM margin / noise band | 0.0073 / ≈ 0.037 |
| Champion, test | PR-AUC 0.8719, ROC-AUC 0.9782 |
| LightGBM + spw=577 collapse | 0.091 PR-AUC (0.907 ROC-AUC) |
| Threshold for ₹2,000 / ₹40,000 fraud (theory) | 0.17 / 0.01 |
| Frozen isotonic / sigmoid Brier | 0.000412 / 0.000378 |

---

## 16. Where you got stuck — the full Revisit list

These are the places where your first answer wobbled. Each has the version to say out loud.

1. **What "local environment" means** (Day 22). The state of the laptop: a conda env + a git repo on disk, not a specific app.
2. **Files saving in the wrong place** (Days 22, 23, 30, 36). The notebook runs from the project root, so `../` escapes the repo. Build paths from `ROOT`.
3. **Empty Artifacts tab / missing model** (Day 23). MLflow 3 stores models as first-class entities under the Models tab. The GenAI tab hides the ML view.
4. **Precision vs recall priority** (Sep 23). The cost matrix decides; in fraud, recall usually matters more because a miss costs far more than a false alarm.
5. **Reading bars instead of the table** (Day 31). Colours repeat on unlabelled MLflow charts. Sort `search_runs` by the metric.
6. **SMOTE "should" have hurt** (Day 29). Report what happened: SMOTE tied with the best. Its mechanism is an open question.
7. **Pruning vs search ranges** (Day 30). Pruning stops losing trials early; ranges bound what gets sampled. Unrelated.
8. **Smaller std = better model** (Day 32). No. The ± is how precisely you know the score. And never "accuracy."
9. **Threshold direction** (Day 34). FN cost halves → threshold rises → fewer flags → recall falls, precision rises.
10. **Undersampling mechanism** (flagged in the Sep 23 review). It discards ~98% of legit rows, so the model loses its picture of normal; it lost on 5 of 5 folds.
11. **Which set the calibrator learns from** (Day 36). `X_cal` / CV folds, both carved from train. Test only scores.
12. **Reading the reliability curve** (Day 36). y = actual fraud rate, not "classified as fraud." Above the diagonal = model underestimates risk.

---

## 17. Teach-back bank (one clean sentence each)

- **Why is accuracy misleading?** An always-legit model scores 99.83% and catches nothing.
- **Why PR-AUC, not ROC-AUC or F1?** ROC-AUC's denominator of ~227k negatives hides false alarms (the collapsed LightGBM still scored 0.907); F1 needs a threshold you haven't chosen yet.
- **Why stratify?** So train and test both keep the 0.17% rate.
- **What does MLflow store that a spreadsheet doesn't?** The model, its parameters, and its environment, linked to the metrics.
- **Why did resampling barely move PR-AUC?** PR-AUC is about ranking, and reweighting changes probabilities more than order.
- **Why resample inside the pipeline?** So synthetic or dropped rows never touch the validation fold.
- **Why keep `n_estimators` out of Optuna?** It trades off against learning rate, and search and refit need different values.
- **Why tune `scale_pos_weight` rather than fix it at 577?** Full weighting over-predicts fraud; PR-AUC prefers a far smaller weight.
- **Why did LightGBM collapse and XGBoost didn't?** LightGBM's `reg_lambda` defaults to 0, so 577× weights produce huge leaf values, saturated probabilities, and ties.
- **Is the champion's win real?** No: 0.0073 against a ~0.037 noise band. It's a tie, chosen on `<fill>`.
- **Why must calibration data come from train?** A calibrator fit on test would be graded on data it already saw.
- **Why sigmoid over isotonic at ~90 frauds?** Two parameters are stable on little data; isotonic's free-form staircase overfits.
- **Why does the optimal threshold land below 0.5?** A miss costs at least 5× a false alarm, so flagging at lower confidence is worth it.

---

## 18. Open items

- [ ] Fill every `<fill>` above: threshold results, raw and `cv=5` Brier, tuned `scale_pos_weight`, tie-break criterion, calibration decision, Day 33 SQL.
- [ ] Record D10 and D11 results in `decisions.md` (daily rhythm; stationarity → random vs temporal split).
- [ ] Test `reg_lambda=1` on LightGBM + spw and close the collapse explanation.
- [ ] Regenerate `reports/figures/05_roc_curves.png`.
- [ ] Rename `champion_refit` (NaN metrics, it only logs an artifact) and fix the `LightBGM` typo in run names.
- [ ] Rename `notebook/` → `notebooks/` to match the plan.
- [ ] Move `sql/` to `ds-portfolio-2026` before Day 52.
- [ ] Finish all 5 hostile questions in `interview_questions.md`.
- [ ] Optional: bootstrap CI on the Brier differences.
- [ ] **D6, by Day 51:** get the model into Render. Git LFS for the joblib, or train on build.
- [ ] Day 37 Block 1: re-run the threshold sweep on the chosen probabilities and record both costs.
