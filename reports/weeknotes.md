# Week notes

## Week 5 — Days 29–35 · Tuning, and learning what "a win" means

### Shipped
- Optuna search on XGBoost (150 trials) and LightGBM (40 trials), both on PR-AUC, persisted in optuna.db
- Champion champion_xgb_400 (tuned XGBoost, 400 trees) registered as fraud-champion@champion
- 5-fold CV estimate of the champion (Day 31) and margin-vs-noise check (Day 32)
- reports/day30_findings.md, sql/week5.sql, reports/interview_questions.md
- Evening S5: WSL 2.7 + Docker Desktop installed, WSL capped at 4 GB, hello-world runs

### The numbers

| Measurement | PR-AUC |
|---|---|
| XGBoost champion, 3-fold CV @400 | 0.8523 ± 0.0096 |
| LightGBM (Optuna), 3-fold CV @400 | 0.8487 ± 0.0087 |
| *XGBoost champion, 5-fold CV @400 — headline number* | *0.8509 ± 0.0186* |
| XGBoost − LightGBM margin, 5-fold | 0.0073 |
| Noise band (2 × 5-fold std) | ≈ 0.037 |

Test set, scored once after the decision: champion_xgb_400 0.8719 PR-AUC / 0.9782 ROC-AUC.
xgb_scale_pos_weight (hand-set, Day 22) scored 0.8751. The gap is 0.0032, inside the noise.

### What I concluded
- *XGBoost and LightGBM are statistically tied.* The 0.0073 margin is about a fifth of the noise band. Rerun with different fold seeds and LightGBM could come out on top.
- *XGBoost is champion on a secondary criterion, not on score:* __ (pick yours: SHAP TreeExplainer work in Week 6 is scoped around it / scale_pos_weight is easier to explain than leaf-wise growth / latency or size).
- *The 3-fold and 5-fold results agree.* 0.8523 vs 0.8509 are two measurements of one model, 0.0014 apart. The 5-fold std is wider because each validation fold holds fewer frauds, not because anything broke.
- *Tuning didn't beat my Day 22 hand-set model.* That's a finding, not a failure. PCA features on a well-separated problem leave little for hyperparameters to find.
- *I never pick the test-set leader.* Selection happened on CV inside X_tr. A tie broken by the test set is a leak.

### Teach-backs, one clean sentence each

*Why PR-AUC and not ROC-AUC or F1?*
ROC-AUC's false-positive rate has ~227k negatives in the denominator, so it barely moves. My collapsed LightGBM run scored 0.09 PR-AUC and still 0.907 ROC-AUC. PR-AUC exposes that failure, and unlike F1 it doesn't depend on a threshold I haven't chosen yet.

*What do mean, std, variance and noise mean here?*
The mean is the average PR-AUC across folds. The std is how far fold scores typically sit from that average. Variance is std², so I don't use it. Noise is the fold-to-fold wobble of one model on the same data. *The ± says how precisely I know the number, not how good the model is.*

*Is the champion's win real?*
No. It won by 0.0073 against a noise band of about 0.037, so I report a tie and name the reason I picked XGBoost anyway.

*Why that imbalance strategy?*
Optuna tuned scale_pos_weight to __ against a class ratio of 577. PR-AUC rewards precision, and full class weighting over-predicts fraud.

*Did SMOTE underperform?*
No. The plan predicted it would, but xgb_smote scored 0.8744 on test, level with the best. Why interpolating in 28-D PCA space didn't hurt is still open.

*If the false-negative cost halves, what happens to the threshold?* (Day 34)
The optimal threshold rises. I flag fewer transactions, so recall falls and precision rises.

### What broke / cost me time
- *My Day 32 reasoning came out backwards.* I read a smaller std as a better model and called PR-AUC "accuracy." Fixed: the ± is precision of the estimate, and on 0.17% fraud, accuracy is the wrong word.
- *On Day 34 I restated the setup instead of answering.* Asked where the threshold moves, I re-explained the cost matrix and got the direction wrong. Lesson: when a question feels uncomfortable, commit to a direction first.
- Eugene Yan's design-patterns post was too advanced for now. I'm deferring it to Days 46–49, after the API exists.
- ../ paths pointed outside the repo, and a stray optuna.db ended up in C:\Users\Legion. Everything now runs from the repo root.
- I misread colours on an unlabelled MLflow chart. From now on I sort mlflow.search_runs() by the metric instead.

### Open items carried forward
- [ ] Rename notebook/ → notebooks/ to match the plan
- [ ] Regenerate reports/figures/05_roc_curves.png (Day 24 asked for PR and ROC)
- [ ] Rename the champion_refit run (NaN metrics) and fix the LightBGM typo in run names
- [ ] Move sql/ to ds-portfolio-2026 before Day 52
- [ ] Finish all 5 hostile questions in interview_questions.md, if not done
- [ ] SMOTE: why didn't it hurt?

### Next week (Week 6)
Cost-sensitive threshold + calibration (Day 36), then SHAP and error analysis. Read the scikit-learn calibration guide.
