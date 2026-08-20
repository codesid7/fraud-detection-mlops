# Decisions log — fraud-detection-mlops

One entry per fork in the road. Each records **what** was decided, **why**, and **what was rejected**.
The rejected column is the point: in six weeks I won't remember what I chose *against*.

Entries marked **[OPEN]** are undecided and have a deadline.
Entries marked **[VERIFY]** were reconstructed after the fact — confirm before trusting.

---

## Day 22 — Sat 11 Aug 2026

### D1 · Environment: Miniforge + conda env `fraud`, Python 3.12
Chose a named conda environment over system Python or `venv`.
**Why:** Phase 1 ships a container and an API; a named env is reproducible and can be destroyed and rebuilt without touching the OS Python. conda-forge also handles the compiled deps (xgboost, lightgbm) more predictably on Windows than pip-only.
**Rejected:** `venv` — fine for pure-Python, weaker for the compiled stack. System Python — no isolation, breaks on the first version conflict.

### D2 · All float64 columns downcast to float32 on load
**Why:** halves the frame, ~68 MB → ~34 MB, and the saving compounds across every copy sklearn makes internally. On an 8 GB machine that matters by Week 7. Precision cost is irrelevant here — float32 carries ~7 significant digits and my cost model rounds to ₹400 and ₹2,000.
**Caveat recorded:** float32 accumulates summation error over long columns. If I ever sum a full 284k-row column, cast back with `dtype="float64"` inside the sum. `expected_cost` sums only ~90 FN values, so it's safe.
**Rejected:** keeping float64 — no benefit at this precision requirement.

### D3 · Data access via `kagglehub` + credentials, not a manually downloaded CSV
Credentials live at `C:\Users\<me>\.kaggle\kaggle.json`, outside the repo.
**Why:** anyone cloning the repo runs the same `dataset_download` call and gets the same data. A manual CSV would mean the repo silently depends on a file only I have.
**Reproducibility note for the README:** a stranger cloning this needs their own Kaggle account and token. That's a real dependency and should be stated, not assumed.
**Rejected:** manual CSV into `data/` — considered as fallback during the 403 troubleshooting, not needed in the end.

### D4 · Gotcha worth remembering: Kaggle tokens are single-active
Creating a new API token **invalidates every previous one**. The 403 I spent time on was a stale token, not a permissions problem — the dataset is public. Kaggle returns 403 rather than 401 when it can't identify you, which is what made it look like an access issue.
**Action:** noted against setup session S2 in the setup guide.

### D5 · Hour feature: `(Time // 3600) % 24`
`Time` is seconds since the first transaction, so `// 3600` gives elapsed hours 0–47 across the dataset's two days; `% 24` folds them onto a single 24-hour clock.
**Why fold:** doubles the rows per bucket, so the per-hour fraud rate is less noisy.
**Limitation that must appear in the model card:** `Time` has no wall-clock anchor. Hour 0 is *some* consistent hour, not necessarily midnight. So "fraud rate varies by time of day" is defensible; "fraud peaks at 2am" is not.

### D6 · **[OPEN]** Model artifact vs Render deployment
`.gitignore` excludes `models/*.joblib`, but Render builds from Git — so the container that works locally on Day 44 will have no model on Day 51.
**Two options:** (a) Git LFS for the joblib, (b) Render runs `python -m src.pipeline` at build time and trains on deploy.
**Deadline:** decide by Day 51. Do not let this surface during the deploy block.

---

## Day 22 (continued) — Wed 12 Aug 2026

### D7 · Plan gap: figure code was missing, wrote my own
The plan listed three PNGs as Block 2 outputs and showed the `savefig` line, but never defined `fig`. Wrote the three plotting cells myself.
**Recorded because:** the plan is a draft, not scripture. Gaps like this are mine to close, and noting them is how the next revision gets better.

### D8 · Log scale on figures 01 and 02
**Why 01 (class balance):** 492 next to 284,315 renders as one bar and an invisible sliver. Log scale is the only way both are legible — and *needing* log scale is itself the story.
**Why 02 (amount by class):** `Amount` is heavily right-skewed; a few very large transactions flatten everything else on a linear axis. Also set `showfliers=False` — with 284k rows the outlier dots form a solid smear.

### D9 · Notebook figure paths use `../reports/figures/`
Notebooks execute with their own folder as the working directory, so paths are relative to `notebooks/`, not the repo root.

### D10 · **[OPEN]** Diagnostic: fraud rate across 48 elapsed hours
Plotting the unwrapped 0–47 hour series *before* trusting the `% 24` fold, to check whether day one and day two actually share a daily rhythm. If they don't, the fold averages away a real difference rather than revealing a pattern.
**Saved as:** `reports/figures/00_fraud_by_elapsed_hour.png` (diagnostic, hence `00_`).
**Result:** _pending — record what the two days looked like._

### D11 · **[OPEN]** Diagnostic: fraud rate by position in the file, binned by 10,000 transactions
`creditcard.csv` is sorted by `Time`, so row position is time order. This asks whether the fraud rate is *stationary across the sequence*.
**Why it matters:** Day 23's `get_splits()` uses a **stratified random** 80/20. That's only defensible if the rate is roughly stable. If it trends, random shuffling leaks future information into training and a **temporal** split (train on early hours, test on late) is the honest evaluation — harder, lower scores, more defensible.
**Bin-size note:** at 10,000 rows a bin holds ~17 frauds, so random wobble is about √17 ≈ 4. Bars swinging ±25% mean nothing; look for 3× or 0.1× the baseline. Re-bin at 20,000 — a pattern that vanishes on re-binning was never there.
**Second use:** this is the reference for Day 51's drift monitor. "Here's what normal variation looked like in training" is exactly what a drift threshold needs.
**Result:** _pending — record whether the rate is flat, and therefore whether the split stays random._

### D12 · Cost matrix: FP = ₹400 flat, FN = `max(Amount, ₹2,000)`
**FP ₹400** — a wrongly declined card costs a support call and possibly a reissue, plus a probabilistic charge for churn: a customer declined at a restaurant may quietly stop using the card, and that lost lifetime value is larger than the call. Unmeasurable exactly, hence a stated assumption.
**FN = the Amount, floored at ₹2,000** — the loss is the transaction value, but never less than ₹2,000, because chargeback processing, investigation, card reissue and lost trust land regardless of size. **Without the floor the model learns that small frauds are nearly free to miss**, which blinds it to card-testing attacks where thieves probe with tiny amounts before the big hit.
**Who bears it:** the cardholder is generally made whole under network rules. The loss sits with the issuer/merchant — which is who this model serves. Do not say "the user loses the money."
**Both numbers are assumptions and the README says so**, so a reviewer can substitute their own.

### D13 · Consequence of D12: the optimal threshold is nowhere near 0.5
Flagging costs `(1-p) × 400`; letting through costs `p × max(Amount, 2000)`. Flag when the first is smaller, which rearranges to:

```
p > 400 / (400 + FN_cost)
```

- ₹2,000 fraud → flag above **p = 0.17**
- ₹40,000 fraud → flag above **p = 0.01**

Two things follow. **The default 0.5 is indefensible** under my own cost model — even the cheapest case says flag at 0.17. And **the correct threshold depends on the transaction amount**, so any single global threshold is a compromise. Day 36's sweep finds the best compromise; an amount-aware threshold is the obvious extension and belongs in "what I'd do next."

---

## Open questions

| # | Question | Decide by |
|---|---|---|
| D6 | Model artifact into Render — Git LFS or train-on-build? | Day 51 |
| D10 | Do day one and day two share a daily rhythm? | Before finalising fig 03 |
| D11 | Is fraud rate stationary across the sequence — random split or temporal? | Day 23, Block 1 |
