# Shiny Dashboard — How It Works

A short explainer for `fraud_dashboard_app.R`, the interactive companion to the main fraud detection script.

## What it is

A 3-tab Shiny dashboard that turns the Random Forest vs. XGBoost comparison from `fraud_detection.R` into something interactive, instead of reading fixed numbers in a table, you can drag a threshold slider and watch both models respond in real time.

## Before running it

This app does **not** train models itself. It reuses model objects already created by `fraud_detection.R`, so you must run that script first, in the same R session:

```r
source("fraud_detection.R")       # trains fraud_fit (Random Forest) and xgb_fit (XGBoost)
source("fraud_dashboard_app.R")   # launches the dashboard
```

It depends on four objects existing in memory: `fraud_fit`, `xgb_fit`, `test_data`, `test_results`, and `xgb_results`.

## Tab 1 — Threshold Explorer

**What it does:** shows precision, recall, F1, accuracy, and a confusion matrix for both models, recalculated live at whatever threshold you set.

**How it works:** every model in this project doesn't just output "fraud" or "not fraud", it outputs a *probability* (e.g. 0.73 chance of fraud). By default, anything ≥ 0.5 gets called "fraud." This tab lets you move that cutoff anywhere from 0.01 to 0.99.

Internally, a helper function (`compute_at_threshold()`) takes the model's raw probabilities, re-applies your chosen cutoff, rebuilds the confusion matrix from scratch, and recalculates every metric, all reactively, so it updates the instant you move the slider.

**Why it matters:** this is the tab that makes the central finding of the project tangible. Lower the threshold and you'll watch recall climb (catching more fraud) while precision drops (more false alarms), and vice versa. It's the live version of the trade-off described in the project write-up.

## Tab 2 — Model Comparison

**What it does:** shows a side-by-side bar chart of both models' metrics at the current threshold, plus the full ROC curve.

**How it works:** the bar chart uses the same live recalculation as Tab 1, so it moves with the slider too. The ROC curve is different on purpose, it's built from `roc_curve()` and plotted across *every possible threshold at once*, which is why it stays fixed no matter where the slider is. That's the whole point of AUROC: it's a threshold-independent summary, shown here next to the threshold-dependent metrics for contrast.

## Tab 3 — Try a Transaction

**What it does:** samples a real transaction from the test set, lets you override its dollar amount, and shows what fraud probability each model assigns it, plus whether that counts as "fraud" at the current threshold.

**How it works:**
1. Click **Sample a Random Transaction** → pulls one random row from `test_data`, including its real (anonymized) `V1`–`V28` values.
2. Optionally adjust the **Amount** field — this is the only feature you can meaningfully edit, since `V1`–`V28` are PCA-transformed and have no real-world interpretation.
3. Click **Predict** → both `fraud_fit` and `xgb_fit` run `predict(..., type = "prob")` on that row, returning each model's fraud probability.
4. Those probabilities are compared against the **same threshold set on Tab 1** (it's a shared value across the whole app), and labeled FRAUD or Not Fraud accordingly.
5. The true label (was it actually fraud in the dataset?) is shown underneath, so you can check whether each model got it right.

**Why it matters:** numbers like "90.6% recall" are abstract. Seeing one specific transaction, with one specific outcome, in a tool you're clicking through yourself, makes the model's behavior concrete rather than statistical.

## The throughline

All three tabs are connected by one shared idea: **a fraud model's "decision" isn't fixed, it's a probability filtered through a threshold you choose.** Change that threshold, and you change what counts as a catch versus a false alarm, for the exact same model. That's the whole project's finding, made clickable.
