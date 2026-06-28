# Credit Card Fraud Detection: Random Forest vs. XGBoost

A small project comparing two tree-based models for detecting fraudulent credit card transactions in a highly imbalanced dataset, built in R using `tidymodels`-style tooling.

## The Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle), anonymized European credit card transactions from September 2013.

* 284,807 transactions total
* Only 492 are fraud ( **about 0.17%** of the data)
* Features `V1`–`V28` are PCA-transformed (anonymized); `Time` and `Amount` are raw

> \*\*Note:\*\* `creditcard.csv` is not included in this repo (file size + data redistribution). Download it from the Kaggle link above and place it in the project root to run the script.

## Why This Dataset Is Tricky

With fraud at 0.17% of transactions, a model that predicts "not fraud" for *everything* scores **99.8% accuracy** while catching zero fraud. This project deliberately ignores accuracy and focuses on **precision** and **recall** instead.

## Approach

1. Split data 75/25, stratified by class so both sets keep the same fraud ratio
2. Built a preprocessing recipe: normalize numeric features, **downsample** the majority class to a 1:1 ratio (training data only — test data stays untouched and realistic)
3. Trained two models on identical preprocessed data:

   * **Random Forest** (`ranger` engine, 100 trees)
   * **XGBoost** (`xgboost` engine, 100 trees)
4. Evaluated both on the same held-out test set

## Results

|Metric|Random Forest|XGBoost|
|-|-|-|
|Accuracy|0.969|0.966|
|Precision|0.0498|0.0460|
|**Recall**|**0.906**|**0.906**|
|F1|0.0944|0.0876|
|**AUROC**|**0.980**|**0.983**|

Both models caught **115 of 127** fraud cases in the test set (90.6% recall) and missed the same 12. Random Forest edged out XGBoost on precision and F1, fewer false alarms for the same fraud catch rate. But on AUROC, which measures ranking quality across every possible decision threshold, **XGBoost comes out slightly ahead**.

## The Interesting Part

This project produced two metrics that disagree, and that disagreement is the actual finding worth noting.

Precision/recall were measured at one fixed threshold (the default 0.5 cutoff) and at that specific cutoff, Random Forest happened to raise fewer false alarms. But AUROC looks across *every* possible threshold, and by that measure, XGBoost's underlying fraud-probability ranking is slightly better (0.983 vs. 0.980).

So which model "wins" depends on which question you're asking:

* *"At the default cutoff, which model raises fewer false alarms?"*: Random Forest
* *"Which model is better at ranking transactions by fraud risk overall?"* : XGBoost (slightly)

This is a good reminder that no single metric tells the whole story and that XGBoost's stronger ranking ability might translate into better precision too, if its decision threshold were tuned instead of left at the default 0.5. Both models were also run with **default hyperparameters** on a **small downsampled training set** (\~730 rows after balancing), so neither has had a fair chance to show its full potential yet.

**Takeaway:** how you handle class imbalance and threshold choice mattered as much as which algorithm was picked and the "best" model depends on which metric you're optimizing for.

## What I'd Explore Next

* Tune the classification threshold for XGBoost specifically its stronger AUROC suggests precision could improve at a non-default cutoff
* Tune XGBoost's hyperparameters properly (learning rate, max depth, nrounds) before declaring a winner
* Try a less aggressive downsample ratio to see if precision improves without sacrificing much recall
* Inspect feature importance to see which `V1`–`V28` components drive predictions

## Tech Stack

R · `tidyverse` · `recipes` · `rsample` · `workflows` · `parsnip` · `ranger` · `xgboost` · `yardstick` · `themis`

## Run It Yourself

```r
# install required packages first if needed:
# install.packages(c("tidyverse", "recipes", "rsample", "workflows",
#                     "parsnip", "ranger", "xgboost", "yardstick", "themis"))

source("fraud\_detection.R")
```

