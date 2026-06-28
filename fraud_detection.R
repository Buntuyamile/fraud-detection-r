
# Credit Card Fraud Detection — Random Forest vs. XGBoost


library(themis)      # for step_downsample() - handles class imbalance
library(tidyverse)    # for read.csv pipe, glimpse(), general data wrangling
library(ranger)        # fast random forest engine
library(parsnip)       # defines model specs (rand_forest, boost_tree, etc.)
library(rsample)       # for initial_split(), training(), testing()
library(workflows)      # for workflow(), add_recipe(), add_model()
library(yardstick)       # for conf_mat() and evaluation metrics
library(xgboost)          # gradient boosting engine, used later for comparison


# 1. Load & Inspect Data

# Read the dataset and convert Class to a factor so R treats it as a
# category (fraud vs. not fraud) rather than a number to do math on.
df <- read.csv("creditcard.csv") |>
  mutate(Class = as.factor(Class))

# Check structure of the data 
glimpse(df)

table(df$Class)


# 2. Split Data into Training and Test Sets

set.seed(123)
df_split   <- initial_split(df, prop = 0.75, strata = Class)
train_data <- training(df_split)
test_data  <- testing(df_split)


# 3. Build Preprocessing Recipe

# step_normalize scales numeric predictors to mean 0 / SD 1 (mainly
# matters for Amount, since V1-V28 are already roughly standardized).
# step_downsample fixes class imbalance by trimming the majority class
# down to match the minority class — applied to training data only,
# so the test set stays realistic for honest evaluation.

fraud_recipe <- recipe(Class ~ ., data = train_data) |>
  step_normalize(all_numeric_predictors()) |>
  step_downsample(Class, under_ratio = 1)


# 4. Model 1 — Random Forest (ranger)

# 100 trees, classification mode. importance = "impurity" tracks which
# variables matter most for splitting decisions.
rf_spec <- rand_forest(trees = 100, mode = "classification") |>
  set_engine("ranger", importance = "impurity")

# Preprocessing and model spec into one workflow,
# so a single fit() call applies both in the correct order.
fraud_wf <- workflow() |>
  add_recipe(fraud_recipe) |>
  add_model(rf_spec)

# Train the model 
fraud_fit <- fit(fraud_wf, data = train_data)
fraud_fit # prints model summary + OOB error estimate

# Generate predictions on the held-out test set:
test_results <- predict(fraud_fit, test_data, type = "class") |>
  bind_cols(predict(fraud_fit, test_data, type = "prob")) |>
  bind_cols(test_data)

# Confusion matrix: how many fraud cases caught vs. missed vs. false alarms
conf_mat(test_results, truth = Class, estimate = .pred_class)

# Precision / recall / F1 / accuracy.
# event_level = "second" tells yardstick to treat "1" (fraud) as the
# positive class we care about — without this it defaults to "0".
test_results |>
  conf_mat(truth = Class, estimate = .pred_class) |>
  summary(event_level = "second") |>
  filter(.metric %in% c("precision", "recall", "f_meas", "accuracy"))


# 5. Model 2 — XGBoost (same recipe, for a fair comparison)

# Reuses fraud_recipe so both models train on identically preprocessed
# data.
xgb_spec <- boost_tree(trees = 100, mode = "classification") |>
  set_engine("xgboost")

xgb_wf <- workflow() |>
  add_recipe(fraud_recipe) |>
  add_model(xgb_spec)

# Train XGBoost
xgb_fit <- fit(xgb_wf, data = train_data)

# Same prediction + evaluation steps as the random forest above
xgb_results <- predict(xgb_fit, test_data, type = "class") |>
  bind_cols(predict(xgb_fit, test_data, type = "prob")) |>
  bind_cols(test_data)

# Confusion matrix for XGBoost
conf_mat(xgb_results, truth = Class, estimate = .pred_class)

# Precision / recall / F1 / accuracy for XGBoost
xgb_results |>
  conf_mat(truth = Class, estimate = .pred_class) |>
  summary(event_level = "second") |>
  filter(.metric %in% c("precision", "recall", "f_meas", "accuracy"))


# 6. ROC Curve & AUROC — Random Forest vs. XGBoost


# Random Forest AUROC
rf_auc <- roc_auc(test_results, truth = Class, .pred_1, event_level = "second")
rf_auc

# XGBoost AUROC
xgb_auc <- roc_auc(xgb_results, truth = Class, .pred_1, event_level = "second")
xgb_auc

# Generate ROC curve data for both models
rf_roc <- roc_curve(test_results, truth = Class, .pred_1, event_level = "second") |>
  mutate(model = "Random Forest")

xgb_roc <- roc_curve(xgb_results, truth = Class, .pred_1, event_level = "second") |>
  mutate(model = "XGBoost")

# Combine both into one data frame for plotting
roc_combined <- bind_rows(rf_roc, xgb_roc)

# Plot both ROC curves on the same chart
roc_combined |>
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  geom_abline(lty = 3, color = "gray") + # diagonal = random guessing baseline
  labs(
    title = "ROC Curve: Random Forest vs. XGBoost",
    subtitle = paste0(
      "AUROC  Random Forest: ", round(rf_auc$.estimate, 4),
      " | XGBoost: ", round(xgb_auc$.estimate, 4)
    ),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity / Recall)",
    color = "Model"
  ) +
  coord_equal() +
  theme_minimal()

