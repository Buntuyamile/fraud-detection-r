
# Shiny App — Fraud Detection Dashboard
# Random Forest vs. XGBoost: Interactive Threshold Explorer

# install.packages(c("shiny", "shinydashboard", "DT")) # run once if needed

library(shiny)
library(shinydashboard)
library(tidyverse)
library(yardstick)
library(DT)


# Helper: compute metrics at a given threshold for a results df


compute_at_threshold <- function(results_df, threshold) {
  preds <- results_df |>
    mutate(
      pred_class = factor(if_else(.pred_1 >= threshold, "1", "0"), levels = c("0", "1"))
    )

  cm <- table(Predicted = preds$pred_class, Truth = preds$Class)

  TP <- ifelse("1" %in% rownames(cm) && "1" %in% colnames(cm), cm["1", "1"], 0)
  FP <- ifelse("1" %in% rownames(cm) && "0" %in% colnames(cm), cm["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(cm) && "1" %in% colnames(cm), cm["0", "1"], 0)
  TN <- ifelse("0" %in% rownames(cm) && "0" %in% colnames(cm), cm["0", "0"], 0)

  precision <- if ((TP + FP) > 0) TP / (TP + FP) else 0
  recall    <- if ((TP + FN) > 0) TP / (TP + FN) else 0
  f1        <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
  accuracy  <- (TP + TN) / (TP + FP + TN + FN)

  list(
    cm = cm,
    metrics = tibble(
      Metric = c("Accuracy", "Precision", "Recall", "F1 score"),
      Value = round(c(accuracy, precision, recall, f1), 4)
    ),
    TP = TP, FP = FP, FN = FN, TN = TN
  )
}


# UI

ui <- dashboardPage(

  dashboardHeader(title = "Fraud Detection: RF vs. XGBoost"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Threshold Explorer", tabName = "threshold", icon = icon("sliders-h")),
      menuItem("Model Comparison", tabName = "comparison", icon = icon("balance-scale")),
      menuItem("Try a Transaction", tabName = "predictor", icon = icon("magnifying-glass-dollar"))
    )
  ),

  dashboardBody(
    tabItems(

      # TAB 1: Threshold Explorer 
      tabItem(tabName = "threshold",
        fluidRow(
          box(
            title = "Decision Threshold", width = 12, status = "primary",
            sliderInput("threshold", "Classify as fraud if predicted probability >=",
                        min = 0.01, max = 0.99, value = 0.50, step = 0.01),
            helpText("Move the slider to see how the precision/recall trade-off shifts",
                     "for both models in real time. Lower thresholds catch more fraud",
                     "but raise more false alarms; higher thresholds do the opposite.")
          )
        ),
        fluidRow(
          box(title = "Random Forest", width = 6, status = "info", solidHeader = TRUE,
              tableOutput("rf_metrics_table"),
              plotOutput("rf_cm_plot", height = "220px")
          ),
          box(title = "XGBoost", width = 6, status = "warning", solidHeader = TRUE,
              tableOutput("xgb_metrics_table"),
              plotOutput("xgb_cm_plot", height = "220px")
          )
        )
      ),

      #  TAB 2: Model Comparison 
      tabItem(tabName = "comparison",
        fluidRow(
          box(title = "Metrics at Current Threshold", width = 12, status = "primary",
              plotOutput("comparison_bar", height = "350px")
          )
        ),
        fluidRow(
          box(title = "ROC Curve (Threshold-Independent)", width = 12, status = "primary",
              plotOutput("roc_plot", height = "400px")
          )
        )
      ),

      #  TAB 3: Predictor 
      tabItem(tabName = "predictor",
        fluidRow(
          box(
            title = "Enter Transaction Details", width = 4, status = "primary",
            numericInput("amount", "Transaction Amount ($)", value = 100, min = 0),
            helpText("V1-V28 are anonymized (PCA-transformed) features from the",
                     "original dataset and can't be entered manually in a meaningful",
                     "way. For this demo, we sample a random real row from the test",
                     "set and let you override just the Amount, to show how the",
                     "models respond."),
            actionButton("sample_row", "Sample a Random Transaction", icon = icon("dice")),
            actionButton("predict_btn", "Predict", icon = icon("play"), class = "btn-primary")
          ),
          box(
            title = "Prediction Results", width = 8, status = "success",
            tableOutput("prediction_table"),
            textOutput("true_label")
          )
        )
      )
    )
  )
)


# Server

server <- function(input, output, session) {

  # ---- Threshold Explorer ----
  rf_calc  <- reactive({ compute_at_threshold(test_results, input$threshold) })
  xgb_calc <- reactive({ compute_at_threshold(xgb_results, input$threshold) })

  output$rf_metrics_table  <- renderTable({ rf_calc()$metrics })
  output$xgb_metrics_table <- renderTable({ xgb_calc()$metrics })

  plot_cm <- function(calc_result, model_name, fill_color) {
    cm_df <- tibble(
      Predicted = c("Not Fraud", "Not Fraud", "Fraud", "Fraud"),
      Actual    = c("Not Fraud", "Fraud", "Not Fraud", "Fraud"),
      Count     = c(calc_result$TN, calc_result$FN, calc_result$FP, calc_result$TP)
    )
    ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Count), size = 5) +
      scale_fill_gradient(low = "white", high = fill_color) +
      labs(title = paste(model_name, "- Confusion Matrix")) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "none")
  }

  output$rf_cm_plot  <- renderPlot({ plot_cm(rf_calc(),  "Random Forest", "skyblue") })
  output$xgb_cm_plot <- renderPlot({ plot_cm(xgb_calc(), "XGBoost",       "darkorange") })

  # ---- Model Comparison Tab ----
  output$comparison_bar <- renderPlot({
    rf_m  <- rf_calc()$metrics  |> mutate(Model = "Random Forest")
    xgb_m <- xgb_calc()$metrics |> mutate(Model = "XGBoost")
    combined <- bind_rows(rf_m, xgb_m) |>
      mutate(Metric = factor(Metric, levels = c("Accuracy", "Precision", "Recall", "F1 score")))

    ggplot(combined, aes(x = Metric, y = Value, fill = Model)) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      geom_text(aes(label = Value), position = position_dodge(width = 0.8),
                 vjust = -0.4, size = 4) +
      scale_fill_manual(values = c("Random Forest" = "skyblue", "XGBoost" = "darkorange")) +
      scale_y_continuous(limits = c(0, 1.1)) +
      labs(title = paste0("Metrics at threshold = ", input$threshold), y = "Score", x = NULL) +
      theme_minimal(base_size = 14)
  })

  output$roc_plot <- renderPlot({
    rf_roc  <- roc_curve(test_results, truth = Class, .pred_1, event_level = "second") |>
      mutate(model = "Random Forest")
    xgb_roc <- roc_curve(xgb_results, truth = Class, .pred_1, event_level = "second") |>
      mutate(model = "XGBoost")
    rf_auc  <- roc_auc(test_results, truth = Class, .pred_1, event_level = "second")$.estimate
    xgb_auc <- roc_auc(xgb_results, truth = Class, .pred_1, event_level = "second")$.estimate

    bind_rows(rf_roc, xgb_roc) |>
      ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
      geom_path(linewidth = 1.1) +
      geom_abline(lty = 3, color = "gray") +
      scale_color_manual(values = c("Random Forest" = "skyblue", "XGBoost" = "darkorange")) +
      labs(
        title = "ROC Curve: Random Forest vs. XGBoost",
        subtitle = paste0("AUROC - RF: ", round(rf_auc, 3), " | XGBoost: ", round(xgb_auc, 3)),
        x = "False Positive Rate", y = "True Positive Rate", color = "Model"
      ) +
      coord_equal() +
      theme_minimal(base_size = 14)
  })

  #  Predictor Tab 
  sampled_row <- eventReactive(input$sample_row, {
    test_data[sample(nrow(test_data), 1), ]
  }, ignoreNULL = FALSE)

  observeEvent(sampled_row(), {
    updateNumericInput(session, "amount", value = round(sampled_row()$Amount, 2))
  })

  prediction_result <- eventReactive(input$predict_btn, {
    row <- sampled_row()
    row$Amount <- input$amount # let the user override just the Amount field

    rf_prob  <- predict(fraud_fit,  row, type = "prob")$.pred_1
    xgb_prob <- predict(xgb_fit, row, type = "prob")$.pred_1

    list(
      table = tibble(
        Model = c("Random Forest", "XGBoost"),
        `Predicted Fraud Probability` = round(c(rf_prob, xgb_prob), 4),
        `Classification (>= threshold)` = if_else(
          c(rf_prob, xgb_prob) >= input$threshold, "FRAUD", "Not Fraud"
        )
      ),
      true_class = as.character(row$Class)
    )
  })

  output$prediction_table <- renderTable({ prediction_result()$table })

  output$true_label <- renderText({
    res <- prediction_result()
    paste0("Actual label for this transaction (from the dataset): ",
           if_else(res$true_class == "1", "FRAUD", "Not Fraud"))
  })
}


# Launch the app

shinyApp(ui, server)
