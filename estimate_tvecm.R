#!/usr/bin/env Rscript

# Estimate a two-regime threshold VECM (TVECM) following Hansen and Seo (2002)
# with the tsDyn package, using the FRED data in fred_data.csv.
#
# The script:
#   1. Loads CPI and FPP from fred_data.csv and takes natural logs.
#   2. Runs ADF tests in levels and first differences to check for I(1) series.
#   3. Runs a Johansen cointegration test with an unrestricted constant.
#   4. Runs the Hansen-Seo threshold cointegration test.
#   5. Estimates a two-regime TVECM and plots the error-correction term.
#
# Interpretation notes:
#   - ADF tests: rejecting a unit root in differences but not in levels is
#     consistent with I(1) behavior.
#   - Johansen test: evidence of rank = 1 supports one cointegrating relation.
#   - Hansen-Seo test: a small bootstrap p-value favors threshold cointegration
#     over a linear VECM.
#   - TVECM regimes: observations with ECT <= threshold are labeled "normal";
#     observations with ECT > threshold are labeled "turbulent" below.

packages <- c("tsDyn", "urca", "vars")
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(tsDyn)
  library(urca)
  library(vars)
})

set.seed(12345)

data_file <- "fred_data.csv"
plot_file <- "tvecm_ect_regimes.png"
bootstrap_replications <- as.integer(Sys.getenv("TVECM_BOOT_REPS", "5000"))
if (is.na(bootstrap_replications) || bootstrap_replications < 1) {
  stop("TVECM_BOOT_REPS must be a positive integer.")
}

section <- function(title) {
  cat("\n", paste(rep("=", nchar(title) + 8), collapse = ""), "\n", sep = "")
  cat("=== ", title, " ===\n", sep = "")
  cat(paste(rep("=", nchar(title) + 8), collapse = ""), "\n\n", sep = "")
}

print_adf <- function(x, name, type = c("trend", "drift", "none"), lags = 12) {
  type <- match.arg(type)
  test <- urca::ur.df(x, type = type, lags = lags, selectlags = "AIC")
  cat("\nADF test for ", name, " (type = ", type, ", max lags = ", lags, ", selected by AIC)\n", sep = "")
  cat("Test statistic(s):\n")
  print(test@teststat)
  cat("Critical values:\n")
  print(test@cval)
  cat("Rule of thumb: reject unit root when the test statistic is more negative than the critical value.\n")
  invisible(test)
}

safe_extract <- function(object, candidates) {
  for (name in candidates) {
    if (!is.null(object[[name]])) {
      return(object[[name]])
    }
  }
  NULL
}

extract_threshold <- function(model) {
  if (!is.null(model$model.specific$Thresh)) {
    return(as.numeric(model$model.specific$Thresh)[1])
  }

  if (exists("getTh", where = asNamespace("tsDyn"), mode = "function")) {
    threshold <- tryCatch(tsDyn::getTh(model), error = function(e) NULL)
    if (!is.null(threshold)) {
      return(as.numeric(threshold)[1])
    }
  }

  threshold <- safe_extract(model, c("Th", "th", "threshold", "Thresh", "gamma"))
  if (!is.null(threshold)) {
    return(as.numeric(threshold)[1])
  }

  stop("Could not extract the TVECM threshold from the fitted model. Inspect str(tvecm_fit).")
}

extract_beta <- function(model) {
  if (!is.null(model$model.specific$coint)) {
    return(model$model.specific$coint)
  }

  if (!is.null(model$model.specific$beta)) {
    return(model$model.specific$beta)
  }

  if (exists("coefB", where = asNamespace("tsDyn"), mode = "function")) {
    beta <- tryCatch(tsDyn::coefB(model), error = function(e) NULL)
    if (!is.null(beta)) {
      return(beta)
    }
  }

  beta <- safe_extract(model, c("beta", "B", "coefficientsB"))
  if (!is.null(beta)) {
    return(beta)
  }

  warning("Could not extract the cointegrating vector from the fitted model; summary(tvecm_fit) may still show it.")
  NULL
}

extract_hstest_stat <- function(test) {
  stat <- safe_extract(test, c("stat", "SupLM", "supLM", "statistic"))
  if (is.null(stat)) {
    return(NA_real_)
  }
  as.numeric(stat)[1]
}

extract_hstest_pvalue <- function(test) {
  pval <- safe_extract(test, c("PvalBoot", "p.value", "pval", "pvalue"))
  if (is.null(pval)) {
    return(NA_real_)
  }
  as.numeric(pval)[1]
}

run_hansen_seo_test <- function(data_matrix, boot_type, nboot) {
  tsDyn::TVECM.HStest(
    data_matrix,
    lag = 2,
    ngridTh = 300,
    trim = 0.05,
    nboot = nboot,
    intercept = TRUE,
    boot.type = boot_type
  )
}

section("Loading and transforming data")

raw <- read.csv(data_file, stringsAsFactors = FALSE)
required_cols <- c("date", "CPI", "FPP")
missing_cols <- setdiff(required_cols, names(raw))
if (length(missing_cols) > 0) {
  stop("fred_data.csv is missing required columns: ", paste(missing_cols, collapse = ", "))
}

raw$date <- as.Date(raw$date)
raw <- raw[order(raw$date), ]
raw <- raw[complete.cases(raw[, required_cols]), ]

df <- data.frame(
  date = raw$date,
  log_CPI = log(raw$CPI),
  log_FPP = log(raw$FPP)
)

if (any(!is.finite(df$log_CPI)) || any(!is.finite(df$log_FPP))) {
  stop("Log transformation produced non-finite values. Check CPI and FPP for zero/negative values.")
}

series <- as.matrix(df[, c("log_CPI", "log_FPP")])
colnames(series) <- c("log_CPI", "log_FPP")

cat("Loaded ", nrow(df), " monthly observations from ", min(df$date), " to ", max(df$date), ".\n", sep = "")
cat("The model uses log(CPI) and log(FPP). CPI_core is available in the file but not used in this bivariate TVECM.\n")

section("Step 1: Preliminary checks")

cat("ADF checks in levels and first differences. These are diagnostic checks, not automatic decisions.\n")
adf_log_cpi_level <- print_adf(df$log_CPI, "log CPI in levels", type = "trend")
adf_log_fpp_level <- print_adf(df$log_FPP, "log FPP in levels", type = "trend")
adf_log_cpi_diff <- print_adf(diff(df$log_CPI), "first difference of log CPI", type = "drift")
adf_log_fpp_diff <- print_adf(diff(df$log_FPP), "first difference of log FPP", type = "drift")

cat("\nJohansen cointegration test on log CPI and log FPP.\n")
cat("Specification: K = 3 VAR lag order with an unrestricted constant outside the cointegrating vector.\n")
cat("In urca::ca.jo this is implemented with ecdet = 'none', which includes a constant among the short-run regressors.\n")
johansen <- urca::ca.jo(
  series,
  type = "trace",
  ecdet = "none",
  K = 3,
  spec = "transitory"
)
print(summary(johansen))
cat("\nRead the Johansen output by comparing trace statistics with critical values; evidence for rank 1 supports one cointegrating relation.\n")

section("Step 2: Hansen-Seo threshold cointegration test")

cat("Testing H0: linear cointegration against H1: threshold cointegration.\n")
cat("Settings: lag = 2, threshold grid = 300, trim = 0.05, bootstrap replications = ", bootstrap_replications, ".\n", sep = "")
cat("This step can take a long time because it runs two ", bootstrap_replications, "-replication bootstrap procedures.\n\n", sep = "")

cat("Running fixed-regressor bootstrap...\n")
hs_fixed <- run_hansen_seo_test(series, boot_type = "FixedReg", nboot = bootstrap_replications)
cat("\nFixed-regressor bootstrap result:\n")
print(hs_fixed)
cat("\nSupLM statistic: ", extract_hstest_stat(hs_fixed), "\n", sep = "")
cat("Fixed-regressor bootstrap p-value: ", extract_hstest_pvalue(hs_fixed), "\n", sep = "")

cat("\nRunning residual bootstrap...\n")
hs_resid <- run_hansen_seo_test(series, boot_type = "ResBoot", nboot = bootstrap_replications)
cat("\nResidual bootstrap result:\n")
print(hs_resid)
cat("\nSupLM statistic: ", extract_hstest_stat(hs_resid), "\n", sep = "")
cat("Residual bootstrap p-value: ", extract_hstest_pvalue(hs_resid), "\n", sep = "")

section("Step 3: Estimate two-regime TVECM")

cat("Estimating TVECM with lag = 2, nthresh = 1, trim = 0.05, ngridBeta = 300, ngridTh = 300.\n")
cat("The model searches over beta and threshold gamma and estimates separate dynamics in each regime.\n\n")

tvecm_fit <- tsDyn::TVECM(
  series,
  lag = 2,
  nthresh = 1,
  trim = 0.05,
  ngridBeta = 300,
  ngridTh = 300,
  plot = FALSE,
  include = "const",
  trace = TRUE
)

cat("\nTVECM summary:\n")
print(summary(tvecm_fit))

threshold <- extract_threshold(tvecm_fit)
beta <- extract_beta(tvecm_fit)

cat("\nEstimated threshold gamma:\n")
print(threshold)

cat("\nEstimated cointegrating vector / beta information:\n")
print(beta)

cat("\nRegime assignment counts from tsDyn::regime():\n")
regime_id <- tryCatch(tsDyn::regime(tvecm_fit), error = function(e) NULL)
if (!is.null(regime_id)) {
  regime_table <- table(regime_id)
  print(regime_table)
  cat("\nRegime proportions:\n")
  print(round(prop.table(regime_table), 4))
} else {
  cat("Could not extract regimes with tsDyn::regime(); using ECT-based counts below.\n")
}

section("Step 4: Interpretation aids and ECT plot")

if (is.null(beta)) {
  stop("Cannot compute the error-correction term because beta was not extracted. Inspect summary(tvecm_fit) for the cointegrating vector.")
}

if (!is.null(tvecm_fit$model.specific$ect)) {
  ect <- as.numeric(tvecm_fit$model.specific$ect)
  plot_dates <- tail(df$date, length(ect))
  beta_numeric <- as.numeric(beta)
  if (!is.null(tvecm_fit$model.specific$coint)) {
    coint_numeric <- as.numeric(tvecm_fit$model.specific$coint)
    ect_label <- paste0(
      "ECT from fitted TVECM: ",
      round(coint_numeric[1], 6),
      " * log_CPI + ",
      round(coint_numeric[2], 6),
      " * log_FPP"
    )
  } else {
    ect_label <- paste0("ECT from fitted TVECM with beta = ", round(beta_numeric[1], 6))
  }
} else {
  beta_numeric <- as.numeric(beta)
  plot_dates <- df$date
  if (length(beta_numeric) == 1) {
    ect <- df$log_CPI - beta_numeric * df$log_FPP
    ect_label <- paste0("ECT = log_CPI - (", round(beta_numeric, 6), ") * log_FPP")
  } else if (length(beta_numeric) >= 2) {
    ect <- beta_numeric[1] * df$log_CPI + beta_numeric[2] * df$log_FPP
    ect_label <- paste0("ECT = (", round(beta_numeric[1], 6), ") * log_CPI + (", round(beta_numeric[2], 6), ") * log_FPP")
  } else {
    stop("Extracted beta is empty; cannot compute the error-correction term.")
  }
}

normal <- ect <= threshold
turbulent <- ect > threshold

cat("Low-deviation / normal regime: ECT <= estimated threshold gamma.\n")
cat("High-deviation / turbulent regime: ECT > estimated threshold gamma.\n")
cat("\nUsing ", ect_label, "\n", sep = "")
cat("Estimated threshold gamma: ", threshold, "\n", sep = "")
cat("Normal observations: ", sum(normal, na.rm = TRUE), " (", round(mean(normal, na.rm = TRUE), 4), ")\n", sep = "")
cat("Turbulent observations: ", sum(turbulent, na.rm = TRUE), " (", round(mean(turbulent, na.rm = TRUE), 4), ")\n", sep = "")

png(plot_file, width = 1200, height = 700)
plot(
  plot_dates,
  ect,
  type = "l",
  col = "gray25",
  lwd = 1.4,
  xlab = "Date",
  ylab = "Error-correction term",
  main = "TVECM Error-Correction Term and Estimated Threshold"
)
abline(h = threshold, col = "red", lwd = 2, lty = 2)
points(plot_dates[turbulent], ect[turbulent], col = "firebrick", pch = 16, cex = 0.55)
legend(
  "topleft",
  legend = c("ECT", "Estimated threshold", "Turbulent periods (ECT > threshold)"),
  col = c("gray25", "red", "firebrick"),
  lty = c(1, 2, NA),
  pch = c(NA, NA, 16),
  lwd = c(1.4, 2, NA),
  bty = "n"
)
dev.off()

cat("\nSaved ECT regime plot to ", plot_file, ".\n", sep = "")
cat("\nDone.\n")
