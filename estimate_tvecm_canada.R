#!/usr/bin/env Rscript

# Estimate a two-regime threshold VECM (TVECM) for CANADA following Hansen and
# Seo (2002), mirroring estimate_tvecm.R (the US version) and the out-of-sample
# methodology in vecm_forecast.py.
#
# Data: fred_data_canada.csv (date, CPI, ENERGY), monthly, natural logs.
#   - CPI    = Canadian Consumer Price Index level (StatCan, 'Prices' sheet).
#   - ENERGY = Canadian energy price index, reconstructed to a base-100 level
#              from the 'Energy Inflation' YoY % series via 12-month chaining.
#
# Steps:
#   1. ADF tests (levels + first diffs) -> confirm both I(1).
#   2. Johansen cointegration (VAR lag by BIC, unrestricted constant).
#   3. BDS test on AR-filtered residuals of each growth series (dims 2-5).
#   4. Hansen-Seo threshold cointegration test (SupLM + 2 bootstrap p-values).
#   5. TVECM estimation (threshold, beta, regime split, adjustment speeds).
#   6. Demeaned-ECT regime classification plot.
#   7. Out-of-sample validation (VECM vs AR(2): RMSE/MAE/direction + HLN-DM).

suppressPackageStartupMessages({
  library(tsDyn)
  library(urca)
  library(vars)
  library(tseries)
})

set.seed(12345)

data_file       <- "fred_data_canada.csv"
train_file      <- "fred_train_canada.csv"
test_file       <- "fred_test_canada.csv"
ect_plot_file   <- "tvecm_ect_regimes_canada.png"
fcst_plot_file  <- "vecm_forecast_canada.png"
nboot <- as.integer(Sys.getenv("TVECM_BOOT_REPS", "5000"))
if (is.na(nboot) || nboot < 1) stop("TVECM_BOOT_REPS must be a positive integer.")

section <- function(title) {
  bar <- paste(rep("=", nchar(title) + 8), collapse = "")
  cat("\n", bar, "\n=== ", title, " ===\n", bar, "\n\n", sep = "")
}

print_adf <- function(x, name, type = c("trend", "drift", "none"), lags = 12) {
  type <- match.arg(type)
  test <- urca::ur.df(x, type = type, lags = lags, selectlags = "AIC")
  cat("\nADF for ", name, " (type=", type, ", maxlags=", lags, ", AIC-selected)\n", sep = "")
  cat("  test statistic(s): ", paste(round(test@teststat, 4), collapse = ", "), "\n", sep = "")
  cat("  critical values (tau):\n"); print(test@cval)
  invisible(test)
}

# --------------------------------------------------------------------------- #
section("Loading and transforming Canadian data")

raw <- read.csv(data_file, stringsAsFactors = FALSE)
required_cols <- c("date", "CPI", "ENERGY")
miss <- setdiff(required_cols, names(raw))
if (length(miss) > 0) stop("fred_data_canada.csv missing columns: ", paste(miss, collapse = ", "))
raw$date <- as.Date(raw$date)
raw <- raw[order(raw$date), ]
raw <- raw[complete.cases(raw[, required_cols]), ]
if (any(raw$CPI <= 0) || any(raw$ENERGY <= 0)) stop("CPI and ENERGY must be positive for logs.")

df <- data.frame(date = raw$date, log_CPI = log(raw$CPI), log_ENERGY = log(raw$ENERGY))
series <- as.matrix(df[, c("log_CPI", "log_ENERGY")])
colnames(series) <- c("log_CPI", "log_ENERGY")
cat("Loaded ", nrow(df), " monthly obs, ", format(min(df$date)), " to ", format(max(df$date)), ".\n", sep = "")
cat("Series: log(CPI_ca) and log(ENERGY_ca). ENERGY is the base-100 reconstructed energy price level.\n")

# --------------------------------------------------------------------------- #
section("Step 1: ADF unit-root tests (confirm I(1))")

a_cpi_l <- print_adf(df$log_CPI,        "log CPI level",        type = "trend")
a_eng_l <- print_adf(df$log_ENERGY,     "log ENERGY level",     type = "trend")
a_cpi_d <- print_adf(diff(df$log_CPI),  "d.log CPI",            type = "drift")
a_eng_d <- print_adf(diff(df$log_ENERGY), "d.log ENERGY",       type = "drift")

is_i1 <- function(level_test, diff_test) {
  lc <- level_test@cval[1, "5pct"]   # tau row (unit-root statistic) critical value
  dc <- diff_test@cval[1, "5pct"]
  ls <- level_test@teststat[1]; ds <- diff_test@teststat[1]
  (ls > lc) && (ds < dc)   # fail to reject in level, reject in diff
}
cpi_i1 <- is_i1(a_cpi_l, a_cpi_d); eng_i1 <- is_i1(a_eng_l, a_eng_d)
cat("\nI(1) verdict (level non-stationary AND diff stationary @5%):\n")
cat("  log CPI    : ", ifelse(cpi_i1, "I(1) confirmed", "NOT cleanly I(1) @5%"), "\n", sep = "")
cat("  log ENERGY : ", ifelse(eng_i1, "I(1) confirmed", "NOT cleanly I(1) @5%"), "\n", sep = "")
if (!cpi_i1 || !eng_i1) {
  cat("\n*** WARNING: at least one series is not cleanly I(1) at the 5% level by the\n",
      "rule-of-thumb (often a seasonality artefact in non-seasonally-adjusted monthly\n",
      "CPI, where the differenced ADF sits near the 10% boundary). Proceeding so the\n",
      "full pipeline can be inspected, but treat the cointegration/TVECM output with\n",
      "this caveat in mind. ***\n", sep = "")
}

# --------------------------------------------------------------------------- #
section("Step 2: Johansen cointegration (VAR lag by BIC, unrestricted const)")

lag_sel <- vars::VARselect(series, lag.max = 12, type = "const")
K_bic <- as.integer(lag_sel$selection["SC(n)"])
if (is.na(K_bic) || K_bic < 2) K_bic <- 2
cat("VARselect BIC (SC) chooses p = ", K_bic, " levels lags.\n", sep = "")
cat("ca.jo uses K = ", K_bic, " (ecdet='none' => unrestricted constant).\n\n", sep = "")

jo_trace <- urca::ca.jo(series, type = "trace",  ecdet = "none", K = K_bic, spec = "transitory")
jo_eigen <- urca::ca.jo(series, type = "eigen",  ecdet = "none", K = K_bic, spec = "transitory")
cat("--- Trace test ---\n");        print(summary(jo_trace))
cat("\n--- Maximum-eigenvalue test ---\n"); print(summary(jo_eigen))

tr_stat <- jo_trace@teststat; tr_cv <- jo_trace@cval
me_stat <- jo_eigen@teststat; me_cv <- jo_eigen@cval
# ca.jo orders hypotheses r<=1 then r=0 (teststat[1]=r<=1, teststat[2]=r=0).
r0_trace_rej  <- tr_stat[length(tr_stat)] > tr_cv[nrow(tr_cv), "5pct"]
r1_trace_keep <- tr_stat[1] < tr_cv[1, "5pct"]
r0_eigen_rej  <- me_stat[length(me_stat)] > me_cv[nrow(me_cv), "5pct"]
r1_eigen_keep <- me_stat[1] < me_cv[1, "5pct"]
cat("\nRank verdict @5%:\n")
cat("  Trace: reject r=0? ", r0_trace_rej, " | fail to reject r<=1? ", r1_trace_keep, "\n", sep = "")
cat("  MaxEig: reject r=0? ", r0_eigen_rej, " | fail to reject r<=1? ", r1_eigen_keep, "\n", sep = "")
cat("  => r=1 confirmed: ", (r0_trace_rej && r1_trace_keep), " (trace), ",
    (r0_eigen_rej && r1_eigen_keep), " (maxeig)\n", sep = "")

# --------------------------------------------------------------------------- #
section("Step 3: BDS nonlinearity test on AR-filtered growth residuals")

bds_on <- function(x, name) {
  ar_fit <- ar(x, order.max = 8, method = "yule-walker", aic = TRUE)
  resid <- na.omit(ar_fit$resid)
  cat("\nBDS for AR-filtered ", name, " (AR order ", ar_fit$order, "):\n", sep = "")
  bt <- tseries::bds.test(as.numeric(resid), m = 5,
                          eps = seq(0.5, 2, length.out = 4) * sd(resid))
  print(bt)
  invisible(bt)
}
bds_cpi <- bds_on(diff(df$log_CPI),    "d.log CPI")
bds_eng <- bds_on(diff(df$log_ENERGY), "d.log ENERGY")
cat("\nInterpretation: small BDS p-values (dims 2-5) reject i.i.d. residuals => ",
    "nonlinear structure, supporting a threshold model over a linear VECM.\n", sep = "")

# --------------------------------------------------------------------------- #
section("Step 4: Hansen-Seo threshold cointegration test")

run_hs <- function(boot_type) {
  tsDyn::TVECM.HStest(series, lag = 2, ngridTh = 300, trim = 0.05,
                      nboot = nboot, intercept = TRUE, boot.type = boot_type)
}
get_stat <- function(t) { for (n in c("stat","SupLM","statistic")) if (!is.null(t[[n]])) return(as.numeric(t[[n]])[1]); NA_real_ }
get_pval <- function(t) { for (n in c("PvalBoot","p.value","pval")) if (!is.null(t[[n]])) return(as.numeric(t[[n]])[1]); NA_real_ }

cat("H0: linear cointegration vs H1: threshold cointegration.\n")
cat("Settings: lag=2, ngridTh=300, trim=0.05, nboot=", nboot, " (x2 bootstraps).\n\n", sep = "")
cat("Running fixed-regressor bootstrap...\n")
hs_fixed <- run_hs("FixedReg"); print(hs_fixed)
cat("Running residual bootstrap...\n")
hs_resid <- run_hs("ResBoot");  print(hs_resid)
suplm <- get_stat(hs_fixed)
p_fixed <- get_pval(hs_fixed); p_resid <- get_pval(hs_resid)
cat("\nSupLM statistic           : ", round(suplm, 4), "\n", sep = "")
cat("Fixed-regressor bootstrap p: ", p_fixed, "\n", sep = "")
cat("Residual bootstrap p       : ", p_resid, "\n", sep = "")
cat("Verdict @5%: ", ifelse(!is.na(p_fixed) && p_fixed < 0.05,
    "REJECT linear cointegration (threshold model preferred)",
    "fail to reject linear"), "\n", sep = "")

# --------------------------------------------------------------------------- #
section("Step 5: TVECM estimation (two regimes)")

tvecm_fit <- tsDyn::TVECM(series, lag = 2, nthresh = 1, trim = 0.05,
                          ngridBeta = 300, ngridTh = 300, plot = FALSE,
                          include = "const", trace = FALSE)
print(summary(tvecm_fit))

ms <- tvecm_fit$model.specific
threshold <- as.numeric(ms$Thresh)[1]
beta_raw  <- as.numeric(ms$coint)        # (log_CPI, log_ENERGY) cointegrating vector
beta_norm <- beta_raw / beta_raw[1]      # normalise on log_CPI
beta_energy <- -beta_norm[2]             # ECT = log_CPI - beta_energy*log_ENERGY
cat("\nEstimated threshold gamma : ", round(threshold, 4), "\n", sep = "")
cat("Cointegrating vector (raw): ", paste(round(beta_raw, 4), collapse = ", "), "\n", sep = "")
cat("Normalised beta (ENERGY)  : ", round(beta_energy, 4),
    "   => ECT = log_CPI - ", round(beta_energy, 4), " * log_ENERGY\n", sep = "")
cat("Long-run energy->CPI elasticity (beta): ", round(beta_energy, 4), "\n", sep = "")

regime_id <- tryCatch(tsDyn::regime(tvecm_fit), error = function(e) NULL)
if (!is.null(regime_id)) {
  rt <- table(regime_id); pr <- round(prop.table(rt), 4)
  cat("\nRegime counts:\n"); print(rt)
  cat("Regime proportions:\n"); print(pr)
}
cat("\nAdjustment speeds (alpha) by regime (from summary above):\n")
cat("  Read the 'ECT' rows: which equation (CPI vs ENERGY) carries the\n",
    "  significant, correctly-signed error-correction term identifies which\n",
    "  variable bears the adjustment burden.\n", sep = "")
coefs <- tryCatch(coef(tvecm_fit), error = function(e) NULL)
if (!is.null(coefs)) { cat("\nTVECM coefficient matrices:\n"); print(coefs) }

# --------------------------------------------------------------------------- #
section("Step 6: Demeaned-ECT regime classification plot")

ect_full <- df$log_CPI - beta_energy * df$log_ENERGY
train_df <- read.csv(train_file, stringsAsFactors = FALSE)
train_n  <- nrow(train_df)
train_mean <- mean(ect_full[seq_len(train_n)])
ect_dm <- ect_full - train_mean
cat("Training ECT mean (demeaning constant): ", round(train_mean, 6), "\n", sep = "")
cat("Demeaned ECT = log_CPI - ", round(beta_energy, 4), " * log_ENERGY - ", round(train_mean, 6), "\n", sep = "")
# Turbulent = ECT at/below threshold (the extreme large-negative-deviation
# regime), matching the US convention in regime_detection.py (ECT <= gamma).
turbulent <- ect_dm <= threshold
cat("Threshold gamma: ", round(threshold, 4), "\n", sep = "")
cat("Turbulent obs (ECT_dm <= gamma): ", sum(turbulent), " (", round(mean(turbulent), 4), ")\n", sep = "")
cat("Normal obs    (ECT_dm >  gamma): ", sum(!turbulent), " (", round(mean(!turbulent), 4), ")\n", sep = "")

png(ect_plot_file, width = 1200, height = 700, res = 110)
plot(df$date, ect_dm, type = "l", col = "gray25", lwd = 1.3,
     xlab = "Date", ylab = "Demeaned error-correction term",
     main = "Canada TVECM: Demeaned ECT and Estimated Threshold")
abline(h = threshold, col = "red", lwd = 2, lty = 2)
points(df$date[turbulent], ect_dm[turbulent], col = "firebrick", pch = 16, cex = 0.5)
legend("topleft", bty = "n",
       legend = c("Demeaned ECT", "Threshold gamma", "Turbulent (ECT <= gamma)"),
       col = c("gray25", "red", "firebrick"), lty = c(1, 2, NA),
       pch = c(NA, NA, 16), lwd = c(1.3, 2, NA))
dev.off()
cat("Saved ", ect_plot_file, "\n", sep = "")

# --------------------------------------------------------------------------- #
section("Step 7: Out-of-sample validation (VECM vs AR(2))")

mk <- function(d) {
  d$date <- as.Date(d$date)
  data.frame(date = d$date, log_CPI = log(d$CPI), log_ENERGY = log(d$ENERGY))
}
train <- mk(read.csv(train_file, stringsAsFactors = FALSE))
test  <- mk(read.csv(test_file,  stringsAsFactors = FALSE))
combined <- rbind(train, test)
logv <- as.matrix(combined[, c("log_CPI", "log_ENERGY")])
ect_c <- logv[, 1] - beta_energy * logv[, 2] - train_mean
diffs <- diff(logv)                        # row i = logv[i+1]-logv[i]
nC <- nrow(logv); train_size <- nrow(train)

# Build training design: Dy_t = c + a*ect_{t-1} + G1*Dy_{t-1} + G2*Dy_{t-2}
Xr <- list(); Yr <- list(); k <- 0
for (t in 4:train_size) {                  # t is 1-based row index into logv
  k <- k + 1
  Xr[[k]] <- c(1, ect_c[t - 1], diffs[t - 2, 1], diffs[t - 2, 2], diffs[t - 3, 1], diffs[t - 3, 2])
  Yr[[k]] <- diffs[t - 1, ]
}
X <- do.call(rbind, Xr); Y <- do.call(rbind, Yr)
vecm_coef <- solve(t(X) %*% X, t(X) %*% Y)  # 6x2; column 1 = CPI equation

# AR(2) on training d.log CPI
dcpi <- diff(train$log_CPI)
Xa <- list(); Ya <- list(); k <- 0
for (t in 3:length(dcpi)) { k <- k + 1; Xa[[k]] <- c(1, dcpi[t - 1], dcpi[t - 2]); Ya[[k]] <- dcpi[t] }
Xa <- do.call(rbind, Xa); ar2_coef <- solve(t(Xa) %*% Xa, t(Xa) %*% do.call(c, Ya))

vecm_step <- function(hist) {
  last <- hist[[length(hist)]]
  d1 <- hist[[length(hist)]] - hist[[length(hist) - 1]]
  d2 <- hist[[length(hist) - 1]] - hist[[length(hist) - 2]]
  ect_last <- last[1] - beta_energy * last[2] - train_mean
  xt <- c(1, ect_last, d1[1], d1[2], d2[1], d2[2])
  as.numeric(last + xt %*% vecm_coef)
}
ar2_step <- function(h) {
  d1 <- h[length(h)] - h[length(h) - 1]; d2 <- h[length(h) - 1] - h[length(h) - 2]
  h[length(h)] + sum(c(1, d1, d2) * ar2_coef)
}

# h-step recursive forecasts
horizon_fc <- function(horizon) {
  rows <- list(); k <- 0; n_test <- nC - train_size
  if (n_test <= horizon) return(NULL)
  for (off in 0:(n_test - horizon - 1)) {
    origin <- train_size + off            # 1-based row index (forecast origin)
    target <- origin + horizon
    vh <- lapply((origin - 2):origin, function(i) logv[i, ])
    for (s in 1:horizon) vh[[length(vh) + 1]] <- vecm_step(vh)
    ah <- as.list(combined$log_CPI[(origin - 2):origin])
    for (s in 1:horizon) ah[[length(ah) + 1]] <- ar2_step(unlist(ah))
    k <- k + 1
    rows[[k]] <- data.frame(horizon = horizon,
      target_date = combined$date[target], actual = exp(logv[target, 1]),
      vecm = exp(vh[[length(vh)]][1]), ar2 = exp(ah[[length(ah)]]),
      prev = exp(logv[origin, 1]))
  }
  do.call(rbind, rows)
}

hln_dm <- function(loss_diff, horizon) {
  v <- as.numeric(loss_diff); n <- length(v); if (n < 2) return(c(NA, NA))
  cen <- v - mean(v); g0 <- mean(cen * cen); lrv <- g0
  maxlag <- min(horizon - 1, n - 1)
  if (maxlag >= 1) for (l in 1:maxlag) {
    ac <- mean(cen[(l + 1):n] * cen[1:(n - l)]); w <- 1 - l / (maxlag + 1); lrv <- lrv + 2 * w * ac
  }
  if (lrv <= 0 || !is.finite(lrv)) return(c(NA, NA))
  dm <- mean(v) / sqrt(lrv / n)
  hln <- (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
  if (hln <= 0) return(c(NA, NA))
  dmh <- dm * sqrt(hln); p <- 2 * pt(-abs(dmh), df = n - 1)
  c(dmh, p)
}

cat("Forecast origin split at 2022-05-01 (train n=", train_size, ", test n=", nC - train_size, ").\n\n", sep = "")
metrics_rows <- list()
for (h in c(1, 3)) {
  fc <- horizon_fc(h); if (is.null(fc)) next
  ve <- fc$vecm - fc$actual; ae <- fc$ar2 - fc$actual
  v_rmse <- sqrt(mean(ve^2)); a_rmse <- sqrt(mean(ae^2))
  v_mae <- mean(abs(ve)); a_mae <- mean(abs(ae))
  dir_v <- mean(sign(fc$actual - fc$prev) == sign(fc$vecm - fc$prev)) * 100
  dir_a <- mean(sign(fc$actual - fc$prev) == sign(fc$ar2 - fc$prev)) * 100
  dm <- hln_dm(ve^2 - ae^2, h)
  imp <- (a_rmse - v_rmse) / a_rmse * 100
  cat("--- Horizon h=", h, " (n=", nrow(fc), " pairs) ---\n", sep = "")
  cat(sprintf("  VECM RMSE %.4f | AR2 RMSE %.4f | RMSE improvement %+.2f%%\n", v_rmse, a_rmse, imp))
  cat(sprintf("  VECM MAE  %.4f | AR2 MAE  %.4f\n", v_mae, a_mae))
  cat(sprintf("  Direction acc: VECM %.1f%% | AR2 %.1f%%\n", dir_v, dir_a))
  cat(sprintf("  HLN Diebold-Mariano stat %.4f, p-value %.4f %s\n\n", dm[1], dm[2],
              ifelse(!is.na(dm[2]) && dm[2] < 0.05, "(*** significant @5%)", "")))
  metrics_rows[[as.character(h)]] <- data.frame(h = h, v_rmse, a_rmse, imp, dir_v, dir_a, dm_stat = dm[1], dm_p = dm[2])
}

# 1-step forecast chart
fc1 <- horizon_fc(1)
if (!is.null(fc1)) {
  png(fcst_plot_file, width = 1200, height = 700, res = 110)
  ylim <- range(c(fc1$actual, fc1$vecm, fc1$ar2))
  plot(fc1$target_date, fc1$actual, type = "l", col = "black", lwd = 2, ylim = ylim,
       xlab = "Date", ylab = "CPI (index level)",
       main = "Canada VECM Out-of-Sample CPI Forecast (1-step): 2022-2026")
  lines(fc1$target_date, fc1$vecm, col = "blue", lwd = 2, lty = 1)
  lines(fc1$target_date, fc1$ar2, col = "firebrick", lwd = 2, lty = 2)
  legend("topleft", bty = "n", legend = c("Actual CPI", "VECM forecast", "AR(2) forecast"),
         col = c("black", "blue", "firebrick"), lwd = 2, lty = c(1, 1, 2))
  dev.off()
  cat("Saved ", fcst_plot_file, "\n", sep = "")
}

# --------------------------------------------------------------------------- #
section("SUMMARY (paste-ready)")
cat("Canada TVECM headline results:\n")
cat("  Cointegrating beta (energy->CPI elasticity): ", round(beta_energy, 4), "\n", sep = "")
cat("  Threshold gamma                            : ", round(threshold, 4), "\n", sep = "")
cat("  SupLM statistic                            : ", round(suplm, 4), "\n", sep = "")
cat("  Bootstrap p-values (fixed / resid)         : ", p_fixed, " / ", p_resid, "\n", sep = "")
if (!is.null(regime_id)) {
  pr <- round(prop.table(table(regime_id)), 4)
  cat("  Regime split                               : ", paste(paste0(names(pr), "=", pr), collapse = ", "), "\n", sep = "")
}
for (h in names(metrics_rows)) {
  m <- metrics_rows[[h]]
  cat(sprintf("  h=%s: RMSE improvement %+.2f%%, DM p=%.4f\n", h, m$imp, m$dm_p))
}
cat("\nDone.\n")
