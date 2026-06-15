#!/usr/bin/env Rscript

# Build a monthly FRED dataset with energy price variants and CPI measures.
#
# Required setup:
#   Sys.setenv(FRED_API_KEY = "your_fred_api_key")
# or from the shell:
#   export FRED_API_KEY="your_fred_api_key"
#
# Notes on MEDCPIM158SFRBCLE:
#   The Cleveland Fed Median CPI series is usually published by FRED as an
#   annualized percent-change rate, not an index level. This script keeps that
#   FRED value in MEDIAN_CPI and prints the FRED units/notes so you can verify
#   it. Because it is a rate, it should generally be tested for stationarity
#   directly rather than treated as an index level. Core CPI (CPILFESL) is kept
#   as an index level.

packages <- c("fredr", "dplyr", "lubridate", "readr")
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(fredr)
  library(dplyr)
  library(lubridate)
  library(readr)
})

api_key <- Sys.getenv("FRED_API_KEY")
if (!nzchar(api_key)) {
  stop(
    "FRED_API_KEY is not set. Set it with Sys.setenv(FRED_API_KEY = '...') ",
    "or export FRED_API_KEY='...' before running this script."
  )
}
fredr_set_key(api_key)

output_path <- "energy_cpi_variations.csv"

series <- tibble::tribble(
  ~series_id,             ~label,       ~type,
  "DCOILBRENTEU",         "BRENT",      "daily_energy",
  "DCOILWTICO",           "WTI",        "daily_energy",
  "CPILFESL",             "CORE_CPI",   "monthly_level",
  "MEDCPIM158SFRBCLE",    "MEDIAN_CPI", "monthly_rate"
)

fetch_series_info <- function(sid) {
  info <- fredr_series(series_id = sid)
  if ("id" %in% names(info) && !"series_id" %in% names(info)) {
    info <- info |> rename(series_id = id)
  }
  if (!"series_id" %in% names(info)) {
    info$series_id <- sid
  }
  if (!"notes" %in% names(info)) {
    info$notes <- NA_character_
  }

  info |>
    mutate(
      notes_preview = substr(gsub("\\s+", " ", notes), 1, 220)
    ) |>
    select(
      all_of("series_id"),
      any_of(c(
        "title",
        "frequency",
        "units",
        "units_short",
        "seasonal_adjustment",
        "observation_start",
        "observation_end",
        "notes_preview"
      ))
    )
}

fetch_observations <- function(series_id) {
  fredr(series_id = series_id) |>
    select(date, value) |>
    filter(!is.na(value))
}

make_daily_monthly_variants <- function(series_id, prefix) {
  daily <- fetch_observations(series_id) |>
    mutate(month = floor_date(date, "month"))

  eom <- daily |>
    arrange(month, date) |>
    group_by(month) |>
    slice_tail(n = 1) |>
    ungroup() |>
    transmute(date = month, !!paste0(prefix, "_eom") := value)

  avg <- daily |>
    group_by(month) |>
    summarise(
      !!paste0(prefix, "_avg") := mean(value, na.rm = TRUE),
      .groups = "drop"
    ) |>
    rename(date = month)

  inner_join(eom, avg, by = "date")
}

make_monthly_series <- function(series_id, column_name) {
  fetch_observations(series_id) |>
    transmute(
      date = floor_date(date, "month"),
      !!column_name := value
    ) |>
    group_by(date) |>
    summarise(
      !!column_name := dplyr::last(.data[[column_name]]),
      .groups = "drop"
    )
}

metadata <- bind_rows(lapply(series$series_id, fetch_series_info)) |>
  left_join(series, by = "series_id") |>
  relocate(series_id, label, type)

brent_monthly <- make_daily_monthly_variants("DCOILBRENTEU", "BRENT")
wti_monthly <- make_daily_monthly_variants("DCOILWTICO", "WTI")
core_cpi <- make_monthly_series("CPILFESL", "CORE_CPI")
median_cpi <- make_monthly_series("MEDCPIM158SFRBCLE", "MEDIAN_CPI")

merged <- Reduce(
  function(x, y) inner_join(x, y, by = "date"),
  list(core_cpi, median_cpi, brent_monthly, wti_monthly)
) |>
  arrange(date) |>
  select(date, CORE_CPI, MEDIAN_CPI, BRENT_eom, BRENT_avg, WTI_eom, WTI_avg)

if (nrow(merged) == 0) {
  stop("No overlapping monthly observations were found after merging all series.")
}

write_csv(merged, output_path)

median_meta <- metadata |> filter(series_id == "MEDCPIM158SFRBCLE")
median_units_text <- paste(
  median_meta$units,
  median_meta$units_short,
  median_meta$title,
  sep = " | "
)
median_is_rate <- grepl("percent|rate|change", median_units_text, ignore.case = TRUE)

cat("\nFRED series metadata / units\n")
cat("============================\n")
metadata_to_print <- metadata |>
  mutate(
    observation_start = as.character(observation_start),
    observation_end = as.character(observation_end)
  )
print(metadata_to_print, n = Inf, width = Inf)

cat("\nMedian CPI handling\n")
cat("===================\n")
if (median_is_rate) {
  cat(
    "MEDCPIM158SFRBCLE appears to be a rate/percent-change series based on FRED metadata.\n",
    "This script keeps the downloaded rate as MEDIAN_CPI and does NOT treat it as an index level.\n",
    "For stationarity work, test MEDIAN_CPI directly; it may already be I(0).\n",
    sep = ""
  )
} else {
  cat(
    "MEDCPIM158SFRBCLE metadata does not look like a rate/percent-change series; ",
    "verify the units above before treating MEDIAN_CPI as an index level.\n",
    sep = ""
  )
}

cat("\nFirst 3 rows\n")
cat("============\n")
print(head(merged, 3))

cat("\nLast 3 rows\n")
cat("===========\n")
print(tail(merged, 3))

cat("\nUsable merged monthly sample\n")
cat("============================\n")
cat("Start date:", format(min(merged$date)), "\n")
cat("End date:  ", format(max(merged$date)), "\n")
cat("Row count: ", nrow(merged), "\n", sep = "")
cat("Saved CSV: ", normalizePath(output_path, mustWork = FALSE), "\n", sep = "")
