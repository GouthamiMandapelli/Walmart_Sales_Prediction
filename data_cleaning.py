"""
Walmart Sales Prediction — Data Cleaning Script
================================================
Stage: Data Cleaning only.
Original CSV files are never modified.
Outputs: train_cleaned.csv, test_cleaned.csv
"""

import pandas as pd
import numpy as np

SEP = "=" * 70

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load original CSV files
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1 — Loading original CSV files")
print(SEP)

train  = pd.read_csv("train.csv")
test   = pd.read_csv("test.csv")
feat   = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

print(f"  train   : {train.shape[0]:>7,} rows x {train.shape[1]} cols")
print(f"  test    : {test.shape[0]:>7,} rows x {test.shape[1]} cols")
print(f"  features: {feat.shape[0]:>7,} rows x {feat.shape[1]} cols")
print(f"  stores  : {stores.shape[0]:>7,} rows x {stores.shape[1]} cols")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Convert Date columns to datetime
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 — Converting Date columns to datetime")
print(SEP)

for df, name in [(train, "train"), (test, "test"), (feat, "features")]:
    before_dtype = df["Date"].dtype
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
    after_dtype = df["Date"].dtype
    print(f"  {name:10s}: Date dtype  {str(before_dtype):10s}  ->  {str(after_dtype)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Verify Date conversion
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 — Verifying Date conversion")
print(SEP)

for df, name in [(train, "train"), (test, "test"), (feat, "features")]:
    nat_count = df["Date"].isna().sum()
    print(f"  {name:10s}: dtype={df['Date'].dtype}  NaT count={nat_count}"
          f"  range=[{df['Date'].min().date()}  ->  {df['Date'].max().date()}]")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Verify merge keys; check for unexpected duplicates
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 — Verifying merge keys before merging")
print(SEP)

# 4a. features.csv should have exactly one row per (Store, Date)
feat_dups = feat.duplicated(subset=["Store", "Date"]).sum()
print(f"  features duplicate (Store+Date) keys : {feat_dups}")

# 4b. stores.csv should have exactly one row per Store
stores_dups = stores.duplicated(subset=["Store"]).sum()
print(f"  stores   duplicate Store keys        : {stores_dups}")

# 4c. Every Store value in train/test should exist in features
train_stores_missing = set(train["Store"].unique()) - set(feat["Store"].unique())
test_stores_missing  = set(test["Store"].unique())  - set(feat["Store"].unique())
print(f"  train Stores missing from features   : {train_stores_missing or 'none'}")
print(f"  test  Stores missing from features   : {test_stores_missing  or 'none'}")

# 4d. Every Store in train/test should exist in stores.csv
train_stores_missing_s = set(train["Store"].unique()) - set(stores["Store"].unique())
test_stores_missing_s  = set(test["Store"].unique())  - set(stores["Store"].unique())
print(f"  train Stores missing from stores.csv : {train_stores_missing_s or 'none'}")
print(f"  test  Stores missing from stores.csv : {test_stores_missing_s  or 'none'}")

# 4e. Check train+test Date coverage vs features Date coverage
train_dates_missing = set(train["Date"].dt.date.unique()) - set(feat["Date"].dt.date.unique())
test_dates_missing  = set(test["Date"].dt.date.unique())  - set(feat["Date"].dt.date.unique())
print(f"  train Dates missing from features    : {len(train_dates_missing)} date(s)")
print(f"  test  Dates missing from features    : {len(test_dates_missing)} date(s)")
if train_dates_missing:
    print(f"    Unmatched train dates: {sorted(train_dates_missing)}")
if test_dates_missing:
    print(f"    Unmatched test  dates: {sorted(test_dates_missing)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Merge train + features  (left join on Store + Date)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 — Merging train with features (Store + Date)")
print(SEP)

rows_before_train = len(train)
train_merged = train.merge(feat, on=["Store", "Date"], how="left", suffixes=("_sales", "_feat"))
rows_after_train = len(train_merged)

print(f"  train rows before merge : {rows_before_train:>7,}")
print(f"  train rows after  merge : {rows_after_train:>7,}")
print(f"  rows gained / lost      : {rows_after_train - rows_before_train:>+7,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Merge result + stores  (left join on Store)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 6 — Merging train_merged with stores (Store)")
print(SEP)

rows_before_stores = len(train_merged)
train_merged = train_merged.merge(stores, on="Store", how="left")
rows_after_stores = len(train_merged)

print(f"  rows before stores merge : {rows_before_stores:>7,}")
print(f"  rows after  stores merge : {rows_after_stores:>7,}")
print(f"  rows gained / lost       : {rows_after_stores - rows_before_stores:>+7,}")
print(f"  final train_merged shape : {train_merged.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Equivalent merge for test data
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 7 — Merging test with features + stores")
print(SEP)

rows_before_test = len(test)
test_merged = test.merge(feat, on=["Store", "Date"], how="left", suffixes=("_sales", "_feat"))
test_merged = test_merged.merge(stores, on="Store", how="left")
rows_after_test = len(test_merged)

print(f"  test rows before merge : {rows_before_test:>7,}")
print(f"  test rows after  merge : {rows_after_test:>7,}")
print(f"  rows gained / lost     : {rows_after_test - rows_before_test:>+7,}")
print(f"  final test_merged shape: {test_merged.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Verify merge quality
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 8 — Verifying merge quality")
print(SEP)

# 8a. Duplicate Store+Dept+Date rows
train_key_dups = train_merged.duplicated(subset=["Store", "Dept", "Date"]).sum()
test_key_dups  = test_merged.duplicated(subset=["Store", "Dept", "Date"]).sum()
print(f"  train_merged duplicate (Store+Dept+Date) : {train_key_dups}")
print(f"  test_merged  duplicate (Store+Dept+Date) : {test_key_dups}")

# 8b. NaN in Temperature after merge (proxy for unmatched Store+Date in features)
train_temp_null = train_merged["Temperature"].isna().sum()
test_temp_null  = test_merged["Temperature"].isna().sum()
print(f"  train_merged Temperature NaN (unmatched rows) : {train_temp_null}")
print(f"  test_merged  Temperature NaN (unmatched rows) : {test_temp_null}")

# 8c. NaN in Type/Size after merge (unmatched Store in stores.csv)
train_type_null = train_merged["Type"].isna().sum()
test_type_null  = test_merged["Type"].isna().sum()
print(f"  train_merged Type NaN (unmatched Store)       : {train_type_null}")
print(f"  test_merged  Type NaN (unmatched Store)       : {test_type_null}")

# 8d. Confirm original row count preserved
rows_ok = (rows_after_train == rows_before_train) and (rows_after_test == rows_before_test)
print(f"  Row count preserved (no fan-out from merge)  : {rows_ok}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Resolve duplicate IsHoliday columns
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 9 — Resolving duplicate IsHoliday columns")
print(SEP)

# After merge with suffixes, the two columns are:
#   IsHoliday_sales (from train/test)
#   IsHoliday_feat  (from features)

def resolve_isholiday(df, label):
    col_sales = "IsHoliday_sales"
    col_feat  = "IsHoliday_feat"

    if col_sales not in df.columns or col_feat not in df.columns:
        # No conflict — only one IsHoliday column exists
        print(f"  {label}: only one IsHoliday column found — no conflict to resolve")
        return df

    total_rows   = len(df)
    match_count  = (df[col_sales] == df[col_feat]).sum()
    mismatch_count = total_rows - match_count

    print(f"  {label}:")
    print(f"    Total rows              : {total_rows:,}")
    print(f"    Matching IsHoliday rows : {match_count:,}")
    print(f"    MISMATCHING rows        : {mismatch_count:,}")

    if mismatch_count > 0:
        print(f"    *** WARNING: {mismatch_count} mismatches found — inspecting: ***")
        mismatches = df[df[col_sales] != df[col_feat]][[
            "Store", "Dept", "Date", col_sales, col_feat
        ]].head(10)
        print(mismatches.to_string(index=False))
    else:
        print(f"    OK All IsHoliday values agree between sales and features tables.")
        print(f"    -> Keeping IsHoliday_sales; dropping IsHoliday_feat.")
        df = df.rename(columns={col_sales: "IsHoliday"})
        df = df.drop(columns=[col_feat])

    return df

train_merged = resolve_isholiday(train_merged, "train_merged")
test_merged  = resolve_isholiday(test_merged,  "test_merged")

print()
print(f"  train_merged columns now: {list(train_merged.columns)}")
print(f"  test_merged  columns now: {list(test_merged.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Investigate MarkDown missing value temporal pattern
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 10 — Investigating MarkDown missing value temporal pattern")
print(SEP)

md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# Use original features (not merged) for cleaner temporal analysis
feat_sorted = feat.sort_values("Date")

print("  MarkDown presence/absence by year:")
print()

for c in md_cols:
    feat_sorted["is_null"] = feat_sorted[c].isna()
    by_year = feat_sorted.groupby(feat_sorted["Date"].dt.year)["is_null"].agg(
        total="count", missing="sum"
    )
    by_year["present"] = by_year["total"] - by_year["missing"]
    by_year["pct_missing"] = (by_year["missing"] / by_year["total"] * 100).round(1)
    print(f"  {c}:")
    print(by_year[["total", "present", "missing", "pct_missing"]].to_string())
    print()

# Find the first date a non-null MarkDown value appears (per column)
print("  Earliest non-null date per MarkDown column (in features.csv):")
for c in md_cols:
    first_valid = feat_sorted.loc[feat_sorted[c].notna(), "Date"].min()
    print(f"    {c}: first non-null date = {first_valid.date()}")

feat_sorted = feat_sorted.drop(columns=["is_null"])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Fill MarkDown NaN with 0
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 11 — Filling MarkDown NaN with 0")
print(SEP)

print("  Rationale: MarkDown columns are missing exclusively (or overwhelmingly)")
print("  in the early period before Walmart introduced promotional markdowns.")
print("  A missing value here means 'no promotion was active', which is")
print("  economically equivalent to a $0 markdown — not 'data unknown'.")
print("  Filling with 0 is defensible and avoids imputing a non-zero promotion.")

for df, label in [(train_merged, "train_merged"), (test_merged, "test_merged")]:
    for c in md_cols:
        if c in df.columns:
            n_filled = df[c].isna().sum()
            df[c] = df[c].fillna(0)
            print(f"  {label:15s} {c}: filled {n_filled:,} NaN -> 0")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — CPI and Unemployment: leakage-safe imputation
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 12 — CPI and Unemployment missing-value treatment")
print(SEP)

econ_cols = ["CPI", "Unemployment"]

print("  Inspection — where are the missing values?")
for c in econ_cols:
    for df, label in [(train_merged, "train_merged"), (test_merged, "test_merged")]:
        if c in df.columns:
            missing_rows = df[df[c].isna()]
            if len(missing_rows) == 0:
                print(f"    {label:15s} {c}: 0 missing rows")
            else:
                print(f"    {label:15s} {c}: {len(missing_rows):,} missing rows  "
                      f"date range [{missing_rows['Date'].min().date()} -> "
                      f"{missing_rows['Date'].max().date()}]")

print()
print("  Method: Forward-fill within each Store group, sorted by Date.")
print("  This carries the last known value forward in time, using only")
print("  past/present information — no future data is used.")
print("  The fill is applied store-by-store to avoid cross-store contamination.")

for df, label in [(train_merged, "train_merged"), (test_merged, "test_merged")]:
    for c in econ_cols:
        if c in df.columns:
            before = df[c].isna().sum()
            # Sort by Store then Date before ffill
            df.sort_values(["Store", "Date"], inplace=True)
            df[c] = df.groupby("Store")[c].transform(
                lambda s: s.ffill()
            )
            after = df[c].isna().sum()
            print(f"  {label:15s} {c}: {before:,} NaN before ffill -> {after:,} after")

# If any remain (e.g., a store has NO prior CPI value at all), back-fill as last resort
for df, label in [(train_merged, "train_merged"), (test_merged, "test_merged")]:
    for c in econ_cols:
        if c in df.columns:
            remaining = df[c].isna().sum()
            if remaining > 0:
                df[c] = df.groupby("Store")[c].transform(lambda s: s.bfill())
                after_bfill = df[c].isna().sum()
                print(f"  {label:15s} {c}: back-filled {remaining:,} residual NaN "
                      f"(store with no prior history) -> {after_bfill:,} remaining")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — Negative Weekly_Sales: keep, add note
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 13 — Negative Weekly_Sales (inspection only — no changes)")
print(SEP)

neg_mask = train_merged["Weekly_Sales"] < 0
n_neg    = neg_mask.sum()
print(f"  Negative Weekly_Sales rows : {n_neg:,}  ({n_neg/len(train_merged)*100:.3f}%)")
print(f"  Min value                  : {train_merged['Weekly_Sales'].min():.2f}")
print(f"  Max negative value         : {train_merged.loc[neg_mask, 'Weekly_Sales'].max():.2f}")
print()
print("  Decision: KEEP negative values unchanged.")
print("  These represent weeks where returns/refunds exceeded gross sales")
print("  for a given department — a legitimate business event.")
print("  Any transformation (clipping, log) will be decided in feature-engineering.")

n_zero = (train_merged["Weekly_Sales"] == 0).sum()
print(f"  Zero Weekly_Sales rows     : {n_zero:,}")
print("  Decision: KEEP zero values unchanged.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — Investigate negative MarkDown values
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 14 — Investigating negative MarkDown values")
print(SEP)

for c in md_cols:
    for df, label in [(train_merged, "train_merged"), (test_merged, "test_merged")]:
        if c in df.columns:
            neg_md = df[df[c] < 0]
            if len(neg_md) > 0:
                print(f"  {label:15s} {c}: {len(neg_md):,} negative rows  "
                      f"min={neg_md[c].min():.2f}  max={neg_md[c].max():.2f}")
                sample = neg_md[["Store", "Dept", "Date", c]].head(5)
                print(sample.to_string(index=False))
                print()
            else:
                print(f"  {label:15s} {c}: 0 negative rows")

print()
print("  Decision: KEEP negative MarkDown values unchanged.")
print("  The counts are very small (< 45 rows per column out of 421k+ rows).")
print("  These may represent promotion reversals or data-entry corrections.")
print("  Clipping them without further business context would introduce bias.")
print("  They will be revisited in feature engineering if needed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — Confirm extreme values are kept
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 15 — Confirming extreme values are preserved")
print(SEP)

print(f"  Weekly_Sales max  : {train_merged['Weekly_Sales'].max():,.2f}  (kept)")
print(f"  MarkDown5 max     : {train_merged['MarkDown5'].max():,.2f}  (kept)")
print(f"  MarkDown3 max     : {train_merged['MarkDown3'].max():,.2f}  (kept)")
print("  These are legitimate high-volume sale/promotion weeks — not errors.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — Final quality checks
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 16 — Final quality checks")
print(SEP)

for df, label in [(train_merged, "train_merged"), (test_merged, "test_merged")]:
    print(f"  [{label}]")
    print(f"    Shape          : {df.shape}")

    # Missing values
    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    if has_missing.empty:
        print("    Remaining NaN  : NONE OK")
    else:
        for col, cnt in has_missing.items():
            print(f"    Remaining NaN  : {col} = {cnt:,}  ({cnt/len(df)*100:.2f}%)")

    # Duplicates
    dups = df.duplicated(subset=["Store", "Dept", "Date"]).sum()
    print(f"    Duplicates (Store+Dept+Date) : {dups}")

    # Invalid dates
    nat_count = df["Date"].isna().sum()
    print(f"    Invalid (NaT) dates          : {nat_count}")

    # Date column dtype
    print(f"    Date dtype                   : {df['Date'].dtype}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — Save cleaned datasets
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STEP 17 — Saving cleaned datasets")
print(SEP)

# Restore index order before saving
train_merged = train_merged.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
test_merged  = test_merged.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

train_merged.to_csv("train_cleaned.csv", index=False)
test_merged.to_csv("test_cleaned.csv",   index=False)

print(f"  Saved train_cleaned.csv  -> {len(train_merged):,} rows x {train_merged.shape[1]} cols")
print(f"  Saved test_cleaned.csv   -> {len(test_merged):,} rows x {test_merged.shape[1]} cols")
print()
print("  Columns in train_cleaned.csv:")
for c in train_merged.columns:
    print(f"    {c}: {train_merged[c].dtype}")
print()
print("  Columns in test_cleaned.csv:")
for c in test_merged.columns:
    print(f"    {c}: {test_merged[c].dtype}")

print()
print(SEP)
print("DATA CLEANING COMPLETE — original CSV files were NOT modified.")
print(SEP)
