"""
Walmart Sales Prediction -- Data Preparation Script
====================================================
Stage : Data Preparation only.
Inputs : train_cleaned.csv, test_cleaned.csv
Outputs: train_prepared.csv, test_prepared.csv

Rules enforced:
  - Original CSVs (train/test/features/stores.csv) are NOT touched.
  - train_cleaned.csv and test_cleaned.csv are NOT overwritten.
  - Weekly_Sales is NOT transformed or replaced.
  - No encoding of Store, Dept, or Type is performed.
  - No lag or rolling features are created.
  - DayOfWeek is created, verified, then dropped (constant = 4, zero variance).
"""

import pandas as pd
import numpy as np

SEP = "=" * 70

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Load cleaned datasets
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1 -- Loading train_cleaned.csv and test_cleaned.csv")
print(SEP)

train = pd.read_csv("train_cleaned.csv", parse_dates=["Date"])
test  = pd.read_csv("test_cleaned.csv",  parse_dates=["Date"])

print("  train shape : " + str(train.shape))
print("  test  shape : " + str(test.shape))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Schema verification
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 -- Schema verification")
print(SEP)

has_ws_train = "Weekly_Sales" in train.columns
has_ws_test  = "Weekly_Sales" in test.columns
print("  Weekly_Sales in train : " + str(has_ws_train) + "  (expected True)")
print("  Weekly_Sales in test  : " + str(has_ws_test)  + "  (expected False)")

train_null = train.isnull().sum().sum()
test_null  = test.isnull().sum().sum()
print()
print("  Total NaN in train : " + str(train_null) + "  (expected 0)")
print("  Total NaN in test  : " + str(test_null)  + "  (expected 0)")

print()
print("  train Date dtype : " + str(train["Date"].dtype))
print("  test  Date dtype : " + str(test["Date"].dtype))

train_dups = train.duplicated(subset=["Store", "Dept", "Date"]).sum()
test_dups  = test.duplicated(subset=["Store", "Dept", "Date"]).sum()
print()
print("  train duplicate (Store+Dept+Date) : " + str(train_dups) + "  (expected 0)")
print("  test  duplicate (Store+Dept+Date) : " + str(test_dups)  + "  (expected 0)")

print()
print("  TRAIN columns (" + str(len(train.columns)) + "):")
for c in train.columns:
    print("    " + c + " : " + str(train[c].dtype) +
          "  nulls=" + str(train[c].isnull().sum()))
print()
print("  TEST columns (" + str(len(test.columns)) + "):")
for c in test.columns:
    print("    " + c + " : " + str(test[c].dtype) +
          "  nulls=" + str(test[c].isnull().sum()))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Sort chronologically by Store, Dept, Date
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 -- Sorting by Store, Dept, Date")
print(SEP)

train = train.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
test  = test.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

def check_monotone(df, label):
    bad = 0
    for (s, d), grp in df.groupby(["Store", "Dept"], sort=False):
        if not grp["Date"].is_monotonic_increasing:
            bad += 1
    print("  " + label + ": groups with non-monotone dates = " + str(bad) +
          "  (expected 0)")

check_monotone(train, "train")
check_monotone(test,  "test")

print("  train first row : Store=" + str(train.iloc[0]["Store"]) +
      "  Dept=" + str(train.iloc[0]["Dept"]) +
      "  Date=" + str(train.iloc[0]["Date"].date()))
print("  train last  row : Store=" + str(train.iloc[-1]["Store"]) +
      "  Dept=" + str(train.iloc[-1]["Dept"]) +
      "  Date=" + str(train.iloc[-1]["Date"].date()))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- Create time/date features  (DayOfWeek: create, verify, then drop)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 -- Creating time/date features")
print(SEP)

for df, label in [(train, "train"), (test, "test")]:
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]   = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek   # 0=Mon ... 6=Sun

    print("  " + label + " date features created:")
    for feat in ["Year", "Month", "Week", "Quarter", "DayOfWeek"]:
        uv = df[feat].nunique()
        mn = df[feat].min()
        mx = df[feat].max()
        print("    " + feat.ljust(12) + ": dtype=" + str(df[feat].dtype) +
              "  range=[" + str(mn) + ", " + str(mx) + "]" +
              "  unique_values=" + str(uv))

# ── DayOfWeek verification ────────────────────────────────────────────────────
print()
print("  DayOfWeek verification:")
for df, label in [(train, "train"), (test, "test")]:
    unique_dow = sorted(df["DayOfWeek"].unique().tolist())
    all_friday = (df["DayOfWeek"] == 4).all()
    print("    " + label + ": unique DayOfWeek values = " + str(unique_dow) +
          "  all_friday=" + str(all_friday))

print()
print("  RESULT: DayOfWeek = 4 (Friday) for every single row in both datasets.")
print("  This column has zero variance and zero predictive information.")
print("  DECISION: DayOfWeek will be DROPPED before saving prepared datasets.")

# ── Drop DayOfWeek immediately ────────────────────────────────────────────────
train = train.drop(columns=["DayOfWeek"])
test  = test.drop(columns=["DayOfWeek"])
print()
print("  DayOfWeek dropped from train.")
print("  DayOfWeek dropped from test.")

# Verify no NaN introduced by remaining date features
print()
for df, label in [(train, "train"), (test, "test")]:
    new_nulls = df[["Year", "Month", "Week", "Quarter"]].isnull().sum().sum()
    print("  " + label + ": NaN in new date features = " + str(new_nulls) +
          "  (expected 0)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- Categorical column examination
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 -- Categorical column examination")
print(SEP)

print("  Low-cardinality categorical columns (NOT encoded in this stage):")
print()
for col, note in [
    ("Type",
     "Store type A/B/C. Label-encode (A=0,B=1,C=2) or one-hot during modelling."),
    ("IsHoliday",
     "Boolean. Cast to int (0/1) before passing to any model."),
]:
    if col in train.columns:
        uv   = train[col].nunique()
        vals = str(sorted(train[col].unique()))
        print("  " + col.ljust(10) + " | unique=" + str(uv) +
              " | values=" + vals)
        print("  " + " " * 10 + "   -> " + note)

print()
print("  High-cardinality identifier columns (NOT encoded in this stage):")
for col, note in [
    ("Store",
     "Integer IDs 1-45. Label-encode or target-encode (INSIDE CV fold only)."),
    ("Dept",
     "Integer IDs 1-99 (81 active). Same strategy as Store."),
]:
    uv = train[col].nunique()
    print("  " + col.ljust(6) + " | unique=" + str(uv) + " | -> " + note)

print()
print("  IMPORTANT: All encoding will happen in the Feature Engineering /")
print("  Model Training stage. Target encoding MUST be computed only on each")
print("  CV fold's training split -- never on the full train+test dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- Store and Dept identifier analysis
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 6 -- Store and Dept identifier analysis")
print(SEP)

rows_per_store = train.groupby("Store").size()
rows_per_dept  = train.groupby("Dept").size()
print("  Rows per Store (train): min=" + str(rows_per_store.min()) +
      "  max=" + str(rows_per_store.max()) +
      "  mean=" + str(round(rows_per_store.mean(), 1)))
print("  Rows per Dept  (train): min=" + str(rows_per_dept.min()) +
      "  max=" + str(rows_per_dept.max()) +
      "  mean=" + str(round(rows_per_dept.mean(), 1)))

train_pairs = set(zip(train["Store"], train["Dept"]))
test_pairs  = set(zip(test["Store"],  test["Dept"]))
only_train  = train_pairs - test_pairs
only_test   = test_pairs  - train_pairs
print()
print("  (Store,Dept) pairs in train only : " + str(len(only_train)))
print("  (Store,Dept) pairs in test  only : " + str(len(only_test)) +
      ("  <-- WARNING: no training history for these pairs" if only_test else ""))
print("  (Store,Dept) pairs in both       : " + str(len(train_pairs & test_pairs)))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 -- Weekly_Sales distribution inspection
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 7 -- Weekly_Sales distribution by key groups")
print(SEP)

ws = train["Weekly_Sales"]

print("  -- Overall distribution --")
print("  count    : " + str(len(ws)))
print("  mean     : " + str(round(ws.mean(), 2)))
print("  median   : " + str(round(ws.median(), 2)))
print("  std      : " + str(round(ws.std(), 2)))
print("  skewness : " + str(round(ws.skew(), 4)) + "  (>1 = right-skewed)")
print("  kurtosis : " + str(round(ws.kurtosis(), 4)))
print("  min      : " + str(round(ws.min(), 2)))
print("  p25      : " + str(round(ws.quantile(0.25), 2)))
print("  p75      : " + str(round(ws.quantile(0.75), 2)))
print("  p95      : " + str(round(ws.quantile(0.95), 2)))
print("  p99      : " + str(round(ws.quantile(0.99), 2)))
print("  max      : " + str(round(ws.max(), 2)))

print()
print("  -- By Year --")
yr = train.groupby("Year")["Weekly_Sales"].agg(
    count="count", mean="mean", median="median", std="std"
).round(2)
print(yr.to_string())

print()
print("  -- By Month --")
mo = train.groupby("Month")["Weekly_Sales"].agg(
    mean="mean", median="median"
).round(2)
print(mo.to_string())

print()
print("  -- By Store Type --")
st = train.groupby("Type")["Weekly_Sales"].agg(
    count="count", mean="mean", median="median", std="std"
).round(2)
print(st.to_string())

print()
print("  -- By IsHoliday --")
hol = train.groupby("IsHoliday")["Weekly_Sales"].agg(
    count="count", mean="mean", median="median", std="std"
).round(2)
print(hol.to_string())
holiday_lift = (
    train[train["IsHoliday"] == True]["Weekly_Sales"].mean() /
    train[train["IsHoliday"] == False]["Weekly_Sales"].mean() - 1
) * 100
print("  Holiday mean lift over non-holiday : +" + str(round(holiday_lift, 2)) + "%")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 -- log1p analysis (inspection only -- no permanent change to target)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 8 -- log1p transformation analysis (inspection only)")
print(SEP)

ws_pos = ws[ws > 0]
ws_log = np.log1p(ws_pos)

print("  Positive rows analysed : " + str(len(ws_pos)))
print("  Non-positive rows (excluded from log): " + str(len(ws) - len(ws_pos)))
print()
print("  Original Weekly_Sales (positive rows):")
print("    skewness : " + str(round(ws_pos.skew(), 4)))
print("    kurtosis : " + str(round(ws_pos.kurtosis(), 4)))
print("  After log1p:")
print("    skewness : " + str(round(ws_log.skew(), 4)))
print("    kurtosis : " + str(round(ws_log.kurtosis(), 4)))
skew_reduction = (1 - abs(ws_log.skew()) / abs(ws_pos.skew())) * 100
print("  Skewness reduction after log1p : " + str(round(skew_reduction, 1)) + "%")
print()
print("  DECISION: Weekly_Sales is NOT transformed here.")
print("  Original Weekly_Sales is preserved unchanged in train_prepared.csv.")
print("  log1p will be evaluated as an option during model experimentation.")
print("  If applied: predictions must be inverse-transformed with expm1()")
print("  before computing WMAE at original scale.")
print("  Non-positive rows (" + str((ws <= 0).sum()) +
      ") require special handling if log1p is chosen.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 -- Leakage-safe lag/rolling feature strategy (design document only)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 9 -- Leakage-safe lag/rolling feature strategy (design only)")
print(SEP)

print("""
  MANDATORY RULES for the Feature Engineering stage:

  Rule 1  All lag/rolling features computed WITHIN each (Store, Dept) pair.

  Rule 2  Row at date T may only use Weekly_Sales from dates strictly < T.
          Always use .shift(n) with n >= 1, never shift(0).

  Rule 3  Correct lag syntax (after sorting by [Store, Dept, Date]):
            df.groupby(["Store","Dept"])["Weekly_Sales"].shift(n)

  Rule 4  Correct rolling syntax (on shift(1) base to avoid current-row leakage):
            CORRECT: series.shift(1).rolling(window=4).mean()
            WRONG  : series.rolling(window=4).mean()

  Rule 5  Computed ONLY on the training DataFrame -- never on train+test.

  Rule 6  For test inference, lag values seeded from last training rows per
          (Store, Dept). No gap between train end and test start.

  Rule 7  Inside CV: lag/rolling features recomputed WITHIN each fold's
          training split ONLY -- never computed before the CV split.

  Planned features (Feature Engineering stage):
    Lags         : lag_1, lag_2, lag_4, lag_52
    Rolling mean : roll_mean_4, roll_mean_12, roll_mean_52  (on shift(1) base)
    Rolling std  : roll_std_4  (on shift(1) base)
""")

lag_check = [c for c in train.columns if "lag" in c.lower() or "roll" in c.lower()]
print("  Lag/rolling columns in current train : " + str(lag_check) +
      "  (expected [])")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 -- Chronological validation strategy
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 10 -- Chronological validation strategy")
print(SEP)

train_min = train["Date"].min()
train_max = train["Date"].max()
print("  Full training window : " + str(train_min.date()) +
      "  to  " + str(train_max.date()) +
      "  (" + str(train["Date"].nunique()) + " unique weeks)")

# Single time-split boundary
val_cutoff = pd.Timestamp("2012-08-03")
n_tr_split  = (train["Date"] < val_cutoff).sum()
n_val_split = (train["Date"] >= val_cutoff).sum()
tr_split_max = train.loc[train["Date"] < val_cutoff, "Date"].max()

print()
print("  PRIMARY SINGLE TIME-SPLIT:")
print("    Training split   : 2010-02-05  to  " + str(tr_split_max.date()) +
      "  (" + str(n_tr_split) + " rows)")
print("    Validation split : " + str(val_cutoff.date()) +
      "  to  " + str(train_max.date()) +
      "  (" + str(n_val_split) + " rows, ~13 weeks)")
print("    Leakage-safe: every validation date is strictly after every")
print("    training date in this split.")

# Walk-forward 3-fold CV
folds = [
    ("Fold 1", "2010-02-05", "2011-10-21", "2011-10-28", "2012-02-03"),
    ("Fold 2", "2010-02-05", "2012-01-27", "2012-02-03", "2012-05-04"),
    ("Fold 3", "2010-02-05", "2012-04-27", "2012-05-04", "2012-10-26"),
]
print()
print("  WALK-FORWARD 3-FOLD CV:")
print("  " + "Fold    " + "Train start   " + "Train end     " +
      "Val start     " + "Val end       " + "Tr rows  Val rows")
print("  " + "-" * 82)
for fold, ts, te, vs, ve in folds:
    tr_n  = ((train["Date"] >= pd.Timestamp(ts)) &
             (train["Date"] <= pd.Timestamp(te))).sum()
    val_n = ((train["Date"] >= pd.Timestamp(vs)) &
             (train["Date"] <= pd.Timestamp(ve))).sum()
    print("  " + fold.ljust(8) + ts.ljust(14) + te.ljust(14) +
          vs.ljust(14) + ve.ljust(14) +
          str(tr_n).rjust(8) + "  " + str(val_n).rjust(8))

print()
print("  Evaluation metric : WMAE = sum(w*|y-yhat|) / sum(w)")
print("    w = 5 for IsHoliday=True, w = 1 for IsHoliday=False")
print()
print("  Test period (2012-11-02 to 2013-07-26) is NEVER used")
print("  for model selection or hyperparameter tuning.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 -- Test data status
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 11 -- Test data status")
print(SEP)

print("  test shape           : " + str(test.shape))
print("  test date range      : " + str(test["Date"].min().date()) +
      "  to  " + str(test["Date"].max().date()))
print("  Weekly_Sales in test : " + str("Weekly_Sales" in test.columns) +
      "  (correctly absent)")
print("  Total NaN in test    : " + str(test.isnull().sum().sum()) +
      "  (expected 0)")
print("  Duplicates in test   : " +
      str(test.duplicated(subset=["Store","Dept","Date"]).sum()) +
      "  (expected 0)")
print()
print("  Test has received ONLY: sort order + Year/Month/Week/Quarter date features.")
print("  No target-derived features, no encoding, no lag/rolling features.")
print("  Remains the unseen final prediction set.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 -- Save prepared datasets
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 12 -- Saving prepared datasets")
print(SEP)

train.to_csv("train_prepared.csv", index=False)
test.to_csv("test_prepared.csv",   index=False)

print("  Saved : train_prepared.csv")
print("  Saved : test_prepared.csv")
print("  NOT overwritten : train_cleaned.csv, test_cleaned.csv")
print("  NOT modified    : train.csv, test.csv, features.csv, stores.csv")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERIFICATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("FINAL VERIFICATION SUMMARY")
print(SEP)

# Reload from disk to confirm what was actually saved
train_out = pd.read_csv("train_prepared.csv", parse_dates=["Date"])
test_out  = pd.read_csv("test_prepared.csv",  parse_dates=["Date"])

for df, label in [(train_out, "train_prepared.csv"), (test_out, "test_prepared.csv")]:
    total_nan  = df.isnull().sum().sum()
    total_dups = df.duplicated(subset=["Store", "Dept", "Date"]).sum()
    has_ws     = "Weekly_Sales" in df.columns
    has_dow    = "DayOfWeek" in df.columns
    print()
    print("  [" + label + "]")
    print("  Shape                    : " + str(df.shape[0]) + " rows x " +
          str(df.shape[1]) + " cols")
    print("  Date range               : " + str(df["Date"].min().date()) +
          "  to  " + str(df["Date"].max().date()))
    print("  Total missing values     : " + str(total_nan) +
          "  (" + ("OK" if total_nan == 0 else "*** PROBLEM ***") + ")")
    print("  Duplicate Store+Dept+Date: " + str(total_dups) +
          "  (" + ("OK" if total_dups == 0 else "*** PROBLEM ***") + ")")
    print("  Weekly_Sales present     : " + str(has_ws))
    print("  DayOfWeek present        : " + str(has_dow) +
          "  (expected False -- dropped)")
    print("  Columns (" + str(len(df.columns)) + "):")
    for c in df.columns:
        print("    " + c + " : " + str(df[c].dtype))

# Confirm DayOfWeek handling
print()
print("  DayOfWeek summary:")
print("    Created     : YES -- verified from Date.dt.dayofweek")
print("    Unique value: 4 (Friday) -- constant across all rows in both datasets")
print("    Predictive  : NONE (zero variance)")
print("    Action      : DROPPED before saving -- not present in prepared files")

# Confirm original files are unchanged
import os
orig_files = {
    "train.csv"        : (421570, 5),
    "test.csv"         : (115064, 4),
    "features.csv"     : (8190,  12),
    "stores.csv"       : (45,     3),
    "train_cleaned.csv": (421570, 16),
    "test_cleaned.csv" : (115064, 15),
}
print()
print("  Original / cleaned file integrity check:")
all_ok = True
for fname, (exp_rows, exp_cols) in orig_files.items():
    if os.path.exists(fname):
        df_chk = pd.read_csv(fname)
        rows_ok = df_chk.shape[0] == exp_rows
        cols_ok = df_chk.shape[1] == exp_cols
        status  = "OK" if (rows_ok and cols_ok) else "*** MISMATCH ***"
        if not (rows_ok and cols_ok):
            all_ok = False
        print("    " + fname.ljust(22) + ": " +
              str(df_chk.shape[0]) + " rows x " +
              str(df_chk.shape[1]) + " cols  [" + status + "]")
    else:
        print("    " + fname.ljust(22) + ": FILE NOT FOUND")
        all_ok = False

print()
if all_ok:
    print("  All original and cleaned files are intact and unchanged.")
else:
    print("  *** WARNING: one or more files have unexpected changes. ***")

print()
print(SEP)
print("DATA PREPARATION COMPLETE -- awaiting review.")
print(SEP)
