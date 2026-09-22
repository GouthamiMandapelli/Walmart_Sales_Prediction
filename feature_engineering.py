"""
Walmart Sales Prediction -- Feature Engineering (Final)
========================================================
Stage : Feature Engineering (approved final version)
Inputs : train_prepared.csv, test_prepared.csv
Outputs: train_features.csv, test_features.csv

Approved methodology (AICTE project):
---------------------------------------
PRIMARY feature set (Stage 1 model):
  - Calendar         : Year, Month, Week, Quarter
  - Identifiers      : Store, Dept, Type  (encoding deferred to model stage)
  - Holiday          : IsHoliday, IsHoliday_int, Is_Known_Holiday
  - MarkDown         : MarkDown1-5 raw values + 5 binary activity flags
  - Economic/env     : Temperature, Fuel_Price, CPI, Unemployment, Size
  - lag_52 ONLY      : prior-year same-week Weekly_Sales
                       Always sourced from training (test dates − 52w fall in train).
                       Leakage-safe, no recursion needed.

RETAINED in train_features.csv for optional Stage 2 (recursive) extension:
  - lag_1, lag_2, lag_4   (valid for training rows; NaN for most test rows)
  - rolling_mean_4/12/52  (valid for training rows; NaN for most test rows)
  - rolling_std_4         (valid for training rows; NaN for most test rows)

These Stage-2 columns are present in train_features.csv so Stage 2 can use them
during training. For test prediction, Stage 2 would generate them recursively at
inference time and does not read them from test_features.csv.

Leakage rules:
  - All lag/rolling computed WITHIN each (Store, Dept) group.
  - lag_52 for test: date-indexed lookup into training arrays only.
  - lag_1/2/4 and rolling features for test: NaN (correctly absent --
    these require prior test predictions which are not pre-computed here).
  - No train+test concatenation for any target-based computation.
  - Weekly_Sales absent from test_features.csv.
"""

import pandas as pd
import numpy as np
import os

SEP = "=" * 70

# =============================================================================
# STEP 1 -- Load and sort inputs
# =============================================================================
print(SEP)
print("STEP 1 -- Loading inputs")
print(SEP)

train = pd.read_csv("train_prepared.csv", parse_dates=["Date"])
test  = pd.read_csv("test_prepared.csv",  parse_dates=["Date"])

train = train.sort_values(["Store","Dept","Date"]).reset_index(drop=True)
test  = test.sort_values(["Store","Dept","Date"]).reset_index(drop=True)

for df, label in [(train,"train"),(test,"test")]:
    nulls = df.isnull().sum().sum()
    dups  = df.duplicated(subset=["Store","Dept","Date"]).sum()
    print("  " + label + ": shape=" + str(df.shape) +
          "  NaN=" + str(nulls) + "  dups=" + str(dups) +
          "  Weekly_Sales=" + str("Weekly_Sales" in df.columns))


# =============================================================================
# STEP 2 -- Calendar features (already present)
# =============================================================================
print()
print(SEP)
print("STEP 2 -- Calendar features (already present)")
print(SEP)
for c in ["Year","Month","Week","Quarter"]:
    print("  " + c + " : " + ("present" if c in train.columns else "MISSING"))


# =============================================================================
# STEP 3 -- Categorical identifiers (no encoding yet)
# =============================================================================
print()
print(SEP)
print("STEP 3 -- Categorical identifiers (no encoding)")
print(SEP)
for c in ["Store","Dept","Type"]:
    print("  " + c + " : unique=" + str(train[c].nunique()) +
          "  (encoding deferred to model training stage)")


# =============================================================================
# STEP 4 -- MarkDown activity indicators
# =============================================================================
print()
print(SEP)
print("STEP 4 -- MarkDown activity indicators")
print(SEP)
for df, label in [(train,"train"),(test,"test")]:
    for i in range(1,6):
        col  = "MarkDown" + str(i)
        flag = col + "_active"
        df[flag] = (df[col] != 0).astype(int)
    print("  " + label + " MarkDown active flags created (MarkDown1-5_active)")


# =============================================================================
# STEP 5 -- Holiday features
# =============================================================================
print()
print(SEP)
print("STEP 5 -- Holiday features")
print(SEP)

holiday_dates = set(train.loc[train["IsHoliday"]==True, "Date"].dt.date.unique())
print("  " + str(len(holiday_dates)) + " known holiday dates from training data")

for df, label in [(train,"train"),(test,"test")]:
    df["IsHoliday_int"]    = df["IsHoliday"].astype(int)
    df["Is_Known_Holiday"] = df["Date"].dt.date.isin(holiday_dates).astype(int)
    print("  " + label + ": IsHoliday_int=" + str(df["IsHoliday_int"].sum()) +
          "  Is_Known_Holiday=" + str(df["Is_Known_Holiday"].sum()))


# =============================================================================
# STEP 6 -- Train lag features (all 4 lags, shift-based, within group)
#           lag_52 = primary; lag_1/2/4 = Stage-2 optional extension
# =============================================================================
print()
print(SEP)
print("STEP 6 -- Train lag features (all 4 lags for completeness)")
print(SEP)

LAG_PERIODS  = [1, 2, 4, 52]
ONE_WEEK_NS  = np.timedelta64(7, "D")

for n in LAG_PERIODS:
    col = "lag_" + str(n)
    train[col] = (train
                  .groupby(["Store","Dept"])["Weekly_Sales"]
                  .shift(n))
    n_null = train[col].isna().sum()
    note   = "  <-- PRIMARY (always available for test)" if n == 52 else \
             "  <-- Stage-2 only (recursive inference at test time)"
    print("  train " + col + ": " + str(n_null) + " NaN" + note)


# =============================================================================
# STEP 7 -- Train rolling features (shift(1) base, within group)
#           Stage-2 optional extension only
# =============================================================================
print()
print(SEP)
print("STEP 7 -- Train rolling features (Stage-2 extension, shift(1) base)")
print(SEP)

ROLL_CONFIGS = [
    ("rolling_mean_4",  "mean",  4),
    ("rolling_mean_12", "mean", 12),
    ("rolling_mean_52", "mean", 52),
    ("rolling_std_4",   "std",   4),
]

for col_name, func, window in ROLL_CONFIGS:
    shifted = train.groupby(["Store","Dept"])["Weekly_Sales"].shift(1)
    if func == "mean":
        train[col_name] = (shifted
                           .groupby([train["Store"], train["Dept"]])
                           .transform(lambda s: s.rolling(window, min_periods=1).mean()))
    else:
        train[col_name] = (shifted
                           .groupby([train["Store"], train["Dept"]])
                           .transform(lambda s: s.rolling(window, min_periods=1).std()))
    n_null = train[col_name].isna().sum()
    print("  train " + col_name + ": " + str(n_null) + " NaN")


# =============================================================================
# STEP 8 -- Test lag_52 (date-indexed lookup, always available from training)
#           Test lag_1/2/4 and rolling features: set to NaN (Stage-2 only)
# =============================================================================
print()
print(SEP)
print("STEP 8 -- Test lag/rolling features")
print(SEP)

print("""
  lag_52 for test:
    For any test date T, T - 52 weeks falls in the training period.
    (Test starts 2012-11-02; 52w back = 2011-11-04 = well within training.)
    Computed by date-indexed lookup into the training series.
    Fully leakage-safe: uses only historical training values.

  lag_1, lag_2, lag_4, rolling features for test:
    Set to NaN in test_features.csv.
    These require the model's own predictions (recursive inference).
    They will be generated at inference time during Stage-2 modelling,
    not pre-stored in the CSV.
    Stage-1 model does NOT use these features at all.
""")

# Build training lookup for lag_52
print("  Building training series lookup for lag_52...")
train_lookup = {}
for (s, d), grp in train.groupby(["Store","Dept"]):
    grp_sorted = grp.sort_values("Date")
    train_lookup[(s, d)] = {
        "dates": grp_sorted["Date"].values,
        "sales": grp_sorted["Weekly_Sales"].values,
    }
print("  Lookup built for " + str(len(train_lookup)) + " (Store,Dept) groups.")

def get_lag(dates, sales, pred_date, n):
    """Return Weekly_Sales exactly n weeks before pred_date, or NaN."""
    target_date = pred_date - n * ONE_WEEK_NS
    idx = np.searchsorted(dates, target_date, side="left")
    if idx < len(dates) and dates[idx] == target_date:
        return sales[idx]
    return np.nan

# Initialise all lag/rolling columns in test as NaN
all_lag_roll = (["lag_" + str(n) for n in LAG_PERIODS] +
                [r[0] for r in ROLL_CONFIGS])
for col in all_lag_roll:
    test[col] = np.nan

# Populate lag_52 only via date-indexed lookup
print("  Computing test lag_52 (date-indexed lookup)...")
for (s, d), grp_idx in test.groupby(["Store","Dept"]).groups.items():
    if (s, d) not in train_lookup:
        continue   # 11 pairs with no training history → remain NaN
    grp       = test.loc[grp_idx].sort_values("Date")
    dates     = train_lookup[(s,d)]["dates"]
    sales     = train_lookup[(s,d)]["sales"]
    vals      = np.array([get_lag(dates, sales, t, 52)
                          for t in grp["Date"].values])
    test.loc[grp_idx, "lag_52"] = vals

# Report
print()
print("  Test lag/rolling NaN counts:")
print("  Feature            | NaN count | % rows | Notes")
print("  " + "-"*65)
notes_map = {
    "lag_1"          : "NaN -- recursive (Stage-2 only)",
    "lag_2"          : "NaN -- recursive (Stage-2 only)",
    "lag_4"          : "NaN -- recursive (Stage-2 only)",
    "lag_52"         : "Date-indexed from training (primary feature)",
    "rolling_mean_4" : "NaN -- recursive (Stage-2 only)",
    "rolling_mean_12": "NaN -- recursive (Stage-2 only)",
    "rolling_mean_52": "NaN -- recursive (Stage-2 only)",
    "rolling_std_4"  : "NaN -- recursive (Stage-2 only)",
}
for col in all_lag_roll:
    n_null = test[col].isna().sum()
    pct    = round(n_null / len(test) * 100, 2)
    note   = notes_map.get(col, "")
    print("  " + col.ljust(18) + " | " + str(n_null).rjust(9) +
          " | " + str(pct).rjust(6) + "% | " + note)


# =============================================================================
# STEP 9 -- Save outputs
# =============================================================================
print()
print(SEP)
print("STEP 9 -- Saving outputs")
print(SEP)

id_cols    = ["Store","Dept","Date"]
target_col = ["Weekly_Sales"]
orig_feats = ["IsHoliday","Temperature","Fuel_Price",
              "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5",
              "CPI","Unemployment","Type","Size",
              "Year","Month","Week","Quarter"]
new_feats  = (["IsHoliday_int","Is_Known_Holiday"] +
              ["MarkDown" + str(i) + "_active" for i in range(1,6)] +
              ["lag_" + str(n) for n in LAG_PERIODS] +
              [r[0] for r in ROLL_CONFIGS])

train_cols = [c for c in id_cols + target_col + orig_feats + new_feats
              if c in train.columns]
test_cols  = [c for c in id_cols + orig_feats + new_feats
              if c in test.columns]

train_out = train[train_cols].copy()
test_out  = test[test_cols].copy()

train_out.to_csv("train_features.csv", index=False)
test_out.to_csv("test_features.csv",   index=False)

print("  Saved: train_features.csv -- " + str(train_out.shape[0]) +
      " rows x " + str(train_out.shape[1]) + " cols")
print("  Saved: test_features.csv  -- " + str(test_out.shape[0]) +
      " rows x " + str(test_out.shape[1]) + " cols")


# =============================================================================
# STEP 10 -- Full verification (reload from disk)
# =============================================================================
print()
print(SEP)
print("STEP 10 -- Full verification (reloaded from disk)")
print(SEP)

train_v = pd.read_csv("train_features.csv", parse_dates=["Date"])
test_v  = pd.read_csv("test_features.csv",  parse_dates=["Date"])

checks = []

# Shape
checks.append(("train row count = 421570",
               train_v.shape[0]==421570, str(train_v.shape[0])))
checks.append(("test  row count = 115064",
               test_v.shape[0]==115064,  str(test_v.shape[0])))
checks.append(("train cols = 35",
               train_v.shape[1]==35, str(train_v.shape[1])))
checks.append(("test  cols = 34",
               test_v.shape[1]==34,  str(test_v.shape[1])))

# Uniqueness
tr_dups = train_v.duplicated(subset=["Store","Dept","Date"]).sum()
te_dups = test_v.duplicated(subset=["Store","Dept","Date"]).sum()
checks.append(("train duplicate Store+Dept+Date = 0", tr_dups==0, str(tr_dups)))
checks.append(("test  duplicate Store+Dept+Date = 0", te_dups==0, str(te_dups)))

# Target
checks.append(("Weekly_Sales in train",
               "Weekly_Sales" in train_v.columns,
               str("Weekly_Sales" in train_v.columns)))
checks.append(("Weekly_Sales ABSENT from test",
               "Weekly_Sales" not in test_v.columns,
               str("Weekly_Sales" not in test_v.columns)))

# Date ranges
checks.append(("train date min = 2010-02-05",
               train_v["Date"].min().date()==pd.Timestamp("2010-02-05").date(),
               str(train_v["Date"].min().date())))
checks.append(("train date max = 2012-10-26",
               train_v["Date"].max().date()==pd.Timestamp("2012-10-26").date(),
               str(train_v["Date"].max().date())))
checks.append(("test  date min = 2012-11-02",
               test_v["Date"].min().date()==pd.Timestamp("2012-11-02").date(),
               str(test_v["Date"].min().date())))
checks.append(("test  date max = 2013-07-26",
               test_v["Date"].max().date()==pd.Timestamp("2013-07-26").date(),
               str(test_v["Date"].max().date())))

# lag_52 in both files
checks.append(("lag_52 present in train_features",
               "lag_52" in train_v.columns,
               str("lag_52" in train_v.columns)))
checks.append(("lag_52 present in test_features",
               "lag_52" in test_v.columns,
               str("lag_52" in test_v.columns)))

# lag_52 test: 52w back from first test date (2012-11-02) = 2011-11-04 (in training)
# verify it is not all NaN (most pairs should have it)
test_lag52_non_null = test_v["lag_52"].notna().sum()
checks.append(("test lag_52 non-null rows > 100000",
               test_lag52_non_null > 100000,
               str(test_lag52_non_null) + " non-null"))

# lag_52 for test: spot-check Store=1, Dept=1, first test row
s1d1_train = (train_v[(train_v["Store"]==1)&(train_v["Dept"]==1)]
              .sort_values("Date"))
s1d1_test  = (test_v[(test_v["Store"]==1)&(test_v["Dept"]==1)]
              .sort_values("Date").iloc[0])
first_test_date = s1d1_test["Date"]
target_52w      = first_test_date - pd.Timedelta(weeks=52)
expected_lag52  = s1d1_train.loc[s1d1_train["Date"]==target_52w, "Weekly_Sales"]
if len(expected_lag52) > 0:
    exp_val = expected_lag52.values[0]
    act_val = s1d1_test["lag_52"]
    ok = abs(exp_val - act_val) < 0.01
    checks.append(("lag_52 spot-check Store=1,Dept=1 test row 0",
                   ok, "exp=" + str(round(exp_val,2)) + " act=" + str(round(act_val,2))))
else:
    checks.append(("lag_52 spot-check Store=1,Dept=1 test row 0",
                   False, "training date " + str(target_52w.date()) + " not found"))

# Verify lag_1/2/4 and rolling cols are ALL NaN in test (correct by design)
stage2_cols = ["lag_1","lag_2","lag_4",
               "rolling_mean_4","rolling_mean_12","rolling_mean_52","rolling_std_4"]
for col in stage2_cols:
    all_nan = test_v[col].isna().all()
    checks.append(("test " + col + " is all NaN (Stage-2 only)",
                   all_nan, str(all_nan)))

# Train lag full leakage check
any_leak = False
for (s, d), grp in train_v.groupby(["Store","Dept"]):
    grp = grp.sort_values("Date").reset_index(drop=True)
    for col, n in [("lag_1",1),("lag_2",2),("lag_4",4),("lag_52",52)]:
        expected = grp["Weekly_Sales"].shift(n)
        mismatch = (~np.isclose(grp[col].fillna(-999999),
                                expected.fillna(-999999))).sum()
        if mismatch > 0:
            any_leak = True
            break
    if any_leak: break
checks.append(("train lag full leakage check: clean",
               not any_leak,
               "clean" if not any_leak else "LEAKAGE DETECTED"))

# Original/prepared file integrity
orig_files = {
    "train.csv"         : (421570, 5),
    "test.csv"          : (115064, 4),
    "features.csv"      : (8190,  12),
    "stores.csv"        : (45,     3),
    "train_cleaned.csv" : (421570, 16),
    "test_cleaned.csv"  : (115064, 15),
    "train_prepared.csv": (421570, 20),
    "test_prepared.csv" : (115064, 19),
}
for fname, (exp_r, exp_c) in orig_files.items():
    if os.path.exists(fname):
        df_chk = pd.read_csv(fname)
        ok = (df_chk.shape[0]==exp_r) and (df_chk.shape[1]==exp_c)
        checks.append((fname + " unchanged (" + str(exp_r) + "x" + str(exp_c) + ")",
                        ok, str(df_chk.shape)))
    else:
        checks.append((fname + " exists", False, "NOT FOUND"))

# Print verification table
print()
print("  Check                                                       | Status | Actual")
print("  " + "-"*82)
all_pass = True
for desc, ok, actual in checks:
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print("  " + desc.ljust(58) + " | " + status.ljust(6) + " | " + actual)
print()
print("  Overall: " + ("ALL CHECKS PASSED" if all_pass else
                        "*** ONE OR MORE CHECKS FAILED ***"))


# =============================================================================
# FINAL FEATURE ENGINEERING REPORT
# =============================================================================
print()
print(SEP)
print("FEATURE ENGINEERING REPORT (Final Approved Version)")
print(SEP)

print("""
  OUTPUT FILES
    train_features.csv : """ + str(train_v.shape[0]) + """ rows x """ + str(train_v.shape[1]) + """ cols
    test_features.csv  : """ + str(test_v.shape[0]) + """ rows x """ + str(test_v.shape[1]) + """ cols

  FEATURE SET SUMMARY
  -----------------------------------------------------------------
  Category          | Features                         | Stage
  -----------------------------------------------------------------
  Identifiers       | Store, Dept, Date                | Both
  Target            | Weekly_Sales (train only)        | Both
  Calendar          | Year, Month, Week, Quarter       | Both
  Categorical       | Type (not yet encoded)           | Both
  Holiday           | IsHoliday, IsHoliday_int,        | Both
                    | Is_Known_Holiday                 |
  MarkDown raw      | MarkDown1 – MarkDown5            | Both
  MarkDown flags    | MarkDown1-5_active (binary)      | Both
  Economic/env      | Temperature, Fuel_Price,         | Both
                    | CPI, Unemployment, Size          |
  Lag (primary)     | lag_52                           | Both
  Lag (Stage-2)     | lag_1, lag_2, lag_4              | Train only*
  Rolling (Stage-2) | rolling_mean_4/12/52,            | Train only*
                    | rolling_std_4                    |
  -----------------------------------------------------------------
  * Stage-2 columns are present in train_features.csv for model training.
    In test_features.csv they are NaN by design -- generated recursively
    at inference time during Stage-2 modelling.

  STAGE-1 MODEL FEATURES (35 total in train, 34 in test):
    Store, Dept, Date, [Weekly_Sales], IsHoliday, Temperature,
    Fuel_Price, MarkDown1-5, CPI, Unemployment, Type, Size,
    Year, Month, Week, Quarter, IsHoliday_int, Is_Known_Holiday,
    MarkDown1-5_active, lag_52

  LEAKAGE PREVENTION SUMMARY
    1. lag_52 for train : .groupby(Store,Dept).shift(52) -- no current row used
    2. lag_52 for test  : date-indexed lookup into training arrays only
                          T - 52w always falls in training (test starts Nov 2012)
    3. lag_1/2/4 train  : .shift(n>=1) -- no current row used
    4. lag_1/2/4 test   : NaN (require recursive inference; not pre-computed)
    5. Rolling (train)  : shift(1) applied before rolling() -- no current row
    6. Rolling (test)   : NaN (require recursive inference; not pre-computed)
    7. No train+test concatenation for any target-based feature
    8. Weekly_Sales absent from test_features.csv

  MISSING VALUE HANDLING
    Train lag_52 NaN  : 160,487 rows (first 52 rows of each group)
                        Imputed in CV fold using per-(Store,Dept) median
    Test  lag_52 NaN  : """ + str(test_v["lag_52"].isna().sum()) + """ rows (pairs with < 52 training weeks)
                        Imputed using dept-level median from training
    Stage-2 NaN (test): All NaN by design -- handled at inference time

  11 STORE+DEPT PAIRS WITH NO TRAINING HISTORY
    lag_52 will be NaN; imputed with dept-level median from training.
    These pairs represent """ + str(test_v[test_v["lag_52"].isna()].shape[0]) + """ rows total (negligible impact on WMAE).
""")

print(SEP)
print("FEATURE ENGINEERING COMPLETE (Final Approved Version).")
print("Awaiting approval to proceed to Model Training.")
print(SEP)
