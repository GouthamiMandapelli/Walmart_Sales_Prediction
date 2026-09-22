"""
Walmart Sales Prediction -- EDA Script
=======================================
Stage : Exploratory Data Analysis
Inputs : train_prepared.csv, test_prepared.csv
Outputs: eda_plots/ directory with PNG charts
         (no CSV files are modified)
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

os.makedirs("eda_plots", exist_ok=True)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("seaborn-whitegrid")

SEP = "=" * 70

# =============================================================================
# A. DATASET OVERVIEW
# =============================================================================
print(SEP)
print("A. DATASET OVERVIEW")
print(SEP)

train = pd.read_csv("train_prepared.csv", parse_dates=["Date"])
test  = pd.read_csv("test_prepared.csv",  parse_dates=["Date"])

for df, label in [(train, "train_prepared"), (test, "test_prepared")]:
    print()
    print("  [" + label + "]")
    print("  Shape        : " + str(df.shape))
    print("  Date range   : " + str(df["Date"].min().date()) +
          "  to  " + str(df["Date"].max().date()))
    print("  Unique weeks : " + str(df["Date"].nunique()))
    print("  Unique stores: " + str(df["Store"].nunique()))
    print("  Unique depts : " + str(df["Dept"].nunique()))
    print("  Total NaN    : " + str(df.isnull().sum().sum()))
    dups = df.duplicated(subset=["Store", "Dept", "Date"]).sum()
    print("  Duplicates (Store+Dept+Date) : " + str(dups))
    print("  Weekly_Sales present : " + str("Weekly_Sales" in df.columns))


# =============================================================================
# B. TARGET DISTRIBUTION
# =============================================================================
print()
print(SEP)
print("B. TARGET DISTRIBUTION  (Weekly_Sales)")
print(SEP)

ws = train["Weekly_Sales"]
print("  count    : " + str(len(ws)))
print("  mean     : " + str(round(ws.mean(), 2)))
print("  median   : " + str(round(ws.median(), 2)))
print("  std      : " + str(round(ws.std(), 2)))
print("  min      : " + str(round(ws.min(), 2)))
print("  p25      : " + str(round(ws.quantile(0.25), 2)))
print("  p75      : " + str(round(ws.quantile(0.75), 2)))
print("  p95      : " + str(round(ws.quantile(0.95), 2)))
print("  p99      : " + str(round(ws.quantile(0.99), 2)))
print("  max      : " + str(round(ws.max(), 2)))
print("  skewness : " + str(round(ws.skew(), 4)))
print("  kurtosis : " + str(round(ws.kurtosis(), 4)))
n_neg  = (ws < 0).sum()
n_zero = (ws == 0).sum()
print("  negative : " + str(n_neg)  + " (" + str(round(n_neg  / len(ws) * 100, 3)) + "%)")
print("  zero     : " + str(n_zero) + " (" + str(round(n_zero / len(ws) * 100, 3)) + "%)")

# Plot 01 -- Histogram (original + log1p)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(ws, bins=100, color="#3b82d4", edgecolor="none", alpha=0.85)
axes[0].set_title("Weekly_Sales Distribution (original)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Weekly Sales ($)")
axes[0].set_ylabel("Frequency")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

ws_log = np.log1p(ws[ws > 0])
axes[1].hist(ws_log, bins=100, color="#7c5cd8", edgecolor="none", alpha=0.85)
axes[1].set_title("Weekly_Sales Distribution (log1p, positive rows only)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("log1p(Weekly Sales)")
axes[1].set_ylabel("Frequency")
fig.tight_layout()
fig.savefig("eda_plots/01_target_distribution.png", dpi=150)
plt.close()
print("  Saved: eda_plots/01_target_distribution.png")

# Plot 02 -- Boxplot
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(ws, vert=False, patch_artist=True,
           boxprops=dict(facecolor="#3b82d4", alpha=0.6),
           medianprops=dict(color="red", linewidth=2),
           flierprops=dict(marker=".", markersize=2, alpha=0.3))
ax.set_title("Weekly_Sales Boxplot", fontsize=11, fontweight="bold")
ax.set_xlabel("Weekly Sales ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/02_target_boxplot.png", dpi=150)
plt.close()
print("  Saved: eda_plots/02_target_boxplot.png")


# =============================================================================
# C. TIME TRENDS AND SEASONALITY
# =============================================================================
print()
print(SEP)
print("C. TIME TRENDS AND SEASONALITY")
print(SEP)

weekly = (train.groupby("Date")["Weekly_Sales"]
          .agg(total="sum", mean="mean", median="median")
          .reset_index()
          .sort_values("Date"))

# Plot 03 -- Weekly total sales over time
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(weekly["Date"], weekly["total"] / 1e6,
        color="#3b82d4", linewidth=1.2, alpha=0.9)
ax.set_title("Total Weekly Sales Over Time (all stores, all depts)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Total Sales ($ millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
fig.tight_layout()
fig.savefig("eda_plots/03_weekly_total_sales_trend.png", dpi=150)
plt.close()
print("  Saved: eda_plots/03_weekly_total_sales_trend.png")

# Plot 04 -- Weekly mean sales over time
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(weekly["Date"], weekly["mean"] / 1e3,
        color="#7c5cd8", linewidth=1.2, alpha=0.9)
ax.set_title("Mean Weekly Sales Per Store-Dept Over Time",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/04_weekly_mean_sales_trend.png", dpi=150)
plt.close()
print("  Saved: eda_plots/04_weekly_mean_sales_trend.png")

# Yearly stats
yr = (train.groupby("Year")["Weekly_Sales"]
      .agg(mean="mean", median="median", total="sum")
      .round(2))
print("  By Year:")
print(yr.to_string())

# Plot 05 -- Monthly seasonality
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mo = train.groupby("Month")["Weekly_Sales"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(mo["Month"], mo["Weekly_Sales"] / 1e3,
              color="#3b82d4", edgecolor="white", alpha=0.85)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title("Mean Weekly Sales by Month (all years)", fontsize=11, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
            f"${h:.1f}k", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig("eda_plots/05_monthly_seasonality.png", dpi=150)
plt.close()
print("  Saved: eda_plots/05_monthly_seasonality.png")

# Plot 06 -- Week-of-year seasonality
wk = train.groupby("Week")["Weekly_Sales"].mean().reset_index()
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(wk["Week"], wk["Weekly_Sales"] / 1e3,
        color="#3b82d4", linewidth=1.5, marker="o", markersize=3)
ax.set_title("Mean Weekly Sales by Week-of-Year", fontsize=11, fontweight="bold")
ax.set_xlabel("ISO Week Number")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/06_week_of_year_seasonality.png", dpi=150)
plt.close()
print("  Saved: eda_plots/06_week_of_year_seasonality.png")

print()
print("  Monthly mean/median sales:")
mo_stats = (train.groupby("Month")["Weekly_Sales"]
            .agg(mean="mean", median="median")
            .round(2))
for idx, row in mo_stats.iterrows():
    print("    " + month_names[idx-1] + ": mean=" + str(row["mean"]) +
          "  median=" + str(row["median"]))

top_month = mo_stats["mean"].idxmax()
bot_month = mo_stats["mean"].idxmin()
print()
print("  Highest mean sales month : " + month_names[top_month-1] +
      " ($" + str(round(mo_stats.loc[top_month, "mean"], 2)) + ")")
print("  Lowest  mean sales month : " + month_names[bot_month-1] +
      " ($" + str(round(mo_stats.loc[bot_month, "mean"], 2)) + ")")


# =============================================================================
# D. STORE ANALYSIS
# =============================================================================
print()
print(SEP)
print("D. STORE ANALYSIS")
print(SEP)

store_stats = (train.groupby("Store")["Weekly_Sales"]
               .agg(mean="mean", median="median", total="sum", std="std")
               .round(2)
               .sort_values("mean", ascending=False))

print("  Top 10 stores by mean Weekly_Sales:")
print(store_stats.head(10).to_string())
print()
print("  Bottom 10 stores by mean Weekly_Sales:")
print(store_stats.tail(10).to_string())

top5_stores = store_stats.head(5).index.tolist()
bot5_stores = store_stats.tail(5).index.tolist()
print()
print("  Top 5 stores by mean sales   : " + str(top5_stores))
print("  Bottom 5 stores by mean sales: " + str(bot5_stores))
print("  Highest mean sales: Store " + str(store_stats.index[0]) +
      "  = $" + str(round(store_stats.iloc[0]["mean"], 2)))
print("  Lowest  mean sales: Store " + str(store_stats.index[-1]) +
      "  = $" + str(round(store_stats.iloc[-1]["mean"], 2)))
print("  Ratio (highest/lowest mean)  : " +
      str(round(store_stats.iloc[0]["mean"] / store_stats.iloc[-1]["mean"], 2)) + "x")

# Plot 07 -- Store mean sales bar
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(store_stats.index.astype(str), store_stats["mean"] / 1e3,
       color="#3b82d4", alpha=0.85)
ax.set_title("Mean Weekly Sales by Store", fontsize=11, fontweight="bold")
ax.set_xlabel("Store ID")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/07_store_mean_sales.png", dpi=150)
plt.close()
print("  Saved: eda_plots/07_store_mean_sales.png")

# Plot 08 -- Store total sales horizontal bar
store_total = store_stats.sort_values("total", ascending=True)
fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(store_total.index.astype(str), store_total["total"] / 1e9,
        color="#3b82d4", alpha=0.85)
ax.set_title("Total Sales by Store (full training period)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Total Sales ($ billions)")
fig.tight_layout()
fig.savefig("eda_plots/08_store_total_sales.png", dpi=150)
plt.close()
print("  Saved: eda_plots/08_store_total_sales.png")


# =============================================================================
# E. DEPARTMENT ANALYSIS
# =============================================================================
print()
print(SEP)
print("E. DEPARTMENT ANALYSIS")
print(SEP)

dept_stats = (train.groupby("Dept")["Weekly_Sales"]
              .agg(mean="mean", median="median", total="sum", std="std")
              .round(2)
              .sort_values("mean", ascending=False))

print("  Top 15 departments by mean Weekly_Sales:")
print(dept_stats.head(15).to_string())
print()
print("  Bottom 10 departments by mean Weekly_Sales:")
print(dept_stats.tail(10).to_string())

top5_depts = dept_stats.head(5).index.tolist()
bot5_depts = dept_stats.tail(5).index.tolist()
print()
print("  Top 5 depts by mean sales   : " + str(top5_depts))
print("  Bottom 5 depts by mean sales: " + str(bot5_depts))
print("  Highest dept mean sales: Dept " + str(dept_stats.index[0]) +
      "  = $" + str(round(dept_stats.iloc[0]["mean"], 2)))
print("  Lowest  dept mean sales: Dept " + str(dept_stats.index[-1]) +
      "  = $" + str(round(dept_stats.iloc[-1]["mean"], 2)))

# Plot 09 -- Top 20 departments
top20_dept = dept_stats.head(20)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(top20_dept.index.astype(str), top20_dept["mean"] / 1e3,
       color="#7c5cd8", alpha=0.85)
ax.set_title("Top 20 Departments by Mean Weekly Sales",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Department")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/09_dept_top20_mean_sales.png", dpi=150)
plt.close()
print("  Saved: eda_plots/09_dept_top20_mean_sales.png")

# Plot 10 -- All departments
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(dept_stats.index.astype(str), dept_stats["mean"] / 1e3,
       color="#3b82d4", alpha=0.75)
ax.set_title("Mean Weekly Sales by Department (all)", fontsize=11, fontweight="bold")
ax.set_xlabel("Department")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
ax.tick_params(axis="x", labelsize=6, rotation=90)
fig.tight_layout()
fig.savefig("eda_plots/10_dept_all_mean_sales.png", dpi=150)
plt.close()
print("  Saved: eda_plots/10_dept_all_mean_sales.png")


# =============================================================================
# F. STORE TYPE ANALYSIS
# =============================================================================
print()
print(SEP)
print("F. STORE TYPE ANALYSIS")
print(SEP)

type_stats = (train.groupby("Type")["Weekly_Sales"]
              .agg(count="count", mean="mean", median="median", std="std",
                   total="sum", min="min", max="max")
              .round(2))
print(type_stats.to_string())

type_size = train.groupby("Type")["Size"].mean().round(0)
print()
print("  Mean store size by Type:")
print(type_size.to_string())

type_ranked = type_stats["mean"].sort_values(ascending=False)
print()
print("  Mean sales by Type (ranked):")
for t, v in type_ranked.items():
    print("    Type " + str(t) + " : $" + str(round(v, 2)))

# Plot 11 -- Boxplot by store type  (tick_labels= for matplotlib >= 3.9)
fig, ax = plt.subplots(figsize=(10, 6))
type_groups = [train.loc[train["Type"] == t, "Weekly_Sales"].values
               for t in ["A", "B", "C"]]
bp = ax.boxplot(type_groups,
                tick_labels=["Type A", "Type B", "Type C"],
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
for patch, color in zip(bp["boxes"], ["#3b82d4", "#7c5cd8", "#57606a"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
ax.set_title("Weekly Sales Distribution by Store Type",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Weekly Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/11_store_type_boxplot.png", dpi=150)
plt.close()
print("  Saved: eda_plots/11_store_type_boxplot.png")

# Plot 12 -- Mean sales bar by type
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(type_stats.index, type_stats["mean"] / 1e3,
              color=["#3b82d4", "#7c5cd8", "#57606a"],
              alpha=0.85, width=0.5)
ax.set_title("Mean Weekly Sales by Store Type", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
            f"${h:.1f}k", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig("eda_plots/12_store_type_mean_sales.png", dpi=150)
plt.close()
print("  Saved: eda_plots/12_store_type_mean_sales.png")


# =============================================================================
# G. HOLIDAY ANALYSIS
# =============================================================================
print()
print(SEP)
print("G. HOLIDAY ANALYSIS")
print(SEP)

hol_stats = (train.groupby("IsHoliday")["Weekly_Sales"]
             .agg(count="count", mean="mean", median="median", std="std")
             .round(2))
print(hol_stats.to_string())

mean_holiday     = train[train["IsHoliday"] == True]["Weekly_Sales"].mean()
mean_non_holiday = train[train["IsHoliday"] == False]["Weekly_Sales"].mean()
lift             = (mean_holiday / mean_non_holiday - 1) * 100
print()
print("  Mean sales (holiday)     : $" + str(round(mean_holiday, 2)))
print("  Mean sales (non-holiday) : $" + str(round(mean_non_holiday, 2)))
print("  Observed holiday lift    : " + str(round(lift, 2)) + "%")

# List actual holiday week dates directly -- no event-name assumptions
holiday_weeks = sorted(train[train["IsHoliday"] == True]["Date"].dt.date.unique())
print()
print("  Holiday weeks in training data (" + str(len(holiday_weeks)) + " dates):")
for d in holiday_weeks:
    print("    " + str(d))

# WMAE weight context
total_rows   = len(train)
holiday_rows = int((train["IsHoliday"] == True).sum())
non_hol_rows = total_rows - holiday_rows
eff_weight   = (holiday_rows * 5 + non_hol_rows * 1) / total_rows
print()
print("  WMAE weight context:")
print("    Holiday rows    : " + str(holiday_rows) +
      " (" + str(round(holiday_rows / total_rows * 100, 1)) + "% of rows, weight=5)")
print("    Non-holiday rows: " + str(non_hol_rows) +
      " (" + str(round(non_hol_rows / total_rows * 100, 1)) + "% of rows, weight=1)")
print("    Effective avg weight : " + str(round(eff_weight, 3)))
print("    Holiday errors penalised " +
      str(round(5 / eff_weight, 2)) + "x more than the average row")

# Plot 13 -- Holiday vs non-holiday boxplot  (tick_labels= for matplotlib >= 3.9)
fig, ax = plt.subplots(figsize=(8, 6))
grp_false = train[train["IsHoliday"] == False]["Weekly_Sales"].values
grp_true  = train[train["IsHoliday"] == True]["Weekly_Sales"].values
bp = ax.boxplot([grp_false, grp_true],
                tick_labels=["Non-Holiday", "Holiday"],
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
bp["boxes"][0].set_facecolor("#3b82d4"); bp["boxes"][0].set_alpha(0.65)
bp["boxes"][1].set_facecolor("#e95c5c"); bp["boxes"][1].set_alpha(0.65)
ax.set_title("Weekly Sales: Holiday vs Non-Holiday", fontsize=11, fontweight="bold")
ax.set_ylabel("Weekly Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/13_holiday_boxplot.png", dpi=150)
plt.close()
print("  Saved: eda_plots/13_holiday_boxplot.png")

# Plot 14 -- Mean sales per actual holiday week (dates on x-axis, no event names)
holiday_week_mean = (train[train["IsHoliday"] == True]
                     .groupby("Date")["Weekly_Sales"]
                     .mean()
                     .reset_index()
                     .sort_values("Date"))
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(holiday_week_mean)),
       holiday_week_mean["Weekly_Sales"] / 1e3,
       color="#e95c5c", alpha=0.85)
ax.set_xticks(range(len(holiday_week_mean)))
ax.set_xticklabels([str(d) for d in holiday_week_mean["Date"].dt.date],
                   rotation=45, ha="right", fontsize=7)
ax.set_title("Mean Weekly Sales on Each Holiday Week",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/14_holiday_week_sales.png", dpi=150)
plt.close()
print("  Saved: eda_plots/14_holiday_week_sales.png")


# =============================================================================
# H. EXTERNAL FEATURE ANALYSIS
# =============================================================================
print()
print(SEP)
print("H. EXTERNAL FEATURE ANALYSIS")
print(SEP)

ext_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment",
            "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "Size"]

print("  External feature statistics (train):")
ext_stats = (train[ext_cols].describe().T
             [["mean", "std", "min", "50%", "max"]]
             .round(3))
print(ext_stats.to_string())

print()
print("  MarkDown zero/positive/negative proportion:")
for c in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
    pct_zero = round((train[c] == 0).sum() / len(train) * 100, 1)
    pct_pos  = round((train[c] >  0).sum() / len(train) * 100, 1)
    pct_neg  = round((train[c] <  0).sum() / len(train) * 100, 1)
    print("    " + c + ": zeros=" + str(pct_zero) + "%"
          "  positive=" + str(pct_pos) + "%"
          "  negative=" + str(pct_neg) + "%")

print()
print("  MarkDown Pearson r with Weekly_Sales (all rows):")
for c in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
    r = train[[c, "Weekly_Sales"]].corr().iloc[0, 1]
    print("    " + c + ": r=" + str(round(r, 4)))

print()
print("  MarkDown Pearson r with Weekly_Sales (non-zero rows only):")
for c in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
    sub = train[train[c] != 0]
    r   = sub[[c, "Weekly_Sales"]].corr().iloc[0, 1]
    print("    " + c + " (n=" + str(len(sub)) + "): r=" + str(round(r, 4)))

# Plot 15 -- External feature distributions
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for ax, col in zip(axes.flatten(),
                   ["Temperature", "Fuel_Price", "CPI",
                    "Unemployment", "Size", "MarkDown1"]):
    ax.hist(train[col], bins=60, color="#3b82d4", edgecolor="none", alpha=0.8)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
axes[2][1].set_visible(False)
fig.suptitle("External Feature Distributions", fontsize=13,
             fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("eda_plots/15_external_feat_distributions.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_plots/15_external_feat_distributions.png")

# Plot 16 -- MarkDown distributions (non-zero only)
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, col in zip(axes,
                   ["MarkDown1", "MarkDown2", "MarkDown3",
                    "MarkDown4", "MarkDown5"]):
    vals = train.loc[train[col] > 0, col]
    ax.hist(vals, bins=60, color="#7c5cd8", edgecolor="none", alpha=0.8)
    ax.set_title(col + "\n(non-zero, n=" + str(len(vals)) + ")", fontsize=9)
    ax.set_xlabel("Value")
fig.suptitle("MarkDown Distributions (non-zero rows only)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("eda_plots/16_markdown_distributions.png", dpi=150)
plt.close()
print("  Saved: eda_plots/16_markdown_distributions.png")

# Plot 17 -- Weekly_Sales vs Temperature scatter
fig, ax = plt.subplots(figsize=(8, 5))
sample = train.sample(min(15000, len(train)), random_state=42)
ax.scatter(sample["Temperature"], sample["Weekly_Sales"] / 1e3,
           alpha=0.15, s=5, color="#3b82d4")
ax.set_title("Weekly_Sales vs Temperature (sampled 15k rows)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Temperature (F)")
ax.set_ylabel("Weekly Sales ($k)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
fig.tight_layout()
fig.savefig("eda_plots/17_sales_vs_temperature.png", dpi=150)
plt.close()
print("  Saved: eda_plots/17_sales_vs_temperature.png")

# Plot 18 -- Store mean sales vs Store Size
store_summary = (train.groupby("Store")
                 .agg(mean_sales=("Weekly_Sales", "mean"),
                      size=("Size", "first"))
                 .reset_index())
size_sales_r = store_summary[["size", "mean_sales"]].corr().iloc[0, 1]
print()
print("  Store Size vs Mean Sales Pearson r : " + str(round(size_sales_r, 4)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(store_summary["size"] / 1e3, store_summary["mean_sales"] / 1e3,
           s=80, color="#7c5cd8", alpha=0.8, edgecolors="white")
for _, row in store_summary.iterrows():
    ax.annotate(str(int(row["Store"])),
                (row["size"] / 1e3, row["mean_sales"] / 1e3),
                textcoords="offset points", xytext=(4, 2), fontsize=6)
ax.set_title("Store Mean Weekly Sales vs Store Size  (r=" +
             str(round(size_sales_r, 3)) + ")",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Store Size (k sq ft)")
ax.set_ylabel("Mean Weekly Sales ($k)")
fig.tight_layout()
fig.savefig("eda_plots/18_sales_vs_size.png", dpi=150)
plt.close()
print("  Saved: eda_plots/18_sales_vs_size.png")

print()
print("  Pearson r with Weekly_Sales (all numerical features, ranked by |r|):")
corr_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size",
             "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
corrs = (train[corr_cols + ["Weekly_Sales"]]
         .corr()["Weekly_Sales"]
         .drop("Weekly_Sales"))
for col, r in corrs.sort_values(key=abs, ascending=False).items():
    print("    " + col.ljust(14) + " : r=" + str(round(r, 4)))


# =============================================================================
# I. CORRELATION ANALYSIS
# =============================================================================
print()
print(SEP)
print("I. CORRELATION ANALYSIS")
print(SEP)

num_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment",
            "Size", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
corr_matrix = train[num_cols].corr().round(3)
print(corr_matrix.to_string())

# Plot 19 -- Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            mask=mask, ax=ax, annot_kws={"size": 8},
            linewidths=0.5)
ax.set_title("Correlation Heatmap -- Numerical Features",
             fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
fig.tight_layout()
fig.savefig("eda_plots/19_correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved: eda_plots/19_correlation_heatmap.png")


# =============================================================================
# J. KEY OBSERVATIONS  (all data-driven)
# =============================================================================
print()
print(SEP)
print("J. KEY OBSERVATIONS")
print(SEP)

nov_dec_mean  = train[train["Month"].isin([11, 12])]["Weekly_Sales"].mean()
other_mean    = train[~train["Month"].isin([11, 12])]["Weekly_Sales"].mean()
seasonal_lift = (nov_dec_mean / other_mean - 1) * 100

store_ratio = store_stats.iloc[0]["mean"] / store_stats.iloc[-1]["mean"]
type_ratio  = type_stats["mean"].max() / type_stats["mean"].min()
dept_ratio  = dept_stats.iloc[0]["mean"] / dept_stats.iloc[-1]["mean"]

print("  1. Weekly_Sales is right-skewed (skewness=" +
      str(round(ws.skew(), 2)) + ", kurtosis=" + str(round(ws.kurtosis(), 2)) + ").")
print("     Median ($" + str(round(ws.median(), 0)) +
      ") is well below mean ($" + str(round(ws.mean(), 0)) + ").")
print()
print("  2. Nov+Dec mean = $" + str(round(nov_dec_mean, 2)) +
      "  vs rest-of-year = $" + str(round(other_mean, 2)) +
      "  (observed lift: +" + str(round(seasonal_lift, 1)) + "%).")
print()
print("  3. Highest mean sales month: " + month_names[top_month-1] +
      " ($" + str(round(mo_stats.loc[top_month, "mean"], 2)) + ")" +
      "  |  Lowest: " + month_names[bot_month-1] +
      " ($" + str(round(mo_stats.loc[bot_month, "mean"], 2)) + ")")
print()
print("  4. Store mean sales range: $" + str(round(store_stats.iloc[-1]["mean"], 2)) +
      " to $" + str(round(store_stats.iloc[0]["mean"], 2)) +
      "  (ratio " + str(round(store_ratio, 1)) + "x).")
print()
print("  5. Store Type mean sales: " +
      "  |  ".join(["Type " + str(t) + "=$" + str(round(v, 2))
                    for t, v in type_ranked.items()]) +
      "  (ratio: " + str(round(type_ratio, 1)) + "x)")
print()
print("  6. Dept mean sales range: $" + str(round(dept_stats.iloc[-1]["mean"], 2)) +
      " to $" + str(round(dept_stats.iloc[0]["mean"], 2)) +
      "  (ratio " + str(round(dept_ratio, 1)) + "x).")
print()
print("  7. Observed holiday mean lift: +" + str(round(lift, 2)) +
      "%.  Holiday errors weighted 5x in WMAE.")
print()
print("  8. Store Size vs mean sales: Pearson r=" + str(round(size_sales_r, 4)) + ".")
print()
print("  9. MarkDown1 zeros=" +
      str(round((train["MarkDown1"] == 0).sum() / len(train) * 100, 1)) + "%" +
      "  MarkDown2 zeros=" +
      str(round((train["MarkDown2"] == 0).sum() / len(train) * 100, 1)) + "%" +
      " (structural: no promotion active).")
print()
print("  10. CPI and Fuel_Price show notable inter-correlation (see heatmap).")


# =============================================================================
# K. MODELING IMPLICATIONS
# =============================================================================
print()
print(SEP)
print("K. MODELING IMPLICATIONS FOR FEATURE ENGINEERING")
print(SEP)

print("  1. Seasonality -- Year, Month, Week, Quarter features are candidates.")
print("     The week-of-year pattern shows clear variation across the year.")
print()
print("  2. Store Size has Pearson r=" + str(round(size_sales_r, 4)) +
      " with mean store sales. It is a candidate numerical feature.")
print()
print("  3. Store and Dept IDs show high variance in mean sales between groups.")
print("     Label encoding or target encoding (inside CV fold only) are")
print("     candidate strategies to evaluate during Feature Engineering.")
print()
print("  4. Store Type shows differentiated sales distributions across A, B, C.")
print("     One-hot encoding is an appropriate candidate for Store Type.")
print()
print("  5. IsHoliday flag is directly relevant because WMAE weights holiday")
print("     weeks 5x. The IsHoliday indicator must be present in all models.")
print("     The " + str(len(holiday_weeks)) + " distinct holiday week dates can also")
print("     be used as a binary indicator feature.")
print()
print("  6. MarkDown columns are zero for a large proportion of rows.")
print("     A binary MarkDown_active flag may be a useful companion feature.")
print("     Observed correlations with Weekly_Sales are reported above;")
print("     causal interpretation requires further investigation.")
print()
print("  7. CPI and Unemployment have low direct Pearson r with weekly sales.")
print("     They provide store-level economic context and may be useful in")
print("     combination with other features.")
print()
print("  8. Lag and rolling features are promising candidates for Feature")
print("     Engineering and should be evaluated using leakage-safe")
print("     chronological validation.")
print()
print("  9. Negative Weekly_Sales rows (" + str(n_neg) + ") are kept.")
print("     Model choice and loss functions should account for these.")
print()
print("  10. Skewness=" + str(round(ws.skew(), 2)) + " suggests log1p transformation")
print("      may improve performance for error-sensitive objectives.")
print("      This will be evaluated during modelling.")


# =============================================================================
# FINAL PLOT INVENTORY
# =============================================================================
print()
print(SEP)
print("EDA COMPLETE -- all plots saved to eda_plots/")
print(SEP)
print()
plot_files = sorted(os.listdir("eda_plots"))
print("  Plots generated (" + str(len(plot_files)) + "):")
for f in plot_files:
    print("    eda_plots/" + f)
print()
print("Awaiting review before proceeding to Feature Engineering.")
