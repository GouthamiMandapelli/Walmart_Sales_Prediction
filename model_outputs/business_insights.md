# Walmart Sales Prediction — Business Insights

*Generated from training data (2010-02-05 to 2012-10-26) and final
test predictions (2012-11-02 to 2013-07-26).*

---

## 1. Overall Sales Pattern

- **Total training records:** 421,570 weekly store/department observations.
- **Weekly Sales range:** $-4,988.94 to $693,099.36.
- **Mean Weekly Sales:** $15,981.26  |  **Median:** $7,612.03.
- The distribution is heavily right-skewed (mean >> median), driven by a
  small number of high-volume departments (e.g. grocery, electronics).
- **1,285 rows** have negative Weekly_Sales, representing product returns
  or refunds; these are valid business records and were retained.
- A clear seasonal pattern is visible: sales spike in weeks 47–52
  (Thanksgiving and Christmas) and dip in January–February.

---

## 2. Store-Level Insights

### Highest average historical Weekly Sales (Top 5)
| Store | Mean Weekly Sales |
|-------|------------------|
| 20 | $29,508.30 |
| 4 | $29,161.21 |
| 14 | $28,784.85 |
| 13 | $27,355.14 |
| 2 | $26,898.07 |

### Lowest average historical Weekly Sales (Bottom 5)
| Store | Mean Weekly Sales |
|-------|------------------|
| 38 | $7,492.48 |
| 3 | $6,373.03 |
| 44 | $6,038.93 |
| 33 | $5,728.41 |
| 5 | $5,053.42 |

### Highest predicted total sales in the test period (Top 5)
| Store | Predicted Total |
|-------|----------------|
| 4 | $87,852,181.19 |
| 20 | $85,776,869.78 |
| 13 | $82,188,321.13 |
| 14 | $77,516,469.62 |
| 2 | $77,274,962.98 |

The stores with the highest historical average sales also tend to have
the highest predicted totals, consistent with the Lag-52 Baseline
methodology.

---

## 3. Department-Level Insights

### Highest average historical Weekly Sales (Top 10)
| Dept | Mean Weekly Sales |
|------|------------------|
| 92 | $75,204.87 |
| 95 | $69,824.42 |
| 38 | $61,090.62 |
| 72 | $50,566.52 |
| 65 | $45,441.71 |
| 90 | $45,232.08 |
| 40 | $44,900.70 |
| 2 | $43,607.02 |
| 91 | $33,687.91 |
| 94 | $33,405.88 |

### Highest predicted total sales in test period (Top 5)
| Dept | Predicted Total |
|------|----------------|
| 92 | $140,370,862.78 |
| 95 | $122,474,646.30 |
| 38 | $110,999,948.93 |
| 72 | $89,446,554.45 |
| 90 | $82,917,163.34 |

- Department sales vary enormously; a handful of departments (likely
  grocery/general merchandise) account for a disproportionate share
  of store revenue.
- Long-tail departments with near-zero sales may be seasonal or specialty
  units that operate intermittently.

---

## 4. Store Type Insights

| Type | Count | Mean Weekly Sales | Median Weekly Sales | Total Sales |
|------|-------|------------------|---------------------|-------------|
| A | 215,478 | $20,099.57 | $10,105.17 | $4,331,014,722.75 |
| B | 163,495 | $12,237.08 | $6,187.87 | $2,000,700,736.82 |
| C | 42,597 | $9,519.53 | $1,149.67 | $405,503,527.54 |

- **Type A** stores are the largest by floor area and show the highest
  average and median weekly sales.
- **Type C** stores are the smallest and show the lowest average sales.
- These differences are consistent with store size: larger floor area
  generally accommodates more departments and higher volume.

---

## 5. Holiday Insights

| Segment | Count | Mean Weekly Sales | Median Weekly Sales |
|---------|-------|------------------|---------------------|
| Holiday weeks | 29,661 | $17,035.82 | $7,947.74 |
| Non-holiday weeks | 391,909 | $15,901.45 | $7,589.95 |

- Holiday weeks represent only **7.0%** of training rows
  but show notably higher average and median sales.
- The four known holiday weeks are: **Super Bowl (Week 6)**,
  **Labour Day (Week 36)**, **Thanksgiving (Week 47)**, and
  **Christmas (Week 52)**.
- The Kaggle competition WMAE metric assigns **5× weight** to holiday
  weeks, reflecting their outsized business importance.

---

## 6. Promotion / Markdown Insights

| MarkDown | % of Weeks Active in Training |
|----------|-------------------------------|
| MarkDown1 | 35.7% |
| MarkDown2 | 26.0% |
| MarkDown3 | 32.4% |
| MarkDown4 | 32.0% |
| MarkDown5 | 35.9% |

- MarkDown promotions were **absent before November 2011** (structural
  zeros) and became active in the latter portion of the training period.
- All MarkDown NaN values were filled with 0 during cleaning, reflecting
  the absence of an active promotion rather than missing data.
- Rows with active MarkDowns tend to coincide with higher sales weeks,
  but this is an **observational pattern** — the data do not establish
  that MarkDowns caused higher sales.

---

## 7. External Factors

| Factor | Min | Max | Mean |
|--------|-----|-----|------|
| Temperature (°F) | -2.06 | 100.14 | 60.09 |
| Fuel Price ($/gal) | 2.472 | 4.468 | 3.361 |
| CPI | 126.06 | 227.23 | 171.20 |
| Unemployment (%) | 3.879 | 14.313 | 7.960 |

- **Temperature** shows seasonal variation; extreme cold/heat weeks
  occasionally align with reduced foot traffic in certain regions.
- **Fuel Price** increased over the training period (2010–2012),
  consistent with macroeconomic trends during that era.
- **CPI** rose steadily, reflecting general inflation; higher CPI weeks
  do not uniformly correspond to higher sales.
- **Unemployment** was elevated throughout the training period (post-2008
  recession recovery). Stores in higher-unemployment areas show more
  variability in sales.
- These are **observational relationships** in the dataset; no causal
  inference is drawn.

---

## 8. Model Performance (Stage 6 Validation)

| Model | Fold 1 WMAE | Fold 2 WMAE | Fold 3 WMAE | Avg CV WMAE | Primary WMAE | Primary MAE | Primary RMSE |
|-------|------------|------------|------------|------------|-------------|------------|-------------|
| Lag-52 Baseline | 2,679.07 | 2,252.93 | 1,902.38 | 2,278.13 | 1,893.62 | 1,867.98 | 3,936.34 |
| Random Forest | 3,970.12 | 1,938.52 | 1,693.97 | 2,534.20 | 1,561.32 | 1,565.34 | 3,209.71 |
| HistGradBoost | 4,441.19 | 2,048.61 | 1,848.10 | 2,779.30 | 1,689.01 | 1,672.26 | 3,345.98 |

**Selected model:** Lag-52 Baseline  
**Validation period (primary):** 2012-08-03 to 2012-10-26  
**Selection criterion:** Lowest average WMAE across three
chronological walk-forward folds.

The Lag-52 Baseline was selected because it achieved the best
**average cross-validated WMAE** (2,278). The Random Forest
outperformed on the single primary hold-out split (WMAE 1,561 vs 1,893)
but had a very high Fold-1 WMAE (3,970) due to the cold-start
effect — Fold-1 training ends before any MarkDown promotion history
was available, exposing the RF's sensitivity to new feature distributions.
The baseline is immune to this because it makes no assumption about
feature distributions.

---

## 9. Final Prediction Summary

- **Test rows predicted:** 115,064
- **Test date range:** 2012-11-02 to 2013-07-26
- **Total predicted Weekly Sales (test period):** $1,903,508,195.43
- **Average predicted Weekly Sales per row:** $16,543.04

### Top 5 Stores by Predicted Total Sales
| Store | Predicted Total |
|-------|----------------|
| 4 | $87,852,181.19 |
| 20 | $85,776,869.78 |
| 13 | $82,188,321.13 |
| 14 | $77,516,469.62 |
| 2 | $77,274,962.98 |

### Top 5 Departments by Predicted Total Sales
| Dept | Predicted Total |
|------|----------------|
| 92 | $140,370,862.78 |
| 95 | $122,474,646.30 |
| 38 | $110,999,948.93 |
| 72 | $89,446,554.45 |
| 90 | $82,917,163.34 |

*Note: Test-set actual Weekly_Sales values are unavailable; these are
model predictions, not confirmed outcomes.*

---

## 10. Business Takeaways

1. **Seasonal planning is critical.** Weeks 47–52 (Thanksgiving through
   Christmas) consistently generate the highest sales across stores and
   departments. Inventory and staffing should be scaled up well in advance
   of this window.

2. **Holiday weeks drive outsized revenue risk.** Holiday weeks carry 5×
   the forecasting importance in the WMAE metric. Mis-forecasting a single
   Thanksgiving or Christmas week costs more than missing five ordinary weeks.
   Dedicated holiday demand models or higher safety stock are warranted.

3. **Store size is the strongest structural predictor of sales volume.**
   Type A stores (largest) consistently lead in both historical average
   and predicted totals. Resource allocation — staff, inventory depth,
   promotional budgets — should scale with store size.

4. **A small number of departments account for the majority of revenue.**
   The top 10 departments (by average weekly sales) should receive
   priority in supply-chain planning, shelf-space optimisation, and
   promotional scheduling.

5. **MarkDown promotions are concentrated in the second half of the year.**
   Because MarkDown activity began in November 2011, the model has limited
   history to learn promotion effects. Future modelling iterations with
   richer MarkDown history may improve accuracy during promotion weeks.

6. **The Lag-52 Baseline is a strong and robust benchmark.** It requires
   no parameter tuning and is fully explainable: "last year's same-week
   sales is the best estimate for this year." It outperformed tree-based
   models when evaluated fairly across all chronological folds. More
   complex models should be expected to meaningfully beat this baseline
   before being deployed.

7. **Economic conditions (CPI, Unemployment) show gradual trends rather
   than sharp signals.** Their predictive contribution is modest in the
   short run but may matter for long-range forecasting. Monitoring
   macro indicators alongside store-level metrics is a prudent approach
   for annual planning cycles.
