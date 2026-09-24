# Drift Monitoring Analysis

This is my write-up of the drift check in `src/monitor_drift.py`.

## How the check works

- **Reference data:** the training split (1,176 employees).
- **Production data:** the 294 held-out employees the model never trained on, with simulated drift applied. The drift settings live in `configs/config.yaml` under `monitoring.simulated_drift`.
- **Tests:** Kolmogorov-Smirnov for numeric features and chi-square for categorical features. A feature counts as drifted when its p-value is below 0.05.
- **Alert rule:** the script exits with code 1 when more than 20% of the features drift (`drift_share_threshold`).

Before adding any drift, I ran Evidently on the untouched held-out rows as a sanity check. Evidently's default distance-based tests flagged 3 features there even though nothing had changed, which I think comes from the production batch being fairly small (294 rows). The K-S and chi-square tests flagged 0 features on the same data, so I switched to them to avoid false alarms. You can repeat this check by setting `changes: []` in the config.

### The scenario I simulated

I imagined the company going through a **hiring wave during a busy season**:

| Feature | Simulated change | Why it could happen |
|---|---|---|
| Age | 6 years younger | Lots of new, younger hires |
| TotalWorkingYears | multiplied by 0.6 | New hires are earlier in their careers |
| YearsAtCompany | multiplied by 0.5 | New hires have short tenure |
| MonthlyIncome | multiplied by 0.8 | More people at entry-level pay |
| OverTime | 35% of rows set to "Yes" | Busy season means more overtime |

## Results

Running `python -m src.monitor_drift` gave:

- **5 of 30 features drifted (16.7%)**: Age, MonthlyIncome, OverTime, TotalWorkingYears, and YearsAtCompany. Their p-values were extremely small, between about 1e-9 and 1e-31.
- The other 25 features did not drift. Their p-values were all above 0.07.
- 16.7% is under the 20% threshold, so the script printed "OK" and exited with code 0.
- Running `python -m src.monitor_drift --threshold 0.1` shows the alert path: same results, but it prints "ALERT" and exits with code 1.

The full HTML report is saved to `reports/drift_report.html` each time the script runs.

## 1. Which features showed drift and why?

Age, TotalWorkingYears, YearsAtCompany, MonthlyIncome, and OverTime drifted. These are exactly the five features I changed, and nothing else was flagged, so the check caught the real changes without false positives.

The cause is the hiring wave scenario: a big group of younger, less experienced, lower-paid employees joining during a stretch of heavy overtime. In a real company I would not assume that right away, though. The same pattern could also come from a data problem, such as income being reported in different units or tenure being calculated from the wrong start date, so I would check the data pipeline too.

One honest limitation: I only changed five columns. In real life a hiring wave would probably also shift related features like JobLevel, YearsInCurrentRole, and YearsWithCurrManager, so real drift would likely be wider than what I simulated.

## 2. Would this drift likely affect model performance?

Yes, I think it would. There are two reasons.

**The drifted features are closely tied to attrition.** In the original data, employees in the groups this drift pushes toward leave about two to three times as often as everyone else (the overall attrition rate is 16.1%):

| Group | Attrition rate | Everyone else |
|---|---|---|
| Works overtime | 30.5% | 10.4% |
| Under 30 years old | 27.9% | 12.8% |
| 2 years or less at the company | 29.8% | 12.0% |
| 3 years or less of total experience | 38.2% | 13.3% |
| Bottom quarter of monthly income | 29.3% | 11.7% |

The model leans on these features too. OverTime = "Yes" is in the top 10 of the model's 51 coefficients by size, and TotalWorkingYears and Age are among its larger numeric coefficients.

**I measured the effect directly.** `python -m src.drift_impact` trains the model and scores it on the held-out rows before and after the drift is applied. The true labels are the same in both cases, so the difference comes only from the shifted inputs:

| Data | ROC-AUC | F1 | Recall | Precision | Accuracy | Flagged to leave |
|---|---|---|---|---|---|---|
| Original test set | 0.814 | 0.477 | 0.660 | 0.373 | 0.769 | 28.2% |
| Drifted batch | 0.772 | 0.407 | 0.787 | 0.274 | 0.633 | 45.9% |

The model goes from flagging about 28% of employees as likely to leave to about 46%. Precision drops from 0.37 to 0.27, which means HR would get a lot more false alarms, and accuracy drops from 0.77 to 0.63.

One caveat: because I did not change the labels, this shows how the model reacts to the shifted inputs, not what would really happen. A younger, overworked group might truly leave more often, which would make some of those extra flags correct. That is exactly why I could not judge the real performance until actual attrition outcomes come in.

## 3. What action would I recommend?

**Investigate first, then plan to retrain.** I would not just continue monitoring.

The automatic check technically passed (16.7% is under 20%), but the five features that drifted are some of the most important ones for predicting attrition. So in this case the overall drift share understates the risk. My plan would be:

1. **Investigate.** Confirm with HR whether a hiring wave or busy season actually happened, and check the data pipeline for errors like unit changes.
2. **If the change is real, retrain.** Start collecting attrition outcomes for the new employees, and retrain on recent data once enough of them are labeled. Before switching models, I would compare the new model against the current one with `compare_experiments.py`.
3. **Use the current model carefully in the meantime.** Since it is flagging almost half of the new batch, I would treat its predictions for recent hires as low confidence.
4. **Improve the monitoring.** Add per-feature alerts for the key predictors (OverTime, tenure, Age, and MonthlyIncome), so a few important features drifting can trigger an alert even when the overall share stays under the threshold.

## How to reproduce

```
dvc pull
python -m src.monitor_drift
python -m src.monitor_drift --threshold 0.1
python -m src.drift_impact
```