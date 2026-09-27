# Model card — PD scorecard (Lending Club, 36-month loans)

**Purpose.** Illustrative probability-of-default model for unsecured consumer loans, built to demonstrate
development and second-line validation practice. Not for production credit decisions.

**Target and horizon.** Default = Charged Off / Default (including loans that did not meet the credit policy).
The model estimates a lifetime PD over the 36-month contractual term (not a 12-month regulatory PD); expected loss is lifetime expected loss.

**Data.** Resolved 36-month Lending Club loans; development 2007-2013, out-of-time 2014-2015. Sample sizes: train 140,225, holdout 35,201, oot 445,596.

**Model.** WoE-binned logistic regression (scorecard, PDO 20, 600 points at 50:1 odds). Challenger: XGBoost.
Lending Club's grade is used only as a benchmark; grade, sub-grade, interest rate and instalment are excluded
from the features because they are the platform's own pricing of risk.

**Performance (out-of-time 2014-2015).** AUC 0.639, Gini 0.277, KS 0.199
(Lending Club grade: AUC 0.661). Calibration ratio mean PD / observed default rate
0.890; Hosmer-Lemeshow p <1e-16. Lifetime EL / realized
loss 0.836. Score PSI 0.003.

**Limitations.** Only resolved 36-month loans (60-month and post-2015 vintages are immature at the 2018Q4 data
cut-off); approved applicants only (no reject inference); US platform data; a thin application feature set
(total IV 0.39); exposure at default is the funded amount times the average EAD/funded ratio of
development defaults (0.586); LGD ignores discounting and collection costs other than the
recorded recovery fee.
