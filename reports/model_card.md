# Model card — PD scorecard (Lending Club, 36-month loans)

**Purpose.** Illustrative probability-of-default model for unsecured consumer loans, built to demonstrate
development and second-line validation practice. Not for production credit decisions.

**Data.** Resolved 36-month Lending Club loans; development 2007-2013, out-of-time 2014-2015. Sample sizes: {'train': 140225, 'holdout': 35201, 'oot': 445596}.

**Model.** WoE-binned logistic regression (scorecard, PDO 20, 600 points at 50:1 odds). Challenger: XGBoost.
Lending Club's own grade and interest rate are excluded from the features and used only as a benchmark.

**Performance (out-of-time 2014-2015).** AUC 0.639, Gini 0.277, KS 0.199.
Hosmer-Lemeshow p-value 0. Score PSI 0.003.

**Limitations.** Only resolved 36-month loans (60-month and post-2015 vintages are immature at the data
cut-off); approved applicants only (no reject inference); US platform data from 2007-2015; EAD at
origination equals the funded amount.
