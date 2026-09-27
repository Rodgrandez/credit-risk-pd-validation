# Credit risk: PD scorecard, challenger and second-line validation

[![CI](https://github.com/Rodgrandez/credit-risk-pd-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodgrandez/credit-risk-pd-validation/actions/workflows/ci.yml)

How would a bank's risk team build, challenge and validate a probability-of-default model? This project does it
end to end on 36-month Lending Club loans: a WoE logistic scorecard, an XGBoost challenger, an LGD model,
expected loss, and a validation report written as a second line of defence would write it.

<!-- RESULTS:START -->
| Out-of-time (2014-2015) | AUC | Gini | KS |
|---|---|---|---|
| Scorecard (WoE logistic) | 0.639 | 0.277 | 0.199 |
| Challenger (XGBoost) | 0.661 | 0.322 | 0.232 |
| Benchmark: Lending Club grade | 0.661 | 0.322 | – |

Score PSI (development vs OOT): **0.003** · Hosmer-Lemeshow OOT p-value: **0** · LGD OOT predicted vs realized: **0.895** vs **0.891**
<!-- RESULTS:END -->

![ROC](reports/figures/roc.png) ![Calibration by vintage](reports/figures/calibration_vintage.png)
![Population stability](reports/figures/psi.png)

**Reports:** [validation report (PDF)](reports/validation_report.pdf) · [model card](reports/model_card.md)

## Design choices
- Only resolved **36-month** loans. Development vintages 2007-2013 (80/20 split by loan id); **out-of-time 2014-2015**.
  Later vintages are still open at the 2018Q4 data cut-off and would bias the default rate upwards.
- Lending Club's grade and interest rate are **excluded** from the model and used only as a benchmark.
- Default = Charged Off / Default, including loans that did not meet the credit policy. The PD is a
  **lifetime PD over the 36-month term** (not a 12-month regulatory PD), and expected loss is lifetime.
- LGD = 1 - net recoveries / exposure at default. Expected loss uses the funded amount times the average
  EAD/funded ratio of development defaults, so it is compared with realized loss on the same exposure base.
- The XGBoost challenger early-stops on a slice of the training sample; the holdout is never used for tuning.

## Reproduce
```bash
conda env create -f environment.yml && conda activate credit-risk-pd
# Kaggle API token in ~/.kaggle/access_token (https://www.kaggle.com/settings/api)
make all        # or: python -m credit_risk.pipeline all
```

Data: [Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (not redistributed here).
License: MIT.
