# Credit risk: PD scorecard, challenger and second-line validation

[![CI](https://github.com/Rodgrandez/credit-risk-pd-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodgrandez/credit-risk-pd-validation/actions/workflows/ci.yml)

How would a bank's risk team build, challenge and validate a probability-of-default model? This project does it
end to end on 36-month Lending Club loans: a WoE logistic scorecard, an XGBoost challenger, an LGD model,
expected loss, and a validation report written as a second line of defence would write it.

<!-- RESULTS:START -->
| Out-of-time (2014-2015) | AUC | Gini | KS |
|---|---|---|---|
| Scorecard (WoE logistic) | 0.639 | 0.277 | 0.199 |
| Challenger (XGBoost, all features) | 0.661 | 0.322 | 0.232 |
| Challenger (XGBoost, scorecard features) | 0.644 | 0.288 | – |
| Benchmark: Lending Club grade | 0.661 | 0.322 | – |

- **Horizon:** lifetime PD over the 36-month contractual term (not a 12-month regulatory PD).
- **Calibration (OOT):** mean PD 0.129 vs observed default rate 0.145 (ratio **0.890**, slope 0.95, largest decile gap 2.4 pp; Hosmer-Lemeshow p <1e-16). The model under-predicts default out of time: recalibration is required.
- **Expected loss backtest (OOT, same exposure-at-default base):** lifetime EL / realized loss **0.836**; LGD predicted 0.895 vs realized 0.891.
- **Stability:** score PSI 0.003; verification_status 0.165 (monitor), purpose 0.106 (monitor).
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
