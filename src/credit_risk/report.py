import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.metrics import format_pvalue

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
HORIZON = "lifetime PD over the 36-month contractual term (not a 12-month regulatory PD)"


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def write_results(results: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=_default), encoding="utf-8")
    return path


def _row(rows, sample):
    return next(r for r in rows if r["sample"] == sample)


def _psi_text(r: dict) -> str:
    flags = r.get("psi_flags", [])
    if not flags:
        return "all variables stable"
    return ", ".join(f"{f['variable']} {f['psi']:.3f} ({f['status']})" for f in flags)


def results_markdown(r: dict) -> str:
    champ, chall = _row(r["discrimination"], "oot"), _row(r["challenger_discrimination"], "oot")
    cal = r["calibration_oot"]
    lines = [
        "| Out-of-time (2014-2015) | AUC | Gini | KS |", "|---|---|---|---|",
        f"| Scorecard (WoE logistic) | {champ['auc']:.3f} | {champ['gini']:.3f} | {champ['ks']:.3f} |",
        f"| Challenger (XGBoost, all features) | {chall['auc']:.3f} | {chall['gini']:.3f} | {chall['ks']:.3f} |",
        (f"| Challenger (XGBoost, scorecard features) | {r['challenger_same_features_auc_oot']:.3f} | "
         f"{2 * r['challenger_same_features_auc_oot'] - 1:.3f} | – |"),
        (f"| Benchmark: Lending Club grade | {r['benchmark_grade_auc_oot']:.3f} | "
         f"{2 * r['benchmark_grade_auc_oot'] - 1:.3f} | – |"),
        "",
        f"- **Horizon:** {HORIZON}.",
        (f"- **Calibration (OOT):** mean PD {cal['mean_pd']:.3f} vs observed default rate {cal['observed_dr']:.3f} "
         f"(ratio **{cal['ratio']:.3f}**, slope {cal['slope']:.2f}, largest decile gap {cal['max_decile_gap_pp']:.1f} pp; "
         f"Hosmer-Lemeshow p {format_pvalue(r['hosmer_lemeshow_oot']['pvalue'])}). "
         f"{'The model under-predicts default out of time: recalibration is required.' if cal['ratio'] < 0.95 else ''}"),
        (f"- **Expected loss backtest (OOT, same exposure-at-default base):** lifetime EL / realized loss "
         f"**{r['el_total_ratio_oot']:.3f}**; LGD predicted {r['lgd_oot']['predicted_mean']:.3f} vs realized "
         f"{r['lgd_oot']['realized_mean']:.3f}."),
        f"- **Stability:** score PSI {r['score_psi']:.3f}; {_psi_text(r)}.",
    ]
    return "\n".join(lines)


def update_readme(readme: Path, results: dict) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    block = f"{START}\n{results_markdown(results)}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL)
    Path(readme).write_text(new, encoding="utf-8")


def write_model_card(r: dict, path) -> Path:
    champ = _row(r["discrimination"], "oot")
    cal = r["calibration_oot"]
    sizes = ", ".join(f"{k} {v:,}" for k, v in r["sample_sizes"].items())
    text = f"""# Model card — PD scorecard (Lending Club, 36-month loans)

**Purpose.** Illustrative probability-of-default model for unsecured consumer loans, built to demonstrate
development and second-line validation practice. Not for production credit decisions.

**Target and horizon.** Default = Charged Off / Default (including loans that did not meet the credit policy).
The model estimates a {HORIZON}; expected loss is lifetime expected loss.

**Data.** {r['data_note']} Sample sizes: {sizes}.

**Model.** WoE-binned logistic regression (scorecard, PDO 20, 600 points at 50:1 odds). Challenger: XGBoost.
Lending Club's grade is used only as a benchmark; grade, sub-grade, interest rate and instalment are excluded
from the features because they are the platform's own pricing of risk.

**Performance (out-of-time 2014-2015).** AUC {champ['auc']:.3f}, Gini {champ['gini']:.3f}, KS {champ['ks']:.3f}
(Lending Club grade: AUC {r['benchmark_grade_auc_oot']:.3f}). Calibration ratio mean PD / observed default rate
{cal['ratio']:.3f}; Hosmer-Lemeshow p {format_pvalue(r['hosmer_lemeshow_oot']['pvalue'])}. Lifetime EL / realized
loss {r['el_total_ratio_oot']:.3f}. Score PSI {r['score_psi']:.3f}.

**Limitations.** Only resolved 36-month loans (60-month and post-2015 vintages are immature at the 2018Q4 data
cut-off); approved applicants only (no reject inference); US platform data; a thin application feature set
(total IV {r['iv_total']:.2f}); exposure at default is the funded amount times the average EAD/funded ratio of
development defaults ({r['ead_factor']:.3f}); LGD ignores discounting and collection costs other than the
recorded recovery fee.
"""
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _severity(value: float, high, medium) -> str:
    return "High" if high(value) else ("Medium" if medium(value) else "Low")


def validation_narrative(r: dict) -> str:
    """Second-line narrative (LaTeX): scope, opinion, rated findings, limitations and recommendations."""
    champ = _row(r["discrimination"], "oot")
    chall = _row(r["challenger_discrimination"], "oot")
    cal, el = r["calibration_oot"], r["el_total_ratio_oot"]
    gap = r["benchmark_grade_auc_oot"] - champ["auc"]
    shifts = [f for f in r.get("psi_flags", []) if f["status"] == "shift"]
    population = ("The score population is stable, so the gap reflects a different default rate in the "
                  "2014--2015 vintages, not a change in the applicant mix." if r["score_psi"] < 0.10 else
                  "The score population has also shifted, so part of the gap may come from the applicant mix.")
    lgd_gap = abs(r["lgd_oot"]["predicted_mean"] - r["lgd_oot"]["realized_mean"])
    lgd_text = ("LGD is close on average, so the difference is driven mainly by PD." if lgd_gap < 0.03 else
                "LGD is also off on average and contributes to the difference.")
    full_gain = chall["auc"] - champ["auc"]
    same_gain = r["challenger_same_features_auc_oot"] - champ["auc"]
    features_drive = full_gain > 0 and same_gain < full_gain / 2
    source = ("most of the challenger's advantage comes from the extra features (total IV "
              f"{r['iv_total']:.2f} for the scorecard), not from the non-linear form" if features_drive else
              "a material part of the challenger's advantage comes from the non-linear form")
    choice = ("The challenger's extra discrimination comes mostly from using more features; on the same "
              "features the gain is small, which supports keeping the interpretable form and investing in "
              "data rather than in model complexity." if features_drive else
              "The non-linear challenger adds discrimination even on the same features; the trade-off between "
              "that gain and the scorecard's transparency should be weighed by the model owner.")
    findings = [
        (_severity(cal["ratio"], lambda v: v < 0.90 or v > 1.10, lambda v: v < 0.95 or v > 1.05),
         (f"Calibration out of time. Mean PD {cal['mean_pd']:.3f} against an observed default rate of "
          f"{cal['observed_dr']:.3f} (ratio {cal['ratio']:.3f}, slope {cal['slope']:.2f}, largest decile gap "
          f"{cal['max_decile_gap_pp']:.1f} pp, Hosmer--Lemeshow p "
          f"{format_pvalue(r['hosmer_lemeshow_oot']['pvalue']).replace('<', '$<$')}). "
          f"{population} Estimates are non-conservative when the ratio is below one.")),
        (_severity(el, lambda v: v < 0.90, lambda v: v < 0.95 or v > 1.20),
         (f"Expected loss backtest. On the same exposure-at-default base, lifetime expected loss is {el:.3f} "
          f"times realized loss out of time (predicted LGD {r['lgd_oot']['predicted_mean']:.3f} vs realized "
          f"{r['lgd_oot']['realized_mean']:.3f}). {lgd_text}")),
        ("Medium" if gap > 0.01 else "Low",
         (f"Discrimination. Out-of-time AUC {champ['auc']:.3f} against {r['benchmark_grade_auc_oot']:.3f} for "
          f"Lending Club's grade and {chall['auc']:.3f} for the XGBoost challenger on all features. Restricted "
          f"to the scorecard's features, XGBoost reaches {r['challenger_same_features_auc_oot']:.3f}: {source}.")),
        ("Medium" if shifts else "Low",
         f"Stability. Score PSI {r['score_psi']:.3f}. Variables flagged: {_esc(_psi_text(r))}."),
        ("Low",
         (f"LGD differentiation. Predicted LGD ranges only from {r['lgd_oot']['pred_min']:.3f} to "
          f"{r['lgd_oot']['pred_max']:.3f}: the LGD model works as a segment average rather than a ranking tool.")),
    ]
    high = any(s == "High" for s, _ in findings)
    opinion = ("Not fit for loss estimation in its current form. Acceptable for rank-ordering applicants, "
               "subject to recalibration before any use of PD levels." if high else
               "Fit for purpose with conditions: address the findings below at the next review.")
    items = "\n".join(f"\\item \\textbf{{[{s}]}} {t}" for s, t in findings)
    return rf"""\section*{{Scope}}
Independent (second-line) review of a probability-of-default scorecard for unsecured 36-month consumer loans.
The target is a {_esc(HORIZON)}. Development: vintages 2007--2013 (80/20 split by loan id); out-of-time
validation: vintages 2014--2015, the latest vintages fully matured at the 2018Q4 data cut-off.

\section*{{Overall opinion}}
{opinion}

\section*{{Findings}}
\begin{{enumerate}}
{items}
\end{{enumerate}}

\section*{{Model choice: accuracy versus explainability}}
The scorecard is additive in weight-of-evidence bins, so every point of the score can be traced to an
applicant attribute and explained in an adverse-action notice. {choice}

\section*{{Limitations}}
Approved applicants only (no reject inference); 60-month loans and vintages after 2015 excluded because
they are immature; US platform data; exposure at default approximated as the funded amount times the average
EAD/funded ratio of development defaults ({r['ead_factor']:.3f}); LGD ignores discounting; the challenger
used an early-stopping set carved out of the training sample, so the holdout remains independent.

\section*{{Recommendations}}
\begin{{enumerate}}
\item Recalibrate the PD level on the most recent matured vintages (calibration-in-the-large, then slope)
and re-run the binomial and Hosmer--Lemeshow tests before using PD levels.
\item Enrich the application data with bureau attributes available at origination (for example
revolving balance-to-limit ratios and recent account openings), and re-assess discrimination.
\item Replace the constant exposure factor with an amortization-based EAD model.
\item Monitor the flagged variables and the score PSI quarterly with the thresholds 0.10 (monitor) and
0.25 (action).
\end{{enumerate}}
"""


TABLE_TITLES = {"discrimination_scorecard": "Discrimination: scorecard",
                "discrimination_challenger": "Discrimination: challenger (XGBoost)",
                "calibration_deciles_oot": "Calibration by PD decile (OOT)",
                "calibration_vintage": "Calibration by vintage", "stability": "Population stability (PSI)",
                "el_backtest": "Expected loss backtest (OOT)", "sensitivity": "Expected loss sensitivity (OOT)",
                "coefficients": "Scorecard coefficients", "information_value": "Information value"}


def _tex_table(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, float_format=lambda x: f"{x:,.3f}" if abs(x) < 1e5 else f"{x:,.0f}",
                       escape=True)


def write_validation_tex(r: dict, tables: dict, path) -> Path:
    champ = _row(r["discrimination"], "oot")
    body = "\n".join(f"\\subsection*{{{TABLE_TITLES.get(name, name.replace('_', ' '))}}}\n{_tex_table(t)}"
                     for name, t in tables.items())
    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.2cm]{{geometry}}\usepackage{{booktabs}}\usepackage{{graphicx}}
\title{{Validation report: PD scorecard on Lending Club 36-month loans}}\author{{Rodrigo Grandez}}\date{{}}
\begin{{document}}\maketitle
\noindent\textbf{{Headline (out-of-time 2014--2015).}} AUC {champ['auc']:.3f}, Gini {champ['gini']:.3f},
KS {champ['ks']:.3f}; calibration ratio {r['calibration_oot']['ratio']:.3f}; lifetime EL / realized loss
{r['el_total_ratio_oot']:.3f}; score PSI {r['score_psi']:.3f}.
{validation_narrative(r)}
\section*{{Supporting tables}}
{body}
\section*{{Figures}}
\includegraphics[width=.48\textwidth]{{figures/roc.png}}\hfill\includegraphics[width=.48\textwidth]{{figures/calibration_vintage.png}}
\includegraphics[width=.48\textwidth]{{figures/psi.png}}\hfill\includegraphics[width=.48\textwidth]{{figures/shap.png}}
\end{{document}}
"""
    path = Path(path)
    path.write_text(tex, encoding="utf-8")
    return path


def compile_pdf(tex: Path):
    exe = shutil.which("pdflatex")
    if not exe:
        return None
    for _ in range(2):
        subprocess.run([exe, "-interaction=nonstopmode", tex.name], cwd=tex.parent, check=False,
                       stdout=subprocess.DEVNULL)
    pdf = tex.with_suffix(".pdf")
    return pdf if pdf.exists() else None
