import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


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


def results_markdown(r: dict) -> str:
    champ, chall = _row(r["discrimination"], "oot"), _row(r["challenger_discrimination"], "oot")
    lines = ["| Out-of-time (2014-2015) | AUC | Gini | KS |", "|---|---|---|---|",
             f"| Scorecard (WoE logistic) | {champ['auc']:.3f} | {champ['gini']:.3f} | {champ['ks']:.3f} |",
             f"| Challenger (XGBoost) | {chall['auc']:.3f} | {chall['gini']:.3f} | {chall['ks']:.3f} |",
             (f"| Benchmark: Lending Club grade | {r['benchmark_grade_auc_oot']:.3f} | "
              f"{2 * r['benchmark_grade_auc_oot'] - 1:.3f} | – |"), "",
             (f"Score PSI (development vs OOT): **{r['score_psi']:.3f}** · Hosmer-Lemeshow OOT p-value: "
              f"**{r['hosmer_lemeshow_oot']['pvalue']:.3g}** · LGD OOT predicted vs realized: "
              f"**{r['lgd_oot']['predicted_mean']:.3f}** vs **{r['lgd_oot']['realized_mean']:.3f}**")]
    return "\n".join(lines)


def update_readme(readme: Path, results: dict) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    block = f"{START}\n{results_markdown(results)}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL)
    Path(readme).write_text(new, encoding="utf-8")


def write_model_card(r: dict, path) -> Path:
    champ = _row(r["discrimination"], "oot")
    text = f"""# Model card — PD scorecard (Lending Club, 36-month loans)

**Purpose.** Illustrative probability-of-default model for unsecured consumer loans, built to demonstrate
development and second-line validation practice. Not for production credit decisions.

**Data.** {r['data_note']} Sample sizes: {r['sample_sizes']}.

**Model.** WoE-binned logistic regression (scorecard, PDO 20, 600 points at 50:1 odds). Challenger: XGBoost.
Lending Club's own grade and interest rate are excluded from the features and used only as a benchmark.

**Performance (out-of-time 2014-2015).** AUC {champ['auc']:.3f}, Gini {champ['gini']:.3f}, KS {champ['ks']:.3f}.
Hosmer-Lemeshow p-value {r['hosmer_lemeshow_oot']['pvalue']:.3g}. Score PSI {r['score_psi']:.3f}.

**Limitations.** Only resolved 36-month loans (60-month and post-2015 vintages are immature at the data
cut-off); approved applicants only (no reject inference); US platform data from 2007-2015; EAD at
origination equals the funded amount.
"""
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path


def _tex_table(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=True)


def write_validation_tex(r: dict, tables: dict, path) -> Path:
    champ = _row(r["discrimination"], "oot")
    body = "\n".join(f"\\subsection*{{{name.replace('_', ' ').capitalize()}}}\n{_tex_table(t)}"
                     for name, t in tables.items())
    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.2cm]{{geometry}}\usepackage{{booktabs}}\usepackage{{graphicx}}
\title{{Validation report: PD scorecard on Lending Club 36-month loans}}\author{{Rodrigo Grandez}}\date{{}}
\begin{{document}}\maketitle
\section*{{Summary}}
Out-of-time (2014--2015) AUC {champ['auc']:.3f}, Gini {champ['gini']:.3f}, KS {champ['ks']:.3f};
Hosmer--Lemeshow p-value {r['hosmer_lemeshow_oot']['pvalue']:.3g}; score PSI {r['score_psi']:.3f}.
\section*{{Findings and tables}}
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
