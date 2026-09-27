import json

import numpy as np
import pandas as pd

from credit_risk import plots, report

RESULTS = {
    "discrimination": [{"sample": "train", "n": 100, "default_rate": 0.12, "auc": 0.71, "gini": 0.42, "ks": 0.31},
                       {"sample": "oot", "n": 80, "default_rate": 0.14, "auc": 0.6912, "gini": 0.3824, "ks": 0.29}],
    "challenger_discrimination": [{"sample": "oot", "n": 80, "default_rate": 0.14, "auc": 0.7034, "gini": 0.4068,
                                   "ks": 0.3}],
    "benchmark_grade_auc_oot": 0.70, "hosmer_lemeshow_oot": {"stat": 1200.0, "pvalue": 0.0},
    "calibration_oot": {"mean_pd": 0.124, "observed_dr": 0.144, "ratio": 0.861, "slope": 0.93, "brier": 0.12,
                        "max_decile_gap_pp": 2.4},
    "challenger_same_features_auc_oot": 0.6451, "iv_total": 0.45,
    "score_psi": 0.03, "psi_flags": [{"variable": "purpose", "psi": 0.106, "status": "monitor"}],
    "lgd_oot": {"predicted_mean": 0.9, "realized_mean": 0.88, "pred_min": 0.85, "pred_max": 0.92},
    "ead_factor": 0.58, "el_total_ratio_oot": 0.847,
    "el_backtest": [{"vintage": 2014, "n": 10, "exposure": 1e5, "expected_loss": 9e3, "realized_loss": 1e4,
                     "ratio": 0.9}],
    "sample_sizes": {"train": 100, "holdout": 20, "oot": 80}, "data_note": "36-month loans.",
}


def test_validation_narrative_has_opinion_findings_and_recommendations():
    text = report.validation_narrative(RESULTS)
    for heading in ["Scope", "Overall opinion", "Findings", "Limitations", "Recommendations"]:
        assert heading in text
    assert "lifetime" in text.lower()
    assert "High" in text                         # calibration ratio 0.861 < 0.90 -> high-severity finding
    assert "0.645" in text and "0.847" in text     # same-feature challenger and EL ratio come from results
    assert "purpose" in text


def test_write_results_roundtrip(tmp_path):
    r = dict(RESULTS, extra=np.float64(1.5))
    p = report.write_results(r, tmp_path / "results.json")
    assert json.loads(p.read_text())["extra"] == 1.5


def test_update_readme_uses_results(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# T\n<!-- RESULTS:START -->\nold 0.99\n<!-- RESULTS:END -->\ntail\n", encoding="utf-8")
    report.update_readme(readme, RESULTS)
    text = readme.read_text(encoding="utf-8")
    assert "old 0.99" not in text and "0.691" in text and "0.703" in text and text.endswith("tail\n")
    assert "<1e-16" in text and "0.861" in text and "0.847" in text and "lifetime" in text.lower()


def test_model_card_and_tex(tmp_path):
    mc = report.write_model_card(RESULTS, tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "0.691" in mc and "Limitations" in mc and "lifetime" in mc.lower() and "{'train'" not in mc
    tables = {"stability": pd.DataFrame({"variable": ["a"], "psi": [0.01], "status": ["stable"]})}
    tex = report.write_validation_tex(RESULTS, tables, tmp_path / "v.tex").read_text(encoding="utf-8")
    assert r"\begin{document}" in tex and "0.691" in tex


def test_plots_create_files(tmp_path):
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 300); p = rng.random(300)
    assert plots.roc_ks({"oot": (y, p)}, tmp_path / "roc.png").stat().st_size > 0
    cal = pd.DataFrame({"group": [2014, 2015], "observed_dr": [.1, .12], "mean_pd": [.11, .11]})
    assert plots.calibration_by_vintage(cal, tmp_path / "cal.png").exists()
    psi_t = pd.DataFrame({"variable": ["a", "score"], "psi": [.02, .3], "status": ["stable", "shift"]})
    assert plots.psi_bars(psi_t, tmp_path / "psi.png").exists()
    vals = rng.normal(size=(50, 2)); sample = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
    assert plots.shap_summary(vals, sample, tmp_path / "shap.png").exists()


def test_pipeline_smoke(tmp_path, monkeypatch):
    from conftest import make_raw

    from credit_risk import config, pipeline
    from credit_risk.data import clean
    df = clean(make_raw(n=6000, seed=5))
    df = df.assign(vintage=np.where(df.index % 2 == 0, 2012, 2014))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES", tmp_path / "reports/figures")
    monkeypatch.setattr(config, "TABLES", tmp_path / "reports/tables")
    monkeypatch.setattr(pipeline, "INTERIM", tmp_path / "i.parquet")
    monkeypatch.setattr(pipeline, "MODELS", tmp_path / "m.pkl")
    (tmp_path / "README.md").write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    df.to_parquet(tmp_path / "i.parquet")
    pipeline.stage_models()
    pipeline.stage_report()
    res = json.loads((tmp_path / "reports/results.json").read_text())
    assert {"discrimination", "score_psi", "el_backtest", "calibration_oot", "el_total_ratio_oot",
            "challenger_same_features_auc_oot", "ead_factor", "psi_flags"} <= set(res)
    assert "Overall opinion" in (tmp_path / "reports/validation_report.tex").read_text(encoding="utf-8")
    assert "Scorecard (WoE logistic)" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_narrative_conclusions_follow_the_numbers():
    shifted = dict(RESULTS, score_psi=0.3, challenger_same_features_auc_oot=0.70,
                   lgd_oot=dict(RESULTS["lgd_oot"], predicted_mean=0.80))
    base, alt = report.validation_narrative(RESULTS), report.validation_narrative(shifted)
    assert "population is stable" in base and "has also shifted" in alt
    assert "extra features" in base and "non-linear form" in alt
    assert "driven mainly by PD" in base and "also off on average" in alt
