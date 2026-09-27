import pickle
import sys

import pandas as pd

from credit_risk import config, plots, report
from credit_risk.challenger import fit_xgb, shap_matrix, to_model_frame
from credit_risk.data import build_interim, download, feature_frame
from credit_risk.lgd import TwoStageLGD, realized_lgd
from credit_risk.metrics import auc, hosmer_lemeshow
from credit_risk.scorecard import Scorecard
from credit_risk.split import split_samples
from credit_risk.validation import (decile_calibration, discrimination_table, el_backtest, sensitivity_table,
                                    stability_table)

INTERIM = config.DATA_INTERIM / "loans_36m.parquet"
MODELS = config.DATA_INTERIM / "models.pkl"
NUM, CAT = config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES


def stage_data():
    build_interim(download(), INTERIM)


def stage_models():
    s = split_samples(pd.read_parquet(INTERIM))
    X = {k: feature_frame(v) for k, v in s.items()}
    y = {k: v["default"].to_numpy() for k, v in s.items()}
    sc = Scorecard.fit(X["train"], y["train"], NUM, CAT)
    tr, cats = to_model_frame(X["train"], CAT)
    ho, _ = to_model_frame(X["holdout"], CAT, cats)
    xgb = fit_xgb(tr, y["train"], ho, y["holdout"])
    bad_train = s["train"][s["train"]["default"] == 1]
    lgd_y = realized_lgd(bad_train)
    lgd = TwoStageLGD(NUM, CAT).fit(feature_frame(bad_train)[lgd_y.notna()], lgd_y.dropna())
    with open(MODELS, "wb") as f:
        pickle.dump({"scorecard": sc, "xgb": xgb, "cats": cats, "lgd": lgd}, f)


def _calibration_by_vintage(frames: dict, y: dict, pd_sc: dict) -> pd.DataFrame:
    rows = []
    for k, df in frames.items():
        for v, idx in df.groupby("vintage").groups.items():
            pos = df.index.get_indexer(idx)
            rows.append({"group": int(v), "n": len(pos), "observed_dr": float(y[k][pos].mean()),
                         "mean_pd": float(pd_sc[k][pos].mean())})
    t = pd.DataFrame(rows)
    w = t.assign(dr_w=t.observed_dr * t.n, pd_w=t.mean_pd * t.n).groupby("group")[["n", "dr_w", "pd_w"]].sum()
    return pd.DataFrame({"group": w.index, "n": w["n"].to_numpy(), "observed_dr": (w.dr_w / w.n).to_numpy(),
                         "mean_pd": (w.pd_w / w.n).to_numpy()})


def stage_report():
    s = split_samples(pd.read_parquet(INTERIM))
    with open(MODELS, "rb") as f:
        m = pickle.load(f)
    X = {k: feature_frame(v) for k, v in s.items()}
    y = {k: v["default"].to_numpy() for k, v in s.items()}
    pd_sc = {k: m["scorecard"].predict_pd(X[k]) for k in s}
    pd_xgb = {k: m["xgb"].predict_proba(to_model_frame(X[k], CAT, m["cats"])[0])[:, 1] for k in s}
    oot = s["oot"]
    grade_rank = oot["grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    lgd_pred_oot = m["lgd"].predict(X["oot"])
    bad_oot = (oot["default"] == 1).to_numpy()
    lgd_real_oot = realized_lgd(oot[bad_oot])

    disc = discrimination_table({k: (y[k], pd_sc[k]) for k in ["train", "holdout", "oot"]})
    disc_x = discrimination_table({k: (y[k], pd_xgb[k]) for k in ["train", "holdout", "oot"]})
    cal_dec = decile_calibration(y["oot"], pd_sc["oot"])
    cal_vint = _calibration_by_vintage({"train": s["train"], "holdout": s["holdout"], "oot": oot}, y, pd_sc)
    stab = stability_table(X["train"], X["oot"], NUM, CAT, m["scorecard"].score(X["train"]),
                           m["scorecard"].score(X["oot"]))
    elbt = el_backtest(oot, pd_sc["oot"], lgd_pred_oot)
    sens = sensitivity_table(oot, pd_sc["oot"], lgd_pred_oot)
    hl_stat, hl_p = hosmer_lemeshow(y["oot"], pd_sc["oot"])
    vals, sample = shap_matrix(m["xgb"], to_model_frame(X["oot"], CAT, m["cats"])[0])

    coefs = m["scorecard"].coefficients().rename("coef").reset_index().rename(columns={"index": "feature"})
    ivs = m["scorecard"].binner.iv_.rename("iv").reset_index().rename(columns={"index": "feature"})
    tables = {"discrimination_scorecard": disc, "discrimination_challenger": disc_x,
              "calibration_deciles_oot": cal_dec, "calibration_vintage": cal_vint, "stability": stab,
              "el_backtest": elbt, "sensitivity": sens, "coefficients": coefs, "information_value": ivs}
    config.TABLES.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(config.TABLES / f"{name}.csv", index=False)
    plots.roc_ks({"Scorecard OOT": (y["oot"], pd_sc["oot"]), "XGBoost OOT": (y["oot"], pd_xgb["oot"])},
                 config.FIGURES / "roc.png")
    plots.calibration_by_vintage(cal_vint, config.FIGURES / "calibration_vintage.png")
    plots.psi_bars(stab, config.FIGURES / "psi.png")
    plots.shap_summary(vals, sample, config.FIGURES / "shap.png")

    results = {"discrimination": disc.to_dict("records"), "challenger_discrimination": disc_x.to_dict("records"),
               "benchmark_grade_auc_oot": auc(y["oot"], grade_rank),
               "hosmer_lemeshow_oot": {"stat": hl_stat, "pvalue": hl_p},
               "score_psi": float(stab.set_index("variable").loc["score", "psi"]),
               "lgd_oot": {"predicted_mean": float(lgd_pred_oot[bad_oot].mean()),
                           "realized_mean": float(lgd_real_oot.mean())},
               "el_backtest": elbt.to_dict("records"),
               "sample_sizes": {k: int(len(v)) for k, v in s.items()},
               "selected_features": m["scorecard"].features_,
               "data_note": "Resolved 36-month Lending Club loans; development 2007-2013, out-of-time 2014-2015."}
    report.write_results(results, config.REPORTS / "results.json")
    report.update_readme(config.ROOT / "README.md", results)
    report.write_model_card(results, config.REPORTS / "model_card.md")
    tex_tables = {k: tables[k] for k in ["discrimination_scorecard", "discrimination_challenger",
                                         "calibration_deciles_oot", "stability", "el_backtest", "sensitivity",
                                         "coefficients"]}
    tex = report.write_validation_tex(results, tex_tables, config.REPORTS / "validation_report.tex")
    report.compile_pdf(tex)


STAGES = {"data": [stage_data], "models": [stage_models], "report": [stage_report],
          "all": [stage_data, stage_models, stage_report]}

if __name__ == "__main__":
    for fn in STAGES[sys.argv[1] if len(sys.argv) > 1 else "all"]:
        print(f"== {fn.__name__}", flush=True)
        fn()
