from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_ks(curves: dict, path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax.legend(frameon=False)
    return _save(fig, path)


def calibration_by_vintage(table, path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(table["group"].astype(str), table["observed_dr"], marker="o", label="Observed default rate")
    ax.plot(table["group"].astype(str), table["mean_pd"], marker="s", label="Mean predicted PD")
    ax.set(xlabel="Vintage", ylabel="Rate", title="Calibration by vintage")
    ax.legend(frameon=False)
    return _save(fig, path)


def psi_bars(table, path) -> Path:
    t = table.sort_values("psi")
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(t) + 1.5))
    ax.barh(t["variable"], t["psi"])
    for x, style in [(0.10, ":"), (0.25, "--")]:
        ax.axvline(x, ls=style, c="grey", lw=1)
    ax.set(xlabel="PSI (development vs OOT)", title="Population stability")
    return _save(fig, path)


def shap_summary(values, sample, path) -> Path:
    shap.summary_plot(values, sample, show=False, plot_size=(6, 4.5))
    return _save(plt.gcf(), path)
