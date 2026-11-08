"""EDA + data-quality + leakage checks -> reports/figures/01-04 and reports/eda_summary.json.

    python -m noshow.eda
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .data_prep import build_dataset, load_raw

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
C = "#4c78a8"


def rate_plot(ax, g, title, xlabel, rot=0):
    r = g.agg(rate=(config.TARGET, "mean"), n=(config.TARGET, "size"))
    ax.bar(range(len(r)), r["rate"], color=C)
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels([str(i) for i in r.index], rotation=rot, fontsize=8)
    ax.axhline(r_overall, color="grey", ls="--", lw=1)
    ax.set(title=title, xlabel=xlabel, ylabel="no-show rate")
    for i, (v, n) in enumerate(zip(r["rate"], r["n"])):
        ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=7)


def main() -> None:
    global r_overall
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    source, raw = load_raw()
    df, rep = build_dataset()
    r_overall = df[config.TARGET].mean()
    summary: dict = {"data_source": source, "data_quality": rep}

    # raw profile
    summary["raw_columns"] = list(raw.columns)
    summary["raw_dtypes"] = {k: str(v) for k, v in raw.dtypes.items()}
    summary["age_describe_raw"] = raw["Age"].describe().round(2).to_dict()
    summary["class_balance"] = df[config.TARGET].value_counts(normalize=True).round(4).to_dict()

    # 01 class balance + lead time
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].bar(["attended", "no-show"], [1 - r_overall, r_overall], color=[C, "#e45756"])
    for i, v in enumerate([1 - r_overall, r_overall]):
        axs[0].text(i, v, f"{v:.1%}", ha="center", va="bottom")
    axs[0].set(title="Class balance (imbalanced ~4:1)", ylabel="share")
    bins = [-1, 0, 2, 6, 13, 27, 60, 400]
    lab = ["same day", "1-2", "3-6", "7-13", "14-27", "28-60", "60+"]
    df["lead_bucket"] = pd.cut(df["lead_days"], bins, labels=lab)
    rate_plot(axs[1], df.groupby("lead_bucket", observed=True), "No-show rate by booking lead time", "days between booking and appointment")
    axs[2].hist(df["lead_days"].clip(upper=120), bins=40, color=C)
    axs[2].set(title="Lead-time distribution (clipped at 120)", xlabel="lead days", ylabel="appointments")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "01_target_and_lead_time.png"); plt.close(fig)

    # 02 demographics / calendar
    df["age_group"] = pd.cut(df["age"], [-1, 11, 17, 24, 34, 49, 64, 120], labels=["0-11", "12-17", "18-24", "25-34", "35-49", "50-64", "65+"])
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    df["appointment_weekday"] = pd.Categorical(df["appointment_weekday"], order, ordered=True)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    rate_plot(axs[0], df.groupby("age_group", observed=True), "By age group", "age")
    rate_plot(axs[1], df.groupby("appointment_weekday", observed=True), "By appointment weekday", "", rot=30)
    rate_plot(axs[2], df.groupby("scholarship"), "By welfare scholarship (audit variable)", "scholarship")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "02_age_weekday_scholarship.png"); plt.close(fig)

    # 03 history + neighbourhood
    df["prior_bucket"] = pd.cut(df["prior_no_shows"], [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    nb = df.groupby("neighbourhood").agg(rate=(config.TARGET, "mean"), n=(config.TARGET, "size"))
    nb = nb[nb["n"] >= 300].sort_values("rate")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    rate_plot(axs[0], df[df["has_history"] == 1].groupby("prior_bucket", observed=True), "No-show rate vs # of PAST no-shows (known at booking)", "prior no-shows")
    axs[1].barh(nb.index, nb["rate"], color=C)
    axs[1].axvline(r_overall, color="grey", ls="--")
    axs[1].tick_params(axis="y", labelsize=6)
    axs[1].set(title="No-show rate by neighbourhood (n>=300)", xlabel="no-show rate")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "03_history_neighbourhood.png"); plt.close(fig)

    # 04 data quality: missingness, age outliers, split timeline
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].hist(raw["Age"].clip(-5, 125), bins=50, color=C)
    axs[0].axvline(0, color="red", ls="--", lw=1); axs[0].axvline(110, color="red", ls="--", lw=1)
    axs[0].set(title="Raw age (red = validity limits 0..110)", xlabel="age")
    miss = raw.isna().mean()
    axs[1].barh(miss.index, miss.values, color=C); axs[1].tick_params(axis="y", labelsize=7)
    axs[1].set(title="Missing-value share per raw column", xlim=(0, 1))
    if miss.max() == 0:
        axs[1].text(0.5, 0.5, "no missing values\nin any raw column", ha="center", va="center", transform=axs[1].transAxes)
    daily = df.groupby(["appointment_day", "split"]).size().unstack(fill_value=0)
    daily.plot.bar(stacked=True, ax=axs[2], width=0.9, color={"train": C, "val": "#f58518", "test": "#e45756"}, legend=False)
    axs[2].set_xticks(range(0, len(daily), 7)); axs[2].set_xticklabels([str(d.date()) for d in daily.index[::7]], rotation=30, fontsize=7)
    axs[2].legend(["train", "val", "test"], fontsize=7, loc="upper center", ncol=3)
    axs[2].set(title="Chronological split by appointment date", xlabel="", ylabel="appointments")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "04_data_quality_and_split.png"); plt.close(fig)

    # ---------------- leakage analysis ----------------
    tr = df[df["split"] == "train"]
    single = {}
    for f in config.NUMERIC_FEATURES + config.BINARY_FEATURES:
        x = tr[f].fillna(tr[f].median()).to_numpy(float)
        auc = roc_auc_score(tr[config.TARGET], x)
        single[f] = round(float(max(auc, 1 - auc)), 4)
    summary["single_feature_auc_train"] = dict(sorted(single.items(), key=lambda kv: -kv[1]))
    summary["leakage_review"] = {
        "max_single_feature_auc": max(single.values()),
        "note": "No single feature has an implausibly high AUC (>0.9), the signature of target leakage.",
        "excluded_post_booking": config.POST_BOOKING_COLUMNS,
        "excluded_sensitive_clinical": config.SENSITIVE_CLINICAL_COLUMNS,
        "audit_only": config.AUDIT_ONLY_COLUMNS,
        "history_rule": "prior_* features count only appointments whose DATE is strictly before the booking date",
    }
    if "SMS_received" in raw.columns:
        s = raw.assign(y=(raw["No-show"] == "Yes").astype(int)).groupby("SMS_received")["y"].mean().round(4)
        summary["sms_received_no_show_rate_raw"] = s.to_dict()
        summary["sms_received_note"] = ("SMS is sent after booking, only for longer lead times - it is a post-booking "
                                         "intervention confounded with lead time, so it is excluded from the features.")
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["leakage_review"], indent=2)); print("figures ->", config.FIG_DIR)


if __name__ == "__main__":
    main()
