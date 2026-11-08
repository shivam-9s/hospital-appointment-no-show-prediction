"""End-to-end, repeatable training pipeline.

    python -m noshow.train

Steps: build dataset (clean + features + chronological split) -> baselines -> 3 candidate models
(tuned on the VALIDATION set only) -> calibrate + choose a cost-based threshold on validation ->
evaluate everything ONCE on the untouched TEST set -> save artifact, metrics.json and figures.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import config
from .data_prep import build_dataset
from .features import PlattCalibrator, build_preprocessor

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------------------
# metric helpers
# ----------------------------------------------------------------------------------------
def business_cost(y, pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE)


def threshold_metrics(y, score, thr) -> dict:
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "flag_rate": round(float(pred.mean()), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "business_cost": float(fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE),
        "cost_per_appointment": round(float((fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE) / len(y)), 4),
    }


def best_threshold(y, score, objective: str) -> float:
    grid = np.unique(np.round(np.quantile(score, np.linspace(0.02, 0.98, 241)), 5))
    if objective == "f1":
        vals = [f1_score(y, (score >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(vals))])
    vals = [business_cost(y, (score >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(vals))])


def rank_metrics(y, score) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "pr_auc": round(float(average_precision_score(y, score)), 4),
    }


# ----------------------------------------------------------------------------------------
# model zoo (tuned on validation only)
# ----------------------------------------------------------------------------------------
def candidate_grids():
    seed = config.SEED
    return {
        "logistic_regression": (
            True,
            [
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
                for c in (0.03, 0.3, 3.0)
            ],
            [{"C": c} for c in (0.03, 0.3, 3.0)],
        ),
        "random_forest": (
            False,
            [
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=msl, max_depth=md, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=seed,
                )
                for msl, md in ((30, 8), (60, 10), (100, None))
            ],
            [{"min_samples_leaf": a, "max_depth": b} for a, b in ((30, 8), (60, 10), (100, None))],
        ),
        "hist_gradient_boosting": (
            False,
            [
                HistGradientBoostingClassifier(
                    learning_rate=lr, max_depth=md, max_iter=it, min_samples_leaf=100, l2_regularization=1.0,
                    class_weight="balanced", early_stopping=False, random_state=seed,
                )
                for lr, md, it in ((0.05, 3, 150), (0.05, 4, 250), (0.03, 3, 300))
            ],
            [{"learning_rate": a, "max_depth": b, "max_iter": c} for a, b, c in ((0.05, 3, 150), (0.05, 4, 250), (0.03, 3, 300))],
        ),
    }


def score_of(pipe, X) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]


def main() -> None:
    np.random.seed(config.SEED)
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    config.FIG_DIR.mkdir(exist_ok=True, parents=True)

    df, data_report = build_dataset()
    df.to_csv(config.DATA_PROCESSED / config.CLEAN_FILENAME, index=False)
    (config.DATA_PROCESSED / "data_quality_report.json").write_text(json.dumps(data_report, indent=2))

    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    Xtr, ytr = tr[config.FEATURES], tr[config.TARGET].to_numpy()
    Xva, yva = va[config.FEATURES], va[config.TARGET].to_numpy()
    Xte, yte = te[config.FEATURES], te[config.TARGET].to_numpy()
    print(f"data source={data_report['data_source']}  train={len(tr):,} val={len(va):,} test={len(te):,}")

    results: dict = {"baselines": {}, "models": {}, "tuning": {}}
    fitted: dict = {}

    # ---------------- baselines ----------------
    # B0: never predict a no-show (what happens today with no model)
    results["baselines"]["never_flag"] = {
        **threshold_metrics(yte, np.zeros(len(yte)), 0.5), "roc_auc": 0.5, "pr_auc": round(float(yte.mean()), 4),
    }
    # B1: flag everybody (remind everyone)
    results["baselines"]["flag_everyone"] = {
        **threshold_metrics(yte, np.ones(len(yte)), 0.5), "roc_auc": 0.5, "pr_auc": round(float(yte.mean()), 4),
    }
    # B2: one-feature rule - longer booking lead time => higher risk; cut-off picked on validation (max F1)
    lead_va, lead_te = Xva["lead_days"].to_numpy(float), Xte["lead_days"].to_numpy(float)
    cut = best_threshold(yva, lead_va, "f1")
    results["baselines"]["lead_time_rule"] = {
        **threshold_metrics(yte, lead_te, cut), **rank_metrics(yte, lead_te), "rule": f"flag if lead_days >= {cut:g}",
    }
    # B3: Dummy classifier that outputs the training prior (ranking-free reference for Brier)
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    results["baselines"]["train_prior_probability"] = {
        "brier": round(float(brier_score_loss(yte, dummy.predict_proba(Xte)[:, 1])), 4)
    }

    # ---------------- candidates, tuned on validation PR-AUC ----------------
    for name, (scale, estimators, grid) in candidate_grids().items():
        best = None
        tuning_rows = []
        for est, params in zip(estimators, grid):
            pipe = Pipeline([("prep", build_preprocessor(scale)), ("clf", est)]).fit(Xtr, ytr)
            ap = average_precision_score(yva, score_of(pipe, Xva))
            tuning_rows.append({"params": params, "val_pr_auc": round(float(ap), 4)})
            if best is None or ap > best[0]:
                best = (ap, pipe, params)
        results["tuning"][name] = tuning_rows
        fitted[name] = {"pipe": best[1], "params": best[2], "val_pr_auc": float(best[0])}
        print(f"{name:24s} best val PR-AUC={best[0]:.4f} params={best[2]}")

    # ---------------- evaluate on TEST (thresholds chosen on validation) ----------------
    for name, f in fitted.items():
        s_va, s_te = score_of(f["pipe"], Xva), score_of(f["pipe"], Xte)
        thr = best_threshold(yva, s_va, "f1")
        results["models"][name] = {
            "best_params": f["params"],
            "val_pr_auc": round(f["val_pr_auc"], 4),
            "selection_threshold_rule": "max F1 on validation (raw, class-weighted scores)",
            **threshold_metrics(yte, s_te, thr),
            **rank_metrics(yte, s_te),
            "brier_raw": round(float(brier_score_loss(yte, s_te)), 4),
        }
        f["s_va"], f["s_te"] = s_va, s_te

    # ---------------- choose the winner on VALIDATION only ----------------
    winner = max(fitted, key=lambda k: fitted[k]["val_pr_auc"])
    w = fitted[winner]
    results["selected_model"] = winner
    print(f"selected model (by validation PR-AUC): {winner}")

    # ---------------- calibration (Platt scaling, fit on validation) + cost-based threshold ----------------
    iso = PlattCalibrator().fit(w["s_va"], yva)
    p_va, p_te = iso.predict(w["s_va"]), iso.predict(w["s_te"])
    cost_thr = best_threshold(yva, p_va, "cost")
    theory_thr = config.COST_FALSE_POSITIVE / (config.COST_FALSE_POSITIVE + config.COST_FALSE_NEGATIVE)
    results["final_model"] = {
        "name": winner,
        "calibration": "Platt scaling (logistic regression on logit score) fitted on the validation set",
        "operating_threshold": round(cost_thr, 4),
        "theoretical_cost_threshold": round(theory_thr, 4),
        "test": {
            **threshold_metrics(yte, p_te, cost_thr),
            **rank_metrics(yte, p_te),
            "brier_calibrated": round(float(brier_score_loss(yte, p_te)), 4),
            "brier_raw_uncalibrated": round(float(brier_score_loss(yte, w["s_te"])), 4),
            "brier_reference_train_prior": results["baselines"]["train_prior_probability"]["brier"],
        },
        "test_at_f1_threshold": threshold_metrics(yte, p_te, best_threshold(yva, p_va, "f1")),
        "test_at_theoretical_threshold": threshold_metrics(yte, p_te, theory_thr),
    }
    # threshold sweep table on TEST for the report / dashboard (read-only, nothing is tuned on it)
    sweep = [threshold_metrics(yte, p_te, t) for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)]
    results["threshold_sweep_test"] = sweep

    # top-k precision (capacity-constrained call centre)
    order = np.argsort(-p_te)
    results["precision_at_top_k_test"] = {
        f"top_{int(q*100)}pct": {
            "precision": round(float(yte[order[: int(len(yte) * q)]].mean()), 4),
            "lift_vs_base_rate": round(float(yte[order[: int(len(yte) * q)]].mean() / yte.mean()), 3),
        }
        for q in (0.05, 0.10, 0.20, 0.30)
    }

    # ---------------- risk bands (uncertainty communication) from VALIDATION ----------------
    edges = [0.0, 0.10, 0.20, 0.30, 0.45, 1.0001]
    labels = ["very low", "low", "medium", "high", "very high"]
    bands = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (p_va >= lo) & (p_va < hi)
        bands.append({"label": lab, "low": lo, "high": min(hi, 1.0), "n_validation": int(m.sum()),
                      "observed_no_show_rate_validation": round(float(yva[m].mean()), 4) if m.sum() else None})
    results["risk_bands_validation"] = bands

    # ---------------- fairness / subgroup audit on TEST (gender is audit-only, never a feature) ----------------
    te_a = te.assign(p=p_te, flag=(p_te >= cost_thr).astype(int))
    te_a["age_group"] = pd.cut(te_a["age"], [-1, 17, 34, 64, 120], labels=["0-17", "18-34", "35-64", "65+"])
    audit = {}
    for col in ("gender", "age_group", "scholarship"):
        rows = []
        for key, g in te_a.groupby(col, observed=True):
            yy, ff = g[config.TARGET].to_numpy(), g["flag"].to_numpy()
            rows.append({
                "group": str(key), "n": int(len(g)),
                "observed_no_show_rate": round(float(yy.mean()), 4),
                "mean_predicted_probability": round(float(g["p"].mean()), 4),
                "flag_rate": round(float(ff.mean()), 4),
                "recall": round(float(recall_score(yy, ff, zero_division=0)), 4),
                "precision": round(float(precision_score(yy, ff, zero_division=0)), 4),
            })
        audit[col] = rows
    results["subgroup_audit_test"] = audit

    # ---------------- permutation importance (validation) ----------------
    pi = permutation_importance(
        w["pipe"], Xva, yva, scoring="average_precision", n_repeats=5, random_state=config.SEED, n_jobs=-1
    )
    imp = (
        pd.DataFrame({"feature": config.FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
        .sort_values("importance", ascending=False)
    )
    results["permutation_importance_validation"] = imp.round(5).to_dict("records")

    # ---------------- figures ----------------
    make_figures(results, fitted, winner, yte, p_te, w["s_te"], p_va, yva, imp, cost_thr, te_a)

    # ---------------- artifact ----------------
    ref = {
        "numeric_median": {c: float(tr[c].median()) for c in config.NUMERIC_FEATURES},
        # "no information" reference used by the occlusion-style explanation
        "explain_reference": {
            "age": float(tr["age"].median()), "lead_days": float(tr["lead_days"].median()),
            "scheduled_hour": float(tr["scheduled_hour"].median()), "prior_appointments": 0,
            "prior_no_shows": 0, "days_since_last_appointment": None, "scholarship": 0,
            "appointment_weekday": tr["appointment_weekday"].mode()[0],
            "scheduled_weekday": tr["scheduled_weekday"].mode()[0],
            "neighbourhood": tr["neighbourhood"].mode()[0],
        },
        "known_neighbourhoods": sorted(tr["neighbourhood"].unique().tolist()),
        "age_range": [int(tr["age"].min()), int(tr["age"].max())],
        "lead_days_range": [int(tr["lead_days"].min()), int(tr["lead_days"].max())],
    }
    artifact = {
        "pipeline": w["pipe"],
        "calibrator": iso,
        "threshold": cost_thr,
        "risk_bands": bands,
        "reference": ref,
        "features": config.FEATURES,
        "metadata": {
            "model_name": winner,
            "model_params": w["params"],
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": data_report["data_source"],
            "seed": config.SEED,
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
            "cost_false_positive": config.COST_FALSE_POSITIVE, "cost_false_negative": config.COST_FALSE_NEGATIVE,
            "test_metrics": results["final_model"]["test"],
        },
    }
    joblib.dump(artifact, config.MODEL_PATH, compress=3)
    results["data_quality"] = data_report
    config.METRICS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved {config.MODEL_PATH}\nsaved {config.METRICS_PATH}")
    print(json.dumps(results["final_model"]["test"], indent=2))


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def make_figures(results, fitted, winner, yte, p_te, s_te_raw, p_va, yva, imp, thr, te_a):
    F = config.FIG_DIR
    colors = {"logistic_regression": "#4c78a8", "random_forest": "#f58518", "hist_gradient_boosting": "#54a24b"}

    # 1. PR curves (test)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n, f in fitted.items():
        pr, rc, _ = precision_recall_curve(yte, f["s_te"])
        ax.plot(rc, pr, label=f"{n} (AP={results['models'][n]['pr_auc']:.3f})", color=colors[n])
    ax.axhline(yte.mean(), ls="--", color="grey", label=f"base rate {yte.mean():.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (held-out test)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "05_pr_curves.png"); plt.close(fig)

    # 2. calibration before / after
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for lab, s, c in (("raw class-weighted score", s_te_raw, "#e45756"), ("after Platt calibration", p_te, "#4c78a8")):
        fr, mp = calibration_curve(yte, s, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", label=lab, color=c)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.set(xlabel="Predicted probability", ylabel="Observed no-show rate", title=f"Calibration ({winner}, test)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "06_calibration.png"); plt.close(fig)

    # 3. cost vs threshold (validation-chosen threshold marked)
    ts = np.linspace(0.05, 0.6, 56)
    costs = [business_cost(yte, (p_te >= t).astype(int)) / len(yte) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, costs, color="#4c78a8")
    ax.axvline(thr, color="#e45756", ls="--", label=f"chosen threshold {thr:.2f} (picked on validation)")
    ax.axhline(results["baselines"]["never_flag"]["cost_per_appointment"], color="grey", ls=":", label="never remind (status quo)")
    ax.set(xlabel="Decision threshold", ylabel="Expected cost per appointment",
           title=f"Business cost (FP={config.COST_FALSE_POSITIVE:g}, FN={config.COST_FALSE_NEGATIVE:g})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "07_cost_vs_threshold.png"); plt.close(fig)

    # 4. confusion matrix at the operating point
    cm = confusion_matrix(yte, (p_te >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["attends", "no-show"], yticklabels=["attends", "no-show"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix @ threshold {thr:.2f} (test)")
    fig.tight_layout(); fig.savefig(F / "08_confusion_matrix.png"); plt.close(fig)

    # 5. permutation importance
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#4c78a8")
    ax.set(xlabel="Drop in PR-AUC when shuffled (validation)", title="Permutation importance")
    fig.tight_layout(); fig.savefig(F / "09_permutation_importance.png"); plt.close(fig)

    # 6. model comparison bars
    names = list(results["models"])
    metrics = ["precision", "recall", "f1", "pr_auc"]
    x = np.arange(len(names)); wdt = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + i * wdt, [results["models"][n][m] for n in names], wdt, label=m)
    ax.set_xticks(x + 1.5 * wdt); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set(title="Candidate models on held-out test (threshold = max-F1 on validation)", ylim=(0, 1))
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(F / "10_model_comparison.png"); plt.close(fig)

    # 7. observed vs predicted by age group (subgroup)
    g = te_a.groupby("age_group", observed=True).agg(obs=(config.TARGET, "mean"), pred=("p", "mean"))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    g.plot.bar(ax=ax, color=["#4c78a8", "#f58518"], rot=0)
    ax.set(title="Subgroup check: observed vs predicted no-show rate (test)", xlabel="age group", ylabel="rate")
    ax.legend(["observed", "mean predicted"], fontsize=8)
    fig.tight_layout(); fig.savefig(F / "11_subgroup_age.png"); plt.close(fig)


if __name__ == "__main__":
    main()
