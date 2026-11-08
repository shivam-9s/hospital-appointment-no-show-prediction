"""Inference: load the saved artifact and score bookings. Used by the API, the dashboard and the tests.

Nothing here is hard-coded: probabilities always come from artifact["pipeline"] + artifact["calibrator"].
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import add_derived_features

DISCLAIMER = (
    "Decision support only. This is an estimate of attendance behaviour from booking data - it says nothing about "
    "diagnosis, urgency or clinical need. A member of staff must review every flag before any reminder is sent, "
    "and no appointment may be cancelled, double-booked or deprioritised because of this score."
)

# which raw inputs are perturbed together when explaining a prediction
EXPLAIN_GROUPS = {
    "booking lead time": ["lead_days"],
    "age": ["age"],
    "neighbourhood": ["neighbourhood"],
    "past attendance history": ["prior_appointments", "prior_no_shows", "days_since_last_appointment"],
    "welfare scholarship": ["scholarship"],
    "booking hour": ["scheduled_hour"],
    "appointment weekday": ["appointment_weekday"],
    "booking weekday": ["scheduled_weekday"],
}


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    p = path or str(config.MODEL_PATH)
    return joblib.load(p)



def raw_frame(records: list[dict]) -> pd.DataFrame:
    """Turn API-level records into model-input columns."""
    rows = []

    for r in records:
        booked = pd.Timestamp(r["booking_datetime"])
        appt = pd.Timestamp(r["appointment_date"]).normalize()

        # Calculate calendar-day difference.
        # This avoids subtracting timezone-aware and timezone-naive timestamps.
        lead_days = (appt.date() - booked.date()).days

        rows.append(
            {
                "age": int(r["age"]),
                "lead_days": lead_days,
                "scheduled_hour": booked.hour,
                "prior_appointments": int(
                    r.get("prior_appointments") or 0
                ),
                "prior_no_shows": int(
                    r.get("prior_no_shows") or 0
                ),
                "days_since_last_appointment":
                    r.get("days_since_last_appointment"),
                "scholarship": int(
                    bool(r.get("scholarship", 0))
                ),
                "appointment_weekday": appt.day_name(),
                "scheduled_weekday": booked.day_name(),
                "neighbourhood": str(
                    r.get("neighbourhood", "")
                ).strip().upper(),
            }
        )

    df = pd.DataFrame(rows)

    df["days_since_last_appointment"] = pd.to_numeric(
        df["days_since_last_appointment"],
        errors="coerce"
    )

    return df



def model_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    return add_derived_features(raw)[config.FEATURES]


def predict_proba(raw: pd.DataFrame, artifact: dict | None = None) -> np.ndarray:
    art = artifact or load_artifact()
    score = art["pipeline"].predict_proba(model_matrix(raw))[:, 1]
    return np.clip(art["calibrator"].predict(score), 0.0, 1.0)


def risk_band(p: float, art: dict) -> dict:
    for b in art["risk_bands"]:
        if b["low"] <= p < b["high"] or (p >= b["high"] and b["high"] >= 1.0):
            return b
    return art["risk_bands"][-1]


def input_warnings(raw_row: pd.Series, art: dict) -> list[str]:
    ref = art["reference"]
    w = []
    if raw_row["neighbourhood"] not in ref["known_neighbourhoods"]:
        w.append("Neighbourhood not seen in training data - the model treats it as 'rare/other'; treat the score with extra caution.")
    lo, hi = ref["lead_days_range"]
    if raw_row["lead_days"] > hi:
        w.append(f"Lead time {raw_row['lead_days']}d is longer than anything seen in training (max {hi}d); the model extrapolates.")
    if raw_row["prior_no_shows"] > raw_row["prior_appointments"]:
        w.append("prior_no_shows exceeds prior_appointments - please check the history fields.")
    if raw_row["prior_appointments"] == 0:
        w.append("No attendance history for this patient: the score relies on population-level patterns only (wider uncertainty).")
    return w


def explain_row(raw_row: pd.Series, art: dict, top_k: int = 5) -> list[dict]:
    """Occlusion explanation: reset one input group to a neutral reference value and measure the probability change.

    contribution = p(actual inputs) - p(group reset). Positive => this input RAISES the predicted no-show risk.
    Model-agnostic and computed with the real model, not a hard-coded table.
    """
    ref = art["reference"]["explain_reference"]
    base = raw_row.to_frame().T.reset_index(drop=True)
    base = base.astype({"age": float, "lead_days": float, "scheduled_hour": float, "prior_appointments": float,
                        "prior_no_shows": float, "scholarship": float})
    p0 = float(predict_proba(base, art)[0])
    out = []
    for label, cols in EXPLAIN_GROUPS.items():
        alt = base.copy()
        for c in cols:
            alt.loc[0, c] = np.nan if ref[c] is None else ref[c]
        if label == "past attendance history":
            alt.loc[0, "prior_no_shows"] = 0
        p1 = float(predict_proba(alt, art)[0])
        shown = ", ".join(f"{c}={raw_row[c]}" for c in cols if not (isinstance(raw_row[c], float) and np.isnan(raw_row[c])))
        out.append({"factor": label, "value": shown, "effect_on_probability": round(p0 - p1, 4),
                    "direction": "raises risk" if p0 - p1 > 0.0005 else "lowers risk" if p0 - p1 < -0.0005 else "neutral"})
    out.sort(key=lambda d: -abs(d["effect_on_probability"]))
    return out[:top_k]


def score_records(records: list[dict], explain: bool = True) -> list[dict]:
    art = load_artifact()
    raw = raw_frame(records)
    probs = predict_proba(raw, art)
    results = []
    for i, p in enumerate(probs):
        row = raw.iloc[i]
        band = risk_band(float(p), art)
        res = {
            "no_show_probability": round(float(p), 4),
            "risk_band": band["label"],
            "historical_no_show_rate_in_band": band["observed_no_show_rate_validation"],
            "flag_for_reminder_review": bool(p >= art["threshold"]),
            "decision_threshold": round(float(art["threshold"]), 4),
            "warnings": input_warnings(row, art),
            "human_review_required": True,
            "disclaimer": DISCLAIMER,
        }
        if explain:
            res["top_factors"] = explain_row(row, art)
        results.append(res)
    return results


def model_info() -> dict:
    art = load_artifact()
    return {"metadata": art["metadata"], "threshold": art["threshold"], "risk_bands": art["risk_bands"],
            "features": art["features"], "known_neighbourhoods": art["reference"]["known_neighbourhoods"],
            "disclaimer": DISCLAIMER}
