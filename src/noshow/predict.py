
"""
Inference: load the saved artifact and score bookings.

Used by the FastAPI API, Streamlit dashboard and tests.

Probabilities always come from:
artifact["pipeline"] + artifact["calibrator"].
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import add_derived_features


DISCLAIMER = (
    "Decision support only. This is an estimate of attendance behaviour "
    "from booking data - it says nothing about diagnosis, urgency or "
    "clinical need. A member of staff must review every flag before any "
    "reminder is sent, and no appointment may be cancelled, double-booked "
    "or deprioritised because of this score."
)


# Raw input groups used for model explanations
EXPLAIN_GROUPS = {
    "booking lead time": ["lead_days"],
    "age": ["age"],
    "neighbourhood": ["neighbourhood"],
    "past attendance history": [
        "prior_appointments",
        "prior_no_shows",
        "days_since_last_appointment",
    ],
    "welfare scholarship": ["scholarship"],
    "booking hour": ["scheduled_hour"],
    "appointment weekday": ["appointment_weekday"],
    "booking weekday": ["scheduled_weekday"],
}


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    """Load the trained model artifact once and cache it."""

    model_path = path or str(config.MODEL_PATH)

    return joblib.load(model_path)


def raw_frame(records: list[dict]) -> pd.DataFrame:
    """Convert API/dashboard records into model input columns."""

    rows = []

    for record in records:

        booked = pd.Timestamp(
            record["booking_datetime"]
        )

        appointment = pd.Timestamp(
            record["appointment_date"]
        ).normalize()

        # Calendar-day difference avoids timezone-aware
        # and timezone-naive timestamp subtraction.
        lead_days = (
            appointment.date() - booked.date()
        ).days

        rows.append(
            {
                "age": int(record["age"]),
                "lead_days": lead_days,
                "scheduled_hour": booked.hour,

                "prior_appointments": int(
                    record.get("prior_appointments") or 0
                ),

                "prior_no_shows": int(
                    record.get("prior_no_shows") or 0
                ),

                "days_since_last_appointment": (
                    record.get("days_since_last_appointment")
                ),

                "scholarship": int(
                    bool(record.get("scholarship", 0))
                ),

                "appointment_weekday": (
                    appointment.day_name()
                ),

                "scheduled_weekday": (
                    booked.day_name()
                ),

                "neighbourhood": str(
                    record.get("neighbourhood", "")
                ).strip().upper(),
            }
        )

    frame = pd.DataFrame(rows)

    frame["days_since_last_appointment"] = pd.to_numeric(
        frame["days_since_last_appointment"],
        errors="coerce",
    )

    return frame


def model_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Generate the exact feature matrix expected by the model."""

    return add_derived_features(raw)[config.FEATURES]


def predict_proba(
    raw: pd.DataFrame,
    artifact: dict | None = None,
) -> np.ndarray:
    """Generate calibrated no-show probabilities."""

    art = artifact or load_artifact()

    scores = art["pipeline"].predict_proba(
        model_matrix(raw)
    )[:, 1]

    calibrated = art["calibrator"].predict(scores)

    return np.clip(calibrated, 0.0, 1.0)


def risk_band(probability: float, art: dict) -> dict:
    """Find the validation risk band containing a probability."""

    for band in art["risk_bands"]:

        if (
            band["low"] <= probability < band["high"]
            or (
                probability >= band["high"]
                and band["high"] >= 1.0
            )
        ):
            return band

    return art["risk_bands"][-1]


def input_warnings(
    raw_row: pd.Series,
    art: dict,
) -> list[str]:
    """Identify inputs that need additional human attention."""

    reference = art["reference"]

    warnings = []

    if (
        raw_row["neighbourhood"]
        not in reference["known_neighbourhoods"]
    ):
        warnings.append(
            "Neighbourhood not seen in training data - "
            "the model treats it as rare/other; treat the score "
            "with extra caution."
        )

    low, high = reference["lead_days_range"]

    if raw_row["lead_days"] > high:
        warnings.append(
            f"Lead time {raw_row['lead_days']}d is longer than "
            f"anything seen in training (max {high}d); "
            "the model extrapolates."
        )

    if (
        raw_row["prior_no_shows"]
        > raw_row["prior_appointments"]
    ):
        warnings.append(
            "prior_no_shows exceeds prior_appointments - "
            "please check the history fields."
        )

    if raw_row["prior_appointments"] == 0:
        warnings.append(
            "No attendance history for this patient: the score "
            "relies on population-level patterns only "
            "(wider uncertainty)."
        )

    return warnings


def explain_row(
    raw_row: pd.Series,
    art: dict,
    top_k: int = 5,
) -> list[dict]:
    """
    Occlusion explanation.

    Reset one input group to its neutral reference value and
    measure the resulting change in calibrated probability.

    Positive contribution means the input raises predicted risk.
    """

    reference = art["reference"]["explain_reference"]

    base = raw_row.to_frame().T.reset_index(drop=True)

    numeric_columns = [
        "age",
        "lead_days",
        "scheduled_hour",
        "prior_appointments",
        "prior_no_shows",
        "scholarship",
    ]

    base = base.astype(
        {
            column: float
            for column in numeric_columns
        }
    )

    original_probability = float(
        predict_proba(base, art)[0]
    )

    explanations = []

    for label, columns in EXPLAIN_GROUPS.items():

        alternative = base.copy()

        for column in columns:

            alternative.loc[0, column] = (
                np.nan
                if reference[column] is None
                else reference[column]
            )

        if label == "past attendance history":
            alternative.loc[0, "prior_no_shows"] = 0

        alternative_probability = float(
            predict_proba(alternative, art)[0]
        )

        effect = (
            original_probability
            - alternative_probability
        )

        shown_values = ", ".join(
            f"{column}={raw_row[column]}"
            for column in columns
            if not (
                isinstance(raw_row[column], float)
                and np.isnan(raw_row[column])
            )
        )

        explanations.append(
            {
                "factor": label,
                "value": shown_values,
                "effect_on_probability": round(effect, 4),

                "direction": (
                    "raises risk"
                    if effect > 0.0005
                    else "lowers risk"
                    if effect < -0.0005
                    else "neutral"
                ),
            }
        )

    explanations.sort(
        key=lambda item: abs(
            item["effect_on_probability"]
        ),
        reverse=True,
    )

    return explanations[:top_k]


def score_records(
    records: list[dict],
    explain: bool = True,
) -> list[dict]:
    """Score appointment records and return review information."""

    art = load_artifact()

    raw = raw_frame(records)

    probabilities = predict_proba(raw, art)

    results = []

    for index, probability in enumerate(probabilities):

        row = raw.iloc[index]

        band = risk_band(
            float(probability),
            art,
        )

        result = {
            "no_show_probability": round(
                float(probability),
                4,
            ),

            "risk_band": band["label"],

            "historical_no_show_rate_in_band": (
                band["observed_no_show_rate_validation"]
            ),

            "flag_for_reminder_review": bool(
                probability >= art["threshold"]
            ),

            "decision_threshold": round(
                float(art["threshold"]),
                4,
            ),

            "warnings": input_warnings(
                row,
                art,
            ),

            "human_review_required": True,

            "disclaimer": DISCLAIMER,
        }

        if explain:
            result["top_factors"] = explain_row(
                row,
                art,
            )

        results.append(result)

    return results


def model_info() -> dict:
    """Return model metadata for the dashboard and API."""

    art = load_artifact()

    return {
        "metadata": art["metadata"],
        "threshold": art["threshold"],
        "risk_bands": art["risk_bands"],
        "features": art["features"],
        "known_neighbourhoods": (
            art["reference"]["known_neighbourhoods"]
        ),
        "disclaimer": DISCLAIMER,
    }
