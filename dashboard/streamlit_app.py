
"""
Hospital Appointment No-Show Prediction
Streamlit Dashboard

Supports:
1. Single appointment prediction
2. Batch appointment prediction
3. Risk ranking and staff review
4. Model evaluation and safeguards

Uses direct model inference.
A separate FastAPI server is not required for this dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from noshow import predict


DISCLAIMER = (
    "Decision support only. This score is not a diagnosis or measure "
    "of clinical need. A staff member must review every flag before "
    "a reminder is sent. Never cancel, double-book, overbook, or "
    "deprioritise care because of this score."
)

REQUIRED_COLUMNS = [
    "booking_datetime",
    "appointment_date",
    "age",
    "neighbourhood",
    "scholarship",
    "prior_appointments",
    "prior_no_shows",
    "days_since_last_appointment",
]


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="No-Show Risk Desk",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1250px;
    }

    [data-testid="stMetric"] {
        background: rgba(100,120,150,.08);
        padding: 14px;
        border-radius: 12px;
    }

    .small-note {
        font-size: .88rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DIRECT MODEL INFERENCE
# ============================================================

def api_get(path: str):
    """
    Compatibility wrapper replacing the previous HTTP GET calls.
    """

    if path == "/health":
        predict.load_artifact()
        return {"status": "ok"}

    if path == "/model-info":
        return predict.model_info()

    raise ValueError(f"Unsupported endpoint: {path}")


def api_post(path: str, payload: dict):
    """
    Compatibility wrapper replacing the previous HTTP POST calls.

    Uses the existing trained model directly.
    """

    if path == "/predict":

        results = predict.score_records(
            [payload],
            explain=True,
        )

        return results[0]

    if path == "/predict/batch":

        records = payload.get("bookings", [])

        if not records:
            raise ValueError(
                "Please provide at least one appointment."
            )

        if len(records) > 2000:
            raise ValueError(
                "Maximum batch size is 2,000 appointments."
            )

        results = predict.score_records(
            records,
            explain=payload.get("explain", False),
        )

        ranked_indices = sorted(
            range(len(results)),
            key=lambda i: results[i]["no_show_probability"],
            reverse=True,
        )

        return {
            "results": results,
            "ranked_indices": ranked_indices,
        }

    raise ValueError(f"Unsupported endpoint: {path}")


# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def pct(value):
    """Convert probability into percentage format."""
    return f"{float(value):.1%}"


def risk_color(label: str) -> str:
    """Return an emoji based on the predicted risk category."""

    return {
        "low": "🟢",
        "very low": "🟢",
        "medium": "🟠",
        "high": "🔴",
        "very high": "🔴",
    }.get(label.lower(), "⚪")


def normalize_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize uploaded appointment records."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
        )

    frame = frame[REQUIRED_COLUMNS].copy()

    for column in [
        "age",
        "prior_appointments",
        "prior_no_shows",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        ).astype(int)

    frame["scholarship"] = (
        frame["scholarship"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y"])
    )

    frame["days_since_last_appointment"] = pd.to_numeric(
        frame["days_since_last_appointment"],
        errors="coerce",
    )

    frame["booking_datetime"] = (
        frame["booking_datetime"].astype(str)
    )

    frame["appointment_date"] = (
        frame["appointment_date"].astype(str)
    )

    return frame


# ============================================================
# 5. APPLICATION HEADER
# ============================================================

st.title("🏥 Hospital Appointment No-Show Risk")

st.caption(
    "Decision-support workspace · calibrated predictions · "
    "human-reviewed reminders only"
)

st.warning(DISCLAIMER, icon="⚠️")


# ============================================================
# 6. LOAD TRAINED MODEL AND METADATA
# ============================================================

model_info = None
meta = {}
metrics = {}

with st.sidebar:

    st.subheader("Model connection")

    st.code(
        "Direct model inference",
        language=None,
    )

    try:
        api_get("/health")

        st.success(
            "Model: loaded successfully"
        )

    except Exception as exc:

        st.error(
            "Could not load the trained model artifact."
        )

        st.caption(str(exc))


try:

    model_info = api_get("/model-info")

    meta = model_info.get("metadata", {})

    metrics = meta.get("test_metrics", {})

except Exception as exc:

    st.error(
        "Could not load model information. "
        "Check the model artifact and installed dependencies."
    )

    st.exception(exc)


with st.sidebar:

    st.markdown("---")

    st.caption(
        "Use synthetic or appropriately de-identified records only. "
        "This demo has no authentication or audit logging and is not "
        "approved for real patient data."
    )


# ============================================================
# 7. MODEL SUMMARY
# ============================================================

if model_info:

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Model",
        str(
            meta.get(
                "model_name",
                "Hist Gradient Boosting",
            )
        ),
    )

    c2.metric(
        "ROC-AUC",
        f"{metrics.get('roc_auc', '—')}",
    )

    c3.metric(
        "PR-AUC",
        f"{metrics.get('pr_auc', '—')}",
    )

    c4.metric(
        "Review threshold",
        pct(model_info.get("threshold", 0)),
    )


# ============================================================
# 8. DASHBOARD TABS
# ============================================================

single_tab, batch_tab, about_tab = st.tabs(
    [
        "Single appointment",
        "Batch review queue",
        "Model & safeguards",
    ]
)


# ============================================================
# 9. SINGLE APPOINTMENT PREDICTION
# ============================================================

with single_tab:

    st.subheader(
        "Score one booked appointment"
    )

    st.caption(
        "Enter information available at booking time. "
        "Booking timestamps are interpreted in UTC, matching model training."
    )

    with st.form("single_booking"):

        a, b, c = st.columns(3)

        booking_day = a.date_input(
            "Booking date (UTC)",
            value=date.today(),
        )

        booking_clock = b.time_input(
            "Booking time (UTC)",
            value=time(9, 0),
        )

        appt_day = c.date_input(
            "Appointment date",
            value=date.today() + timedelta(days=7),
            min_value=date.today(),
        )

        d, e, f = st.columns(3)

        age = d.number_input(
            "Age",
            min_value=0,
            max_value=110,
            value=35,
        )

        neighbourhood = e.text_input(
            "Neighbourhood",
            value="UNKNOWN",
            max_chars=60,
        )

        scholarship = f.selectbox(
            "Scholarship",
            [False, True],
            format_func=lambda x: "Yes" if x else "No",
        )

        g, h, i = st.columns(3)

        prior_appointments = g.number_input(
            "Prior appointments",
            min_value=0,
            max_value=500,
            value=3,
        )

        prior_no_shows = h.number_input(
            "Prior no-shows",
            min_value=0,
            max_value=500,
            value=1,
        )

        days_since = i.number_input(
            "Days since last appointment",
            min_value=0.0,
            value=45.0,
        )

        submitted = st.form_submit_button(
            "Calculate no-show risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if appt_day < booking_day:

            st.error(
                "Appointment date cannot be before booking date."
            )

        elif (appt_day - booking_day).days > 365:

            st.error(
                "Appointment date cannot be more than 365 days "
                "after booking."
            )

        elif prior_no_shows > prior_appointments:

            st.error(
                "Prior no-shows cannot exceed prior appointments."
            )

        else:

            booking_dt = datetime.combine(
                booking_day,
                booking_clock,
            ).replace(tzinfo=timezone.utc)

            payload = {
                "booking_datetime": (
                    booking_dt.isoformat().replace(
                        "+00:00",
                        "Z",
                    )
                ),
                "appointment_date": appt_day.isoformat(),
                "age": int(age),
                "neighbourhood": (
                    neighbourhood.strip() or "UNKNOWN"
                ),
                "scholarship": scholarship,
                "prior_appointments": int(
                    prior_appointments
                ),
                "prior_no_shows": int(
                    prior_no_shows
                ),
                "days_since_last_appointment": float(
                    days_since
                ),
            }

            try:

                with st.spinner(
                    "Calculating no-show probability..."
                ):

                    result = api_post(
                        "/predict",
                        payload,
                    )

                left, right, third = st.columns(3)

                left.metric(
                    "Estimated no-show probability",
                    pct(result["no_show_probability"]),
                )

                right.metric(
                    "Risk band",
                    (
                        f"{risk_color(result['risk_band'])} "
                        f"{result['risk_band'].title()}"
                    ),
                )

                third.metric(
                    "Reminder review",
                    (
                        "Flagged"
                        if result["flag_for_reminder_review"]
                        else "Not flagged"
                    ),
                )

                st.progress(
                    float(result["no_show_probability"])
                )

                st.caption(
                    "Validation no-show rate in this risk band: "
                    f"{pct(result['historical_no_show_rate_in_band'])}"
                    " · Threshold: "
                    f"{pct(result['decision_threshold'])}"
                )

                if result.get("warnings"):

                    for warning in result["warnings"]:

                        st.warning(warning)

                factors = result.get("top_factors", [])

                if factors:

                    st.markdown(
                        "#### Model explanation "
                        "(reference-perturbation estimates)"
                    )

                    factor_frame = pd.DataFrame(factors)

                    factor_frame["effect_pp"] = (
                        factor_frame[
                            "effect_on_probability"
                        ] * 100
                    )

                    st.dataframe(
                        factor_frame[
                            [
                                "factor",
                                "value",
                                "effect_pp",
                                "direction",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                st.info(
                    "Human review is required. This score does not "
                    "authorize an automated reminder or change to care."
                )

                with st.expander("Raw prediction response"):

                    st.json(result)

            except Exception as exc:

                st.error(
                    f"Prediction failed: {exc}"
                )


# ============================================================
# 10. BATCH APPOINTMENT REVIEW
# ============================================================

with batch_tab:

    st.subheader(
        "Prioritize a batch of appointments"
    )

    st.write(
        "Upload a CSV with the required booking-time fields. "
        "The trained model scores the records and displays a "
        "descending risk ranking."
    )

    sample = pd.DataFrame(
        [
            [
                "2026-11-08T08:00:00Z",
                "2026-11-15",
                35,
                "UNKNOWN",
                False,
                3,
                1,
                45,
            ],
            [
                "2026-11-08T10:30:00Z",
                "2026-11-09",
                52,
                "UNKNOWN",
                True,
                8,
                0,
                30,
            ],
            [
                "2026-11-08T14:00:00Z",
                "2026-11-20",
                24,
                "UNKNOWN",
                False,
                2,
                2,
                90,
            ],
        ],
        columns=REQUIRED_COLUMNS,
    )

    st.download_button(
        "Download sample CSV",
        sample.to_csv(index=False),
        "sample_bookings.csv",
        "text/csv",
    )

    uploaded = st.file_uploader(
        "Upload appointment CSV",
        type=["csv"],
    )

    explain = st.checkbox(
        "Include per-appointment explanations (slower)",
        value=False,
    )

    if uploaded:

        try:

            input_frame = normalize_csv(
                pd.read_csv(uploaded)
            )

            st.write(
                f"Loaded **{len(input_frame):,}** appointment rows."
            )

            st.dataframe(
                input_frame.head(10),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "Score and rank appointments",
                type="primary",
            ):

                if input_frame.empty:

                    st.error(
                        "The uploaded CSV contains no appointments."
                    )

                elif len(input_frame) > 2000:

                    st.error(
                        "Maximum batch size is 2,000 appointments."
                    )

                else:

                    records = (
                        input_frame.astype(object)
                        .where(pd.notna(input_frame), None)
                        .to_dict(orient="records")
                    )

                    with st.spinner(
                        "Scoring appointments using the trained model..."
                    ):

                        batch = api_post(
                            "/predict/batch",
                            {
                                "bookings": records,
                                "explain": explain,
                            },
                        )

                    ranked = []

                    for rank, idx in enumerate(
                        batch["ranked_indices"],
                        start=1,
                    ):

                        row = input_frame.iloc[idx].to_dict()

                        prediction_result = batch["results"][idx]

                        ranked.append(
                            {
                                "review_rank": rank,
                                "original_row": idx + 1,
                                "appointment_date": (
                                    row["appointment_date"]
                                ),
                                "age": row["age"],
                                "neighbourhood": (
                                    row["neighbourhood"]
                                ),
                                "no_show_probability": (
                                    prediction_result[
                                        "no_show_probability"
                                    ]
                                ),
                                "risk_band": (
                                    prediction_result["risk_band"]
                                ),
                                "flag_for_reminder_review": (
                                    prediction_result[
                                        "flag_for_reminder_review"
                                    ]
                                ),
                                "staff_decision": "Pending review",
                                "warnings": " | ".join(
                                    prediction_result.get(
                                        "warnings",
                                        [],
                                    )
                                ),
                            }
                        )

                    ranked_frame = pd.DataFrame(ranked)

                    st.session_state["batch_results"] = ranked_frame

                    st.session_state["batch_input"] = input_frame

                    # Clear the old editor state for a new batch.
                    st.session_state.pop(
                        "staff_review_editor",
                        None,
                    )

        except Exception as exc:

            st.error(
                f"Could not process this CSV: {exc}"
            )


    # --------------------------------------------------------
    # BATCH RESULTS AND STAFF REVIEW
    # --------------------------------------------------------

    if "batch_results" in st.session_state:

        results_frame = (
            st.session_state["batch_results"].copy()
        )

        total = len(results_frame)

        flagged = int(
            results_frame[
                "flag_for_reminder_review"
            ].sum()
        )

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Appointments scored",
            total,
        )

        k2.metric(
            "Flagged for staff review",
            flagged,
        )

        k3.metric(
            "Not flagged",
            total - flagged,
        )

        st.markdown(
            "#### Ranked review queue"
        )

        st.dataframe(
            results_frame,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Rank is a prioritization aid only. Staff must independently "
            "review every case; the model must not determine access to care."
        )

        st.markdown(
            "#### Record staff review decisions"
        )

        editable = st.data_editor(
            results_frame,
            use_container_width=True,
            hide_index=True,
            disabled=[
                column
                for column in results_frame.columns
                if column != "staff_decision"
            ],
            column_config={
                "staff_decision": (
                    st.column_config.SelectboxColumn(
                        "Staff decision",
                        options=[
                            "Pending review",
                            "Approve reminder",
                            "Dismiss",
                        ],
                        required=True,
                    )
                )
            },
            key="staff_review_editor",
        )

        approved = editable[
            editable["staff_decision"]
            == "Approve reminder"
        ]

        st.download_button(
            "Export staff-approved reminders CSV",
            approved.to_csv(index=False),
            "staff_approved_reminder_reviews.csv",
            "text/csv",
            disabled=approved.empty,
        )


# ============================================================
# 11. MODEL INFORMATION AND SAFEGUARDS
# ============================================================

with about_tab:

    st.subheader(
        "Model information"
    )

    if model_info:

        st.json(
            {
                "metadata": model_info.get("metadata"),
                "threshold": model_info.get("threshold"),
                "risk_bands": model_info.get("risk_bands"),
                "features": model_info.get("features"),
            }
        )

        st.markdown(
            "**Feature inputs:** "
            + ", ".join(
                model_info.get("features", [])
            )
        )

    st.markdown(
        "#### Intended use"
    )

    st.write(
        "Prioritize booked appointments for supportive, human-reviewed "
        "reminders. The model estimates attendance behaviour, not "
        "diagnosis, urgency, severity, or clinical need."
    )

    st.markdown(
        "#### Safeguards and limitations"
    )

    st.markdown(
        """
        - No automated reminders are sent by this dashboard.
        - Never cancel, double-book, overbook, or deprioritize an appointment based on a score.
        - Historical data is from public hospitals in Vitória, Brazil, in 2016; it may not represent another hospital or current patient population.
        - Neighbourhood and scholarship are socio-economic proxies and require fairness monitoring.
        - This demonstration has no authentication or audit logging.
        - Do not use real patient data.
        """
    )

    st.markdown(
        "#### Evaluation"
    )

    if metrics:

        metric_frame = pd.DataFrame(
            [
                {
                    "Metric": key,
                    "Test value": value,
                }
                for key, value in metrics.items()
            ]
        )

        st.dataframe(
            metric_frame,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Evaluation metrics are not available in the model metadata."
        )


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Hospital appointment no-show risk · "
    "Research / portfolio decision-support prototype · "
    "Human review required"
)
