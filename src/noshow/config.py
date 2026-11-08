"""Central configuration: paths, seed, split dates, cost assumptions, feature lists.

Everything that defines the *problem* (prediction time, target, allowed features)
lives here so it is documented in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Real Kaggle file (joniarroba/noshowappointments). If present it is used instead of synthetic data.
KAGGLE_FILENAME = "KaggleV2-May-2016.csv"
SYNTHETIC_FILENAME = "appointments_synthetic.csv"
CLEAN_FILENAME = "appointments_clean.csv"

MODEL_PATH = MODELS_DIR / "noshow_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

SEED = 42

# ----------------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------------
# Prediction unit      : one scheduled appointment
# Prediction time      : the moment the appointment is BOOKED (ScheduledDay)
# Prediction horizon   : the appointment date, 0..~100 days later
# Target               : no_show = 1 if the patient did not attend
# Business decision    : which booked appointments get a human-reviewed reminder call
TARGET = "no_show"

# Chronological split on the appointment date (fractions of the date range by row count)
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20  # remainder (0.20) is the untouched test set

# Columns that are clinical / sensitive. Dropped on load and NEVER used as features.
SENSITIVE_CLINICAL_COLUMNS = ["Hipertension", "Diabetes", "Alcoholism", "Handcap"]

# SMS_received is dropped: the SMS is sent AFTER booking (it is the very intervention
# we want to plan), so it is not known at prediction time and would mask the true effect.
POST_BOOKING_COLUMNS = ["SMS_received"]

# Gender is kept in the clean table ONLY for the fairness audit, never as a model input.
AUDIT_ONLY_COLUMNS = ["gender"]

NUMERIC_FEATURES = [
    "age",
    "lead_days",
    "log_lead_days",
    "scheduled_hour",
    "prior_appointments",
    "prior_no_shows",
    "prior_no_show_rate",
    "days_since_last_appointment",
]
BINARY_FEATURES = ["scholarship", "same_day", "has_history"]
CATEGORICAL_FEATURES = ["appointment_weekday", "scheduled_weekday", "neighbourhood"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Neighbourhoods with fewer training rows than this are grouped into "OTHER"
MIN_NEIGHBOURHOOD_COUNT = 150

# ----------------------------------------------------------------------------
# Business cost assumptions (ILLUSTRATIVE - replace with the hospital's real numbers)
# ----------------------------------------------------------------------------
COST_FALSE_POSITIVE = 1.0  # a reminder call made to someone who would have attended
COST_FALSE_NEGATIVE = 4.0  # an unreminded no-show: an empty slot nobody else could use
# Result: reminding is worthwhile when P(no-show) > cost_fp / (cost_fp + cost_fn) = 0.20
