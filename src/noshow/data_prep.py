"""Load -> validate -> clean -> leakage-safe feature engineering -> chronological split.

PREDICTION TIME = booking time (ScheduledDay). Every feature below must be computable
at that moment. In particular the patient-history features only look at appointments
whose *date had already passed* on the booking day, i.e. whose outcome was known.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config

RAW_RENAME = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_ts",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "No-show": "no_show_raw",
}
# a few Kaggle versions spell columns differently
# canonical (Kaggle v2) name -> alternative spelling seen in other copies of the file
ALT_NAMES = {"Hipertension": "Hypertension", "Handcap": "Handicap", "No-show": "No_show", "PatientId": "PatientID"}

MAX_PLAUSIBLE_AGE = 110


def find_raw_file():
    """Prefer the real Kaggle CSV (any file name containing 'noshow' + 'appointment', e.g. KaggleV2-May-2016.csv
    or noshowappointments-kagglev2-may-2016.csv); otherwise fall back to the synthetic file."""
    for f in sorted(config.DATA_RAW.glob("*.csv")):
        n = f.name.lower()
        if "kaggle" in n or ("noshow" in n and "appointment" in n and "synthetic" not in n):
            return "kaggle", f
    synth = config.DATA_RAW / config.SYNTHETIC_FILENAME
    if not synth.exists():
        from .data_generator import generate

        config.DATA_RAW.mkdir(parents=True, exist_ok=True)
        generate().to_csv(synth, index=False)
    return "synthetic", synth


def load_raw() -> tuple[str, pd.DataFrame]:
    source, path = find_raw_file()
    df = pd.read_csv(path)
    df = df.rename(columns={alt: canon for canon, alt in ALT_NAMES.items() if alt in df.columns})
    return source, df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quality checks + cleaning. Returns the cleaned frame and a quality report."""
    report: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()

    # ---- sensitive / post-booking columns are removed up front ----------------------
    dropped = [c for c in config.SENSITIVE_CLINICAL_COLUMNS + config.POST_BOOKING_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    report["dropped_columns"] = dropped

    df = df.rename(columns=RAW_RENAME)
    report["missing_values_raw"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}

    dup_ids = int(df.duplicated("appointment_id").sum())
    df = df.drop_duplicates("appointment_id")
    report["duplicate_appointment_ids_removed"] = dup_ids

    df["scheduled_ts"] = pd.to_datetime(df["scheduled_ts"], utc=True).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True).dt.tz_localize(None).dt.normalize()
    df["scheduled_day"] = df["scheduled_ts"].dt.normalize()

    # ---- validity rules --------------------------------------------------------------
    bad_age = (df["age"] < 0) | (df["age"] > config_max_age())
    bad_dates = df["appointment_day"] < df["scheduled_day"]
    report["invalid_age_rows_removed"] = int(bad_age.sum())
    report["appointment_before_booking_rows_removed"] = int(bad_dates.sum())
    df = df[~bad_age & ~bad_dates].copy()

    df["no_show"] = (df["no_show_raw"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df = df.drop(columns="no_show_raw")
    df["gender"] = df["gender"].astype(str).str.upper().str[0]
    df["neighbourhood"] = df["neighbourhood"].astype(str).str.strip().str.upper()
    df["scholarship"] = df["scholarship"].astype(int)
    df["age"] = df["age"].astype(int)

    report["rows_clean"] = int(len(df))
    report["no_show_rate"] = round(float(df["no_show"].mean()), 4)
    report["date_range"] = [str(df["appointment_day"].min().date()), str(df["appointment_day"].max().date())]
    report["unique_patients"] = int(df["patient_id"].nunique())
    return df.reset_index(drop=True), report


def config_max_age() -> int:
    return MAX_PLAUSIBLE_AGE


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient history using ONLY appointments whose date < booking date of the current row.

    That guarantees the outcome of every counted appointment was already known when the
    current appointment was booked (no target leakage, no peeking at the future).
    """
    df = df.sort_values("scheduled_ts").reset_index(drop=True)
    prior_n = np.zeros(len(df), dtype=int)
    prior_ns = np.zeros(len(df), dtype=int)
    since = np.full(len(df), np.nan)

    for _, g in df.groupby("patient_id", sort=False):
        if len(g) == 1:
            continue
        order = np.argsort(g["appointment_day"].to_numpy(), kind="stable")
        a_sorted = g["appointment_day"].to_numpy()[order]
        ns_sorted = g["no_show"].to_numpy()[order]
        cum_ns = np.concatenate([[0], np.cumsum(ns_sorted)])
        k = np.searchsorted(a_sorted, g["scheduled_day"].to_numpy(), side="left")  # count of A_j < booking date
        pos = g.index.to_numpy()
        prior_n[pos] = k
        prior_ns[pos] = cum_ns[k]
        last = np.where(k > 0, a_sorted[np.maximum(k - 1, 0)], np.datetime64("NaT", "ns"))
        delta = (g["scheduled_day"].to_numpy() - last).astype("timedelta64[D]").astype(float)
        since[pos] = np.where(k > 0, delta, np.nan)

    df["prior_appointments"] = prior_n
    df["prior_no_shows"] = prior_ns
    df["prior_no_show_rate"] = np.where(prior_n > 0, prior_ns / np.maximum(prior_n, 1), np.nan)
    df["days_since_last_appointment"] = since
    df["has_history"] = (prior_n > 0).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_history_features(df)
    df["lead_days"] = (df["appointment_day"] - df["scheduled_day"]).dt.days
    df["log_lead_days"] = np.log1p(df["lead_days"])
    df["same_day"] = (df["lead_days"] == 0).astype(int)
    df["scheduled_hour"] = df["scheduled_ts"].dt.hour
    df["scheduled_weekday"] = df["scheduled_ts"].dt.day_name()
    df["appointment_weekday"] = df["appointment_day"].dt.day_name()
    return df


def chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test by APPOINTMENT DATE (whole days, never splitting one day across sets).

    Test = the most recent ~20 % of days -> simulates deploying on tomorrow's bookings.
    """
    day_counts = df.groupby("appointment_day").size().sort_index()
    cum = day_counts.cumsum() / day_counts.sum()
    train_days = cum.index[cum <= config.TRAIN_FRAC]
    val_days = cum.index[(cum > config.TRAIN_FRAC) & (cum <= config.TRAIN_FRAC + config.VAL_FRAC)]
    split = pd.Series("test", index=df.index)
    split[df["appointment_day"].isin(train_days)] = "train"
    split[df["appointment_day"].isin(val_days)] = "val"
    out = df.copy()
    out["split"] = split
    return out


def build_dataset() -> tuple[pd.DataFrame, dict]:
    source, raw = load_raw()
    df, report = clean(raw)
    df = chronological_split(add_features(df))
    report["data_source"] = source
    report["split_rows"] = df["split"].value_counts().to_dict()
    report["split_no_show_rate"] = df.groupby("split")["no_show"].mean().round(4).to_dict()
    report["split_date_ranges"] = {
        s: [str(g["appointment_day"].min().date()), str(g["appointment_day"].max().date())]
        for s, g in df.groupby("split")
    }
    return df, report


def main() -> None:
    argparse.ArgumentParser(description="Build the cleaned, feature-engineered, split dataset").parse_args()
    df, report = build_dataset()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.DATA_PROCESSED / config.CLEAN_FILENAME
    df.to_csv(out, index=False)
    (config.DATA_PROCESSED / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
