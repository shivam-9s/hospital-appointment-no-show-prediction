"""Preprocessing, split and leakage tests (use tiny hand-made frames, so they are fast and exact)."""
import numpy as np
import pandas as pd
import pytest

from noshow import config
from noshow.data_prep import add_features, chronological_split, clean, load_raw
from noshow.features import add_derived_features


def mk(rows):
    return pd.DataFrame(rows, columns=["PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age",
                                       "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
                                       "SMS_received", "No-show"])


def base_rows():
    # one patient, three appointments, booked on 1 Jun, 5 Jun, 7 Jun
    return mk([
        [1.0, 1, "F", "2016-06-01T09:00:00Z", "2016-06-03T00:00:00Z", 30, "centro", 0, 1, 0, 0, 0, 1, "Yes"],
        [1.0, 2, "F", "2016-06-05T09:00:00Z", "2016-06-10T00:00:00Z", 30, "centro", 0, 1, 0, 0, 0, 1, "No"],
        [1.0, 3, "F", "2016-06-07T09:00:00Z", "2016-06-12T00:00:00Z", 30, "centro", 0, 1, 0, 0, 0, 0, "Yes"],
    ])


def test_sensitive_and_post_booking_columns_are_dropped():
    df, rep = clean(base_rows())
    for col in config.SENSITIVE_CLINICAL_COLUMNS + config.POST_BOOKING_COLUMNS:
        assert col not in df.columns
    assert set(rep["dropped_columns"]) == set(config.SENSITIVE_CLINICAL_COLUMNS + config.POST_BOOKING_COLUMNS)


def test_model_features_contain_no_forbidden_columns():
    forbidden = set(config.SENSITIVE_CLINICAL_COLUMNS + config.POST_BOOKING_COLUMNS + config.AUDIT_ONLY_COLUMNS
                    + [config.TARGET, "no_show_raw", "patient_id", "appointment_id", "Hipertension", "SMS_received"])
    assert not forbidden & set(config.FEATURES)


def test_cleaning_removes_invalid_rows_and_encodes_target():
    raw = base_rows()
    raw.loc[0, "Age"] = -1                                   # invalid age
    raw.loc[1, "AppointmentDay"] = "2016-06-01T00:00:00Z"    # appointment before booking (booked 5 Jun)
    df, rep = clean(raw)
    assert rep["invalid_age_rows_removed"] == 1 and rep["appointment_before_booking_rows_removed"] == 1
    assert len(df) == 1 and df["no_show"].tolist() == [1]
    assert df.loc[0, "neighbourhood"] == "CENTRO"


def test_history_features_only_use_outcomes_known_at_booking_time():
    df, _ = clean(base_rows())
    f = add_features(df).sort_values("scheduled_ts").reset_index(drop=True)
    # booking 1 (1 Jun): no past appointments
    assert f.loc[0, ["prior_appointments", "prior_no_shows"]].tolist() == [0, 0]
    # booking 2 (5 Jun): appointment 1 took place 3 Jun (a no-show) -> known
    assert f.loc[1, ["prior_appointments", "prior_no_shows"]].tolist() == [1, 1]
    assert f.loc[1, "days_since_last_appointment"] == 2
    # booking 3 (7 Jun): appt 2 is on 10 Jun - still in the FUTURE - must NOT be counted (its outcome is unknown)
    assert f.loc[2, ["prior_appointments", "prior_no_shows"]].tolist() == [1, 1]


def test_changing_a_future_outcome_never_changes_current_features():
    """Time-travel test: flip the outcome of the LAST appointment; features of earlier bookings must not move."""
    a, _ = clean(base_rows())
    r = base_rows(); r.loc[2, "No-show"] = "No"
    b, _ = clean(r)
    fa = add_features(a).sort_values("scheduled_ts").reset_index(drop=True)
    fb = add_features(b).sort_values("scheduled_ts").reset_index(drop=True)
    cols = config.FEATURES
    pd.testing.assert_frame_equal(fa.loc[:1, cols], fb.loc[:1, cols])


def test_lead_time_and_same_day_flag():
    df, _ = clean(base_rows())
    f = add_features(df).sort_values("scheduled_ts").reset_index(drop=True)
    assert f["lead_days"].tolist() == [2, 5, 5]
    assert f["same_day"].tolist() == [0, 0, 0]


def test_chronological_split_has_no_overlap_and_is_ordered():
    df, _ = clean(pd.concat([base_rows().assign(AppointmentID=lambda d: d.AppointmentID + 100 * k,
                                                 AppointmentDay=lambda d, k=k: (pd.to_datetime(d.AppointmentDay) + pd.Timedelta(days=3 * k)).dt.strftime("%Y-%m-%dT00:00:00Z"),
                                                 ScheduledDay=lambda d, k=k: (pd.to_datetime(d.ScheduledDay) + pd.Timedelta(days=3 * k)).dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
                             for k in range(10)]))
    s = chronological_split(add_features(df))
    assert set(s["split"]) <= {"train", "val", "test"}
    tr, va, te = (s[s.split == k]["appointment_day"] for k in ("train", "val", "test"))
    if len(tr) and len(va):
        assert tr.max() < va.min()
    if len(va) and len(te):
        assert va.max() < te.min()
    # a whole day never spans two splits
    assert (s.groupby("appointment_day")["split"].nunique() == 1).all()


def test_real_dataset_split_is_chronological_and_leak_free():
    from noshow.data_prep import build_dataset
    df, _ = build_dataset()
    mx = df.groupby("split")["appointment_day"].agg(["min", "max"])
    assert mx.loc["train", "max"] < mx.loc["val", "min"] <= mx.loc["val", "max"] < mx.loc["test", "min"]
    assert (df["lead_days"] >= 0).all() and df["age"].between(0, 110).all()
    assert df[config.FEATURES].drop(columns=["days_since_last_appointment", "prior_no_show_rate"]).notna().all().all()
