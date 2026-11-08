"""Inference tests: the API must load the saved artifact and return real, deterministic model output."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from noshow import config, predict
from noshow.api import app
from noshow.features import add_derived_features

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python -m noshow.train` first")
client = TestClient(app)

BOOKING = {"booking_datetime": "2016-06-01T09:30:00", "appointment_date": "2016-06-20", "age": 24,
           "neighbourhood": "CENTRO", "scholarship": True, "prior_appointments": 3, "prior_no_shows": 2,
           "days_since_last_appointment": 12}


def test_artifact_contents():
    art = joblib.load(config.MODEL_PATH)
    assert {"pipeline", "calibrator", "threshold", "features", "metadata", "risk_bands"} <= set(art)
    assert art["features"] == config.FEATURES
    assert art["metadata"]["seed"] == config.SEED


def test_train_serve_consistency_on_training_rows():
    """Features rebuilt from API-style raw inputs must equal the features used in training (no skew)."""
    df = pd.read_csv(config.DATA_PROCESSED / config.CLEAN_FILENAME, parse_dates=["scheduled_ts", "appointment_day"])
    s = df[df.split == "test"].sample(200, random_state=1)
    recs = [{"booking_datetime": r.scheduled_ts.isoformat(), "appointment_date": str(r.appointment_day.date()), "age": r.age,
             "neighbourhood": r.neighbourhood, "scholarship": r.scholarship, "prior_appointments": r.prior_appointments,
             "prior_no_shows": r.prior_no_shows,
             "days_since_last_appointment": None if np.isnan(r.days_since_last_appointment) else r.days_since_last_appointment}
            for r in s.itertuples()]
    rebuilt = predict.model_matrix(predict.raw_frame(recs)).reset_index(drop=True)
    orig = s[config.FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt.astype(float, errors="ignore"), orig.astype(float, errors="ignore"), check_dtype=False, check_exact=False, atol=1e-9)


def test_predict_endpoint_returns_real_probability():
    r = client.post("/predict", json=BOOKING)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["no_show_probability"] <= 1
    assert j["human_review_required"] is True and "diagnosis" in j["disclaimer"]
    assert len(j["top_factors"]) > 0
    # same input -> same output (deterministic)
    assert client.post("/predict", json=BOOKING).json()["no_show_probability"] == j["no_show_probability"]


def test_prediction_equals_direct_artifact_call():
    art = joblib.load(config.MODEL_PATH)
    raw = predict.raw_frame([{**BOOKING, "scholarship": True}])
    expected = art["calibrator"].predict(art["pipeline"].predict_proba(add_derived_features(raw)[config.FEATURES])[:, 1])[0]
    assert client.post("/predict", json=BOOKING).json()["no_show_probability"] == pytest.approx(round(float(expected), 4))


def test_model_is_not_a_constant_and_responds_sensibly_to_inputs():
    """Not hard-coded: a long lead time with bad history must score above a same-day booking with good history."""
    risky = {**BOOKING, "appointment_date": "2016-08-20"}
    safe = {**BOOKING, "appointment_date": "2016-06-01", "prior_appointments": 6, "prior_no_shows": 0, "age": 70}
    pr, ps = (client.post("/predict", json=x).json()["no_show_probability"] for x in (risky, safe))
    assert pr > ps + 0.15


@pytest.mark.parametrize("patch", [
    {"age": -3}, {"age": 150}, {"appointment_date": "2016-05-01"}, {"appointment_date": "2018-01-01"}, {"neighbourhood": ""},
])
def test_invalid_input_rejected(patch):
    assert client.post("/predict", json={**BOOKING, **patch}).status_code == 422


def test_unknown_neighbourhood_and_no_history_produce_warnings():
    r = client.post("/predict", json={**BOOKING, "neighbourhood": "NOWHERE", "prior_appointments": 0, "prior_no_shows": 0,
                                       "days_since_last_appointment": None}).json()
    assert any("Neighbourhood" in w for w in r["warnings"]) and any("history" in w for w in r["warnings"])


def test_batch_endpoint_ranks_by_risk():
    safe = {**BOOKING, "appointment_date": "2016-06-01", "prior_appointments": 0, "prior_no_shows": 0}
    r = client.post("/predict/batch", json={"bookings": [safe, BOOKING]}).json()
    assert r["count"] == 2 and r["ranked_indices"][0] == 1


def test_no_sensitive_fields_accepted_or_returned():
    r = client.post("/predict", json={**BOOKING, "diabetes": 1, "hypertension": 1})
    assert r.status_code == 200  # unknown fields are ignored...
    text = str(r.json()).lower()
    assert "diabet" not in text and "hypertens" not in text  # ...and never used
    assert client.get("/health").json()["status"] == "ok"


def test_threshold_follows_cost_assumptions():
    art = joblib.load(config.MODEL_PATH)
    theory = config.COST_FALSE_POSITIVE / (config.COST_FALSE_POSITIVE + config.COST_FALSE_NEGATIVE)
    assert abs(art["threshold"] - theory) < 0.10  # validation-optimal threshold is near the theoretical one

def test_timezone_aware_booking_datetime():
    booking = {
        **BOOKING,
        "booking_datetime": "2016-06-01T09:30:00Z"
    }

    raw = predict.raw_frame([booking])

    assert raw.loc[0, "lead_days"] == 19
    assert raw.loc[0, "scheduled_hour"] == 9


def test_timezone_offset_booking_datetime():
    booking = {
        **BOOKING,
        "booking_datetime": "2016-06-01T15:00:00+05:30"
    }

    raw = predict.raw_frame([booking])

    assert raw.loc[0, "lead_days"] == 19
    assert raw.loc[0, "scheduled_hour"] == 15


def test_timezone_naive_booking_datetime():
    booking = {
        **BOOKING,
        "booking_datetime": "2016-06-01T09:30:00"
    }

    raw = predict.raw_frame([booking])

    assert raw.loc[0, "lead_days"] == 19
    assert raw.loc[0, "scheduled_hour"] == 9
