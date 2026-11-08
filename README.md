# Hospital Appointment No-Show Prediction (decision support)

Predicts the probability that a **booked appointment will be missed**, so front-desk staff can plan
**human-reviewed reminder calls** and scheduling buffers. It never predicts diagnosis, urgency or clinical outcome.

| | |
|---|---|
| **Business decision** | Which booked appointments get a reminder call/SMS (a person approves each one) |
| **Prediction unit** | One scheduled appointment |
| **Prediction time** | The moment of **booking** (`ScheduledDay`). Only information known then is used |
| **Horizon** | Appointment date, 0-179 days after booking |
| **Target** | `no_show = 1` if the patient did not attend |
| **Split** | Chronological by appointment date: 60 % train / 20 % validation / 20 % test (most recent days) |
| **Models** | Baselines (never-flag, flag-everyone, lead-time rule) vs Logistic Regression, Random Forest, Hist-Gradient-Boosting (selected) |
| **Imbalance** | `class_weight` (balanced) + Platt probability calibration + cost-based threshold |
| **Interface** | FastAPI (`/predict`, `/predict/batch`, `/model-info`) + browser dashboard with a staff review queue |

> **Dataset.** Kaggle **"Medical Appointment No Shows"** (`joniarroba/noshowappointments`, 110,527 appointments, Vitória, Brazil, Apr-Jun 2016).
> Download the CSV from Kaggle (unzip it) and place it in `data/raw/` - **any file name works** (e.g. `KaggleV2-May-2016.csv` or `noshowappointments-kagglev2-may-2016.csv`).
> The raw file is git-ignored; check the dataset's licence on its Kaggle page before redistributing. If no Kaggle file is present, the pipeline falls back to a
> schema-identical synthetic generator (`src/noshow/data_generator.py`) so everything can still be reproduced. All numbers in `reports/` and `docs/` come from the **real Kaggle data**.

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # Python 3.10-3.13; exact tested versions: requirements-tested.txt
export PYTHONPATH=src                                   # Windows PowerShell: $env:PYTHONPATH="src"
```

> **Important:** `models/noshow_model.joblib` in the zip was saved with scikit-learn 1.9.1. If your scikit-learn version differs,
> loading it can fail - just retrain (step 3, `python -m noshow.train`, ~35 s) before starting the API or running tests.

## 2. Data preparation (reproduces the dataset + split)
```bash
# Real Kaggle data: put the downloaded CSV in data/raw/  (used automatically)
# (optional CLI: pip install kaggle ; kaggle datasets download -d joniarroba/noshowappointments -p data/raw --unzip ; copy .env.example -> .env)
# No Kaggle file? synthetic fallback:  python -m noshow.data_generator   # -> data/raw/appointments_synthetic.csv
python -m noshow.data_prep                 # -> data/processed/appointments_clean.csv + data_quality_report.json
```
Cleaning drops invalid ages (<0 or >110) and appointments dated before their booking; removes clinical columns
(`Hipertension, Diabetes, Alcoholism, Handcap`) and `SMS_received` (post-booking); builds leakage-safe history features;
assigns the chronological `split` column.

## 3. EDA and training
```bash
python -m noshow.eda                       # figures 01-04 + reports/eda_summary.json (incl. leakage review)
python -m noshow.train                     # ~35 s; trains, tunes on validation, evaluates ONCE on test
                                           # -> models/noshow_model.joblib, reports/metrics.json, figures 05-11
python scripts/make_sample_bookings.py     # data/sample_bookings.csv (test-period bookings, no outcomes) for the dashboard
```
Same seed + same data => **identical `metrics.json`** (verified by running the whole pipeline twice from scratch).
Notebook walkthrough: `notebooks/01_eda_and_results.ipynb` (already executed).

## 4. Tests
```bash
python -m pytest -q tests                  # 22 tests: cleaning, leakage/time-travel, split, artifact, API, validation
```

## 5. Inference API / dashboard
```bash
python -m uvicorn noshow.api:app --port 8000
# dashboard: http://localhost:8000      interactive API docs: http://localhost:8000/docs
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "booking_datetime":"2016-06-01T09:30:00","appointment_date":"2016-06-20","age":24,"neighbourhood":"JARDIM CAMBURI",
  "scholarship":true,"prior_appointments":3,"prior_no_shows":2,"days_since_last_appointment":12}'
```
Every response comes from `models/noshow_model.joblib` (pipeline + calibrator): calibrated probability, risk band with the
historically observed rate for that band, review flag, the **top influencing inputs** (occlusion explanation), warnings
(unknown neighbourhood, no history, extrapolated lead time) and a disclaimer. `human_review_required` is always `true`.
Dashboard tabs: *Single booking*, *Review queue* (upload CSV -> ranked list -> staff clicks **Approve reminder / Dismiss** -> export approved list), *About the model*.

## 6. Repository layout
```
src/noshow/   config, data_generator, data_prep, features, train, eda, predict, api
app/static/   index.html (dashboard)          tests/   pytest suite        scripts/  sample-file helper
data/         raw/, processed/, sample_bookings.csv        models/  saved artifact
reports/      REPORT.md (technical report), metrics.json, eda_summary.json, figures/
docs/         MODEL_CARD.md, DEMO_VIDEO_SCRIPT.md, SUBMISSION_FORM_ANSWERS.md
notebooks/    01_eda_and_results.ipynb
```

## 7. Known limitations / unfinished
* The 3-5 min demo video is not included (cannot be recorded here); `docs/DEMO_VIDEO_SCRIPT.md` is a shot-by-shot script.
* Cost figures (reminder = 1, missed slot = 4) are illustrative assumptions; replace with the hospital's real numbers in `config.py`.
* No authentication/audit logging on the API - it is an MVP for local use. Not for production with real patient data.
