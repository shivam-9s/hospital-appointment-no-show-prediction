# Draft answers for the submission form (fill the bracketed items yourself)

* **Candidate identification:** [Full name] · [Registration no.] · [University email]
* **Assignment confirmation:** AI & Machine Learning - Hospital Appointment No-Show Prediction
* **Dataset source / license / generator:** Kaggle "Medical Appointment No Shows" (joniarroba/noshowappointments, `KaggleV2-May-2016.csv`; licence as shown on the Kaggle dataset page - raw file not committed). A schema-identical synthetic generator (`src/noshow/data_generator.py`, seed 42) is also included.
* **Notebook / report link:** `notebooks/01_eda_and_results.ipynb`, `reports/REPORT.md` [repo URL]
* **Model artifact location:** `models/noshow_model.joblib` (regenerate: `python -m noshow.train`)
* **Evaluation metrics (held-out test, 26,449 appts):** ROC-AUC 0.729 · PR-AUC 0.330 · precision 0.274 · recall 0.869 · F1 0.417 @ threshold 0.175 · Brier 0.136 (calibrated) · cost/appointment 0.521 vs 0.738 (no reminders)
* **Inference API/app URL:** local-only: `python -m uvicorn noshow.api:app --port 8000` -> http://localhost:8000 (declare local-only unless you deploy)
* **Repository / branch / commit:** [URL] · main · [commit id]
* **Video:** [link] (see `docs/DEMO_VIDEO_SCRIPT.md`)
* **Technology stack:** Python 3.13, pandas, scikit-learn, FastAPI, pytest, matplotlib
* **Known incomplete / assumptions:** see README §7 (illustrative costs, no auth, six-week data window, weak recall for 65+)
* **AI & attribution:** AI assistant (Claude) used to scaffold code, tests and documentation; dataset: Kaggle "Medical Appointment No Shows" (joniarroba) if used. [Edit to reflect your actual use - and make sure you can explain every file in the walkthrough.]
* **Declarations:** [originality · can explain the code · no real sensitive data/secrets · links work]
