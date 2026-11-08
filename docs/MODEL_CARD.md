# Model card - appointment no-show risk (histogram gradient boosting, Platt-calibrated)

**Intended use.** Rank booked appointments by estimated no-show probability so staff can decide, case by case, whom to remind. Decision support only.
**Out of scope / prohibited.** Predicting diagnosis, urgency, severity or outcomes; cancelling, double-booking, overbooking or deprioritising any patient because of a score; automated action without staff review; use on populations/clinics very different from the training data without re-validation.
**Inputs (known at booking).** lead time, age, booking hour/weekday, appointment weekday, neighbourhood, welfare-scholarship flag, past appointment count / past no-shows / days since last visit. No clinical fields; gender is audit-only.
**Training data.** Kaggle "Medical Appointment No Shows" (110,516 appointments after cleaning, Vitória, Brazil, Apr-Jun 2016). Chronological split: train 29 Apr-19 May, validation 20-31 May, test 1-8 Jun.
**Performance (test, 26,449 appointments).** ROC-AUC 0.729, PR-AUC 0.330, Brier 0.136 (calibrated). At the 0.175 operating threshold: precision 0.274, recall 0.869, F1 0.417, flags 58 % of bookings. Cost per appointment 0.521 vs 0.738 (no reminders) and 0.542 ("not same-day" rule).
**Known weaknesses.** Low precision (most flags attend anyway); blind to same-day bookings (43 % of misses); recall much lower for patients 65+ (0.59 vs 0.95 for 18-34); slightly higher flag burden for scholarship holders; only ~4 % cheaper than a one-line lead-time rule; six-week data window.
**Ethical considerations.** Socio-economic proxies (neighbourhood, scholarship) are present and audited; review subgroup gaps after every retrain. Reminders should be supportive, not punitive.
**Monitoring.** Track weekly: base rate, calibration (observed vs predicted by band), flag rate and recall by subgroup, share of unknown neighbourhoods. Retrain when calibration slope leaves 0.8-1.2 or base rate shifts > 3 pp.
**Artifact.** `models/noshow_model.joblib` (scikit-learn pipeline + calibrator + threshold + metadata). Regenerate with `python -m noshow.train`.
