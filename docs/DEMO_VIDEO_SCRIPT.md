# 3-5 minute demo video - shot list (record with any screen recorder)

| Time | Show | Say |
|---|---|---|
| 0:00-0:30 | README table + `reports/REPORT.md` §1 | Decision, unit, **prediction time = booking**, target, why human-reviewed |
| 0:30-1:15 | Terminal: `python -m noshow.data_prep`; open `figures/01`, `04` | Source (Kaggle "Medical Appointment No Shows"), cleaning counts, chronological split, imbalance 20 % |
| 1:15-1:45 | `figures/02-03`; `eda_summary.json` leakage block | Lead time dominates; dropped SMS + clinical columns; time-travel leakage test |
| 1:45-2:30 | Terminal: `python -m noshow.train` (or show `metrics.json`); `figures/10`, `06`, `07` | Baselines vs 3 models, test-set metrics, calibration, cost threshold, honest note: only ~4 % cheaper than a "not same-day" rule |
| 2:30-3:00 | Terminal: `pytest -q` | 22 tests incl. leakage + API |
| 3:00-4:15 | Browser `localhost:8000`: score a booking; change lead time / history and show score move; unknown neighbourhood warning; Review-queue tab: upload `data/sample_bookings.csv`, Approve 2 reminders, export | Real saved model, top factors, uncertainty note, human approves every reminder |
| 4:15-4:45 | `REPORT.md` §7-8 | Subgroup gaps (65+ recall 0.59), limitations, what is unfinished (auth, six-week window) |
