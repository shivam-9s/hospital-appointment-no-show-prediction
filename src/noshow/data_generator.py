"""Synthetic generator that reproduces the schema of the Kaggle dataset
"Medical Appointment No Shows" (joniarroba/noshowappointments, KaggleV2-May-2016.csv).

Why it exists: the sandbox that built this project could not reach kaggle.com. The pipeline
automatically prefers the real CSV when it is placed in data/raw/. This generator lets
anyone reproduce the full pipeline without a Kaggle account, and is explicitly allowed by the brief
("synthetic data generator"). No real person is represented.

Generative story (so the data has realistic but *known* structure):
  * patients have a latent reliability that persists across visits  -> prior-history features matter
  * no-show odds rise with booking lead time, are lowest for same-day visits
  * young adults miss more than children/older adults
  * neighbourhood and welfare-scholarship shift the base rate a little
  * a lot of irreducible noise, so the achievable AUC is modest (like the real data, ~0.7)
Also injects the data-quality problems found in the real file (age < 0, age > 110,
appointment-before-booking rows, date-only AppointmentDay) so the cleaning code is exercised.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

NEIGHBOURHOODS = [
    "JARDIM CAMBURI", "MARIA ORTIZ", "RESISTÊNCIA", "JARDIM DA PENHA", "ITARARÉ",
    "CENTRO", "TABUAZEIRO", "SANTA MARTHA", "JESUS DE NAZARETH", "BONFIM",
    "SANTO ANTÔNIO", "CARATOÍRA", "SÃO PEDRO", "ILHA DO PRÍNCIPE", "ANDORINHAS",
    "NOVA PALESTINA", "DA PENHA", "ROMÃO", "GURIGICA", "SÃO CRISTÓVÃO",
    "MARUÍPE", "SANTOS DUMONT", "ILHA DE SANTA MARIA", "GRANDE VITÓRIA",
    "PRAIA DO SUÁ", "MÁRIO CYPRESTE", "AEROPORTO", "ILHAS OCEÂNICAS DE TRINDADE",
]


def generate(n_rows: int = 60_000, seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- patients & neighbourhoods -------------------------------------------------
    n_patients = int(n_rows * 0.55)
    patient_ids = rng.integers(10**11, 10**13, size=n_patients).astype(float)
    p_age = rng.gamma(shape=4.2, scale=10.5, size=n_patients).astype(int)
    p_age = np.where(p_age > 100, rng.integers(55, 100, n_patients), p_age)
    p_gender = rng.choice(["F", "M"], size=n_patients, p=[0.65, 0.35])
    nb_weights = rng.dirichlet(np.ones(len(NEIGHBOURHOODS)) * 0.6)
    nb_weights = 0.15 + nb_weights  # avoid vanishing categories
    nb_weights /= nb_weights.sum()
    p_nb = rng.choice(len(NEIGHBOURHOODS), size=n_patients, p=nb_weights)
    nb_effect = rng.normal(0, 0.22, size=len(NEIGHBOURHOODS))
    p_scholar = rng.random(n_patients) < 0.10
    # latent personal reliability (heavy-tailed: a few chronic no-showers)
    p_latent = rng.normal(0, 0.9, size=n_patients)

    # ---- appointments ---------------------------------------------------------------
    pat_idx = rng.choice(n_patients, size=n_rows, p=rng.dirichlet(np.ones(n_patients) * 2.0))
    start = pd.Timestamp("2016-04-29")
    n_days = 41  # 29 Apr -> 08 Jun, like the Kaggle file
    all_days = pd.date_range(start, periods=n_days, freq="D")
    wd = all_days[all_days.dayofweek < 5]  # Mon-Fri
    sat = all_days[all_days.dayofweek == 5]  # Saturdays are rare
    is_sat = rng.random(n_rows) < 0.01
    appt_date = pd.DatetimeIndex(np.where(is_sat, rng.choice(sat, n_rows), rng.choice(wd, n_rows)))

    same_day = rng.random(n_rows) < 0.38
    lead = np.where(
        same_day,
        0,
        np.clip(np.round(rng.lognormal(mean=2.4, sigma=1.0, size=n_rows)), 1, 179),
    ).astype(int)
    sched_date = appt_date - pd.to_timedelta(lead, unit="D")
    hour = np.clip(rng.normal(10.8, 3.0, n_rows), 6, 20).astype(int)
    minute = rng.integers(0, 60, n_rows)
    second = rng.integers(0, 60, n_rows)
    # scheduled timestamp = booking date + time of day (same-day bookings happen earlier than clinic)
    scheduled = sched_date + pd.to_timedelta(hour * 3600 + minute * 60 + second, unit="s")

    age = p_age[pat_idx]
    nb = p_nb[pat_idx]

    # ---- true no-show probability ---------------------------------------------------
    logit = -1.78
    logit += np.where(same_day, -1.25, 0.0)
    logit += np.where(~same_day, 0.55 * np.log1p(lead) - 0.9, 0.0)  # ~ +0.1 .. +0.8
    logit += np.where(age < 12, -0.15, 0) + np.where((age >= 15) & (age < 35), 0.35, 0)
    logit += np.where(age >= 65, -0.35, 0)
    logit += nb_effect[nb]
    logit += np.where(p_scholar[pat_idx], 0.30, 0)
    logit += p_latent[pat_idx]
    logit += np.where(pd.DatetimeIndex(appt_date).dayofweek == 4, 0.12, 0)  # Fridays slightly worse
    logit += rng.normal(0, 0.35, n_rows)  # irreducible noise
    p = 1 / (1 + np.exp(-logit))
    no_show = rng.random(n_rows) < p

    # SMS: sent only for lead >= 3 days, ~ 32 % of rows; mild real effect is NOT simulated on purpose
    sms = ((lead >= 3) & (rng.random(n_rows) < 0.62)).astype(int)

    # sensitive clinical columns (present in the raw Kaggle file; included so the pipeline proves it drops them)
    hyper = (rng.random(n_rows) < np.clip((age - 20) / 120, 0, 0.55)).astype(int)
    diab = (rng.random(n_rows) < np.clip((age - 30) / 300, 0, 0.25)).astype(int)
    alc = (rng.random(n_rows) < 0.03).astype(int)
    handcap = rng.choice([0, 1, 2, 3, 4], size=n_rows, p=[0.98, 0.016, 0.003, 0.0007, 0.0003])

    df = pd.DataFrame(
        {
            "PatientId": patient_ids[pat_idx],
            "AppointmentID": np.arange(5_600_000, 5_600_000 + n_rows),
            "Gender": p_gender[pat_idx],
            "ScheduledDay": scheduled.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "AppointmentDay": appt_date.strftime("%Y-%m-%dT00:00:00Z"),
            "Age": age,
            "Neighbourhood": np.array(NEIGHBOURHOODS)[nb],
            "Scholarship": p_scholar[pat_idx].astype(int),
            "Hipertension": hyper,
            "Diabetes": diab,
            "Alcoholism": alc,
            "Handcap": handcap,
            "SMS_received": sms,
            "No-show": np.where(no_show, "Yes", "No"),
        }
    )

    # ---- inject real-world data-quality defects --------------------------------------
    idx = rng.choice(n_rows, size=3, replace=False)
    df.loc[idx[0], "Age"] = -1
    df.loc[idx[1:], "Age"] = [115, 102]
    bad = rng.choice(n_rows, size=5, replace=False)  # appointment day before booking day
    for i in bad:
        d = pd.Timestamp(df.loc[i, "AppointmentDay"]) - pd.Timedelta(days=int(rng.integers(1, 4)))
        df.loc[i, "ScheduledDay"] = (d + pd.Timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
        df.loc[i, "AppointmentDay"] = (d - pd.Timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    df = df.sort_values("ScheduledDay").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    out = config.DATA_RAW / config.SYNTHETIC_FILENAME
    df.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(df):,}  no-show rate={(df['No-show']=='Yes').mean():.3f}")


if __name__ == "__main__":
    main()
