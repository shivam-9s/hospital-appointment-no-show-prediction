"""Write data/sample_bookings.csv: 150 bookings from the held-out TEST period, with NO outcome column,
formatted exactly like the API input (use it in the dashboard's "Review queue" tab)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from noshow import config

df = pd.read_csv(config.DATA_PROCESSED / config.CLEAN_FILENAME, parse_dates=["scheduled_ts", "appointment_day"])
te = df[df["split"] == "test"].sample(150, random_state=config.SEED)
out = pd.DataFrame({
    "booking_datetime": te["scheduled_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
    "appointment_date": te["appointment_day"].dt.strftime("%Y-%m-%d"),
    "age": te["age"], "neighbourhood": te["neighbourhood"], "scholarship": te["scholarship"],
    "prior_appointments": te["prior_appointments"], "prior_no_shows": te["prior_no_shows"],
    "days_since_last_appointment": te["days_since_last_appointment"],
}).sort_values("booking_datetime")
out.to_csv(config.ROOT / "data" / "sample_bookings.csv", index=False)
print("wrote data/sample_bookings.csv", out.shape)
