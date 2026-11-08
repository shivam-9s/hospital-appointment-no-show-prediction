"""FastAPI service + dashboard.

    uvicorn noshow.api:app --reload --port 8000      # then open http://localhost:8000
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import config, predict

STATIC = Path(__file__).resolve().parents[2] / "app" / "static"

app = FastAPI(
    title="Hospital appointment no-show risk - decision support",
    version="1.0.0",
    description="Returns a calibrated no-show probability for a booked appointment. Human-reviewed reminders only.",
)


class Booking(BaseModel):
    booking_datetime: datetime = Field(..., description="When the appointment was booked (the prediction time).")
    appointment_date: date
    age: int = Field(..., ge=0, le=110)
    neighbourhood: str = Field(..., min_length=1, max_length=60)
    scholarship: bool = False
    prior_appointments: int = Field(0, ge=0, le=500, description="Past appointments whose date had already passed at booking time.")
    prior_no_shows: int = Field(0, ge=0, le=500)
    days_since_last_appointment: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _dates(self):
        if self.appointment_date < self.booking_datetime.date():
            raise ValueError("appointment_date cannot be before booking_datetime")
        if (self.appointment_date - self.booking_datetime.date()).days > 365:
            raise ValueError("lead time over 365 days is not supported")
        return self


class BatchRequest(BaseModel):
    bookings: list[Booking] = Field(..., min_length=1, max_length=2000)
    explain: bool = False


def _records(bs: list[Booking]) -> list[dict]:
    return [b.model_dump(mode="json") for b in bs]


@app.get("/health")
def health():
    ok = config.MODEL_PATH.exists()
    return {"status": "ok" if ok else "model_missing", "model_path": str(config.MODEL_PATH.name)}


@app.get("/model-info")
def model_info():
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Model artifact missing. Run: python -m noshow.train")
    return predict.model_info()


@app.post("/predict")
def predict_one(b: Booking):
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Model artifact missing. Run: python -m noshow.train")
    return predict.score_records(_records([b]), explain=True)[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Model artifact missing. Run: python -m noshow.train")
    res = predict.score_records(_records(req.bookings), explain=req.explain)
    order = sorted(range(len(res)), key=lambda i: -res[i]["no_show_probability"])
    return {"count": len(res), "ranked_indices": order, "results": res, "disclaimer": predict.DISCLAIMER}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")
