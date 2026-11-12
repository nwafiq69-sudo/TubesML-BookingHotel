"""FastAPI service for the Hotel Booking Cancellation Predictor."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
import sklearn.compose._column_transformer as _column_transformer


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model_kelompok_4.joblib"

app = FastAPI(
    title="Staywise | Hotel Booking Cancellation Predictor",
    description="Prototipe akademik untuk memperkirakan peluang pembatalan reservasi hotel.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class BookingInput(BaseModel):
    """Selected Hotel Booking Demand features used by the model pipeline."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0, le=1000, description="Jarak hari antara pemesanan dan kedatangan")
    arrival_date_month: Literal[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    stays_in_weekend_nights: int = Field(ge=0, le=60)
    stays_in_week_nights: int = Field(ge=0, le=100)
    adults: int = Field(ge=0, le=20)
    children: float = Field(ge=0, le=20)
    babies: int = Field(ge=0, le=10)
    meal: Literal["BB", "HB", "FB", "SC", "Undefined"]
    market_segment: Literal[
        "Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation", "Undefined",
    ]
    distribution_channel: Literal["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    is_repeated_guest: Literal[0, 1]
    previous_cancellations: int = Field(ge=0, le=100)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"]


_model = None
_model_error = "Model belum tersedia. Ganti placeholder model dengan pipeline .joblib kelompok."
try:
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as model_file:
            is_placeholder = model_file.read(3) == b"cd "
        if not is_placeholder:
            # scikit-learn 1.9 removed this private list subclass used in 1.6 pickles.
            # Its serialized value is an ordinary list of remainder column indices.
            if not hasattr(_column_transformer, "_RemainderColsList"):
                _column_transformer._RemainderColsList = type("_RemainderColsList", (list,), {})
            _model = joblib.load(MODEL_PATH)
            # SimpleImputer renamed this fitted dtype attribute after the model's
            # training version. Preserve the fitted dtype expected by transform().
            for _, transformer, _ in _model.named_steps["preprocessor"].transformers_:
                imputer = getattr(transformer, "named_steps", {}).get("imputer")
                if imputer is not None and not hasattr(imputer, "_fill_dtype"):
                    imputer._fill_dtype = imputer._fit_dtype
    else:
        _model_error = "File model belum dikirim."
except Exception as exc:  # Model load errors should be visible through /health.
    _model_error = f"Model gagal dimuat: {type(exc).__name__}"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_message": _model_error if _model is None else "Model siap"}


@app.post("/predict")
def predict(booking: BookingInput):
    if _model is None:
        raise HTTPException(status_code=503, detail=_model_error)

    frame = pd.DataFrame([booking.model_dump()])
    try:
        prediction = _model.predict(frame)[0]
        probability = None
        if hasattr(_model, "predict_proba"):
            probabilities = _model.predict_proba(frame)[0]
            classes = list(getattr(_model, "classes_", []))
            positive_index = next((i for i, label in enumerate(classes) if str(label).lower() in {"1", "true", "yes", "canceled", "cancelled"}), 1)
            positive_index = min(positive_index, len(probabilities) - 1)
            probability = float(probabilities[positive_index])
        is_canceled = str(prediction).strip().lower() in {"1", "true", "yes", "canceled", "cancelled"}
        return {
            "prediction": "cancelled" if is_canceled else "not_cancelled",
            "label": "Berisiko dibatalkan" if is_canceled else "Diperkirakan tetap aktif",
            "cancellation_probability": probability,
            "confidence": max(float(probability), 1 - float(probability)) if probability is not None else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediksi gagal: {type(exc).__name__}: {exc}") from exc


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="web")
