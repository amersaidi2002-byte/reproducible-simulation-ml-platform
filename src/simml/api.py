"""Docker-friendly FastAPI service. Model is verified once at startup."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .inference import Predictor
from .simulator import OUTPUTS, PARAMETERS


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    natural_frequency_hz: float = Field(ge=2, le=15)
    damping_ratio: float = Field(ge=0.03, le=0.3)
    drive_frequency_hz: float = Field(ge=1, le=20)
    force_n: float = Field(ge=0.2, le=3)
    mass_kg: float = Field(ge=0.5, le=3)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameters: list[Parameters] = Field(min_length=1, max_length=256)


def create_app(bundle=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = Predictor(
            bundle or os.environ.get("SIMML_BUNDLE", "artifacts/bundle")
        )
        yield

    app = FastAPI(
        title="Reproducible Simulation & ML Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Exclude raw input values so NaN/Infinity errors remain valid JSON.
        errors = [
            {k: error[k] for k in ("type", "loc", "msg")} for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health")
    def health():
        m = app.state.predictor.manifest
        return {
            "status": "ready",
            "bundle_id": m["bundle_id"],
            "dataset_id": m["dataset_id"],
        }

    @app.post("/predict")
    def predict(request: PredictionRequest):
        rows = [[getattr(p, name) for name in PARAMETERS] for p in request.parameters]
        predictor = app.state.predictor
        values = predictor.predict(rows)
        return {
            "bundle_id": predictor.manifest["bundle_id"],
            "predictions": [dict(zip(OUTPUTS, row.tolist())) for row in values],
        }

    return app


app = create_app()
