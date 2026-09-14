from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.graph import build_graph
from app.routers import sessions
from telemetry import configure_telemetry

logging.basicConfig(level=logging.INFO)

configure_telemetry(
    settings.otel_service_name,
    console_export=settings.otel_console_export,
    otlp_endpoint=settings.otel_exporter_otlp_endpoint,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield


app = FastAPI(title="Monologue Coach", lifespan=lifespan)

app.include_router(sessions.router)

_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
