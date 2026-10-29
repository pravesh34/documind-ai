"""Celery application for background document enrichment."""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()

BROKER_URL = settings.celery_broker_url or settings.redis_url or "redis://localhost:6379/0"
RESULT_BACKEND = settings.celery_result_backend or BROKER_URL

celery_app = Celery(
    "documind",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    broker_connection_retry_on_startup=True,
    result_expires=3_600,
    task_default_queue="enrichment",
    task_routes={
        "documind.reembed_document": {"queue": "enrichment"},
    },
)
