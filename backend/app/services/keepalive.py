"""Optional self-ping watchdog that keeps free-tier instances warm.

Render's free web tier spins an instance down after ~15 minutes without
inbound HTTP traffic. An HTTPS self-request through the public proxy counts as
inbound traffic, so a scheduled ``GET /healthz`` can keep the service warm
without any external cron:

    SELF_PING_ENABLED=true
    SELF_PING_URL=https://documind-api.onrender.com
    SELF_PING_INTERVAL_SECONDS=420   # every 7 minutes

Disabled by default; enable only on platforms where the cost is free.
"""

from __future__ import annotations

import asyncio

import httpx

from app.config import Settings
from app.core.logging import get_logger

logger = get_logger("services.keepalive")


class KeepAlive:
    def __init__(self, settings: Settings) -> None:
        self.url = (settings.self_ping_url or "").rstrip("/")
        self.interval = settings.self_ping_interval_seconds
        self.enabled = settings.self_ping_enabled and bool(self.url)
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        if self.interval < 30:
            logger.warning(
                "self-ping interval too short; clamping to 30s",
                requested=self.interval,
            )
            self.interval = 30
        self._task = asyncio.create_task(self._loop(), name="self-ping")
        logger.info("self-ping started", url=self.url, interval=self.interval)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("self-ping stopped")

    async def _loop(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                while True:
                    await asyncio.sleep(self.interval)
                    try:
                        response = await client.get(f"{self.url}/healthz")
                        if response.status_code != 200:
                            logger.warning(
                                "self-ping received unexpected status",
                                status=response.status_code,
                            )
                        else:
                            logger.debug("self-ping ok")
                    except Exception as error:  # noqa: BLE001 - watchdog failures are non-fatal
                        logger.warning("self-ping failed", error=str(error))
        except asyncio.CancelledError:
            logger.debug("self-ping task cancelled")
            raise
