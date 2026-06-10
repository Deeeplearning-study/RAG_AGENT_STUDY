from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.core.health import get_fallback_health


def get_health(settings: Settings | None = None) -> dict[str, Any]:
    return get_fallback_health(settings or get_settings())


def check_health(settings: Settings | None = None) -> dict[str, Any]:
    return get_health(settings)


def health_check(settings: Settings | None = None) -> dict[str, Any]:
    return get_health(settings)
