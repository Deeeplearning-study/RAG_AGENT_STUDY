from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.errors import ServiceContractError, ServiceNotWiredError
from app.core.health import get_fallback_health
from app.core.service_loader import call_service
from app.models.documents import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    try:
        result = await call_service(
            candidates=[
                ("app.services.health", "get_health"),
                ("app.services.health", "check_health"),
                ("app.services.health", "health_check"),
            ],
            settings=settings,
        )
    except ServiceNotWiredError:
        result = get_fallback_health(settings)
    return HealthResponse.model_validate(_normalize_health_result(result))


def _normalize_health_result(result: object) -> object:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, dict):
        raise ServiceContractError(
            "Health service must return a mapping or an object with to_dict().",
            {"received_type": type(result).__name__},
        )
    return result
