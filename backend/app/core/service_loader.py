import importlib
import inspect
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from app.core.errors import (
    ApiError,
    ServiceContractError,
    ServiceFailureError,
    ServiceNotWiredError,
)

ServiceCandidate = tuple[str, str]


async def call_service(candidates: list[ServiceCandidate], **kwargs: Any) -> Any:
    service = _resolve_service(candidates)
    selected_kwargs = _select_kwargs(service, kwargs)
    try:
        result = service(**selected_kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    except ApiError:
        raise
    except HTTPException as exc:
        raise _from_http_exception(exc) from exc
    except Exception as exc:
        raise _from_service_exception(exc) from exc


def _resolve_service(candidates: list[ServiceCandidate]) -> Callable[..., Any]:
    expected = [f"{module}.{name}" for module, name in candidates]
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise ServiceFailureError(
                "Service module import failed.",
                {"module": module_name, "error": str(exc)},
            ) from exc

        service = _resolve_attr(module, function_name)
        if callable(service):
            return service

    raise ServiceNotWiredError(expected)


def _resolve_attr(module: object, dotted_name: str) -> object | None:
    target: object = module
    for part in dotted_name.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
        if inspect.isclass(target):
            target = target()
    return target


def _select_kwargs(service: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(service)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    accepted = {
        name
        for name, param in parameters.items()
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {name: value for name, value in kwargs.items() if name in accepted}


def _from_http_exception(exc: HTTPException) -> ApiError:
    code = "service_http_error"
    message = str(exc.detail) if exc.detail else "Service returned an HTTP error."
    return ApiError(status_code=exc.status_code, code=code, message=message)


def _from_service_exception(exc: Exception) -> ApiError:
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    details = getattr(exc, "details", None)

    if isinstance(status_code, int) and isinstance(code, str):
        return ApiError(
            status_code=status_code,
            code=code,
            message=message or "Service failed.",
            details=details,
        )

    name = exc.__class__.__name__.lower()
    if "empty" in name and "index" in name:
        return ApiError(
            status_code=409,
            code="empty_index",
            message=message or "No indexed documents are available. Run ingestion first.",
            details=details,
        )
    if "ollama" in name or "connection" in name:
        return ApiError(
            status_code=503,
            code="ollama_unavailable",
            message=message or "Ollama is unavailable.",
            details=details,
        )
    if "model" in name:
        return ApiError(
            status_code=503,
            code="model_unavailable",
            message=message or "Required Ollama model is unavailable.",
            details=details,
        )
    if "contract" in name:
        return ServiceContractError(message or "Service contract error.", details)

    return ServiceFailureError(message or "Service failed.", details)
