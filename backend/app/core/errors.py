from typing import Any


class ApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details

    def to_response(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.details is not None:
            payload["error"]["details"] = self.details
        return payload


class ServiceNotWiredError(ApiError):
    def __init__(self, expected: list[str]) -> None:
        super().__init__(
            status_code=503,
            code="service_not_wired",
            message="Required backend service is not wired yet.",
            details={"expected_callables": expected},
        )


class ServiceContractError(ApiError):
    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(
            status_code=500,
            code="service_contract_error",
            message=message,
            details=details,
        )


class ServiceFailureError(ApiError):
    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(
            status_code=502,
            code="service_failure",
            message=message,
            details=details,
        )

