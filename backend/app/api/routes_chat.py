from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.errors import ServiceContractError
from app.core.service_loader import call_service
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, settings: Settings = Depends(get_settings)
) -> ChatResponse:
    top_k = request.top_k if request.top_k is not None else settings.default_top_k
    result = await call_service(
        candidates=[
            ("app.services.chat", "answer_question"),
            ("app.services.chat", "chat"),
            ("app.agents.crew", "answer_question"),
            ("app.agents.crew", "run_chat"),
        ],
        settings=settings,
        message=request.message,
        top_k=top_k,
        include_agent_trace=settings.enable_agent_trace,
    )
    response = ChatResponse.model_validate(_normalize_chat_result(result))
    if not settings.enable_agent_trace:
        response.agent_trace = None
    return response


def _normalize_chat_result(result: object) -> object:
    if hasattr(result, "to_chat_response"):
        result = result.to_chat_response()
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, dict):
        raise ServiceContractError(
            "Chat service must return a mapping, to_dict(), or to_chat_response().",
            {"received_type": type(result).__name__},
        )
    return result
