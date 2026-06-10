from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.errors import ServiceContractError
from app.core.service_loader import call_service
from app.models.documents import DocumentResponse

router = APIRouter()


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    settings: Settings = Depends(get_settings),
) -> list[DocumentResponse]:
    result = await call_service(
        candidates=[
            ("app.services.documents", "list_documents"),
            ("app.services.ingestion", "list_documents"),
            ("app.services.ingestion", "DocumentService.list_documents"),
            ("app.services.pdf_ingestion", "list_documents"),
        ],
        settings=settings,
        processed_dir=settings.processed_dir,
        chroma_dir=settings.chroma_dir,
    )
    return [DocumentResponse.model_validate(item) for item in _normalize_documents(result)]


def _normalize_documents(result: object) -> list[object]:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if result is None:
        return []
    if isinstance(result, dict) and isinstance(result.get("documents"), list):
        result = result["documents"]
    if not isinstance(result, list):
        raise ServiceContractError(
            "Document service must return a list or a mapping with a documents list.",
            {"received_type": type(result).__name__},
        )

    documents: list[object] = []
    for item in result:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, dict):
            documents.append(item)
            continue
        if {"document_id", "file_name", "pages", "chunks"} <= item.keys():
            documents.append(item)
            continue
        documents.append(
            {
                "document_id": item.get("document_id", ""),
                "file_name": item.get("file_name", ""),
                "pages": int(item.get("pages") or item.get("page_count") or 0),
                "chunks": int(item.get("chunks") or item.get("chunk_count") or 0),
                "indexed_at": item.get("indexed_at"),
            }
        )
    return documents
