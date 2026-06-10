from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.errors import ServiceContractError
from app.core.service_loader import call_service
from app.models.documents import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest, settings: Settings = Depends(get_settings)
) -> IngestResponse:
    result = await call_service(
        candidates=[
            ("app.services.ingestion", "ingest_documents"),
            ("app.services.ingestion", "IngestionService.ingest"),
            ("app.services.ingest", "ingest_documents"),
            ("app.services.pdf_ingestion", "ingest_documents"),
        ],
        settings=settings,
        force=request.force,
        pdf_dir=settings.pdf_dir,
        chroma_dir=settings.chroma_dir,
        processed_dir=settings.processed_dir,
    )
    return IngestResponse.model_validate(_normalize_ingest_result(result))


def _normalize_ingest_result(result: object) -> object:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, dict):
        raise ServiceContractError(
            "Ingestion service must return a mapping or an object with to_dict().",
            {"received_type": type(result).__name__},
        )
    if {"status", "documents_total", "documents_indexed", "chunks_total"} <= result.keys():
        return result

    processed = result.get("processed") if isinstance(result.get("processed"), list) else []
    skipped = result.get("skipped") if isinstance(result.get("skipped"), list) else []
    failed = result.get("failed") if isinstance(result.get("failed"), list) else []
    documents_indexed = len(processed)
    chunks_total = sum(int(item.get("chunk_count") or 0) for item in processed + skipped)
    status = "failed" if failed and not processed and not skipped else "completed"
    if skipped and not processed and not failed:
        status = "skipped"
    return {
        "status": status,
        "documents_total": int(result.get("total_pdfs") or len(processed) + len(skipped) + len(failed)),
        "documents_indexed": documents_indexed,
        "chunks_total": chunks_total,
    }
