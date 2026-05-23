from fastapi import APIRouter, HTTPException
from rediris.validator.services.audit_validator import AuditValidator
from rediris.validator.schemas.audit import AuditTaskRequest, AuditTaskResponse
from rediris.common.utils.logging import setup_logger
import httpx
from rediris.common.config import settings

router = APIRouter()
logger = setup_logger(__name__)
audit_validator = AuditValidator()

@router.post("/validate", response_model=AuditTaskResponse)
async def validate_audit_task(request: AuditTaskRequest):
    try:
        result = await audit_validator.process_audit_task(request)
        return AuditTaskResponse(**result)
    except Exception as e:
        logger.error(f"Audit validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending")
async def get_pending_tasks(validator_key: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.TASK_CENTER_URL}/v1/validators/pending",
                params={"validator_key": validator_key}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to get pending tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pending tasks")


@router.post("/receive")
async def receive_audit_task(
    audit_task_id: str,
    validator_key: str
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.TASK_CENTER_URL}/v1/validators/receive",
                json={
                    "audit_task_id": audit_task_id,
                    "validator_key": validator_key
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to receive audit task: {e}")
        raise HTTPException(status_code=500, detail="Failed to receive audit task")

