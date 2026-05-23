from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from rediris.common.database import get_db
from rediris.common.models.audit_task import AuditTask
from rediris.common.models.miner_dataset import MinerDataset
from rediris.task_center.services.audit_task_creator import AuditTaskCreator
from rediris.task_center.schemas.audit import AuditTaskResponse, AuditTaskListResponse
from rediris.task_center.schemas.dataset import DatasetValidationRequest, DatasetValidationResponse
from rediris.common.utils.logging import setup_logger
from rediris.common.auth.signature_auth import verify_node_signature

router = APIRouter()
logger = setup_logger(__name__)

@router.get("/pending", response_model=AuditTaskListResponse)
async def get_pending_audit_tasks(
    validator_key: str,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != validator_key:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    
    creator = AuditTaskCreator(db)
    tasks = creator.get_pending_tasks_for_validator(validator_key)
    
    return AuditTaskListResponse(
        tasks=[AuditTaskResponse.from_orm(t) for t in tasks]
    )


@router.post("/receive")
async def receive_audit_task(
    audit_task_id: str,
    validator_key: str,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != validator_key:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    
    creator = AuditTaskCreator(db)
    task = creator.assign_audit_task_to_validator(audit_task_id, validator_key)

    if not task:
        raise HTTPException(status_code=404, detail="Audit task not found")

    return AuditTaskResponse.from_orm(task)


@router.post("/dataset/validation", response_model=DatasetValidationResponse)
async def submit_dataset_validation(
    request: DatasetValidationRequest,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != request.validator_hotkey:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    
    try:
        audit_task = db.query(AuditTask).filter(
            AuditTask.audit_task_id == request.audit_task_id
        ).first()

        if not audit_task:
            raise HTTPException(status_code=404, detail="Audit task not found")

        if audit_task.audit_type != "dataset":
            raise HTTPException(status_code=400, detail="Not a dataset audit task")

        audit_task.is_completed = True
        audit_task.completed_at = datetime.now(timezone.utc)
        audit_task.result = {
            "is_approved": request.is_approved,
            "validation_result": request.validation_result,
            "rejection_reason": request.rejection_reason,
            "validator_hotkey": request.validator_hotkey
        }

        if request.is_approved:
            miner_dataset = db.query(MinerDataset).filter(
                MinerDataset.task_id == audit_task.original_task_id,
                MinerDataset.miner_hotkey == audit_task.miner_hotkey
            ).first()

            if miner_dataset:
                if miner_dataset.validation_status == "pending":
                    miner_dataset.validation_status = "approved"
                    miner_dataset.validated_by = request.validator_hotkey
                    miner_dataset.validation_result = request.validation_result
                    miner_dataset.validated_at = datetime.now(timezone.utc)
                    logger.info(f"Dataset approved for miner {audit_task.miner_hotkey[:16]}... by validator {request.validator_hotkey[:16]}...")

        db.commit()

        return DatasetValidationResponse(
            audit_task_id=request.audit_task_id,
            status="completed",
            message="Dataset validation result submitted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting dataset validation: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")