from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime, timezone
from typing import List
from rediris.common.database import get_db
from rediris.common.models.task import Task, TaskStatus
from rediris.common.models.miner_dataset import MinerDataset
from rediris.common.models.audit_task import AuditTask
from rediris.task_center.services.task_dispatcher import TaskDispatcher
from rediris.task_center.services.audit_task_creator import AuditTaskCreator
from rediris.task_center.shared import miner_cache
from rediris.task_center.schemas.miner import (
    MinerTaskReceive,
    MinerTaskResponse,
    MinerSubmitRequest,
    MinerSubmitResponse,
    AvailableTasksResponse,
    AvailableTaskItem
)
from rediris.task_center.schemas.dataset import (
    DatasetSubmitRequest,
    DatasetSubmitResponse,
    DatasetStatusResponse,
    DatasetListResponse
)
from rediris.common.utils.logging import setup_logger
from rediris.common.auth.signature_auth import verify_node_signature

router = APIRouter()
logger = setup_logger(__name__)


@router.post("/receive", response_model=MinerTaskResponse)
async def receive_task(
    request: MinerTaskReceive,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    logger.info("=" * 80)
    logger.info("receive_task endpoint CALLED!")
    logger.info(f"task_id={request.task_id}, miner_key={request.miner_key[:20]}...")
    logger.info("=" * 80)
    
    if hotkey != request.miner_key:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    
    try:
        dispatcher = TaskDispatcher(db, miner_cache)
        task = dispatcher.assign_task_to_miner(request.task_id, request.miner_key)

        if not task:
            logger.warning(f"Task not found or already assigned: task_id={request.task_id}")
            raise HTTPException(status_code=404, detail="Task not found or already assigned")

        logger.info(f"Task assigned successfully: task_id={request.task_id}")
        return MinerTaskResponse.from_task(task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in receive_task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/submit", response_model=MinerSubmitResponse)
async def submit_result(
    request: MinerSubmitRequest,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != request.miner_key:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    dispatcher = TaskDispatcher(db, miner_cache)
    submission = dispatcher.receive_miner_submission(request)

    return MinerSubmitResponse(
        submission_id=submission["submission_id"],
        task_id=submission["task_id"],
        status=submission["status"],
        estimated_reward=submission.get("estimated_reward", 0.0)
    )


@router.post("/dataset/submit", response_model=DatasetSubmitResponse)
async def submit_dataset(
    request: DatasetSubmitRequest,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != request.miner_hotkey:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    
    try:
        async def ensure_dataset_audit_task(task_id: str, miner_hotkey: str, dataset_url: str, workflow_spec: dict):
            existing_dataset_audit = db.query(AuditTask).filter(
                AuditTask.original_task_id == task_id,
                AuditTask.miner_hotkey == miner_hotkey,
                AuditTask.audit_type == "dataset",
                AuditTask.is_completed == False
            ).first()
            if existing_dataset_audit:
                logger.info(
                    f"Dataset audit task already exists for miner {miner_hotkey[:16]}... "
                    f"(audit_task_id={existing_dataset_audit.audit_task_id})"
                )
                return existing_dataset_audit

            audit_creator = AuditTaskCreator(db)
            created = await audit_creator.create_dataset_audit_task(
                task_id=task_id,
                miner_hotkey=miner_hotkey,
                dataset_url=dataset_url,
                task_info=workflow_spec
            )
            logger.info(f"Dataset audit task created for miner {miner_hotkey[:16]}... (audit_task_id={created.audit_task_id})")
            return created

        task = db.query(Task).filter(Task.task_id == request.task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status not in [TaskStatus.ANNOUNCEMENT, TaskStatus.DATASET_VALIDATION]:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset submission not allowed in {task.status.value} phase"
            )

        existing = db.query(MinerDataset).filter(
            MinerDataset.task_id == request.task_id,
            MinerDataset.miner_hotkey == request.miner_hotkey
        ).first()

        if existing:
            if existing.validation_status == "pending":
                existing.dataset_url = request.dataset_url
                existing.dataset_description = request.dataset_description
                db.commit()
                try:
                    await ensure_dataset_audit_task(
                        task_id=request.task_id,
                        miner_hotkey=request.miner_hotkey,
                        dataset_url=request.dataset_url,
                        workflow_spec=task.workflow_spec
                    )
                except Exception as e:
                    logger.error(f"Failed to ensure dataset audit task after dataset update: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Dataset submitted but failed to create audit task")
                logger.info(f"Updated dataset submission for miner {request.miner_hotkey[:16]}... on task {request.task_id}")
                return DatasetSubmitResponse(
                    submission_id=existing.id,
                    task_id=existing.task_id,
                    miner_hotkey=existing.miner_hotkey,
                    status=existing.validation_status,
                    message="Dataset submission updated successfully"
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset already submitted and {existing.validation_status}. Cannot resubmit."
                )

        miner_dataset = MinerDataset(
            task_id=request.task_id,
            miner_hotkey=request.miner_hotkey,
            dataset_url=request.dataset_url,
            dataset_description=request.dataset_description,
            validation_status="pending"
        )
        db.add(miner_dataset)
        db.commit()
        db.refresh(miner_dataset)

        logger.info(f"Dataset submitted by miner {request.miner_hotkey[:16]}... for task {request.task_id}")

        try:
            await ensure_dataset_audit_task(
                task_id=request.task_id,
                miner_hotkey=request.miner_hotkey,
                dataset_url=request.dataset_url,
                workflow_spec=task.workflow_spec
            )
        except Exception as e:
            logger.error(f"Failed to ensure dataset audit task after dataset submit: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Dataset submitted but failed to create audit task")

        return DatasetSubmitResponse(
            submission_id=miner_dataset.id,
            task_id=miner_dataset.task_id,
            miner_hotkey=miner_dataset.miner_hotkey,
            status=miner_dataset.validation_status,
            message="Dataset submitted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in submit_dataset: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/dataset/status", response_model=DatasetStatusResponse)
async def get_dataset_status(
    task_id: str = Query(...),
    miner_hotkey: str = Query(...),
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != miner_hotkey:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    try:
        dataset = db.query(MinerDataset).filter(
            MinerDataset.task_id == task_id,
            MinerDataset.miner_hotkey == miner_hotkey
        ).first()

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset submission not found")

        return DatasetStatusResponse(
            task_id=dataset.task_id,
            miner_hotkey=dataset.miner_hotkey,
            dataset_url=dataset.dataset_url,
            validation_status=dataset.validation_status,
            validated_by=dataset.validated_by,
            validation_result=dataset.validation_result,
            rejection_reason=dataset.rejection_reason,
            created_at=dataset.created_at,
            validated_at=dataset.validated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_dataset_status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/dataset/list", response_model=DatasetListResponse)
async def list_datasets(
    task_id: str = Query(...),
    status: str = Query(None),
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    try:
        query = db.query(MinerDataset).filter(MinerDataset.task_id == task_id)

        if status:
            query = query.filter(MinerDataset.validation_status == status)

        datasets = query.all()

        return DatasetListResponse(
            datasets=[
                DatasetStatusResponse(
                    task_id=d.task_id,
                    miner_hotkey=d.miner_hotkey,
                    dataset_url=d.dataset_url,
                    validation_status=d.validation_status,
                    validated_by=d.validated_by,
                    validation_result=d.validation_result,
                    rejection_reason=d.rejection_reason,
                    created_at=d.created_at,
                    validated_at=d.validated_at
                )
                for d in datasets
            ],
            total=len(datasets)
        )

    except Exception as e:
        logger.error(f"Error in list_datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/tasks/available", response_model=AvailableTasksResponse)
async def get_available_tasks(
    miner_hotkey: str = Query(...),
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != miner_hotkey:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    try:
        tasks = db.query(Task).filter(
            or_(
                Task.status == TaskStatus.ANNOUNCEMENT,
                Task.status == TaskStatus.DATASET_VALIDATION
            )
        ).all()

        available_tasks = []
        for task in tasks:
            submitted_dataset = db.query(MinerDataset).filter(
                MinerDataset.task_id == task.task_id,
                MinerDataset.miner_hotkey == miner_hotkey
            ).first()

            dataset_status = None
            if submitted_dataset:
                dataset_status = submitted_dataset.validation_status

            available_tasks.append(AvailableTaskItem(
                task_id=task.task_id,
                workflow_type=task.workflow_type,
                status=task.status.value,
                workflow_spec=task.workflow_spec,
                announcement_start=task.announcement_start,
                dataset_validation_start=task.dataset_validation_start,
                execution_start=task.execution_start,
                dataset_submitted=submitted_dataset is not None,
                dataset_status=dataset_status
            ))

        return AvailableTasksResponse(
            tasks=available_tasks,
            total=len(available_tasks)
        )

    except Exception as e:
        logger.error(f"Error in get_available_tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
