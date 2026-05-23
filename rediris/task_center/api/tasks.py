from fastapi import APIRouter, Depends, HTTPException, Security, Request
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta, timezone

from rediris.common.database import get_db
from rediris.common.models.task import Task, TaskStatus, PublishStatus
from rediris.common.auth.api_key import verify_api_key
from rediris.common.auth.signature_auth import verify_node_signature
from rediris.task_center.services.task_dispatcher import TaskDispatcher
from rediris.task_center.services.task_repository import TaskRepository
from rediris.task_center.services.task_validator import TaskValidator, TaskValidationError
from rediris.task_center.schemas.task import TaskCreate, TaskResponse, TaskListResponse
from rediris.common.utils.logging import setup_logger
from rediris.task_center.shared import miner_cache

router = APIRouter()
logger = setup_logger(__name__)

@router.post("/publish", response_model=TaskResponse)
async def publish_task(
    task_data: TaskCreate,
    request: Request,
    api_key: str = Security(verify_api_key),
    db: Session = Depends(get_db)
):
    logger.info(f"Task publish request received from authorized source (API key verified): {task_data.task_id}")

    workflow_spec_dict = task_data.workflow_spec.dict() if hasattr(task_data.workflow_spec, 'dict') else task_data.workflow_spec
    logger.info(f"Task {task_data.task_id}: workflow_spec keys: {list(workflow_spec_dict.keys())}")
    logger.info(f"Task {task_data.task_id}: test_spec from request: {workflow_spec_dict.get('test_spec')}")
    
    try:
        if hasattr(task_data, 'model_dump'):
            task_dict = task_data.model_dump()
        elif hasattr(task_data, 'dict'):
            task_dict = task_data.dict()
        else:
            task_dict = task_data
        
        workflow_spec = task_dict.get('workflow_spec')
        if workflow_spec:
            if hasattr(workflow_spec, 'model_dump'):
                workflow_spec_dict = workflow_spec.model_dump()
            elif hasattr(workflow_spec, 'dict'):
                workflow_spec_dict = workflow_spec.dict()
            else:
                workflow_spec_dict = workflow_spec
            
            is_valid, errors = TaskValidator.validate_workflow_spec(workflow_spec_dict)
            if not is_valid:
                error_message = "; ".join(errors)
                logger.warning(f"Task validation failed for {task_data.task_id}: {error_message}")
                raise HTTPException(status_code=400, detail=f"Task validation failed: {error_message}")

        is_valid, errors = TaskValidator.validate_task_create(task_dict)
        if not is_valid:
            error_message = "; ".join(errors)
            logger.warning(f"Task validation failed for {task_data.task_id}: {error_message}")
            raise HTTPException(status_code=400, detail=f"Task validation failed: {error_message}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating task {task_data.task_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Task validation error: {str(e)}")
    
    dispatcher = TaskDispatcher(db, miner_cache)
    repository = TaskRepository(db)

    existing_task = repository.get_by_task_id(task_data.task_id)
    if existing_task:
        db.refresh(existing_task)
        if existing_task.publish_status == PublishStatus.PUBLISHED:
            raise HTTPException(status_code=400, detail="Task already published")

        now = datetime.now(timezone.utc)
        announcement_duration = task_data.announcement_duration
        dataset_validation_duration = getattr(task_data, "dataset_validation_duration", 0.0)
        execution_duration = task_data.execution_duration
        review_duration = task_data.review_duration
        reward_duration = getattr(task_data, "reward_duration", 0.0)

        announcement_start = now
        dataset_validation_start = announcement_start + timedelta(hours=announcement_duration)
        execution_start = dataset_validation_start + timedelta(hours=dataset_validation_duration)
        review_start = execution_start + timedelta(hours=execution_duration)
        reward_start = review_start + timedelta(hours=review_duration)
        workflow_end = reward_start + timedelta(hours=reward_duration)

        existing_task.workflow_type = task_data.workflow_type
        existing_task.workflow_spec = task_data.workflow_spec.dict()
        existing_task.status = TaskStatus.ANNOUNCEMENT
        existing_task.publish_status = PublishStatus.PUBLISHED
        existing_task.start_date = task_data.start_date if task_data.start_date is not None else existing_task.start_date
        existing_task.end_date = task_data.end_date if task_data.end_date is not None else existing_task.end_date
        existing_task.description = task_data.description if task_data.description is not None else existing_task.description
        existing_task.description_ja = task_data.description_ja if task_data.description_ja is not None else existing_task.description_ja
        existing_task.hf_dataset_url = task_data.hf_dataset_url if task_data.hf_dataset_url is not None else existing_task.hf_dataset_url
        existing_task.pdf_file_url = task_data.pdf_file_url if task_data.pdf_file_url is not None else existing_task.pdf_file_url
        existing_task.announcement_start = announcement_start
        existing_task.dataset_validation_start = dataset_validation_start
        existing_task.execution_start = execution_start
        existing_task.review_start = review_start
        existing_task.reward_start = reward_start
        existing_task.workflow_end = workflow_end
        
        from rediris.common.config import settings
        default_reward_miners = getattr(task_data, 'default_reward_miners', None)
        if default_reward_miners is not None:
            if default_reward_miners < settings.MIN_REWARD_MINERS:
                default_reward_miners = settings.MIN_REWARD_MINERS
            elif default_reward_miners > settings.MAX_REWARD_MINERS:
                default_reward_miners = settings.MAX_REWARD_MINERS
            existing_task.default_reward_miners = default_reward_miners

        db.commit()
        db.refresh(existing_task)
        task = existing_task

    else:
        task = dispatcher.create_task(task_data)

    return TaskResponse.from_orm(task)


@router.get("/active/count")
async def get_active_task_count(
    request: Request,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    reward_tasks = db.query(Task).filter(
        Task.status == TaskStatus.REWARD
    ).all()

    active_count = len(reward_tasks)
    is_idle = active_count == 0

    reward_task_ids = []
    task_types = {"text": [], "image": []}
    task_configs = {}

    for task in reward_tasks:
        reward_task_ids.append(task.task_id)

        task_type_str = task.workflow_type.value if hasattr(task.workflow_type, 'value') else str(task.workflow_type)
        task_type_lower = task_type_str.lower()

        if "image" in task_type_lower:
            task_types["image"].append(task.task_id)
        else:
            task_types["text"].append(task.task_id)

        task_configs[task.task_id] = {
            "workflow_type": task_type_str,
            "pool_ratio": getattr(task, 'reward_pool_ratio', 1.0) or 1.0,
            "pool_name": getattr(task, 'reward_pool_name', None) or task_type_lower.split('_')[0],
            "min_score_threshold": getattr(task, 'min_score_threshold', 3.5) or 3.5,
            "quality_exponent": getattr(task, 'quality_exponent', 2) or 2
        }

    return {
        "active_count": active_count,
        "is_idle": is_idle,
        "reward_task_ids": reward_task_ids,
        "task_types": task_types,
        "task_configs": task_configs
    }

@router.get("/{task_id}/participants")
async def get_task_participants(
    task_id: str,
    request: Request,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    from rediris.common.models.miner_submission import MinerSubmission

    submissions = db.query(MinerSubmission).filter(
        MinerSubmission.task_id == task_id
    ).all()

    miner_hotkeys = list(set(s.miner_hotkey for s in submissions))

    return {
        "task_id": task_id,
        "miner_hotkeys": miner_hotkeys,
        "participant_count": len(miner_hotkeys)
    }

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    request: Request,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    repository = TaskRepository(db)
    task = repository.get_by_task_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse.from_orm(task)

@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: TaskStatus = None,
    page: int = 1,
    page_size: int = 20,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    repository = TaskRepository(db)
    tasks, total = repository.list_tasks(status=status, page=page, page_size=page_size)
    
    return TaskListResponse(
        workflows=[TaskResponse.from_orm(t) for t in tasks],
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.get("/pending")
async def get_pending_tasks(
    request: Request,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):

    from rediris.common.models.miner_submission import MinerSubmission
    from rediris.common.models.audit_task import AuditTask
    
    repository = TaskRepository(db)
    active_tasks = db.query(Task).filter(
        Task.status.in_([TaskStatus.EXECUTION, TaskStatus.REVIEW])
    ).all()
    
    pending_tasks = []
    for task in active_tasks:
        submissions = db.query(MinerSubmission).filter(
            MinerSubmission.task_id == task.task_id,
            MinerSubmission.status == "pending_verification"
        ).all()

        for submission in submissions:
            audit_tasks = db.query(AuditTask).filter(
                AuditTask.original_task_id == task.task_id,
                AuditTask.miner_hotkey == submission.miner_hotkey,
                AuditTask.validator_hotkey.isnot(None)
            ).all()

            if not audit_tasks or not all(t.is_completed for t in audit_tasks):
                pending_tasks.append({
                    "task_id": task.task_id,
                    "workflow_type": task.workflow_type,
                    "workflow_spec": task.workflow_spec,
                    "miner_hotkey": submission.miner_hotkey,
                    "lora_url": submission.model_url,
                    "submission_id": submission.id,
                    "submitted_at": submission.created_at.isoformat() if submission.created_at else None,
                    "task_status": task.status.value,
                    "audit_status": {
                        "total_audits": len(audit_tasks),
                        "completed_audits": sum(1 for t in audit_tasks if t.is_completed),
                        "pending_audits": sum(1 for t in audit_tasks if not t.is_completed)
                    }
                })
    
    return {
        "pending_tasks": pending_tasks,
        "total": len(pending_tasks)
    }


@router.get("/{task_id}/config")
async def get_task_config(
    task_id: str,
    request: Request,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    repository = TaskRepository(db)
    task = repository.get_by_task_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    from rediris.common.config import settings
    
    return {
        "task_id": task_id,
        "min_reward_miners": settings.MIN_REWARD_MINERS,
        "max_reward_miners": settings.MAX_REWARD_MINERS,
        "default_reward_miners": getattr(task, 'default_reward_miners', 6)
    }

@router.get("/{task_id}/phase")
async def get_task_phase(
    task_id: str,
    request: Request,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    from rediris.task_center.services.task_lifecycle_manager import TaskLifecycleManager

    lifecycle_manager = TaskLifecycleManager(db)
    phase_info = lifecycle_manager.get_task_phase(task_id)

    if not phase_info:
        raise HTTPException(status_code=404, detail="Task not found")

    if phase_info.get("phase_start"):
        phase_info["phase_start"] = phase_info["phase_start"].isoformat()
    if phase_info.get("phase_end"):
        phase_info["phase_end"] = phase_info["phase_end"].isoformat()

    return {
        "task_id": task_id,
        **phase_info
    }

