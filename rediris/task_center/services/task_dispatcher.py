from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from rediris.common.models.task import Task, TaskStatus
from rediris.common.config import settings
from rediris.common.models.miner import Miner
from rediris.task_center.schemas.task import TaskCreate
from rediris.task_center.schemas.miner import MinerSubmitRequest
from rediris.task_center.services.task_repository import TaskRepository
from rediris.task_center.services.audit_task_creator import AuditTaskCreator
from rediris.task_center.services.miner_selector import MinerSelector
from rediris.task_center.services.miner_cache import MinerCache

from rediris.common.utils.logging import setup_logger
from rediris.task_center import shared
import uuid
import asyncio

logger = setup_logger(__name__)

class TaskDispatcher:
    def __init__(self, db: Session, miner_cache: MinerCache):
        self.db = db
        self.repository = TaskRepository(db)
        self.audit_creator = AuditTaskCreator(db)
        self.miner_selector = MinerSelector(db, miner_cache)
        self.bittensor_client = shared.bittensor_client
    
    def create_task(self, task_data: TaskCreate) -> Task:
        now = datetime.now(timezone.utc)

        announcement_duration = task_data.announcement_duration
        dataset_validation_duration = getattr(task_data, 'dataset_validation_duration', 0.0)
        execution_duration = task_data.execution_duration
        review_duration = task_data.review_duration
        reward_duration = getattr(task_data, 'reward_duration', 0.0)

        announcement_start = now
        dataset_validation_start = announcement_start + timedelta(hours=announcement_duration)
        execution_start = dataset_validation_start + timedelta(hours=dataset_validation_duration)
        review_start = execution_start + timedelta(hours=execution_duration)
        reward_start = review_start + timedelta(hours=review_duration)
        workflow_end = reward_start + timedelta(hours=reward_duration)

        initial_status = TaskStatus.ANNOUNCEMENT
        if hasattr(task_data, 'publish_status') and task_data.publish_status == 'draft':
            initial_status = TaskStatus.PENDING

        reward_config = getattr(task_data, 'reward_config', None)
        reward_pool_ratio = 1.0
        reward_pool_name = None
        min_score_threshold = 3.5
        quality_exponent = 2

        if reward_config:
            reward_pool_ratio = getattr(reward_config, 'pool_ratio', 1.0) or 1.0
            reward_pool_name = getattr(reward_config, 'pool_name', None)
            min_score_threshold = getattr(reward_config, 'min_score_threshold', 3.5) or 3.5
            quality_exponent = getattr(reward_config, 'quality_exponent', 2) or 2

        default_reward_miners = getattr(task_data, 'default_reward_miners', None)
        if default_reward_miners is None:
            default_reward_miners = 6
        else:
            if default_reward_miners < settings.MIN_REWARD_MINERS:
                default_reward_miners = settings.MIN_REWARD_MINERS
            elif default_reward_miners > settings.MAX_REWARD_MINERS:
                default_reward_miners = settings.MAX_REWARD_MINERS

        task = Task(
            task_id=task_data.task_id,
            workflow_id=task_data.task_id,
            workflow_type=task_data.workflow_type,
            workflow_spec=task_data.workflow_spec.dict(),
            status=initial_status,
            publish_status=getattr(task_data, 'publish_status', 'draft'),
            start_date=getattr(task_data, 'start_date', None),
            end_date=getattr(task_data, 'end_date', None),
            description=getattr(task_data, 'description', None),
            description_ja=getattr(task_data, 'description_ja', None),
            hf_dataset_url=getattr(task_data, 'hf_dataset_url', None),
            pdf_file_url=getattr(task_data, 'pdf_file_url', None),
            announcement_start=announcement_start if initial_status != TaskStatus.PENDING else None,
            dataset_validation_start=dataset_validation_start if initial_status != TaskStatus.PENDING else None,
            execution_start=execution_start if initial_status != TaskStatus.PENDING else None,
            review_start=review_start if initial_status != TaskStatus.PENDING else None,
            reward_start=reward_start if initial_status != TaskStatus.PENDING else None,
            workflow_end=workflow_end if initial_status != TaskStatus.PENDING else None,
            reward_pool_ratio=reward_pool_ratio,
            reward_pool_name=reward_pool_name,
            min_score_threshold=min_score_threshold,
            quality_exponent=quality_exponent,
            default_reward_miners=default_reward_miners
        )

        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)

        logger.info(f"Task created: {task.task_id} (status={initial_status.value})")
        logger.info(f"Task will be assigned to miners when EXECUTION phase starts at {execution_start}")

        return task
    
    async def assign_task_to_validated_miners(self, task: Task):
        validated_miners = self.miner_selector.select_validated_miners(task.task_id)

        if not validated_miners:
            logger.warning(f"No miners with validated datasets found for task {task.task_id}")
            return

        logger.info(f"Assigning task {task.task_id} to {len(validated_miners)} validated miners")

        for miner_info in validated_miners:
            miner_hotkey = miner_info["hotkey"]
            dataset_url = miner_info.get("dataset_url")

            task_data = {
                "task_id": task.task_id,
                "workflow_type": task.workflow_type,
                "workflow_spec": task.workflow_spec,
                "dataset_url": dataset_url,
                "announcement_start": task.announcement_start.isoformat() if task.announcement_start else None,
                "dataset_validation_start": task.dataset_validation_start.isoformat() if task.dataset_validation_start else None,
                "execution_start": task.execution_start.isoformat() if task.execution_start else None,
                "review_start": task.review_start.isoformat() if task.review_start else None,
                "reward_start": task.reward_start.isoformat() if task.reward_start else None,
                "workflow_end": task.workflow_end.isoformat() if task.workflow_end else None
            }

            results = await self.miner_selector.assign_task_to_miners(
                task.task_id,
                task_data,
                [miner_hotkey]
            )

            if results.get(miner_hotkey):
                logger.info(f"Task {task.task_id} assigned to validated miner {miner_hotkey[:16]}...")
            else:
                logger.warning(f"Failed to assign task {task.task_id} to miner {miner_hotkey[:16]}...")
    
    def assign_task_to_miner(self, task_id: str, miner_key: str) -> Task:
        task = self.repository.get_by_task_id(task_id)
        if not task:
            return None
        
        stake = self.bittensor_client.get_miner_stake(miner_key)
        
        if stake < settings.MINER_MIN_STAKE:
            logger.warning(f"Miner {miner_key} stake too low: {stake}")
            return None
        
        miner = self.db.query(Miner).filter(Miner.hotkey == miner_key).first()
        if not miner:
            miner = Miner(hotkey=miner_key, stake=stake, reputation=0.0)
            self.db.add(miner)
        else:
            miner.stake = stake
            self.db.commit()
        
        return task
    
    def select_miners_for_task(self, task_id: str, count: int = 10) -> list:
        return self.miner_selector.select_miners(task_id, count)
    
    def receive_miner_submission(self, request: MinerSubmitRequest) -> dict:
        task = self.repository.get_by_task_id(request.task_id)
        if not task:
            raise ValueError("Task not found")

        from rediris.task_center.services.task_lifecycle_manager import TaskLifecycleManager
        lifecycle_manager = TaskLifecycleManager(self.db)
        can_submit, reason = lifecycle_manager.can_miner_submit(request.task_id)

        if not can_submit:
            logger.warning(f"Miner submission rejected for task {request.task_id}: {reason}")
            raise ValueError(f"Submission not allowed: {reason}")

        submission_id = str(uuid.uuid4())

        from rediris.common.models.miner_submission import MinerSubmission

        submit_time = datetime.now(timezone.utc)

        submission = MinerSubmission(
            id=submission_id,
            task_id=request.task_id,
            miner_hotkey=request.miner_key,
            model_url=request.model_url,
            sample_images=request.sample_images,
            submission_data=request.dict(),
            status="pending_verification"
        )

        self.db.add(submission)
        self.db.commit()

        logger.info(f"Miner submission stored: {submission_id} for task {request.task_id} at {submit_time}")

        audit_task = self.audit_creator.create_audit_task(
            task_id=request.task_id,
            miner_hotkey=request.miner_key,
            lora_url=request.model_url
        )

        self.audit_creator.auto_assign_audit_tasks(request.task_id)

        asyncio.create_task(self._notify_validators(audit_task.audit_task_id))

        time_coefficient = 1.0
        if task.execution_start and task.review_start:
            from rediris.common.utils.time import calculate_time_coefficient
            time_coefficient = calculate_time_coefficient(
                submit_time, task.execution_start, task.review_start
            )

        return {
            "submission_id": submission_id,
            "task_id": request.task_id,
            "status": "pending_verification",
            "estimated_reward": 0.0,
            "submit_time": submit_time.isoformat(),
            "time_coefficient": time_coefficient
        }
    
    async def _notify_validators(self, audit_task_id: str):
        from rediris.common.models.audit_task import AuditTask
        
        audit_tasks = self.db.query(AuditTask).filter(
            AuditTask.audit_task_id.like(f"{audit_task_id}%")
        ).all()
        
        for audit_task in audit_tasks:
            if not audit_task.validator_hotkey:
                continue
            
            try:
                import httpx
                validator_url = f"http://{audit_task.validator_hotkey}:8000"
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{validator_url}/v1/audit/receive",
                        json={
                            "audit_task_id": audit_task.audit_task_id,
                            "validator_key": audit_task.validator_hotkey
                        }
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Notified validator {audit_task.validator_hotkey} about audit task {audit_task.audit_task_id}")
            except Exception as e:
                logger.error(f"Failed to notify validator {audit_task.validator_hotkey}: {e}")

