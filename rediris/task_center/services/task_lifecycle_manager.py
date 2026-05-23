from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from typing import Optional
from rediris.common.models.task import Task, TaskStatus
from rediris.common.utils.logging import setup_logger
from rediris.common.config.yaml_config import YamlConfig
import bittensor as bt
from rediris.task_center.services.miner_cache import MinerCache
import asyncio

logger = setup_logger(__name__)

class TaskLifecycleManager:

    def __init__(
        self,
        db: Session,
        miner_cache: Optional[MinerCache] = None,
        wallet: Optional[bt.wallet] = None,
        wallet_name: Optional[str] = None,
        hotkey_name: Optional[str] = None,
        yaml_config: Optional[YamlConfig] = None
    ):
        self.db = db
        self.miner_cache = miner_cache
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.yaml_config = yaml_config
        self.is_running = False
        self._lifecycle_task = None
    
    async def start(self):
        if self.is_running:
            logger.warning("Task lifecycle manager is already running")
            return
        
        self.is_running = True
        self._lifecycle_task = asyncio.create_task(self._lifecycle_loop())
        logger.info("Task lifecycle manager started")
    
    async def stop(self):
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._lifecycle_task:
            self._lifecycle_task.cancel()
            try:
                await self._lifecycle_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task lifecycle manager stopped")
    
    async def _lifecycle_loop(self):
        while self.is_running:
            try:
                await self._update_task_statuses()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("Lifecycle loop cancelled")
                break
            except Exception as e:
                logger.error(f"Task lifecycle loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _update_task_statuses(self):
        from rediris.common.database import SessionLocal
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)

            active_tasks = db.query(Task).filter(
                Task.status.in_([
                    TaskStatus.ANNOUNCEMENT,
                    TaskStatus.DATASET_VALIDATION,
                    TaskStatus.EXECUTION,
                    TaskStatus.REVIEW,
                    TaskStatus.REWARD
                ])
            ).all()

            for task in active_tasks:
                try:
                    if task.status == TaskStatus.ANNOUNCEMENT:
                        if task.dataset_validation_start and now >= task.dataset_validation_start:
                            task.status = TaskStatus.DATASET_VALIDATION
                            db.commit()
                            logger.info(f"Task {task.task_id} entered DATASET_VALIDATION phase")

                    elif task.status == TaskStatus.DATASET_VALIDATION:
                        if task.execution_start and now >= task.execution_start:
                            task.status = TaskStatus.EXECUTION
                            db.commit()
                            logger.info(f"Task {task.task_id} entered EXECUTION phase")
                            await self._dispatch_task_to_validated_miners(db, task)

                    elif task.status == TaskStatus.EXECUTION:
                        if task.review_start and now >= task.review_start:
                            task.status = TaskStatus.REVIEW
                            db.commit()
                            logger.info(f"Task {task.task_id} entered REVIEW phase")

                    elif task.status == TaskStatus.REVIEW:
                        if task.reward_start and now >= task.reward_start:
                            task.status = TaskStatus.REWARD
                            db.commit()
                            logger.info(f"Task {task.task_id} entered REWARD phase")

                    elif task.status == TaskStatus.REWARD:
                        if task.workflow_end and now >= task.workflow_end:
                            task.status = TaskStatus.ENDED
                            db.commit()
                            logger.info(f"Task {task.task_id} ended")

                except Exception as e:
                    logger.error(f"Error updating task {task.task_id} status: {e}", exc_info=True)
                    db.rollback()
        except Exception as e:
            logger.error(f"Error in update_task_statuses: {e}", exc_info=True)
        finally:
            db.close()

    async def _dispatch_task_to_validated_miners(self, db: Session, task: Task):
        try:
            from rediris.task_center.services.task_dispatcher import TaskDispatcher

            if not self.miner_cache:
                from rediris.task_center.shared import miner_cache
                self.miner_cache = miner_cache

            dispatcher = TaskDispatcher(db, self.miner_cache)
            await dispatcher.assign_task_to_validated_miners(task)

        except Exception as e:
            logger.error(f"Error dispatching task {task.task_id} to validated miners: {e}", exc_info=True)
    
    def is_task_in_execution_or_review(self, task_id: str) -> bool:
        try:
            task = self.db.query(Task).filter(Task.task_id == task_id).first()
            if not task:
                return False

            return task.status in [TaskStatus.EXECUTION, TaskStatus.REVIEW]
        except Exception as e:
            logger.error(f"Error checking task status: {e}", exc_info=True)
            return False

    def is_task_ended(self, task_id: str) -> bool:
        try:
            task = self.db.query(Task).filter(Task.task_id == task_id).first()
            if not task:
                return False

            return task.status == TaskStatus.ENDED
        except Exception as e:
            logger.error(f"Error checking if task ended: {e}", exc_info=True)
            return False

    def get_task_phase(self, task_id: str) -> dict:
        try:
            task = self.db.query(Task).filter(Task.task_id == task_id).first()
            if not task:
                return None

            now = datetime.now(timezone.utc)

            result = {
                "phase": task.status.value,
                "can_submit": False,
                "can_score": False,
                "can_distribute_rewards": False,
                "time_remaining": 0,
                "phase_start": None,
                "phase_end": None,
            }

            if task.status == TaskStatus.ANNOUNCEMENT:
                result["phase_start"] = task.announcement_start
                result["phase_end"] = task.dataset_validation_start
                result["can_submit"] = False
                result["can_score"] = False
                result["can_submit_dataset"] = True

            elif task.status == TaskStatus.DATASET_VALIDATION:
                result["phase_start"] = task.dataset_validation_start
                result["phase_end"] = task.execution_start
                result["can_submit"] = False
                result["can_score"] = False
                result["can_submit_dataset"] = True

            elif task.status == TaskStatus.EXECUTION:
                result["phase_start"] = task.execution_start
                result["phase_end"] = task.review_start
                result["can_submit"] = True
                result["can_score"] = False

            elif task.status == TaskStatus.REVIEW:
                result["phase_start"] = task.review_start
                result["phase_end"] = task.reward_start
                result["can_submit"] = False
                result["can_score"] = True

            elif task.status == TaskStatus.REWARD:
                result["phase_start"] = task.reward_start
                result["phase_end"] = task.workflow_end
                result["can_submit"] = False
                result["can_score"] = False
                result["can_distribute_rewards"] = True

            elif task.status == TaskStatus.ENDED:
                result["phase_start"] = task.workflow_end
                result["phase_end"] = None

            if result["phase_end"]:
                remaining = (result["phase_end"] - now).total_seconds()
                result["time_remaining"] = max(0, remaining)

            return result

        except Exception as e:
            logger.error(f"Error getting task phase: {e}", exc_info=True)
            return None

    def can_miner_submit(self, task_id: str) -> tuple:
        phase_info = self.get_task_phase(task_id)
        if not phase_info:
            return False, "Task not found"

        if phase_info["phase"] == TaskStatus.ANNOUNCEMENT.value:
            return False, "Task is in announcement phase. Submission not allowed yet. Please wait for execution phase."

        if phase_info["phase"] == TaskStatus.DATASET_VALIDATION.value:
            return False, "Task is in dataset validation phase. Submission not allowed yet. Please wait for execution phase."

        if phase_info["phase"] == TaskStatus.EXECUTION.value:
            return True, "OK"

        if phase_info["phase"] == TaskStatus.REVIEW.value:
            return False, "Execution phase has ended. Submission window closed."

        if phase_info["phase"] in [TaskStatus.REWARD.value, TaskStatus.ENDED.value]:
            return False, "Task has ended. No more submissions allowed."

        return False, f"Unknown task phase: {phase_info['phase']}"

    def can_validator_score(self, task_id: str) -> tuple:
        phase_info = self.get_task_phase(task_id)
        if not phase_info:
            return False, "Task not found"

        if phase_info["phase"] == TaskStatus.ANNOUNCEMENT.value:
            return False, "Task is in announcement phase. Scoring not started."

        if phase_info["phase"] == TaskStatus.DATASET_VALIDATION.value:
            return False, "Task is in dataset validation phase. Scoring not started."

        if phase_info["phase"] == TaskStatus.EXECUTION.value:
            return True, "OK"

        if phase_info["phase"] == TaskStatus.REVIEW.value:
            return True, "OK"

        if phase_info["phase"] == TaskStatus.REWARD.value:
            return True, "OK"

        if phase_info["phase"] == TaskStatus.ENDED.value:
            return False, "Task has ended. No more scoring allowed."

        return False, f"Unknown task phase: {phase_info['phase']}"

    def can_distribute_rewards(self, task_id: str) -> tuple:
        phase_info = self.get_task_phase(task_id)
        if not phase_info:
            return False, "Task not found"

        if phase_info["phase"] in [TaskStatus.EXECUTION.value, TaskStatus.REVIEW.value, TaskStatus.REWARD.value]:
            return True, "OK"

        if phase_info["phase"] == TaskStatus.ANNOUNCEMENT.value:
            return False, "Task is in announcement phase. Reward distribution not started."

        if phase_info["phase"] == TaskStatus.DATASET_VALIDATION.value:
            return False, "Task is in dataset validation phase. Reward distribution not started."

        if phase_info["phase"] == TaskStatus.ENDED.value:
            return False, "Task has ended."

        return False, f"Unknown task phase: {phase_info['phase']}"
