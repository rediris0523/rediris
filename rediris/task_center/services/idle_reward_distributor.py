
import asyncio
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

from rediris.common.models.task import Task, TaskStatus
from rediris.common.models.score import Score
from rediris.common.bittensor.client import BittensorClient
import bittensor as bt
from rediris.common.config.yaml_config import YamlConfig
from rediris.common.config import settings
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class IdleRewardDistributor:

    IDLE_BASE_SCORE = 1.0
    IDLE_WORKFLOW_ID = "IDLE_PERIOD"
    IDLE_VALIDATOR_HOTKEY = "TASK_CENTER_IDLE"

    def __init__(
        self,
        db: Session,
        wallet: Optional[bt.wallet] = None,
        wallet_name: Optional[str] = None,
        hotkey_name: Optional[str] = None,
        yaml_config: Optional[YamlConfig] = None,
        score_interval: int = 360
    ):
        self.db = db
        self.wallet = wallet
        self.wallet_name = wallet_name or "task_center"
        self.hotkey_name = hotkey_name or "default"
        self.yaml_config = yaml_config
        self.score_interval = score_interval

        self.bittensor_client = BittensorClient(self.wallet_name, self.hotkey_name, yaml_config=yaml_config)

        self.is_running = False
        self._score_task = None

    async def start(self):
        if self.is_running:
            logger.warning("Idle reward distributor is already running")
            return

        self.is_running = True
        self._score_task = asyncio.create_task(self._score_loop())
        logger.info(f"Idle reward distributor started (interval={self.score_interval}s)")

    async def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self._score_task:
            self._score_task.cancel()
            try:
                await self._score_task
            except asyncio.CancelledError:
                pass

        logger.info("Idle reward distributor stopped")

    async def _score_loop(self):
        await asyncio.sleep(300)

        while self.is_running:
            try:
                if self._is_system_idle():
                    await self._save_idle_score_for_uid()

                await asyncio.sleep(self.score_interval)

            except asyncio.CancelledError:
                logger.info("Idle score loop cancelled")
                break
            except Exception as e:
                logger.error(f"Idle score error: {e}", exc_info=True)
                await asyncio.sleep(self.score_interval)

    def _is_system_idle(self) -> bool:
        try:
            from rediris.common.database import SessionLocal
            db = SessionLocal()
            try:
                active_tasks = db.query(Task).filter(
                    Task.status.in_([
                        TaskStatus.ANNOUNCEMENT,
                        TaskStatus.EXECUTION,
                        TaskStatus.REVIEW,
                        TaskStatus.REWARD
                    ])
                ).count()

                is_idle = active_tasks == 0

                if is_idle:
                    logger.debug("System is in idle state (no active tasks)")

                return is_idle
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error checking system idle state: {e}", exc_info=True)
            return False

    async def _save_idle_score_for_uid(self):
        try:
            from rediris.common.database import SessionLocal
            db = SessionLocal()
            try:
                idle_uid = settings.IDLE_REWARD_UID
                miner_info = self.bittensor_client.get_miner_by_uid(idle_uid)

                if not miner_info:
                    logger.warning(f"No miner found for idle UID: {idle_uid}")
                    return

                miner_hotkey = miner_info.get("hotkey")
                if not miner_hotkey:
                    logger.warning(f"No hotkey found for UID: {idle_uid}")
                    return

                score = Score(
                    task_id=self.IDLE_WORKFLOW_ID,
                    miner_hotkey=miner_hotkey,
                    validator_hotkey=self.IDLE_VALIDATOR_HOTKEY,
                    cosine_similarity=self.IDLE_BASE_SCORE,
                    quality_score=self.IDLE_BASE_SCORE,
                    final_score=self.IDLE_BASE_SCORE,
                )
                db.add(score)
                db.commit()

                logger.info(f"Idle score saved: UID={idle_uid} hotkey={miner_hotkey[:16]}... score={self.IDLE_BASE_SCORE:.2f}")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to save idle score for UID: {e}", exc_info=True)

    def is_system_currently_idle(self) -> bool:
        return self._is_system_idle()

    def get_last_distribution_time(self) -> Optional[datetime]:
        """
        Return the latest time when an idle score distribution was recorded.

        Used by the /health endpoint to report whether idle rewards have been
        distributed recently.
        """
        try:
            from rediris.common.database import SessionLocal
            db = SessionLocal()
            try:
                # Lazy import to avoid circular deps at module import time.
                from rediris.common.models.score import Score

                latest = (
                    db.query(Score.created_at)
                    .filter(
                        Score.task_id == self.IDLE_WORKFLOW_ID,
                        Score.validator_hotkey == self.IDLE_VALIDATOR_HOTKEY,
                    )
                    .order_by(Score.created_at.desc())
                    .first()
                )
                if not latest:
                    return None
                return latest[0]
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to get last idle distribution time: {e}", exc_info=True)
            return None
