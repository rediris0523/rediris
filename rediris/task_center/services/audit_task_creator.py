from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional, Dict, Any
from rediris.common.models.audit_task import AuditTask
from rediris.common.models.task import Task
from rediris.common.models.validator import Validator
from rediris.common.utils.logging import setup_logger
from rediris.common.config import settings
from rediris.task_center import shared
from datetime import datetime, timezone
import uuid
import random

logger = setup_logger(__name__)

try:
    import bittensor as bt
    BITTENSOR_AVAILABLE = True
except ImportError:
    BITTENSOR_AVAILABLE = False
    logger.warning("Bittensor not available, will use fallback validators from config")


class AuditTaskCreator:

    def __init__(self, db: Session):
        self.db = db
        self.bittensor_client = shared.bittensor_client
        self.config = shared.yaml_config

    def _get_validators_with_vtrust(self) -> List[Dict[str, Any]]:

        validators = []

        if BITTENSOR_AVAILABLE and self.config:
            try:
                network = self.config.get('bittensor.network', 'finney')
                netuid = self.config.get('bittensor.netuid', settings.BITNETWORK_NETUID)

                logger.info(f"Fetching validators from bittensor network={network}, netuid={netuid}")

                subtensor = bt.Subtensor(network=network)
                metagraph = subtensor.metagraph(netuid)

                vtrust_list = metagraph.validator_trust
                hotkeys = metagraph.hotkeys
                stakes = metagraph.stake

                for i, vtrust in enumerate(vtrust_list):
                    if vtrust > 0:
                        validators.append({
                            "uid": i,
                            "hotkey": hotkeys[i],
                            "vtrust": float(vtrust),
                            "stake": float(stakes[i]) if i < len(stakes) else 0.0,
                            "is_active": True
                        })

                logger.info(f"Found {len(validators)} validators with vtrust > 0 from bittensor network")

            except Exception as e:
                logger.error(f"Failed to get validators from bittensor: {e}", exc_info=True)
                validators = []

        if not validators:
            validators = self._get_fallback_validators()

        return validators

    def _get_fallback_validators(self) -> List[Dict[str, Any]]:
        validators = []

        if self.config:
            fallback_list = self.config.get('task_center.fallback_validators', [])
            if fallback_list:
                logger.info(f"Using {len(fallback_list)} fallback validators from config")
                for v in fallback_list:
                    if isinstance(v, dict) and v.get('hotkey'):
                        validators.append({
                            "hotkey": v['hotkey'],
                            "vtrust": v.get('vtrust', 1.0),
                            "stake": v.get('stake', 0.0),
                            "is_active": True
                        })
                    elif isinstance(v, str):
                        validators.append({
                            "hotkey": v,
                            "vtrust": 1.0,
                            "stake": 0.0,
                            "is_active": True
                        })

        if not validators:
            logger.warning("No fallback validators configured in config.yml")

        return validators
    
    def create_audit_task(
        self,
        task_id: str,
        miner_hotkey: str,
        lora_url: str
    ) -> AuditTask:
        task = self.db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise ValueError("Task not found")

        audit_task_id = f"audit_{uuid.uuid4().hex[:12]}"

        task_info = {
            "prompt": task.workflow_spec.get("prompt", ""),
            "seed": task.workflow_spec.get("seed", 42),
            "base_model": task.workflow_spec.get("training_spec", {}).get("base_model", ""),
            "target_vector": task.workflow_spec.get("target_vector", [])
        }

        audit_task = AuditTask(
            audit_task_id=audit_task_id,
            original_task_id=task_id,
            miner_hotkey=miner_hotkey,
            audit_type="lora",
            lora_url=lora_url,
            task_info=task_info,
            is_completed=False
        )

        self.db.add(audit_task)
        self.db.commit()
        self.db.refresh(audit_task)

        logger.info(f"Audit task created: {audit_task_id} for miner {miner_hotkey}")

        self.auto_assign_audit_tasks(task_id)

        return audit_task

    async def create_dataset_audit_task(
        self,
        task_id: str,
        miner_hotkey: str,
        dataset_url: str,
        task_info: Dict[str, Any]
    ) -> AuditTask:
        audit_task_id = f"dataset_audit_{uuid.uuid4().hex[:12]}"

        validation_task_info = {
            "workflow_spec": task_info,
            "dataset_url": dataset_url,
            "validation_criteria": {
                "format_check": True,
                "quality_check": True,
                "safety_check": True,
                "relevance_check": True,
            }
        }

        audit_task = AuditTask(
            audit_task_id=audit_task_id,
            original_task_id=task_id,
            miner_hotkey=miner_hotkey,
            audit_type="dataset",
            dataset_url=dataset_url,
            task_info=validation_task_info,
            is_completed=False
        )

        self.db.add(audit_task)
        self.db.commit()
        self.db.refresh(audit_task)

        logger.info(f"Dataset audit task created: {audit_task_id} for miner {miner_hotkey[:16]}...")

        self.auto_assign_dataset_audit_task(audit_task.audit_task_id)

        return audit_task
    
    def assign_audit_task_to_validator(
        self,
        audit_task_id: str,
        validator_key: str
    ) -> Optional[AuditTask]:
        base_audit_task = self.db.query(AuditTask).filter(
            AuditTask.audit_task_id == audit_task_id
        ).first()
        
        if not base_audit_task:
            return None
        
        existing_assignment = self.db.query(AuditTask).filter(
            and_(
                AuditTask.original_task_id == base_audit_task.original_task_id,
                AuditTask.miner_hotkey == base_audit_task.miner_hotkey,
                AuditTask.validator_hotkey == validator_key
            )
        ).first()
        
        if existing_assignment:
            return None
        
        validator = self.db.query(Validator).filter(Validator.hotkey == validator_key).first()
        if not validator:
            stake = self.bittensor_client.get_validator_stake(validator_key)
            validator = Validator(hotkey=validator_key, stake=stake, reputation=0.0)
            self.db.add(validator)
            self.db.commit()
        
        new_audit_task = AuditTask(
            audit_task_id=f"{audit_task_id}_{validator_key}_{uuid.uuid4().hex[:8]}",
            original_task_id=base_audit_task.original_task_id,
            miner_hotkey=base_audit_task.miner_hotkey,
            validator_hotkey=validator_key,
            audit_type=base_audit_task.audit_type,
            lora_url=base_audit_task.lora_url,
            dataset_url=base_audit_task.dataset_url,
            task_info=base_audit_task.task_info,
            is_completed=False
        )
        
        self.db.add(new_audit_task)
        self.db.commit()
        self.db.refresh(new_audit_task)
        
        return new_audit_task
    
    def _get_eligible_validators(self) -> List[Dict[str, Any]]:
        all_validators_data = self._get_validators_with_vtrust()
        eligible_validators = []

        for v_data in all_validators_data:
            if not v_data.get("is_active", False):
                continue

            validator = self.db.query(Validator).filter(
                Validator.hotkey == v_data["hotkey"]
            ).first()

            if not validator:
                validator = Validator(
                    hotkey=v_data["hotkey"],
                    stake=v_data.get("stake", 0.0),
                    reputation=0.0
                )
                self.db.add(validator)
                self.db.commit()

            eligible_validators.append({
                "hotkey": v_data["hotkey"],
                "stake": v_data.get("stake", 0.0),
                "vtrust": v_data.get("vtrust", 0.0),
                "reputation": validator.reputation
            })

        return eligible_validators

    def auto_assign_dataset_audit_task(self, audit_task_id: str):
        audit_task = self.db.query(AuditTask).filter(
            AuditTask.audit_task_id == audit_task_id
        ).first()

        if not audit_task:
            return

        eligible_validators = self._get_eligible_validators()

        if not eligible_validators:
            logger.warning(f"No eligible validators found for dataset audit task {audit_task_id}")
            return

        selected_validator = random.choice(eligible_validators)

        assigned = self.assign_audit_task_to_validator(
            audit_task_id,
            selected_validator["hotkey"]
        )

        if assigned:
            logger.info(
                f"Assigned dataset audit task {audit_task_id} to validator "
                f"{selected_validator['hotkey'][:16]}..."
            )
        else:
            logger.warning(f"Failed to assign dataset audit task {audit_task_id}")

    def auto_assign_audit_tasks(self, task_id: str):
        audit_tasks = self.db.query(AuditTask).filter(
            and_(
                AuditTask.original_task_id == task_id,
                AuditTask.validator_hotkey.is_(None),
                AuditTask.audit_type == "lora"
            )
        ).all()

        if not audit_tasks:
            return

        eligible_validators = self._get_eligible_validators()

        if not eligible_validators:
            logger.warning(f"No eligible validators found for task {task_id}")
            return

        max_validators = settings.CONSENSUS_MAX_VALIDATORS
        min_validators = settings.CONSENSUS_MIN_VALIDATORS

        if len(eligible_validators) > max_validators:
            logger.info(
                f"Found {len(eligible_validators)} eligible validators, randomly selecting {max_validators} "
                f"for consensus (max={max_validators})"
            )
            eligible_validators = random.sample(eligible_validators, max_validators)

        if len(eligible_validators) < min_validators:
            logger.warning(
                f"Number of eligible validators ({len(eligible_validators)}) is below minimum ({min_validators}). "
                f"Consensus may not be reliable."
            )

        logger.info(f"Found {len(eligible_validators)} eligible validators for task {task_id}")

        total_assigned = 0
        for audit_task in audit_tasks:
            assigned_count = 0
            for validator in eligible_validators:
                # Check if we've already assigned max validators for this audit task
                existing_count = self.db.query(AuditTask).filter(
                    and_(
                        AuditTask.original_task_id == audit_task.original_task_id,
                        AuditTask.miner_hotkey == audit_task.miner_hotkey,
                        AuditTask.validator_hotkey.isnot(None)
                    )
                ).count()

                if existing_count >= max_validators:
                    break

                existing = self.db.query(AuditTask).filter(
                    and_(
                        AuditTask.original_task_id == audit_task.original_task_id,
                        AuditTask.miner_hotkey == audit_task.miner_hotkey,
                        AuditTask.validator_hotkey == validator["hotkey"]
                    )
                ).first()

                if existing:
                    continue

                assigned = self.assign_audit_task_to_validator(
                    audit_task.audit_task_id,
                    validator["hotkey"]
                )

                if assigned:
                    total_assigned += 1
                    assigned_count += 1
                    logger.info(
                        f"Assigned audit task {audit_task.audit_task_id} to validator "
                        f"{validator['hotkey'][:16]}... (vtrust={validator.get('vtrust', 0):.3f})"
                    )

                    # Stop if we've reached max validators for this audit task
                    if assigned_count >= max_validators:
                        break

        logger.info(
            f"Task {task_id}: assigned {total_assigned} audit tasks to "
            f"up to {max_validators} validators per audit task"
        )
    
    def update_audit_task_status(
        self,
        audit_task_id: str,
        status: str,
        result: Optional[Dict] = None
    ):
        audit_task = self.db.query(AuditTask).filter(
            AuditTask.audit_task_id == audit_task_id
        ).first()
        
        if not audit_task:
            logger.warning(f"Audit task not found: {audit_task_id}")
            return
        
        audit_task.is_completed = (status == "completed")
        if result:
            audit_task.result = result
        
        if status == "completed":
            audit_task.completed_at = datetime.now(timezone.utc)
        
        self.db.commit()
        logger.info(f"Audit task {audit_task_id} status updated to {status}")
    
    def get_audit_task_status(self, task_id: str) -> Dict[str, Any]:
        audit_tasks = self.db.query(AuditTask).filter(
            and_(
                AuditTask.original_task_id == task_id,
                AuditTask.validator_hotkey.isnot(None)
            )
        ).all()

        total = len(audit_tasks)
        completed = sum(1 for t in audit_tasks if t.is_completed)
        pending = total - completed

        return {
            "task_id": task_id,
            "total_audit_tasks": total,
            "completed": completed,
            "pending": pending,
            "completion_rate": completed / total if total > 0 else 0.0
        }
    
    def get_pending_tasks_for_validator(self, validator_key: str) -> List[AuditTask]:
        return self.db.query(AuditTask).filter(
            and_(
                AuditTask.validator_hotkey == validator_key,
                AuditTask.is_completed == False
            )
        ).all()

