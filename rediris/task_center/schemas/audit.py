from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from rediris.common.models.audit_task import AuditTask

class AuditTaskResponse(BaseModel):
    audit_task_id: str
    original_task_id: str
    miner_hotkey: str
    validator_hotkey: Optional[str] = None
    audit_type: str = "lora"
    lora_url: Optional[str] = None
    dataset_url: Optional[str] = None
    task_info: Dict[str, Any]
    is_completed: bool
    created_at: datetime

    @classmethod
    def from_orm(cls, audit_task):
        return cls(
            audit_task_id=audit_task.audit_task_id,
            original_task_id=audit_task.original_task_id,
            miner_hotkey=audit_task.miner_hotkey,
            validator_hotkey=audit_task.validator_hotkey,
            audit_type=audit_task.audit_type or "lora",
            lora_url=audit_task.lora_url,
            dataset_url=audit_task.dataset_url,
            task_info=audit_task.task_info,
            is_completed=audit_task.is_completed,
            created_at=audit_task.created_at
        )


class AuditTaskListResponse(BaseModel):
    tasks: list[AuditTaskResponse]

