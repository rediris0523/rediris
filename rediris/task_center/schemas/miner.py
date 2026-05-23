from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class MinerTaskReceive(BaseModel):
    task_id: str
    miner_key: str

class MinerTaskResponse(BaseModel):
    task_id: str
    workflow_type: str
    workflow_spec: Dict[str, Any]
    deadline: datetime
    review_deadline: datetime

    @classmethod
    def from_task(cls, task):
        if task.workflow_type is None:
            raise ValueError("Task workflow_type is None")

        if isinstance(task.workflow_type, Enum):
            workflow_type_value = task.workflow_type.value
        elif hasattr(task.workflow_type, 'value'):
            workflow_type_value = task.workflow_type.value
        else:
            workflow_type_value = str(task.workflow_type)

        return cls(
            task_id=task.task_id,
            workflow_type=workflow_type_value,
            workflow_spec=task.workflow_spec,
            deadline=task.execution_start,
            review_deadline=task.review_start
        )


class MinerSubmitRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task_id: str
    miner_key: str
    training_mode: str
    model_url: str
    model_metadata: Dict[str, Any]
    sample_images: Optional[list[str]] = None


class MinerSubmitResponse(BaseModel):
    submission_id: str
    task_id: str
    status: str
    estimated_reward: float


class AvailableTaskItem(BaseModel):
    task_id: str
    workflow_type: str
    status: str
    workflow_spec: Dict[str, Any]
    announcement_start: Optional[datetime] = None
    dataset_validation_start: Optional[datetime] = None
    execution_start: Optional[datetime] = None
    dataset_submitted: bool = False
    dataset_status: Optional[str] = None


class AvailableTasksResponse(BaseModel):
    tasks: List[AvailableTaskItem]
    total: int
