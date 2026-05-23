from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from rediris.common.models.task import TaskStatus, Task

class WorkflowSpec(BaseModel):
    theme: str
    target_platform: str
    deployment_target: str
    training_mode: str
    dataset_spec: Dict[str, Any]
    training_spec: Dict[str, Any]
    test_spec: Optional[Dict[str, Any]] = None
    base_lora_url: Optional[str] = None
    base_lora_filename: Optional[str] = None
    prompt: Optional[str] = None
    seed: Optional[int] = 42
    target_vector: Optional[list] = None


class RewardConfig(BaseModel):
    pool_ratio: float = 1.0
    pool_name: Optional[str] = None
    min_score_threshold: float = 3.5
    quality_exponent: int = 2


class TaskCreate(BaseModel):
    task_id: str
    workflow_type: str
    workflow_spec: WorkflowSpec
    publish_status: Optional[str] = "draft"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: Optional[str] = None
    description_ja: Optional[str] = None
    hf_dataset_url: Optional[str] = None
    pdf_file_url: Optional[str] = None
    announcement_duration: float
    dataset_validation_duration: float = 0.0
    execution_duration: float
    review_duration: float
    reward_duration: float
    reward_config: Optional[RewardConfig] = None
    default_reward_miners: Optional[int] = None


class TaskResponse(BaseModel):
    task_id: str
    workflow_type: str
    status: TaskStatus
    display_status: Optional[str] = None
    publish_status: Optional[str] = "draft"
    workflow_spec: Optional[Dict[str, Any]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: Optional[str] = None
    description_ja: Optional[str] = None
    hf_dataset_url: Optional[str] = None
    pdf_file_url: Optional[str] = None
    announcement_start: Optional[datetime] = None
    dataset_validation_start: Optional[datetime] = None
    execution_start: Optional[datetime] = None
    review_start: Optional[datetime] = None
    reward_start: Optional[datetime] = None
    workflow_end: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_orm(cls, task):
        display_status = None
        if hasattr(task, 'status'):
            if task.status == TaskStatus.PENDING or task.status == TaskStatus.NOT_STARTED:
                display_status = TaskStatus.NOT_STARTED.value
            elif task.status in [TaskStatus.ANNOUNCEMENT, TaskStatus.EXECUTION, TaskStatus.REVIEW, TaskStatus.REWARD, TaskStatus.IN_PROGRESS]:
                display_status = TaskStatus.IN_PROGRESS.value
            elif task.status in [TaskStatus.ENDED, TaskStatus.COMPLETED]:
                display_status = TaskStatus.COMPLETED.value
        
        return cls(
            task_id=task.task_id,
            workflow_type=task.workflow_type,
            status=task.status,
            display_status=display_status,
            publish_status=getattr(task, 'publish_status', 'draft'),
            workflow_spec=getattr(task, 'workflow_spec', None),
            start_date=getattr(task, 'start_date', None),
            end_date=getattr(task, 'end_date', None),
            description=getattr(task, 'description', None),
            description_ja=getattr(task, 'description_ja', None),
            hf_dataset_url=getattr(task, 'hf_dataset_url', None),
            pdf_file_url=getattr(task, 'pdf_file_url', None),
            announcement_start=task.announcement_start,
            dataset_validation_start=getattr(task, 'dataset_validation_start', None),
            execution_start=task.execution_start,
            review_start=task.review_start,
            reward_start=task.reward_start,
            workflow_end=task.workflow_end,
            created_at=getattr(task, 'created_at', None)
        )

class TaskListResponse(BaseModel):
    workflows: list[TaskResponse]
    pagination: Dict[str, Any]

