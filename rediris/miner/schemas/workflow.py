from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime

class WorkflowReceive(BaseModel):
    task_id: str
    miner_key: str
    workflow_type: Optional[str] = None
    workflow_spec: Optional[Dict[str, Any]] = None
    dataset_url: Optional[str] = None
    announcement_start: Optional[str] = None
    dataset_validation_start: Optional[str] = None
    execution_start: Optional[str] = None
    review_start: Optional[str] = None
    reward_start: Optional[str] = None
    workflow_end: Optional[str] = None


class WorkflowReceiveResponse(BaseModel):
    task_id: str
    workflow_type: str
    workflow_spec: Dict[str, Any]
    deadline: datetime
    review_deadline: datetime


class WorkflowSubmit(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task_id: str
    miner_key: str
    training_mode: str
    model_url: str
    model_metadata: Dict[str, Any]
    sample_images: Optional[list[str]] = None


class WorkflowSubmitResponse(BaseModel):
    submission_id: str
    task_id: str
    status: str
    estimated_reward: float

