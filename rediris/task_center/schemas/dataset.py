from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class DatasetSubmitRequest(BaseModel):
    task_id: str
    miner_hotkey: str
    dataset_url: str
    dataset_description: Optional[str] = None


class DatasetSubmitResponse(BaseModel):
    submission_id: int
    task_id: str
    miner_hotkey: str
    status: str
    message: str


class DatasetValidationRequest(BaseModel):
    audit_task_id: str
    validator_hotkey: str
    is_approved: bool
    validation_result: Dict[str, Any]
    rejection_reason: Optional[str] = None


class DatasetValidationResponse(BaseModel):
    audit_task_id: str
    status: str
    message: str


class DatasetStatusResponse(BaseModel):
    task_id: str
    miner_hotkey: str
    dataset_url: str
    validation_status: str
    validated_by: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    created_at: datetime
    validated_at: Optional[datetime] = None


class DatasetListResponse(BaseModel):
    datasets: List[DatasetStatusResponse]
    total: int
