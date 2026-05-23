from pydantic import BaseModel
from typing import Dict, Any

class AuditTaskRequest(BaseModel):
    audit_task_id: str
    miner_hotkey: str
    lora_url: str
    task_info: Dict[str, Any]


class AuditTaskResponse(BaseModel):
    audit_task_id: str
    miner_hotkey: str
    cosine_similarity: float
    quality_score: float
    final_score: float

