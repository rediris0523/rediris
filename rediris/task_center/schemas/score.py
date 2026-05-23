from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ScoreSubmit(BaseModel):
    task_id: str
    miner_hotkey: str
    validator_hotkey: str
    cosine_similarity: Optional[float]
    quality_score: Optional[float]
    final_score: float

class ScoreQueryResponse(BaseModel):
    miner_hotkey: str
    scores: List[Dict[str, Any]]
    ema_score: float

