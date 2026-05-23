from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any


class TestCase(BaseModel):
    prompt: str
    seed: int
    inference_steps: int = 30
    guidance_scale: float = 7.0


class InferenceTestRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_url: str
    test_cases: List[TestCase]


class InferenceTestResponse(BaseModel):
    results: List[Dict[str, Any]]

