from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from rediris.validator.services.score_calculator import ScoreCalculator
from rediris.validator.schemas.score import ScoreSubmitRequest, ScoreSubmitResponse
from rediris.common.utils.logging import setup_logger
import httpx
from rediris.common.config import settings
import random

router = APIRouter()
logger = setup_logger(__name__)
score_calculator = ScoreCalculator()

class CacheScoresRequest(BaseModel):
    task_id: str
    seed: int | None = None


@router.post("/submit", response_model=ScoreSubmitResponse)
async def submit_score(request: ScoreSubmitRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.TASK_CENTER_URL}/v1/scores/submit",
                json=request.dict()
            )
            response.raise_for_status()
            return ScoreSubmitResponse(**response.json())
    except httpx.HTTPError as e:
        logger.error(f"Failed to submit score: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit score")


@router.get("/query")
async def query_scores(task_id: str, miner_hotkeys: str = None):
    try:
        async with httpx.AsyncClient() as client:
            params = {"task_id": task_id}
            if miner_hotkeys:
                params["miner_hotkeys"] = miner_hotkeys

            response = await client.get(
                f"{settings.TASK_CENTER_URL}/v1/scores/all",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to query scores: {e}")
        raise HTTPException(status_code=500, detail="Failed to query scores")

