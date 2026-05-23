from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from rediris.common.database import get_db
from rediris.task_center.services.score_archive import ScoreArchive
from rediris.task_center.schemas.score import ScoreSubmit, ScoreQueryResponse
from rediris.task_center import shared
from rediris.common.utils.logging import setup_logger
from rediris.common.auth.signature_auth import verify_node_signature

router = APIRouter()
logger = setup_logger(__name__)

@router.post("/submit")
async def submit_score(
    score_data: ScoreSubmit,
    hotkey: str = Depends(verify_node_signature),
    db: Session = Depends(get_db)
):
    if hotkey != score_data.validator_hotkey:
        raise HTTPException(status_code=403, detail="Hotkey mismatch")
    
    from rediris.task_center.services.task_lifecycle_manager import TaskLifecycleManager
    lifecycle_manager = TaskLifecycleManager(db)
    can_score, reason = lifecycle_manager.can_validator_score(score_data.task_id)

    if not can_score:
        logger.warning(f"Score submission rejected for task {score_data.task_id}: {reason}")
        raise HTTPException(status_code=400, detail=f"Scoring not allowed: {reason}")

    archive = ScoreArchive(db, wallet=shared.wallet, wallet_name=shared.wallet_name, hotkey_name=shared.hotkey_name, yaml_config=shared.yaml_config)
    archive.submit_score(score_data)
    return {"status": "success", "message": "Score submitted successfully"}


@router.get("/query/{miner_hotkey}", response_model=ScoreQueryResponse)
async def query_score(
    miner_hotkey: str,
    hotkey: str = Depends(verify_node_signature),
    task_id: str = None,
    request: Request = None,
    db: Session = Depends(get_db)
):
    archive = ScoreArchive(db, wallet=shared.wallet, wallet_name=shared.wallet_name, hotkey_name=shared.hotkey_name, yaml_config=shared.yaml_config)
    scores = archive.get_miner_scores(miner_hotkey, task_id)

    if not scores:
        raise HTTPException(status_code=404, detail="No scores found")

    return ScoreQueryResponse(
        miner_hotkey=miner_hotkey,
        scores=scores,
        ema_score=archive.calculate_ema_score(miner_hotkey, task_id)
    )


@router.get("/all")
async def get_all_scores(
    task_id: str,
    hotkey: str = Depends(verify_node_signature),
    request: Request = None,
    db: Session = Depends(get_db)
):
    archive = ScoreArchive(db, wallet=shared.wallet, wallet_name=shared.wallet_name, hotkey_name=shared.hotkey_name, yaml_config=shared.yaml_config)

    scores = archive.get_all_scores_for_task(task_id)

    return {
        "task_id": task_id,
        "scores": scores
    }


@router.get("/query")
async def query_all_scores(
    task_id: str = None,
    hotkey: str = Depends(verify_node_signature),
    request: Request = None,
    db: Session = Depends(get_db)
):
    archive = ScoreArchive(db, wallet=shared.wallet, wallet_name=shared.wallet_name, hotkey_name=shared.hotkey_name, yaml_config=shared.yaml_config)

    if task_id:
        scores = archive.get_all_scores_for_task(task_id)

        return {
            "task_id": task_id,
            "miner_scores": {
                item["miner_hotkey"]: {
                    "consensus_score": item["consensus_score"],
                    "ema_score": item["ema_score"],
                    "validator_count": item["validator_count"]
                }
                for item in scores
            }
        }
    else:
        from rediris.common.models.score import Score
        from rediris.common.bittensor.client import BittensorClient

        wallet_name = shared.wallet_name if hasattr(shared, 'wallet_name') and shared.wallet_name else "task_center"
        hotkey_name = shared.hotkey_name if hasattr(shared, 'hotkey_name') and shared.hotkey_name else "default"
        bittensor_client = BittensorClient(wallet_name, hotkey_name, yaml_config=shared.yaml_config)

        registered_miners = bittensor_client.get_all_miners()
        registered_hotkeys = {m.get("hotkey") for m in registered_miners if m.get("hotkey")}

        logger.info(f"Found {len(registered_hotkeys)} registered miners on subnet")

        miners = db.query(Score.miner_hotkey).distinct().all()

        all_miner_scores = {}
        excluded_count = 0
        for (miner_hotkey,) in miners:
            if miner_hotkey not in registered_hotkeys:
                excluded_count += 1
                continue

            ema_score = archive.calculate_ema_score(miner_hotkey)
            history = archive.get_miner_history_scores(miner_hotkey, limit=100)

            all_miner_scores[miner_hotkey] = {
                "ema_score": ema_score,
                "history_count": len(history),
                "latest_score": history[0]["final_score"] if history else 0.0
            }

        logger.info(f"Returning scores for {len(all_miner_scores)} registered miners, excluded {excluded_count} deregistered miners")

        return {
            "all_miners": all_miner_scores,
            "total_miners": len(all_miner_scores)
        }

