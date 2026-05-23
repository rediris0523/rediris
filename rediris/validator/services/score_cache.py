from typing import Dict, List, Optional, Any
import time
import asyncio
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class ScoreCache:

    def __init__(self):
        self._cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    def cache_score(
        self,
        task_id: str,
        miner_hotkey: str,
        validator_hotkey: str,
        score: float,
        score_details: Dict[str, Any]
    ):

        if task_id not in self._cache:
            self._cache[task_id] = {}
        
        self._cache[task_id][miner_hotkey] = {
            "task_id": task_id,
            "miner_hotkey": miner_hotkey,
            "validator_hotkey": validator_hotkey,
            "score": score,
            **score_details,
            "timestamp": int(time.time())
        }
        
        logger.debug(
            f"Cached score for task {task_id}, miner {miner_hotkey[:20]}..., "
            f"score={score:.2f}"
        )
    
    def get_cached_scores_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        if task_id not in self._cache:
            return []
        return list(self._cache[task_id].values())
    
    def get_all_cached_scores(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            task_id: list(scores.values())
            for task_id, scores in self._cache.items()
        }
    
    def clear_scores(self, task_id: Optional[str] = None):

        if task_id:
            if task_id in self._cache:
                del self._cache[task_id]
                logger.info(f"Cleared cache for task {task_id}")
        else:
            self._cache.clear()
            logger.info("Cleared all cached scores")
    
    def has_cached_scores(self, task_id: str) -> bool:
        return task_id in self._cache and len(self._cache[task_id]) > 0
