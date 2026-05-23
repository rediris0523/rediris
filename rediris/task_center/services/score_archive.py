from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Optional
import bittensor as bt
from rediris.common.models.score import Score
from rediris.common.config.yaml_config import YamlConfig
from rediris.task_center.schemas.score import ScoreSubmit
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class ScoreArchive:
    def __init__(self, db: Session, wallet: Optional[bt.wallet] = None, wallet_name: Optional[str] = None, hotkey_name: Optional[str] = None, yaml_config: Optional[YamlConfig] = None):
        self.db = db
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.yaml_config = yaml_config
    
    def submit_score(self, score_data: ScoreSubmit):

        score = Score(
            task_id=score_data.task_id,
            miner_hotkey=score_data.miner_hotkey,
            validator_hotkey=score_data.validator_hotkey,
            cosine_similarity=score_data.cosine_similarity,
            quality_score=score_data.quality_score,
            final_score=score_data.final_score
        )
        
        self.db.add(score)
        self.db.commit()
        
        logger.info(f"Score submitted (for record only): miner={score_data.miner_hotkey}, score={score_data.final_score}")
        
    def get_miner_scores(
        self,
        miner_hotkey: str,
        task_id: Optional[str] = None
    ) -> List[Dict]:
        query = self.db.query(Score).filter(Score.miner_hotkey == miner_hotkey)

        if task_id:
            query = query.filter(Score.task_id == task_id)
        
        scores = query.all()
        
        return [
            {
                "task_id": s.task_id,
                "validator_hotkey": s.validator_hotkey,
                "cosine_similarity": s.cosine_similarity,
                "quality_score": s.quality_score,
                "final_score": s.final_score,
                "created_at": s.created_at.isoformat()
            }
            for s in scores
        ]
    
    def _calculate_consensus_score_for_workflow(
        self,
        scores: List[Dict]
    ) -> float:

        if not scores:
            return 0.0

        final_scores = [s["final_score"] for s in scores]

        if len(final_scores) >= 3:
            sorted_scores = sorted(final_scores)
            filtered_scores = sorted_scores[1:-1]
            return sum(filtered_scores) / len(filtered_scores)
        else:
            return sum(final_scores) / len(final_scores)

    def calculate_ema_score(
        self,
        miner_hotkey: str,
        task_id: Optional[str] = None,
        alpha: float = 0.4
    ) -> float:

        if task_id:
            scores = self.get_miner_scores(miner_hotkey, task_id)
            return self._calculate_consensus_score_for_workflow(scores)

        all_scores = self.get_miner_scores(miner_hotkey)

        if not all_scores:
            return 0.0

        task_scores: Dict[str, List[Dict]] = {}
        task_times: Dict[str, str] = {}

        for score in all_scores:
            t_id = score["task_id"]
            if t_id not in task_scores:
                task_scores[t_id] = []
                task_times[t_id] = score["created_at"]
            task_scores[t_id].append(score)
            if score["created_at"] < task_times[t_id]:
                task_times[t_id] = score["created_at"]

        consensus_scores: List[tuple] = []
        for t_id, scores in task_scores.items():
            consensus = self._calculate_consensus_score_for_workflow(scores)
            consensus_scores.append((task_times[t_id], consensus))

        consensus_scores.sort(key=lambda x: x[0])

        if not consensus_scores:
            return 0.0

        ema_score = consensus_scores[0][1]
        for _, score in consensus_scores[1:]:
            ema_score = alpha * score + (1 - alpha) * ema_score

        logger.debug(
            f"EMA score for {miner_hotkey}: {ema_score:.4f} "
            f"(from {len(consensus_scores)} tasks, {len(all_scores)} total scores)"
        )

        return ema_score
    
    def get_all_scores_for_task(self, task_id: str) -> List[Dict]:

        scores = self.db.query(Score).filter(Score.task_id == task_id).all()

        miner_scores = {}
        for score in scores:
            if score.miner_hotkey not in miner_scores:
                miner_scores[score.miner_hotkey] = []

            miner_scores[score.miner_hotkey].append({
                "validator_hotkey": score.validator_hotkey,
                "cosine_similarity": score.cosine_similarity,
                "quality_score": score.quality_score,
                "final_score": score.final_score,
                "created_at": score.created_at.isoformat()
            })

        result = []
        for miner_hotkey, score_list in miner_scores.items():
            consensus_score = self._calculate_consensus_score_for_workflow(score_list)

            result.append({
                "miner_hotkey": miner_hotkey,
                "scores": score_list,
                "consensus_score": consensus_score,
                "ema_score": self.calculate_ema_score(miner_hotkey),
                "validator_count": len(score_list)
            })

        return result
    
    def get_miner_history_scores(self, miner_hotkey: str, limit: int = 100) -> List[Dict]:
        scores = self.db.query(Score).filter(
            Score.miner_hotkey == miner_hotkey
        ).order_by(Score.created_at.desc()).limit(limit).all()
        
        return [
            {
                "task_id": s.task_id,
                "validator_hotkey": s.validator_hotkey,
                "final_score": s.final_score,
                "created_at": s.created_at.isoformat()
            }
            for s in scores
        ]
