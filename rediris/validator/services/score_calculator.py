
import math
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from rediris.common.utils.logging import setup_logger
from rediris.common.config import settings

logger = setup_logger(__name__)

DEFAULT_BASE_PRICE = 0.01
DEFAULT_PRICE_SENSITIVITY = 0.5

class ScoreCalculator:

    def __init__(self):
        self.quality_exponent = 3
        self.base_threshold = 3.5

        self.optimal_window_hours = 24
        self.decay_rate = 0.1

        self.max_lora_size_mb = 100
        self.max_inference_time_sec = 60

    def calculate_score(self, audit_result: Dict[str, Any]) -> float:

        if audit_result.get("rejected", False):
            return 0.0

        cosine_similarity = audit_result.get("cosine_similarity", 0.0)
        quality_score = audit_result.get("quality_score", 0.0)

        time_coefficient = self._calculate_time_coefficient(audit_result)

        constraint_coefficient = self._calculate_constraint_coefficient(audit_result)

        return self.calculate_final_score(
            cosine_similarity=cosine_similarity,
            quality_score=quality_score,
            time_coefficient=time_coefficient,
            constraint_coefficient=constraint_coefficient
        )

    def calculate_final_score(
        self,
        cosine_similarity: float,
        quality_score: float,
        time_coefficient: float = 1.0,
        constraint_coefficient: float = 1.0,
        k: int = None
    ) -> float:

        if k is None:
            k = self.quality_exponent

        base_score = cosine_similarity * 10.0

        if base_score < self.base_threshold:
            logger.debug(f"Score below threshold: {base_score:.2f} < {self.base_threshold}")
            return 0.0


        combined_score = base_score * 0.7 + quality_score * 0.3

        normalized_score = combined_score / 10.0
        quality_weight = normalized_score ** k

        final_weight = quality_weight * time_coefficient * constraint_coefficient

        final_score = final_weight * 10.0

        logger.debug(f"Score calculation: cosine={cosine_similarity:.4f}, quality={quality_score:.2f}, "
                    f"time_coef={time_coefficient:.4f}, constraint_coef={constraint_coefficient:.4f}, "
                    f"final={final_score:.2f}")

        return max(0.0, min(10.0, final_score))

    def _calculate_time_coefficient(self, audit_result: Dict[str, Any]) -> float:

        submit_time = audit_result.get("submit_time")
        task_start_time = audit_result.get("task_start_time")
        task_end_time = audit_result.get("task_end_time")

        if not submit_time:
            return 1.0

        try:
            if isinstance(submit_time, str):
                submit_dt = datetime.fromisoformat(submit_time.replace("Z", "+00:00"))
            elif isinstance(submit_time, (int, float)):
                submit_dt = datetime.fromtimestamp(submit_time, tz=timezone.utc)
            else:
                submit_dt = submit_time

            if task_start_time:
                if isinstance(task_start_time, str):
                    start_dt = datetime.fromisoformat(task_start_time.replace("Z", "+00:00"))
                elif isinstance(task_start_time, (int, float)):
                    start_dt = datetime.fromtimestamp(task_start_time, tz=timezone.utc)
                else:
                    start_dt = task_start_time
            else:
                return 1.0

            delay_hours = (submit_dt - start_dt).total_seconds() / 3600

            if delay_hours <= self.optimal_window_hours:
                return 1.0
            else:
                excess_hours = delay_hours - self.optimal_window_hours
                decay = 1.0 - (self.decay_rate * excess_hours / 24)
                return max(0.5, min(1.0, decay))

        except Exception as e:
            logger.warning(f"Failed to calculate time coefficient: {e}")
            return 1.0

    def _calculate_constraint_coefficient(self, audit_result: Dict[str, Any]) -> float:

        coefficient = 1.0

        lora_size_mb = audit_result.get("lora_size_mb", 0)
        if lora_size_mb > self.max_lora_size_mb:
            size_penalty = min(0.3, (lora_size_mb - self.max_lora_size_mb) / self.max_lora_size_mb * 0.3)
            coefficient -= size_penalty
            logger.debug(f"LoRA size penalty: {size_penalty:.4f} (size={lora_size_mb}MB)")

        inference_time_sec = audit_result.get("inference_time_sec", 0)
        if inference_time_sec > self.max_inference_time_sec:
            time_penalty = min(0.2, (inference_time_sec - self.max_inference_time_sec) / self.max_inference_time_sec * 0.2)
            coefficient -= time_penalty
            logger.debug(f"Inference time penalty: {time_penalty:.4f} (time={inference_time_sec}s)")

        return max(0.0, min(1.0, coefficient))

    def calculate_weight_from_scores(self, scores: Dict[str, float]) -> Dict[str, float]:

        total_score = sum(scores.values())

        if total_score <= 0:
            logger.warning("Total score is zero, returning zero weights")
            return {hotkey: 0.0 for hotkey in scores}

        weights = {}
        for hotkey, score in scores.items():
            if score >= self.base_threshold:
                weights[hotkey] = score / total_score
            else:
                weights[hotkey] = 0.0

        return weights

    def calculate_quality_index(
        self,
        score: float,
        quality_exponent: int = 2,
        max_score: float = 10.0
    ) -> float:

        if score <= 0:
            return 0.0

        normalized = min(score / max_score, 1.0)
        return normalized ** quality_exponent

    def calculate_quality_weighted_scores(
        self,
        scores: Dict[str, float],
        min_threshold: float = 3.5,
        quality_exponent: int = 2
    ) -> Dict[str, float]:

        weighted_indices = {}

        for hotkey, score in scores.items():
            if score >= min_threshold:
                weighted_indices[hotkey] = self.calculate_quality_index(
                    score,
                    quality_exponent=quality_exponent
                )
            else:
                weighted_indices[hotkey] = 0.0

        return weighted_indices

    def normalize_pool_weights(
        self,
        weighted_indices: Dict[str, float],
        pool_ratio: float
    ) -> Dict[str, float]:

        if not weighted_indices:
            return {}

        pool_total = sum(weighted_indices.values())
        if pool_total <= 0:
            return {}

        normalized = {}
        for hotkey, index in weighted_indices.items():
            if index > 0:
                normalized[hotkey] = (index / pool_total) * pool_ratio

        return normalized

    def calculate_price_multiplier(
        self,
        alpha_price: Optional[float],
        base_price: float = DEFAULT_BASE_PRICE,
        sensitivity: float = DEFAULT_PRICE_SENSITIVITY
    ) -> float:

        if alpha_price is None or alpha_price <= 0:
            logger.debug("Alpha price not available, using multiplier=1.0")
            return 1.0

        if base_price <= 0:
            base_price = DEFAULT_BASE_PRICE

        price_factor = alpha_price / base_price

        try:
            log_factor = math.log(price_factor)
            multiplier = 1.0 + (log_factor * sensitivity)

            multiplier = max(0.5, min(2.0, multiplier))

            logger.debug(f"Price multiplier: alpha={alpha_price}, base={base_price}, factor={price_factor:.4f}, multiplier={multiplier:.4f}")
            return multiplier

        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating price multiplier: {e}")
            return 1.0

    def calculate_price_weighted_quality_index(
        self,
        score: float,
        alpha_price: Optional[float],
        quality_exponent: int = 2,
        base_price: float = DEFAULT_BASE_PRICE,
        sensitivity: float = DEFAULT_PRICE_SENSITIVITY,
        max_score: float = 10.0
    ) -> float:

        quality_index = self.calculate_quality_index(
            score,
            quality_exponent=quality_exponent,
            max_score=max_score
        )

        price_multiplier = self.calculate_price_multiplier(
            alpha_price,
            base_price=base_price,
            sensitivity=sensitivity
        )

        return quality_index * price_multiplier

    def calculate_price_weighted_scores(
        self,
        scores: Dict[str, float],
        alpha_price: Optional[float],
        min_threshold: float = 3.5,
        quality_exponent: int = 2,
        base_price: float = DEFAULT_BASE_PRICE,
        sensitivity: float = DEFAULT_PRICE_SENSITIVITY
    ) -> Dict[str, float]:

        weighted_indices = {}

        price_multiplier = self.calculate_price_multiplier(
            alpha_price,
            base_price=base_price,
            sensitivity=sensitivity
        )

        for hotkey, score in scores.items():
            if score >= min_threshold:
                quality_index = self.calculate_quality_index(
                    score,
                    quality_exponent=quality_exponent
                )
                weighted_indices[hotkey] = quality_index * price_multiplier
            else:
                weighted_indices[hotkey] = 0.0

        return weighted_indices

    def apply_consensus(
        self,
        validator_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:

        if not validator_scores:
            return {}

        max_validators = settings.CONSENSUS_MAX_VALIDATORS
        min_validators = settings.CONSENSUS_MIN_VALIDATORS
        
        if len(validator_scores) > max_validators:
            logger.warning(
                f"Number of validators ({len(validator_scores)}) exceeds maximum ({max_validators}). "
                f"Limiting to {max_validators} validators."
            )
            validator_scores = dict(list(validator_scores.items())[:max_validators])
        
        if len(validator_scores) < min_validators:
            logger.warning(
                f"Number of validators ({len(validator_scores)}) is below minimum ({min_validators}). "
                f"Consensus may not be reliable."
            )

        miner_scores: Dict[str, list] = {}
        for validator_key, scores in validator_scores.items():
            for miner_key, score in scores.items():
                if miner_key not in miner_scores:
                    miner_scores[miner_key] = []
                miner_scores[miner_key].append(score)

        consensus_scores = {}
        for miner_key, scores in miner_scores.items():
            if len(scores) > max_validators:
                sorted_scores = sorted(scores)
                filtered_scores = sorted_scores[:max_validators]
            else:
                filtered_scores = scores

            if filtered_scores:
                consensus_scores[miner_key] = sum(filtered_scores) / len(filtered_scores)
            else:
                consensus_scores[miner_key] = 0.0

        logger.info(f"Consensus calculated for {len(consensus_scores)} miners from {len(validator_scores)} validators")
        return consensus_scores
