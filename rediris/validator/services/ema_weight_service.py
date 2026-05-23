
from typing import Dict, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from rediris.common.models.miner_ema_weight import MinerEmaWeight
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)


class EmaWeightService:

    DEFAULT_ALPHA = 0.4

    INACTIVE_THRESHOLD_DAYS = 7

    def __init__(self, db: Session, validator_hotkey: str, alpha: float = None):

        self.db = db
        self.validator_hotkey = validator_hotkey
        self.alpha = alpha or self.DEFAULT_ALPHA

    def get_ema_weights(self) -> Dict[str, float]:

        try:
            records = self.db.query(MinerEmaWeight).filter(
                MinerEmaWeight.validator_hotkey == self.validator_hotkey
            ).all()

            return {record.miner_hotkey: record.ema_weight for record in records}

        except Exception as e:
            logger.error(f"Error getting EMA weights: {e}", exc_info=True)
            return {}

    def get_ema_weight(self, miner_hotkey: str) -> Optional[float]:

        try:
            record = self.db.query(MinerEmaWeight).filter(
                MinerEmaWeight.validator_hotkey == self.validator_hotkey,
                MinerEmaWeight.miner_hotkey == miner_hotkey
            ).first()

            return record.ema_weight if record else None

        except Exception as e:
            logger.error(f"Error getting EMA weight for {miner_hotkey}: {e}", exc_info=True)
            return None

    def update_ema_weights(
        self,
        current_weights: Dict[str, float],
        miner_uids: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:

        if not current_weights:
            return {}

        new_ema_weights = {}
        now = datetime.now(timezone.utc)
        inactive_threshold = now - timedelta(days=self.INACTIVE_THRESHOLD_DAYS)

        try:
            for miner_hotkey, current_weight in current_weights.items():
                record = self.db.query(MinerEmaWeight).filter(
                    MinerEmaWeight.validator_hotkey == self.validator_hotkey,
                    MinerEmaWeight.miner_hotkey == miner_hotkey
                ).first()

                if record:
                    if record.updated_at and record.updated_at < inactive_threshold:
                        new_ema = current_weight
                        logger.debug(f"Reset EMA for inactive miner {miner_hotkey[:16]}...")
                    else:
                        old_ema = record.ema_weight
                        new_ema = self.alpha * current_weight + (1 - self.alpha) * old_ema

                    record.ema_weight = new_ema
                    record.last_raw_weight = current_weight
                    record.update_count += 1
                    record.updated_at = now

                    if miner_uids and miner_hotkey in miner_uids:
                        record.miner_uid = miner_uids[miner_hotkey]

                else:
                    new_ema = current_weight

                    record = MinerEmaWeight(
                        validator_hotkey=self.validator_hotkey,
                        miner_hotkey=miner_hotkey,
                        miner_uid=miner_uids.get(miner_hotkey) if miner_uids else None,
                        ema_weight=new_ema,
                        last_raw_weight=current_weight,
                        update_count=1,
                        created_at=now,
                        updated_at=now
                    )
                    self.db.add(record)
                    logger.debug(f"Created new EMA record for miner {miner_hotkey[:16]}...")

                new_ema_weights[miner_hotkey] = new_ema

            existing_records = self.db.query(MinerEmaWeight).filter(
                MinerEmaWeight.validator_hotkey == self.validator_hotkey,
                MinerEmaWeight.miner_hotkey.notin_(list(current_weights.keys()))
            ).all()

            for record in existing_records:
                old_ema = record.ema_weight
                new_ema = (1 - self.alpha) * old_ema

                if new_ema > 0.0001:
                    record.ema_weight = new_ema
                    record.last_raw_weight = 0.0
                    record.update_count += 1
                    record.updated_at = now
                    new_ema_weights[record.miner_hotkey] = new_ema
                    logger.debug(f"Decayed EMA for inactive miner {record.miner_hotkey[:16]}... ({old_ema:.6f} -> {new_ema:.6f})")

            self.db.commit()

            logger.info(f"Updated EMA weights for {len(new_ema_weights)} miners (alpha={self.alpha})")
            return new_ema_weights

        except Exception as e:
            logger.error(f"Error updating EMA weights: {e}", exc_info=True)
            self.db.rollback()
            return current_weights  # Fallback to current weights

    def apply_ema_smoothing(
        self,
        current_weights: Dict[str, float],
        miner_uids: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:

        return self.update_ema_weights(current_weights, miner_uids)

    def cleanup_old_records(self, days: int = 30):

        try:
            threshold = datetime.now(timezone.utc) - timedelta(days=days)

            deleted = self.db.query(MinerEmaWeight).filter(
                MinerEmaWeight.validator_hotkey == self.validator_hotkey,
                MinerEmaWeight.updated_at < threshold
            ).delete()

            self.db.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old EMA records")

        except Exception as e:
            logger.error(f"Error cleaning up old EMA records: {e}", exc_info=True)
            self.db.rollback()
