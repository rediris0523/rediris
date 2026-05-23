from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Union
from datetime import datetime, timezone
from rediris.common.models.miner import Miner
from rediris.common.models.score import Score
from rediris.common.models.task_assignment import TaskAssignment
from rediris.common.models.account import Account
from rediris.common.models.miner_dataset import MinerDataset
from rediris.common.utils.logging import setup_logger
from rediris.task_center.services.miner_cache import MinerCache
from rediris.task_center import shared
import random
import httpx
from rediris.common.config import settings

logger = setup_logger(__name__)

class MinerSelector:
    def __init__(self, db: Session, miner_cache: MinerCache):
        self.db = db
        self.bittensor_client = shared.bittensor_client
        self.miner_cache = miner_cache
    
    def select_miners(
        self,
        task_id: str,
        count: Optional[int] = 10,
        min_stake: float = 200.0
    ) -> List[str]:

        online_miners = self.miner_cache.get_online_miners()
        logger.info(f"[MinerSelector] select_miners() for task {task_id}: found {len(online_miners)} online miners")

        eligible_miners = []
        for miner_data in online_miners:
            hotkey = miner_data["hotkey"]
            stake = miner_data.get("stake", 0.0)
            is_active = miner_data.get("is_active", False)

            if stake < min_stake:
                logger.debug(f"[MinerSelector] Miner {hotkey[:16]}... skipped: stake {stake} < min_stake {min_stake}")
                continue

            if not is_active:
                logger.debug(f"[MinerSelector] Miner {hotkey[:16]}... skipped: is_active={is_active}")
                continue
            
            miner = self.db.query(Miner).filter(
                Miner.hotkey == hotkey
            ).first()
            
            if not miner:
                miner = Miner(
                    hotkey=hotkey,
                    stake=miner_data.get("stake", 0.0),
                    reputation=miner_data.get("reputation", 0.0),
                    is_online=True
                )
                self.db.add(miner)
                self.db.commit()
            
            recent_scores = self.db.query(Score).filter(
                Score.miner_hotkey == hotkey
            ).order_by(Score.created_at.desc()).limit(10).all()
            
            avg_score = sum(s.final_score for s in recent_scores) / len(recent_scores) if recent_scores else 0.0
            
            weight = miner_data.get("stake", 0.0) * (1.0 + avg_score / 10.0)
            
            eligible_miners.append({
                "hotkey": hotkey,
                "stake": miner_data.get("stake", 0.0),
                "reputation": miner_data.get("reputation", 0.0),
                "weight": weight
            })
        
        if not eligible_miners:
            logger.warning(f"No eligible online miners found for task {task_id}")
            return []

        if count is None:
            logger.info(f"Selecting all {len(eligible_miners)} eligible miners for task {task_id}")
            return [m["hotkey"] for m in eligible_miners]
        
        weights = [m["weight"] for m in eligible_miners]
        selected = random.choices(eligible_miners, weights=weights, k=min(count, len(eligible_miners)))
        
        return [m["hotkey"] for m in selected]
    
    async def assign_task_to_miners(
        self,
        task_id: str,
        task_data: Dict,
        miner_hotkeys: List[str]
    ) -> Dict[str, bool]:
        logger.info(f"[MinerSelector] assign_task_to_miners() for task {task_id}: {len(miner_hotkeys)} miners")
        results = {}

        for miner_hotkey in miner_hotkeys:
            try:
                miner = self.db.query(Miner).filter(Miner.hotkey == miner_hotkey).first()
                if not miner:
                    continue
                
                from rediris.common.models.task_assignment import TaskAssignment
                existing_assignment = self.db.query(TaskAssignment).filter(
                    TaskAssignment.task_id == task_id,
                    TaskAssignment.miner_hotkey == miner_hotkey
                ).first()

                if existing_assignment:
                    results[miner_hotkey] = False
                    continue

                import uuid
                assignment = TaskAssignment(
                    id=str(uuid.uuid4()),
                    task_id=task_id,
                    miner_hotkey=miner_hotkey,
                    assigned_at=datetime.now(timezone.utc),
                    status="assigned"
                )
                self.db.add(assignment)
                self.db.commit()

                # Update account record
                self._update_account_on_assignment(miner_hotkey, miner)
                
                miner_url = self._get_miner_url(miner_hotkey)
                logger.info(f"[MinerSelector] Miner {miner_hotkey[:16]}... URL: {miner_url}")

                if miner_url:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        endpoints = ["/v1/workflows/receive"]
                        response = None

                        for endpoint in endpoints:
                            try:
                                response = await client.post(
                                    f"{miner_url}{endpoint}",
                                    json={
                                        "task_id": task_id,
                                        "miner_key": miner_hotkey,
                                        **task_data
                                    },
                                )
                                if response.status_code == 200:
                                    break
                            except Exception as e:
                                logger.debug(f"Failed to post to {endpoint}: {e}")
                                continue

                        if response is not None and response.status_code == 200:
                            results[miner_hotkey] = True
                            assignment.status = "delivered"
                            logger.info(f"Task assigned to miner {miner_hotkey}")
                        else:
                            results[miner_hotkey] = False
                            assignment.status = "failed"
                            status_code = response.status_code if response is not None else "no_response"
                            logger.warning(f"Failed to assign task to miner {miner_hotkey}: {status_code}")
                else:
                    results[miner_hotkey] = True
                    assignment.status = "pending"
                    logger.info(f"Task queued for miner {miner_hotkey} (URL not available)")
                
                self.db.commit()
            except Exception as e:
                logger.error(f"Error assigning task to miner {miner_hotkey}: {e}")
                results[miner_hotkey] = False
        
        return results
    
    def _get_miner_url(self, miner_hotkey: str) -> Optional[str]:
        miner_url = self.miner_cache.get_miner_url(miner_hotkey)
        if miner_url:
            return miner_url

        try:
            miner = self.db.query(Miner).filter(Miner.hotkey == miner_hotkey).first()
            if miner and miner.miner_url:
                return miner.miner_url

            miners = self.bittensor_client.get_all_miners()

            for miner_data in miners:
                if miner_data["hotkey"] == miner_hotkey:
                    if miner_data.get("uid") is not None:
                        uid = miner_data.get("uid")
                        if self.bittensor_client.metagraph and uid < len(self.bittensor_client.metagraph.axons):
                            axon = self.bittensor_client.metagraph.axons[uid]
                            ip = axon.ip
                            port = axon.port
                            if ip and ip != "0.0.0.0" and port:
                                url = f"http://{ip}:{port}"
                                if miner:
                                    miner.miner_url = url
                                    self.db.commit()
                                return url
        except Exception as e:
            logger.error(f"Error getting miner URL: {e}", exc_info=True)

        return None

    def _update_account_on_assignment(self, hotkey: str, miner: Optional[Miner] = None):
        try:
            account = self.db.query(Account).filter(Account.hotkey == hotkey).first()

            if not account:
                # Create new account
                account = Account(
                    hotkey=hotkey,
                    is_registered=miner is not None,
                    first_seen_at=datetime.now(timezone.utc),
                    last_active_at=datetime.now(timezone.utc),
                    total_tasks_assigned=1,
                    current_stake=miner.stake if miner else 0.0,
                    current_reputation=miner.reputation if miner else 0.0,
                    peak_stake=miner.stake if miner else 0.0,
                    peak_reputation=miner.reputation if miner else 0.0,
                )
                self.db.add(account)
            else:
                # Update existing account
                account.total_tasks_assigned = (account.total_tasks_assigned or 0) + 1
                account.last_active_at = datetime.now(timezone.utc)

                if miner:
                    account.is_registered = True
                    account.current_stake = miner.stake or 0.0
                    account.current_reputation = miner.reputation or 0.0
                    if (miner.stake or 0) > (account.peak_stake or 0):
                        account.peak_stake = miner.stake
                    if (miner.reputation or 0) > (account.peak_reputation or 0):
                        account.peak_reputation = miner.reputation

            self.db.commit()
            logger.debug(f"Account updated for miner {hotkey[:16]}...")
        except Exception as e:
            logger.error(f"Error updating account for miner {hotkey}: {e}")

    def select_validated_miners(self, task_id: str) -> List[Dict]:
        try:
            validated_datasets = self.db.query(MinerDataset).filter(
                MinerDataset.task_id == task_id,
                MinerDataset.validation_status == "approved"
            ).all()

            if not validated_datasets:
                logger.warning(f"No validated datasets found for task {task_id}")
                return []

            online_miners = self.miner_cache.get_online_miners()
            online_hotkeys = {m["hotkey"] for m in online_miners}

            validated_miners = []
            for dataset in validated_datasets:
                miner_hotkey = dataset.miner_hotkey

                if miner_hotkey not in online_hotkeys:
                    logger.info(f"Miner {miner_hotkey[:16]}... has validated dataset but is offline, skipping")
                    continue

                miner = self.db.query(Miner).filter(Miner.hotkey == miner_hotkey).first()
                miner_data = next((m for m in online_miners if m["hotkey"] == miner_hotkey), None)

                validated_miners.append({
                    "hotkey": miner_hotkey,
                    "stake": miner_data.get("stake", 0.0) if miner_data else (miner.stake if miner else 0.0),
                    "reputation": miner_data.get("reputation", 0.0) if miner_data else (miner.reputation if miner else 0.0),
                    "dataset_url": dataset.dataset_url,
                    "dataset_description": dataset.dataset_description
                })

            logger.info(f"Found {len(validated_miners)} validated miners for task {task_id}")
            return validated_miners

        except Exception as e:
            logger.error(f"Error selecting validated miners: {e}", exc_info=True)
            return []
