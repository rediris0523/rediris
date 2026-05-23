
import asyncio
from typing import Dict, Any, Optional, List
import httpx
from sqlalchemy.orm import Session
from rediris.common.utils.logging import setup_logger
from rediris.common.config import settings
from rediris.common.config.yaml_config import YamlConfig
import bittensor as bt
import secrets
import time
from rediris.common.crypto.signature import SignatureAuth
from rediris.validator.services.bittensor_sync import BittensorSyncService
from rediris.validator.services.score_calculator import ScoreCalculator
from rediris.validator.services.ema_weight_service import EmaWeightService
from rediris.validator.services.score_cache import ScoreCache

logger = setup_logger(__name__)

class WeightSyncService:

    def __init__(
        self,
        wallet: bt.wallet,
        wallet_name: str,
        hotkey_name: str,
        bittensor_sync: BittensorSyncService,
        score_cache: ScoreCache,
        sync_interval: int = 3600,
        yaml_config: Optional[YamlConfig] = None,
        db_session: Optional[Session] = None
    ):
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.bittensor_sync = bittensor_sync
        self.score_cache = score_cache
        self.score_calculator = ScoreCalculator()
        self.sync_interval = sync_interval
        self.is_running = False
        self._sync_task = None
        self.yaml_config = yaml_config
        self.db_session = db_session
        self._ema_service = None

        if yaml_config:
            self.task_center_url = yaml_config.get_task_center_url() or settings.TASK_CENTER_URL
            self.api_key = yaml_config.get_task_center_api_key()
        else:
            self.task_center_url = settings.TASK_CENTER_URL
            self.api_key = getattr(settings, 'API_KEY', None)
        
        self.signature_auth = SignatureAuth(wallet)
    
    def _get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def _get_sig_headers(self, endpoint: str) -> Dict[str, str]:
        nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
        return self.signature_auth.create_auth_headers_with_nonce(endpoint, nonce)

    def _get_ema_service(self) -> Optional[EmaWeightService]:
        if self._ema_service is None and self.db_session is not None:
            if self.wallet is None:
                logger.warning("Wallet not initialized, cannot create EMA service")
                return None
            try:
                validator_hotkey = self.wallet.hotkey.ss58_address
                self._ema_service = EmaWeightService(
                    db=self.db_session,
                    validator_hotkey=validator_hotkey
                )
            except Exception as e:
                logger.error(f"Failed to create EMA service: {e}", exc_info=True)
                return None
        return self._ema_service

    async def start(self):
        if self.is_running:
            logger.warning("Weight sync service is already running")
            return

        self.is_running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"Weight sync service started (interval={self.sync_interval}s)")

    async def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info("Weight sync service stopped")

    async def _sync_loop(self):
        await asyncio.sleep(60)

        while self.is_running:
            try:
                await self.sync_weights()
                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                logger.info("Weight sync loop cancelled")
                break
            except Exception as e:
                logger.error(f"Weight sync error: {e}", exc_info=True)
                await asyncio.sleep(self.sync_interval)

    async def sync_weights(self):

        logger.info("Starting weight synchronization...")

        try:
            await self.clear_ended_task_cache()
            
            idle_status = await self._check_system_idle()
            is_idle = idle_status.get("is_idle", True)
            reward_task_ids = idle_status.get("reward_task_ids", [])
            task_types = idle_status.get("task_types", {})
            task_configs = idle_status.get("task_configs", {})

            if is_idle:
                logger.info(f"System is idle (no REWARD phase tasks), setting weight only for UID {settings.IDLE_REWARD_UID}")
                await self._set_idle_weight_to_chain()
                logger.info("Idle weight synchronization completed successfully")
                return

            logger.info(f"Found {len(reward_task_ids)} tasks in REWARD phase: {reward_task_ids}")

            miner_task_types = await self._fetch_participating_miners_with_types(reward_task_ids, task_types)

            if not miner_task_types:
                logger.warning("No participating miners found for REWARD phase tasks, setting idle weight")
                await self._set_idle_weight_to_chain()
                return

            logger.info(f"Found {len(miner_task_types)} participating miners")

            miner_scores = await self._get_miner_scores_from_cache(reward_task_ids)

            if not miner_scores:
                logger.warning("No miner scores found in ScoreCache for REWARD phase tasks, setting idle weight")
                await self._set_idle_weight_to_chain()
                return

            logger.info(f"Fetched scores for {len(miner_scores)} miners from ScoreCache")

            if task_configs:
                weights = self._calculate_pool_weights(
                    miner_scores=miner_scores,
                    miner_task_types=miner_task_types,
                    task_configs=task_configs
                )
            else:
                weights = self._calculate_type_weights(
                    miner_scores=miner_scores,
                    miner_task_types=miner_task_types,
                    task_types=task_types,
                    task_configs=task_configs
                )

            if not weights:
                logger.warning("No valid weights calculated, setting idle weight")
                await self._set_idle_weight_to_chain()
                return

            ema_service = self._get_ema_service()
            if ema_service:
                miners = self.bittensor_sync.get_all_miners()
                miner_uids = {m["hotkey"]: m["uid"] for m in miners}

                treasury_hotkey = self._get_treasury_hotkey()
                miner_weights = {k: v for k, v in weights.items() if k != treasury_hotkey}
                treasury_weight = weights.get(treasury_hotkey, settings.TREASURY_RATIO)

                smoothed_weights = ema_service.apply_ema_smoothing(miner_weights, miner_uids)

                final_weights = smoothed_weights.copy()
                if treasury_hotkey:
                    final_weights[treasury_hotkey] = treasury_weight

                logger.info(f"Applied EMA smoothing to {len(smoothed_weights)} miner weights")
                weights = final_weights
            else:
                logger.warning("EMA service not available, using raw weights")

            logger.info(f"Calculated weights for {len(weights)} entities (including treasury)")

            await self._set_weights_to_chain(weights)

            logger.info("Weight synchronization completed successfully")

        except Exception as e:
            logger.error(f"Weight synchronization failed: {e}", exc_info=True)
            raise

    async def _check_system_idle(self) -> Dict[str, Any]:
        try:
            headers = {**self._get_headers(), **self._get_sig_headers("/v1/tasks/active/count")}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.task_center_url}/v1/tasks/active/count",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        f"System idle check: active_count={data.get('active_count', 0)}, "
                        f"is_idle={data.get('is_idle', True)}, "
                        f"task_types={data.get('task_types', {})}"
                    )
                    return data
                else:
                    logger.warning(f"Failed to check system idle status: HTTP {response.status_code}")
                    return {"is_idle": True, "reward_task_ids": [], "task_types": {}}

        except Exception as e:
            logger.error(f"Error checking system idle status: {e}", exc_info=True)
            return {"is_idle": True, "reward_task_ids": [], "task_types": {}}

    async def _fetch_participating_miners_with_types(
        self,
        task_ids: List[str],
        task_types: Dict[str, List[str]]
    ) -> Dict[str, str]:
        try:
            task_id_to_type = {}
            for task_type, ids in task_types.items():
                for task_id in ids:
                    task_id_to_type[task_id] = task_type

            miner_task_types = {}

            async with httpx.AsyncClient(timeout=60.0) as client:
                for task_id in task_ids:
                    headers = {**self._get_headers(), **self._get_sig_headers(f"/v1/tasks/{task_id}/participants")}
                    response = await client.get(
                        f"{self.task_center_url}/v1/tasks/{task_id}/participants",
                        headers=headers
                    )

                    if response.status_code == 200:
                        data = response.json()
                        miners = data.get("miner_hotkeys", [])
                        task_type = task_id_to_type.get(task_id, "text")

                        for miner in miners:
                            if miner not in miner_task_types:
                                miner_task_types[miner] = task_type
                            elif task_type == "image":
                                miner_task_types[miner] = "image"

                        logger.debug(f"Task {task_id} ({task_type}): found {len(miners)} participants")
                    else:
                        logger.warning(f"Failed to fetch participants for {task_id}: HTTP {response.status_code}")

            return miner_task_types

        except Exception as e:
            logger.error(f"Error fetching participating miners with types: {e}", exc_info=True)
            return {}

    async def _get_miner_scores_from_cache(self, task_ids: List[str]) -> Dict[str, float]:
        miner_scores = {}
        
        for task_id in task_ids:
            task_scores = self.score_cache.get_cached_scores_for_task(task_id)
            
            for score in task_scores:
                miner_hotkey = score["miner_hotkey"]
                if miner_hotkey in miner_scores:
                    miner_scores[miner_hotkey] = max(miner_scores[miner_hotkey], score["score"])
                else:
                    miner_scores[miner_hotkey] = score["score"]
        
        return miner_scores
    
    async def clear_ended_task_cache(self):

        cached_task_ids = list(self.score_cache.get_all_cached_scores().keys())
        
        if not cached_task_ids:
            return
        
        for task_id in cached_task_ids:
            try:
                task_status = await self._get_task_status(task_id)
                
                from rediris.common.models.task import TaskStatus
                if task_status == TaskStatus.ENDED.value or task_status == "ended":
                    self.score_cache.clear_scores(task_id)
                    logger.info(f"Cleared cache for ended task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to check status for task {task_id}: {e}")
                continue
    
    async def _get_task_status(self, task_id: str) -> Optional[str]:
        try:
            headers = {**self._get_headers(), **self._get_sig_headers(f"/v1/tasks/{task_id}")}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.task_center_url}/v1/tasks/{task_id}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    if hasattr(status, 'value'):
                        return status.value
                    return str(status) if status else None
                else:
                    logger.warning(f"Failed to get task status: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Error getting task status: {e}", exc_info=True)
            return None

    async def _set_idle_weight_to_chain(self):
        try:
            miners = self.bittensor_sync.get_all_miners()

            treasury_miner = None
            for miner in miners:
                if miner.get("uid") == settings.IDLE_REWARD_UID:
                    treasury_miner = miner
                    break

            if not treasury_miner:
                logger.warning(f"UID {settings.IDLE_REWARD_UID} not found on subnet, skipping idle weight")
                return

            uids = [settings.IDLE_REWARD_UID]
            weights = [1.0]

            logger.info(f"Setting idle weight for UID {settings.IDLE_REWARD_UID} (hotkey={treasury_miner.get('hotkey', 'unknown')[:16]}...)")

            self.bittensor_sync.set_weights(uids, weights)

            logger.info(f"Idle weight set successfully for UID {settings.IDLE_REWARD_UID}")

        except Exception as e:
            logger.error(f"Failed to set idle weight to chain: {e}", exc_info=True)
            raise

    def _calculate_pool_weights(
        self,
        miner_scores: Dict[str, float],
        miner_task_types: Dict[str, str],
        task_configs: Dict[str, Dict[str, Any]],
        use_price_weighting: bool = True
    ) -> Dict[str, float]:
        if not miner_scores:
            logger.warning("No miner scores provided for pool weight calculation")
            return {}

        if not task_configs:
            logger.warning("No task configs provided for pool weight calculation")
            return {}

        alpha_price = None
        if use_price_weighting:
            alpha_price = self._get_alpha_price()
            if alpha_price:
                logger.info(f"Using alpha price: {alpha_price} for price-weighted calculation")
            else:
                logger.info("Alpha price not available, using pure quality index")

        pools: Dict[str, Dict[str, Any]] = {}
        for task_id, config in task_configs.items():
            pool_name = config.get("pool_name", config.get("workflow_type", "default"))
            if pool_name not in pools:
                pools[pool_name] = {
                    "configured_ratio": config.get("pool_ratio", 1.0),
                    "min_score_threshold": config.get("min_score_threshold", 3.5),
                    "quality_exponent": config.get("quality_exponent", 2),
                    "task_ids": []
                }
            pools[pool_name]["task_ids"].append(task_id)

        total_configured = sum(pool["configured_ratio"] for pool in pools.values())
        miner_pool_ratio = 1.0 - settings.TREASURY_RATIO

        if total_configured <= 0:
            logger.warning("No configured pool ratios found, using equal distribution")
            total_configured = len(pools) or 1

        for pool_name, pool in pools.items():
            pool["actual_ratio"] = (pool["configured_ratio"] / total_configured) * miner_pool_ratio

        pool_ratio_strs = [f"{name}={p['actual_ratio']:.2%}" for name, p in pools.items()]
        logger.info(f"Pool ratios: {', '.join(pool_ratio_strs)}")

        pool_miners: Dict[str, Dict[str, float]] = {pool_name: {} for pool_name in pools}

        for miner_hotkey, score in miner_scores.items():
            task_type = miner_task_types.get(miner_hotkey, "text")

            target_pool = None
            for pool_name in pools:
                if task_type in pool_name.lower() or pool_name.lower() in task_type:
                    target_pool = pool_name
                    break

            if not target_pool:
                target_pool = list(pools.keys())[0] if pools else "default"

            if target_pool in pool_miners:
                pool_miners[target_pool][miner_hotkey] = score

        final_weights: Dict[str, float] = {}

        for pool_name, miners in pool_miners.items():
            if not miners:
                continue

            pool_config = pools.get(pool_name, {})
            min_threshold = pool_config.get("min_score_threshold", 3.5)
            quality_exp = pool_config.get("quality_exponent", 2)
            pool_ratio = pool_config.get("actual_ratio", 0.0)

            if alpha_price is not None:
                quality_indices = self.score_calculator.calculate_price_weighted_scores(
                    scores=miners,
                    alpha_price=alpha_price,
                    min_threshold=min_threshold,
                    quality_exponent=quality_exp
                )
            else:
                quality_indices = self.score_calculator.calculate_quality_weighted_scores(
                    scores=miners,
                    min_threshold=min_threshold,
                    quality_exponent=quality_exp
                )

            pool_weights = self.score_calculator.normalize_pool_weights(
                weighted_indices=quality_indices,
                pool_ratio=pool_ratio
            )

            final_weights.update(pool_weights)

            logger.debug(f"Pool {pool_name}: {len(pool_weights)} miners, ratio={pool_ratio:.2%}")

        treasury_hotkey = self._get_treasury_hotkey()
        if treasury_hotkey:
            final_weights[treasury_hotkey] = settings.TREASURY_RATIO

        total_weight = sum(final_weights.values())
        price_info = f", alpha_price={alpha_price}" if alpha_price else ""
        logger.info(
            f"Pool weight calculation: pools={len(pools)}, miners={len(final_weights) - 1}, "
            f"treasury={settings.TREASURY_RATIO:.0%}, total={total_weight:.4f}{price_info}"
        )

        return final_weights

    def _get_alpha_price(self) -> Optional[float]:
        try:
            if hasattr(self.bittensor_sync, 'client') and self.bittensor_sync.client:
                return self.bittensor_sync.client.get_alpha_price()
            return None
        except Exception as e:
            logger.warning(f"Failed to get alpha price: {e}")
            return None

    def _calculate_type_weights(
        self,
        miner_scores: Dict[str, float],
        miner_task_types: Dict[str, str],
        task_types: Dict[str, List[str]],
        task_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        valid_miners = {
            hotkey: score
            for hotkey, score in miner_scores.items()
            if score >= self.score_calculator.base_threshold
        }

        if not valid_miners:
            return {}

        raw_weights = self.score_calculator.calculate_weight_from_scores(valid_miners)

        active_types = {t for t, ids in task_types.items() if len(ids) > 0}
        if not active_types:
            active_types = set(miner_task_types.values())

        type_miners: Dict[str, Dict[str, float]] = {t: {} for t in active_types}

        for hotkey, weight in raw_weights.items():
            task_type = miner_task_types.get(hotkey)
            if task_type and task_type in type_miners:
                type_miners[task_type][hotkey] = weight
            elif active_types:
                default_type = list(active_types)[0]
                type_miners[default_type][hotkey] = weight

        miner_pool_ratio = settings.MINER_POOL_RATIO
        type_ratios = {}

        if task_configs:
            type_configured_ratios: Dict[str, List[float]] = {t: [] for t in active_types}

            for task_id, config in task_configs.items():
                pool_ratio = config.get("pool_ratio", 1.0)
                workflow_type = config.get("workflow_type", "").lower()

                if "image" in workflow_type:
                    if "image" in type_configured_ratios:
                        type_configured_ratios["image"].append(pool_ratio)
                else:
                    if "text" in type_configured_ratios:
                        type_configured_ratios["text"].append(pool_ratio)

            total_configured = 0.0
            for task_type in active_types:
                ratios = type_configured_ratios.get(task_type, [])
                if ratios:
                    avg_ratio = sum(ratios) / len(ratios)
                    type_ratios[task_type] = avg_ratio
                    total_configured += avg_ratio

            if total_configured > 0:
                for task_type in type_ratios:
                    type_ratios[task_type] = (type_ratios[task_type] / total_configured) * miner_pool_ratio
                logger.info(f"Using task config pool_ratios: {type_ratios}")

        if not type_ratios:
            num_active_types = len(active_types)
            if num_active_types > 0:
                equal_ratio = miner_pool_ratio / num_active_types
                for task_type in active_types:
                    type_ratios[task_type] = equal_ratio
                logger.info(f"Using default equal distribution: {type_ratios}")

        final_weights = {}

        for task_type, miners in type_miners.items():
            if not miners:
                continue

            type_total = sum(miners.values())
            type_ratio = type_ratios.get(task_type, 0.0)

            if type_total > 0 and type_ratio > 0:
                for hotkey, weight in miners.items():
                    final_weights[hotkey] = (weight / type_total) * type_ratio

        treasury_hotkey = self._get_treasury_hotkey()
        if treasury_hotkey:
            final_weights[treasury_hotkey] = settings.TREASURY_RATIO

        total_weight = sum(final_weights.values())
        type_info = ", ".join([f"{t}={len(m)}({type_ratios.get(t, 0):.0%})" for t, m in type_miners.items() if m])
        logger.info(
            f"Weight distribution: treasury={settings.TREASURY_RATIO:.0%}, "
            f"{type_info}, total_weight={total_weight:.4f}"
        )

        return final_weights

    def _get_treasury_hotkey(self) -> Optional[str]:
        try:
            miners = self.bittensor_sync.get_all_miners()
            for miner in miners:
                if miner.get("uid") == settings.IDLE_REWARD_UID:
                    return miner.get("hotkey")
            logger.warning(f"UID {settings.IDLE_REWARD_UID} not found on subnet")
            return None
        except Exception as e:
            logger.error(f"Error getting treasury hotkey: {e}", exc_info=True)
            return None

    async def _set_weights_to_chain(self, weights: Dict[str, float]):
        try:
            miners = self.bittensor_sync.get_all_miners()
            hotkey_to_uid = {miner["hotkey"]: miner["uid"] for miner in miners}

            uids = []
            weight_values = []

            for hotkey, weight in weights.items():
                if hotkey in hotkey_to_uid:
                    uids.append(hotkey_to_uid[hotkey])
                    weight_values.append(weight)

            if not uids:
                logger.warning("No valid miners found for weight setting")
                return

            # Log the exact mapping hotkey -> weight that will be sent on-chain.
            uid_to_hotkey = {miner["uid"]: miner["hotkey"] for miner in miners}
            hotkey_weight_map = {
                uid_to_hotkey.get(uid, f"<uid:{uid}>"): w
                for uid, w in zip(uids, weight_values)
            }
            logger.info(
                f"Setting weights for {len(uids)} miners to chain. "
                f"hotkeys_and_weights={hotkey_weight_map}"
            )

            self.bittensor_sync.set_weights(uids, weight_values)

            logger.info(f"Weights set successfully for {len(uids)} miners")

        except Exception as e:
            logger.error(f"Failed to set weights to chain: {e}", exc_info=True)
            raise

    async def get_current_weights(self) -> Dict[str, float]:
        try:
            miners = self.bittensor_sync.get_all_miners()
            return {
                miner["hotkey"]: miner.get("weight", 0.0)
                for miner in miners
            }
        except Exception as e:
            logger.error(f"Failed to get current weights: {e}", exc_info=True)
            return {}
