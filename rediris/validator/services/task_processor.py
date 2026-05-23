import asyncio
from typing import Dict, Any, Optional
from rediris.validator.services.audit_validator import AuditValidator
from rediris.validator.services.score_calculator import ScoreCalculator
from rediris.validator.services.score_cache import ScoreCache
from rediris.validator.schemas.audit import AuditTaskRequest
from rediris.common.utils.logging import setup_logger
from rediris.common.config.yaml_config import YamlConfig
import httpx
from rediris.common.config import settings
import bittensor as bt
import secrets
import time
from rediris.common.crypto.signature import SignatureAuth

logger = setup_logger(__name__)

class TaskProcessor:

    def __init__(
        self,
        wallet: bt.wallet,
        wallet_name: str,
        hotkey_name: str,
        score_cache: Optional[ScoreCache] = None,
        yaml_config: Optional[YamlConfig] = None
    ):
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.yaml_config = yaml_config
        self.audit_validator = AuditValidator()
        self.score_calculator = ScoreCalculator()
        self.score_cache = score_cache or ScoreCache()  # 注入 ScoreCache
        self.is_running = False
        self.process_interval = 60
        self._process_task = None

        if yaml_config:
            self.task_center_url = yaml_config.get_task_center_url() or settings.TASK_CENTER_URL
            self.api_key = yaml_config.get_task_center_api_key()
        else:
            self.task_center_url = settings.TASK_CENTER_URL
            self.api_key = getattr(settings, 'API_KEY', None)
        
        self.signature_auth = SignatureAuth(wallet)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API Key if configured"""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def _get_sig_headers(self, endpoint: str) -> Dict[str, str]:
        nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
        return self.signature_auth.create_auth_headers_with_nonce(endpoint, nonce)

    async def start(self):
        if self.is_running:
            logger.warning("Task processor is already running")
            return

        self.is_running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("Task processor started")

    async def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("Task processor stopped")

    async def _process_loop(self):
        while self.is_running:
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(self.process_interval)
            except asyncio.CancelledError:
                logger.info("Process loop cancelled")
                break
            except Exception as e:
                logger.error(f"Task processing loop error: {e}", exc_info=True)
                await asyncio.sleep(self.process_interval)

    async def _process_pending_tasks(self):
        try:
            validator_key = self.wallet.hotkey.ss58_address
            task_center_url = self.task_center_url

            logger.debug(f"Fetching pending audit tasks for validator {validator_key[:20]}...")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{task_center_url}/v1/validators/pending",
                    params={"validator_key": validator_key},
                    headers={**self._get_headers(), **self._get_sig_headers("/v1/validators/pending")}
                )

                if response.status_code == 200:
                    data = response.json()
                    tasks = data.get("tasks", data) if isinstance(data, dict) else data

                    if tasks:
                        logger.info(f"Found {len(tasks)} pending audit tasks")

                    for task in tasks:
                        try:
                            await self._process_audit_task(task)
                        except Exception as e:
                            audit_task_id = task.get("audit_task_id", task.get("id", "unknown"))
                            logger.error(f"Failed to process audit task {audit_task_id}: {e}", exc_info=True)
                            continue
                elif response.status_code == 404:
                    logger.debug("No pending audit tasks found")
                else:
                    logger.warning(f"Unexpected response from task center: {response.status_code}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching pending tasks: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing pending tasks: {e}", exc_info=True)

    async def _process_audit_task(self, task: Dict[str, Any]):
        audit_task_id = task.get("audit_task_id", task.get("id", "unknown"))
        miner_hotkey = task.get("miner_hotkey", "")
        task_id = task.get("original_task_id", task.get("task_id", ""))
        audit_type = task.get("audit_type", "lora")
        lora_url = task.get("lora_url", "")
        dataset_url = task.get("dataset_url", "")
        task_info = task.get("task_info", {})

        logger.info(f"Processing audit task {audit_task_id} (type={audit_type}) for miner {miner_hotkey[:20]}...")

        try:
            if audit_type == "dataset":
                await self._process_dataset_audit_task(
                    audit_task_id=audit_task_id,
                    task_id=task_id,
                    miner_hotkey=miner_hotkey,
                    dataset_url=dataset_url,
                    task_info=task_info
                )
            else:
                await self._process_lora_audit_task(
                    audit_task_id=audit_task_id,
                    task_id=task_id,
                    miner_hotkey=miner_hotkey,
                    lora_url=lora_url,
                    task_info=task_info
                )

        except Exception as e:
            logger.error(f"Failed to process audit task {audit_task_id}: {e}", exc_info=True)
            await self._update_audit_task_status(audit_task_id, "failed", {"error": str(e)})
            raise

    async def _process_lora_audit_task(
        self,
        audit_task_id: str,
        task_id: str,
        miner_hotkey: str,
        lora_url: str,
        task_info: Dict[str, Any]
    ):
        should_score = await self._check_miner_limit(task_id, miner_hotkey)
        
        if not should_score:
            logger.info(
                f"Task {task_id}: Miner {miner_hotkey[:20]}... "
                f"exceeds reward limit, assigning 0 score"
            )
            self.score_cache.cache_score(
                task_id=task_id,
                miner_hotkey=miner_hotkey,
                validator_hotkey=self.wallet.hotkey.ss58_address,
                score=0.0,
                score_details={"reason": "exceeds_reward_limit"}
            )
            await self._update_audit_task_status(audit_task_id, "completed", {"final_score": 0.0})
            return
        
        audit_request = AuditTaskRequest(
            audit_task_id=audit_task_id,
            miner_hotkey=miner_hotkey,
            lora_url=lora_url,
            task_info=task_info
        )

        result = await self.audit_validator.process_audit_task(audit_request)

        score = result.get("final_score", 0.0)

        logger.info(f"LoRA audit task {audit_task_id} completed: "
                   f"cosine_similarity={result.get('cosine_similarity', 0):.4f}, "
                   f"quality_score={result.get('quality_score', 0):.2f}, "
                   f"final_score={score:.2f}")

        self.score_cache.cache_score(
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            validator_hotkey=self.wallet.hotkey.ss58_address,
            score=score,
            score_details={
                "cosine_similarity": result.get("cosine_similarity", 0.0),
                "quality_score": result.get("quality_score", 0.0),
                "content_safety_score": result.get("content_safety_score", 0.0),
                "rejected": result.get("rejected", False),
                "rejection_reason": result.get("reason", None)
            }
        )

        await self._update_audit_task_status(audit_task_id, "completed", result)

    async def _process_dataset_audit_task(
        self,
        audit_task_id: str,
        task_id: str,
        miner_hotkey: str,
        dataset_url: str,
        task_info: Dict[str, Any]
    ):
        result = await self.audit_validator.process_audit_task({
            "audit_task_id": audit_task_id,
            "miner_hotkey": miner_hotkey,
            "dataset_url": dataset_url,
            "audit_type": "dataset",
            "task_info": task_info
        })

        is_valid = result.get("is_valid", False)

        logger.info(f"Dataset audit task {audit_task_id} completed: is_valid={is_valid}")

        await self._submit_dataset_validation(
            audit_task_id=audit_task_id,
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            is_approved=is_valid,
            validation_result=result,
            rejection_reason=result.get("rejection_reason")
        )

        await self._update_audit_task_status(audit_task_id, "completed", result)

    async def _check_miner_limit(self, task_id: str, miner_hotkey: str) -> bool:
        try:
            task_config = await self._get_task_config(task_id)
            default_reward_miners = task_config.get("default_reward_miners", 6)
            
            if default_reward_miners < settings.MIN_REWARD_MINERS:
                default_reward_miners = settings.MIN_REWARD_MINERS
            if default_reward_miners > settings.MAX_REWARD_MINERS:
                default_reward_miners = settings.MAX_REWARD_MINERS
            
            cached_scores = self.score_cache.get_cached_scores_for_task(task_id)
            
            sorted_miners = sorted(
                cached_scores,
                key=lambda x: x["score"],
                reverse=True
            )
            
            if len(sorted_miners) >= default_reward_miners:
                top_miners = [m["miner_hotkey"] for m in sorted_miners[:default_reward_miners]]
                if miner_hotkey not in top_miners:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking miner limit for task {task_id}: {e}", exc_info=True)
            return True
    
    async def _get_task_config(self, task_id: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.task_center_url}/v1/tasks/{task_id}/config",
                    headers={**self._get_headers(), **self._get_sig_headers(f"/v1/tasks/{task_id}/config")}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to get task config: {response.status_code}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting task config: {e}", exc_info=True)
            return {}

    async def _submit_dataset_validation(
        self,
        audit_task_id: str,
        task_id: str,
        miner_hotkey: str,
        is_approved: bool,
        validation_result: Dict[str, Any],
        rejection_reason: Optional[str] = None
    ):
        validator_hotkey = self.wallet.hotkey.ss58_address
        task_center_url = self.task_center_url

        validation_data = {
            "audit_task_id": audit_task_id,
            "validator_hotkey": validator_hotkey,
            "is_approved": is_approved,
            "validation_result": validation_result,
            "rejection_reason": rejection_reason
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{task_center_url}/v1/validators/dataset/validation",
                    json=validation_data,
                    headers={**self._get_headers(), **self._get_sig_headers("/v1/validators/dataset/validation")}
                )

                if response.status_code == 200:
                    logger.info(f"Dataset validation submitted: audit_task={audit_task_id}, approved={is_approved}")
                else:
                    logger.warning(f"Dataset validation submission returned {response.status_code}: {response.text[:200]}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error submitting dataset validation: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error submitting dataset validation: {e}", exc_info=True)

    async def _update_audit_task_status(
        self,
        audit_task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None
    ):
        task_center_url = self.task_center_url

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{task_center_url}/v1/audit/update_status",
                    json={
                        "audit_task_id": audit_task_id,
                        "status": status,
                        "result": result
                    },
                    headers={**self._get_headers(), **self._get_sig_headers("/v1/audit/update_status")}
                )

                if response.status_code == 200:
                    logger.debug(f"Audit task {audit_task_id} status updated to {status}")
                elif response.status_code == 404:
                    logger.warning(f"Audit task {audit_task_id} not found for status update")
                else:
                    logger.warning(f"Status update returned {response.status_code}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error updating audit task status: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error updating audit task status: {e}", exc_info=True)
