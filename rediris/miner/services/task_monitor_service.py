import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import httpx
import secrets
import time
from rediris.common.crypto.signature import SignatureAuth
from rediris.common.utils.logging import setup_logger
from rediris.common.config import settings
from rediris.miner.services.dataset_service import DatasetService

logger = setup_logger(__name__)

TASK_POLL_INTERVAL = 60
DATASET_SUBMISSION_RETRY_INTERVAL = 300

class TaskMonitorService:

    def __init__(self, wallet=None, wallet_name=None, hotkey_name=None, yaml_config=None):
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.yaml_config = yaml_config
        self.task_center_url = settings.TASK_CENTER_URL
        if yaml_config:
            self.task_center_url = yaml_config.get_task_center_url() or self.task_center_url

        self.dataset_service = DatasetService(wallet, wallet_name, hotkey_name, yaml_config)
        self.is_running = False
        self._monitor_task = None
        self.submitted_datasets: Dict[str, str] = {}
        self.poll_interval = TASK_POLL_INTERVAL
        self.task_datasets = self._load_task_datasets()

    def _load_task_datasets(self) -> Dict[str, Dict[str, str]]:
        result = {}
        if not self.yaml_config:
            return result

        tasks_config = self.yaml_config.get("datasets.tasks", [])
        if not tasks_config:
            return result

        for task_config in tasks_config:
            task_id = task_config.get("task_id")
            if task_id:
                result[task_id] = {
                    "url": task_config.get("url", ""),
                    "description": task_config.get("description", "")
                }

        logger.info(f"Loaded {len(result)} task-specific dataset configurations")
        return result

    def reload_config(self):
        self.task_datasets = self._load_task_datasets()

    async def start(self):
        if self.is_running:
            logger.warning("Task monitor service is already running")
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Task monitor service started")

    async def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Task monitor service stopped")

    async def _monitor_loop(self):
        while self.is_running:
            try:
                await self._check_and_submit_datasets()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                logger.info("Task monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Task monitor loop error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def _check_and_submit_datasets(self):
        tasks = await self._get_available_tasks()

        if not tasks:
            return

        for task in tasks:
            task_id = task.get("task_id")
            task_status = task.get("status", "")

            if task_status not in ["dataset_validation"]:
                continue

            if task_id in self.submitted_datasets:
                continue

            if task_id not in self.task_datasets:
                logger.debug(f"Task {task_id}: No dataset configured in config.yml, skipping")
                continue

            dataset_config = self.task_datasets[task_id]
            dataset_url = dataset_config.get("url")
            dataset_description = dataset_config.get("description", "")

            if not dataset_url:
                logger.warning(f"Task {task_id}: Dataset URL is empty in config, skipping")
                continue

            logger.info(f"Task {task_id}: Submitting dataset in {task_status} phase, url={dataset_url}")

            result = await self.dataset_service.submit_dataset(
                task_id=task_id,
                dataset_url=dataset_url,
                dataset_description=dataset_description
            )

            if result.get("success"):
                self.submitted_datasets[task_id] = "submitted"
                logger.info(f"Task {task_id}: Dataset submitted successfully")
            else:
                logger.warning(f"Task {task_id}: Dataset submission failed: {result.get('error')}")

    async def _get_available_tasks(self) -> List[Dict[str, Any]]:
        if not self.wallet:
            return []

        miner_hotkey = self.wallet.hotkey.ss58_address
        tasks_url = f"{self.task_center_url}/v1/miners/tasks/available"
        tasks_endpoint = "/v1/miners/tasks/available"
        logger.info(f"tasks_url  {tasks_url}")
        try:
            async with httpx.AsyncClient(timeout=80.0) as client:
                signature_auth = SignatureAuth(self.wallet)
                nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
                auth_headers = signature_auth.create_auth_headers_with_nonce(tasks_endpoint, nonce)
                response = await client.get(
                    tasks_url,
                    params={"miner_hotkey": miner_hotkey},
                    headers=auth_headers
                )

                if response.status_code == 200:
                    data = response.json()
                    tasks = data.get("tasks", [])
                    if tasks:
                        logger.debug(f"Found {len(tasks)} available tasks")
                    return tasks
                elif response.status_code == 404:
                    return []
                else:
                    logger.warning(f"Failed to get available tasks: HTTP {response.status_code}")
                    return []

        except Exception as e:
            logger.error(f"Error getting available tasks: {e}", exc_info=True)
            return []

    async def check_dataset_status(self, task_id: str) -> Dict[str, Any]:
        return await self.dataset_service.check_validation_status(task_id)

    def get_submitted_datasets(self) -> Dict[str, str]:
        return self.submitted_datasets.copy()
