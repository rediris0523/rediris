import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import httpx
from rediris.common.utils.logging import setup_logger
from rediris.common.config import settings
import secrets
import time
from rediris.common.crypto.signature import SignatureAuth

logger = setup_logger(__name__)

DATASET_SUBMIT_MAX_RETRIES = 3
DATASET_SUBMIT_RETRY_DELAY = 5
DATASET_VALIDATION_CHECK_INTERVAL = 30
DATASET_VALIDATION_MAX_WAIT_TIME = 3600

class DatasetService:

    def __init__(self, wallet=None, wallet_name=None, hotkey_name=None, yaml_config=None):
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.yaml_config = yaml_config
        self.task_center_url = settings.TASK_CENTER_URL
        if yaml_config:
            self.task_center_url = yaml_config.get_task_center_url() or self.task_center_url

    async def submit_dataset(
        self,
        task_id: str,
        dataset_url: str,
        dataset_description: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.wallet:
            return {"success": False, "error": "Wallet not initialized"}

        miner_hotkey = self.wallet.hotkey.ss58_address

        submission_data = {
            "task_id": task_id,
            "miner_hotkey": miner_hotkey,
            "dataset_url": dataset_url,
            "dataset_description": dataset_description
        }

        submit_url = f"{self.task_center_url}/v1/miners/dataset/submit"
        submit_endpoint = "/v1/miners/dataset/submit"

        last_error = None
        for attempt in range(1, DATASET_SUBMIT_MAX_RETRIES + 1):
            try:
                logger.info(f"Submitting dataset for task {task_id} (attempt {attempt}/{DATASET_SUBMIT_MAX_RETRIES})")

                async with httpx.AsyncClient(timeout=60.0) as client:
                    signature_auth = SignatureAuth(self.wallet)
                    nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
                    auth_headers = signature_auth.create_auth_headers_with_nonce(submit_endpoint, nonce)
                    response = await client.post(
                        submit_url,
                        json=submission_data,
                        headers={**auth_headers, "Content-Type": "application/json"}
                    )

                    if response.status_code >= 400:
                        error_body = response.text[:500]
                        logger.warning(f"Dataset submission failed with status {response.status_code}: {error_body}")
                        last_error = f"HTTP {response.status_code}: {error_body}"
                        if attempt < DATASET_SUBMIT_MAX_RETRIES:
                            await asyncio.sleep(DATASET_SUBMIT_RETRY_DELAY)
                            continue
                        else:
                            return {"success": False, "error": last_error}

                    result_data = response.json()
                    logger.info(f"Dataset submitted successfully for task {task_id}: submission_id={result_data.get('submission_id')}")

                    return {
                        "success": True,
                        "submission_id": result_data.get("submission_id"),
                        "status": result_data.get("status"),
                        "message": result_data.get("message")
                    }

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                logger.warning(f"Dataset submission timeout (attempt {attempt}/{DATASET_SUBMIT_MAX_RETRIES})")
            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                logger.warning(f"Dataset submission request error (attempt {attempt}/{DATASET_SUBMIT_MAX_RETRIES}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected dataset submission error (attempt {attempt}/{DATASET_SUBMIT_MAX_RETRIES}): {e}", exc_info=True)

            if attempt < DATASET_SUBMIT_MAX_RETRIES:
                await asyncio.sleep(DATASET_SUBMIT_RETRY_DELAY)

        return {"success": False, "error": last_error}

    async def check_validation_status(self, task_id: str) -> Dict[str, Any]:
        if not self.wallet:
            return {"status": "error", "error": "Wallet not initialized"}

        miner_hotkey = self.wallet.hotkey.ss58_address

        status_url = f"{self.task_center_url}/v1/miners/dataset/status"
        status_endpoint = "/v1/miners/dataset/status"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                signature_auth = SignatureAuth(self.wallet)
                nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
                auth_headers = signature_auth.create_auth_headers_with_nonce(status_endpoint, nonce)
                response = await client.get(
                    status_url,
                    params={"task_id": task_id, "miner_hotkey": miner_hotkey},
                    headers=auth_headers
                )

                if response.status_code == 404:
                    return {"status": "not_found", "is_validated": False}

                if response.status_code != 200:
                    return {"status": "error", "error": f"HTTP {response.status_code}"}

                result = response.json()
                return {
                    "status": result.get("validation_status", "pending"),
                    "is_validated": result.get("validation_status") == "approved",
                    "validated_by": result.get("validated_by"),
                    "validation_result": result.get("validation_result"),
                    "rejection_reason": result.get("rejection_reason")
                }

        except Exception as e:
            logger.error(f"Error checking dataset validation status: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def wait_for_validation(
        self,
        task_id: str,
        max_wait_time: int = DATASET_VALIDATION_MAX_WAIT_TIME
    ) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)

        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed >= max_wait_time:
                logger.warning(f"Dataset validation timeout for task {task_id}")
                return {"is_validated": False, "status": "timeout", "error": "Validation timeout"}

            status = await self.check_validation_status(task_id)

            if status.get("status") == "approved":
                logger.info(f"Dataset approved for task {task_id}")
                return {"is_validated": True, "status": "approved", "validated_by": status.get("validated_by")}

            if status.get("status") == "rejected":
                logger.warning(f"Dataset rejected for task {task_id}: {status.get('rejection_reason')}")
                return {
                    "is_validated": False,
                    "status": "rejected",
                    "rejection_reason": status.get("rejection_reason")
                }

            if status.get("status") == "error":
                logger.warning(f"Error checking validation status: {status.get('error')}")

            logger.debug(f"Waiting for dataset validation for task {task_id}, status={status.get('status')}")
            await asyncio.sleep(DATASET_VALIDATION_CHECK_INTERVAL)

    async def submit_and_wait_for_validation(
        self,
        task_id: str,
        dataset_url: str,
        dataset_description: Optional[str] = None,
        max_wait_time: int = DATASET_VALIDATION_MAX_WAIT_TIME
    ) -> Dict[str, Any]:
        submit_result = await self.submit_dataset(task_id, dataset_url, dataset_description)

        if not submit_result.get("success"):
            return {
                "is_validated": False,
                "status": "submission_failed",
                "error": submit_result.get("error")
            }

        logger.info(f"Dataset submitted for task {task_id}, waiting for validation...")

        return await self.wait_for_validation(task_id, max_wait_time)
