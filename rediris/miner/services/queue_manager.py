import asyncio
import concurrent.futures
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import secrets
import time
import httpx
from rediris.common.utils.logging import setup_logger
from rediris.miner.services.training_service import TrainingService
from rediris.miner.services.gpu_manager import GPUManager
from rediris.common.config import settings

logger = setup_logger(__name__)

_training_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="training")

SUBMISSION_MAX_RETRIES = 3
SUBMISSION_RETRY_DELAY = 5
PHASE_CHECK_INTERVAL = 60
PHASE_WAIT_MAX_TIME = 86400


class TaskPriority(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_MANUAL_UPLOAD = "waiting_manual_upload"


class QueuedTask:
    def __init__(self, task_id: str, priority: TaskPriority, task_type: str, task_data: Dict[str, Any], miner_key: str = None):
        self.task_id = task_id
        self.priority = priority
        self.task_type = task_type
        self.task_data = task_data
        self.miner_key = miner_key
        self.status = TaskStatus.PENDING
        self.enqueued_at = datetime.now(timezone.utc)
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.submission_status = None
        self.submission_error = None


class QueueManager:
    def __init__(self, max_queue_size: int = 100, max_training_jobs: int = 2, max_test_jobs: int = 4):
        self.high_priority_queue = asyncio.Queue()
        self.medium_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        self.running_tasks: Dict[str, QueuedTask] = {}
        self.completed_tasks: Dict[str, QueuedTask] = {}
        self.max_queue_size = max_queue_size
        self.max_training_jobs = max_training_jobs
        self.max_test_jobs = max_test_jobs
        self.scheduler_running = False
        self.training_service = None
        self.gpu_manager = None

    async def enqueue_task(self, task_data: Dict[str, Any]):
        task_id = task_data.get("task_id")
        workflow_type = task_data.get("workflow_type", "")
        logger.info(f"[QueueManager] enqueue_task() called - task_id: {task_id}, workflow_type: {workflow_type}")

        if self.get_total_queue_size() >= self.max_queue_size:
            raise Exception("Task queue is full")

        from rediris.common.models.workflow_type import WorkflowType

        try:
            workflow_type_enum = WorkflowType(workflow_type)
            if workflow_type_enum == WorkflowType.TEXT_LORA_CREATION:
                task_type = "text_lora_training"
                priority = TaskPriority.MEDIUM
            elif workflow_type_enum == WorkflowType.IMAGE_LORA_CREATION:
                task_type = "image_lora_training"
                priority = TaskPriority.MEDIUM
            else:
                task_type = "unknown"
                priority = TaskPriority.LOW
        except ValueError:
            logger.warning(f"Unknown workflow type: {workflow_type}, using LOW priority")
            task_type = "unknown"
            priority = TaskPriority.LOW

        miner_key = None
        try:
            from rediris.miner import shared
            if shared.wallet:
                miner_key = shared.wallet.hotkey.ss58_address
        except Exception as e:
            logger.warning(f"Failed to get miner_key from wallet: {e}")

        queue_task = QueuedTask(task_id, priority, task_type, task_data, miner_key=miner_key)

        if priority == TaskPriority.HIGH:
            await self.high_priority_queue.put(queue_task)
        elif priority == TaskPriority.MEDIUM:
            await self.medium_priority_queue.put(queue_task)
        else:
            await self.low_priority_queue.put(queue_task)

        logger.info(f"Task {task_id} enqueued with priority {priority}, miner_key={miner_key[:20] if miner_key else 'None'}...")
    
    def get_total_queue_size(self) -> int:
        return (
            self.high_priority_queue.qsize() +
            self.medium_priority_queue.qsize() +
            self.low_priority_queue.qsize()
        )
    
    def get_queue_length(self) -> int:
        return self.get_total_queue_size()
    
    def get_running_tasks_count(self) -> int:
        return len(self.running_tasks)
    
    async def start_scheduler(self):
        if self.scheduler_running:
            logger.warning("Scheduler is already running")
            return

        self.scheduler_running = True
        logger.info(f"Starting scheduler with gpu_manager={self.gpu_manager is not None}, training_service={self.training_service is not None}")
        if self.gpu_manager:
            logger.info(f"GPU count: {len(self.gpu_manager.gpus)}, available: {self.gpu_manager.get_available_gpu_count()}")
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Queue scheduler started successfully")
    
    async def stop_scheduler(self):
        if not self.scheduler_running:
            return
        
        self.scheduler_running = False
        
        if hasattr(self, '_scheduler_task') and self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Queue scheduler stopped")
    
    async def _scheduler_loop(self):
        logger.info("Scheduler loop started")
        loop_count = 0
        while self.scheduler_running:
            try:
                loop_count += 1
                await self._process_queue()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_queue(self):
        if self.gpu_manager is None:
            from rediris.miner.services.gpu_manager import GPUManager
            self.gpu_manager = GPUManager()
            logger.warning(f"GPU manager was None, created new one with {self.gpu_manager.get_available_gpu_count()} GPUs")

        if self.training_service is None:
            from rediris.miner.services.training_service import TrainingService
            self.training_service = TrainingService()
            logger.warning("Training service was None, created new one")

        available_training_workers = self.gpu_manager.get_available_gpu_count()
        queue_size = self.get_total_queue_size()

        if queue_size > 0:
            logger.info(f"Queue status: {queue_size} tasks pending, {available_training_workers} GPUs available")

        if available_training_workers == 0:
            if queue_size > 0:
                logger.warning(f"No available GPUs, {queue_size} tasks waiting")
            return

        task = None

        try:
            task = self.high_priority_queue.get_nowait()
            logger.info(f"Got task from HIGH priority queue: {task.task_id}")
        except asyncio.QueueEmpty:
            try:
                task = self.medium_priority_queue.get_nowait()
                logger.info(f"Got task from MEDIUM priority queue: {task.task_id}")
            except asyncio.QueueEmpty:
                try:
                    task = self.low_priority_queue.get_nowait()
                    logger.info(f"Got task from LOW priority queue: {task.task_id}")
                except asyncio.QueueEmpty:
                    pass

        if task is None:
            return

        if task.task_type in ["text_lora_training", "image_lora_training"]:
            training_count = len([
                t for t in self.running_tasks.values()
                if t.task_type in ["text_lora_training", "image_lora_training"]
            ])
            if training_count >= self.max_training_jobs:
                logger.info(f"Max training jobs reached ({training_count}/{self.max_training_jobs}), putting task {task.task_id} back")
                await self._put_back(task)
                return

        gpu_id = self.gpu_manager.allocate_gpu(task.task_type)
        if gpu_id is None:
            logger.warning(f"Could not allocate GPU for task {task.task_id}, putting back")
            await self._put_back(task)
            return

        logger.info(f"Starting task {task.task_id} on GPU {gpu_id}")
        asyncio.create_task(self._execute_task(task, gpu_id))
    
    async def _put_back(self, task: QueuedTask):
        if task.priority == TaskPriority.HIGH:
            await self.high_priority_queue.put(task)
        elif task.priority == TaskPriority.MEDIUM:
            await self.medium_priority_queue.put(task)
        else:
            await self.low_priority_queue.put(task)
    
    async def _execute_task(self, task: QueuedTask, gpu_id: int):
        logger.info(f"Executing task {task.task_id} on GPU {gpu_id}")
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now(timezone.utc)
        self.running_tasks[task.task_id] = task

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _training_executor,
                self._run_training_sync,
                task,
                gpu_id
            )

            if result.get("status") == "failed":
                task.status = TaskStatus.FAILED
                task.error = result.get("error", "Unknown error")
                logger.error(f"Task {task.task_id} failed: {task.error}")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                logger.info(f"Task {task.task_id} completed successfully")

                if task.miner_key and result.get("model_path"):
                    workflow_type = task.task_data.get("workflow_type", "")

                    logger.info(f"Task {task.task_id}: Step 1 - Starting local testing")
                    test_result = await self._test_model_locally(task, result, workflow_type)

                    if not test_result.get("test_passed", False):
                        logger.warning(f"Task {task.task_id}: Local testing failed, skipping HF upload and submission")
                        task.submission_status = "test_failed"
                        task.submission_error = test_result.get("error", "Local test failed")
                    else:
                        logger.info(f"Task {task.task_id}: Local testing passed")
                        result["test_results"] = test_result.get("test_results", [])

                        logger.info(f"Task {task.task_id}: Step 2 - Uploading to HuggingFace")
                        upload_result = await self._upload_to_huggingface(task, result, workflow_type)

                        if not upload_result.get("success", False):
                            logger.error(f"Task {task.task_id}: HuggingFace upload failed, skipping submission")
                            task.submission_status = "upload_failed"
                            task.submission_error = upload_result.get("error", "HF upload failed")
                        else:
                            result["model_url"] = upload_result.get("model_url")
                            logger.info(f"Task {task.task_id}: HuggingFace upload completed")

                            logger.info(f"Task {task.task_id}: Step 3 - Submitting to task center")
                            await self._submit_to_task_center(task, result)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)

        finally:
            task.completed_at = datetime.now(timezone.utc)
            self.gpu_manager.release_gpu(gpu_id)
            self.running_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task

    def _run_training_sync(self, task: QueuedTask, gpu_id: int) -> Dict[str, Any]:
        import asyncio

        logger.info(f"Running training for task {task.task_id} on GPU {gpu_id}")

        try:
            from rediris.common.models.workflow_type import WorkflowType

            workflow_type = task.task_data.get("workflow_type", "")
            logger.debug(f"Task {task.task_id} workflow_type: {workflow_type}")

            workflow_type_enum = WorkflowType(workflow_type)
            logger.info(f"[QueueManager] Task {task.task_id} workflow_type: {workflow_type}, workflow_type_enum: {workflow_type_enum}")

            logger.info(f"[QueueManager] Starting training for task {task.task_id}, workflow_type: {workflow_type}")
            logger.debug(f"[QueueManager] Task data: {task.task_data}")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                logger.info(f"[QueueManager] Calling training_service.train() for task {task.task_id}")
                result = loop.run_until_complete(self.training_service.train(task.task_data))
                logger.info(f"[QueueManager] training_service.train() returned for task {task.task_id}")
            finally:
                loop.close()

            logger.info(f"[QueueManager] Training completed for task {task.task_id}, result keys: {result.keys() if result else 'None'}")
            return result

        except ValueError as ve:
            logger.error(f"ValueError for task {task.task_id}: {ve}")
            return {"status": "failed", "error": str(ve)}
        except Exception as e:
            logger.error(f"Training failed for task {task.task_id}: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    async def _upload_to_huggingface(self, task: QueuedTask, result: Dict[str, Any], workflow_type: str) -> Dict[str, Any]:

        try:
            # Optional: allow skipping HuggingFace uploads (debug / air-gapped runs).
            # When enabled, we pretend upload succeeded and keep the rest of the pipeline unchanged.
            skip_upload = False
            try:
                from rediris.miner import shared
                if shared.yaml_config:
                    skip_upload = bool(shared.yaml_config.get("huggingface.skip_upload", False))
            except Exception:
                pass

            if skip_upload:
                import os
                model_path = result.get("model_path", "")
                model_url = result.get("model_url")
                if not model_url:
                    model_url = f"file://{os.path.abspath(model_path)}" if model_path else "file://"

                return {"success": True, "model_url": model_url, "repo_id": None, "skipped": True}

            from huggingface_hub import HfApi
            try:
                from huggingface_hub.errors import HfHubHTTPError
            except ImportError:
                from huggingface_hub.utils import HfHubHTTPError
            from rediris.common.models.workflow_type import WorkflowType
            import re
            import os
            import secrets
            import string

            model_path = result.get("model_path", "")
            if not model_path or not os.path.exists(model_path):
                return {"success": False, "model_url": "", "error": f"Model path not found: {model_path}"}

            yaml_config = None
            try:
                from rediris.miner import shared
                yaml_config = shared.yaml_config
            except Exception:
                pass

            hf_token = None
            if yaml_config:
                hf_token = yaml_config.get("huggingface.token")
            if not hf_token:
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

            if not hf_token:
                return {"success": False, "model_url": "", "error": "No HuggingFace token available"}

            api = HfApi(token=hf_token)

            existing_model_url = result.get("model_url")
            repo_id = None
            if existing_model_url:
                if existing_model_url.startswith("https://huggingface.co/"):
                    potential_repo_id = existing_model_url.replace("https://huggingface.co/", "").strip("/")
                    if "/" in potential_repo_id and len(potential_repo_id.split("/")) == 2:
                        try:
                            api.model_info(repo_id=potential_repo_id, token=hf_token)
                            logger.info(f"Task {task.task_id}: Model already exists on HuggingFace: {existing_model_url}")
                            return {"success": True, "model_url": existing_model_url, "repo_id": potential_repo_id}
                        except HfHubHTTPError as e:
                            if e.response.status_code == 404:
                                logger.info(f"Task {task.task_id}: Model URL provided but not found on HuggingFace, will upload: {existing_model_url}")
                                repo_id = potential_repo_id
                            else:
                                logger.warning(f"Task {task.task_id}: Error checking existing model: {e}, will create new repo")
                        except Exception as e:
                            logger.warning(f"Task {task.task_id}: Error checking existing model: {e}, will create new repo")

            if not repo_id:
                username = None
                if yaml_config:
                    username = yaml_config.get("huggingface.username")
                if not username:
                    try:
                        who = api.whoami(token=hf_token)
                        username = who.get("name")
                        if not username:
                            email = who.get("email", "")
                            if email and "@" in email:
                                username = email.split("@")[0]
                            else:
                                username = email
                    except Exception as e:
                        logger.warning(f"Failed to get HuggingFace username: {e}")

                if not username:
                    return {"success": False, "model_url": "", "error": "Unable to resolve HuggingFace username"}

                username = re.sub(r'[^a-zA-Z0-9\-_.]', '', username)
                if not username:
                    username = "user"

                random_chars = secrets.choice(string.ascii_uppercase)
                random_chars += ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(15))
                
                try:
                    workflow_type_enum = WorkflowType(workflow_type)
                    if workflow_type_enum == WorkflowType.TEXT_LORA_CREATION:
                        repo_id = f"{username}/{random_chars}"
                    elif workflow_type_enum == WorkflowType.IMAGE_LORA_CREATION:
                        repo_id = f"{username}/{random_chars}"
                    else:
                        repo_id = f"{username}/{random_chars}"
                except ValueError:
                    repo_id = f"{username}/Lora{random_chars}"

            try:
                api.model_info(repo_id=repo_id, token=hf_token)
                model_url = f"https://huggingface.co/{repo_id}"
                logger.info(f"Task {task.task_id}: Model already exists on HuggingFace: {model_url}")
                return {"success": True, "model_url": model_url, "repo_id": repo_id}
            except HfHubHTTPError as e:
                if e.response.status_code == 404:
                    logger.info(f"Task {task.task_id}: Model does not exist, will upload to HuggingFace repo {repo_id}")
                else:
                    logger.warning(f"Task {task.task_id}: Error checking model existence: {e}")
            except Exception as e:
                logger.warning(f"Task {task.task_id}: Error checking model existence: {e}")

            private_repo = False
            try:
                api.create_repo(repo_id=repo_id, private=private_repo, exist_ok=True)
            except Exception as e:
                logger.error(f"Task {task.task_id}: Failed to create HuggingFace repo {repo_id}: {e}", exc_info=True)
                return {"success": False, "model_url": "", "error": str(e)}

            try:
                logger.info(f"Task {task.task_id}: Uploading all files from {model_path} to HuggingFace repo {repo_id} (private={private_repo})")
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=model_path,
                    commit_message=f"Upload model for task {task.task_id}"
                )
            except Exception as e:
                logger.error(f"Task {task.task_id}: Failed to upload files to HuggingFace repo {repo_id}: {e}", exc_info=True)
                return {"success": False, "model_url": "", "error": str(e)}

            model_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"Task {task.task_id}: Upload complete: {model_url}")

            return {"success": True, "model_url": model_url, "repo_id": repo_id}

        except Exception as e:
            logger.error(f"Task {task.task_id}: HuggingFace upload failed: {e}", exc_info=True)
            return {"success": False, "model_url": "", "error": str(e)}

    async def _submit_to_task_center(self, task: QueuedTask, result: Dict[str, Any]):
        workflow_type = task.task_data.get("workflow_type", "")
        training_mode = result.get("training_mode", "new")
        model_url = result.get("model_url", "")

        if not model_url:
            logger.error(f"Task {task.task_id}: Cannot submit - no model_url in result")
            task.submission_status = "failed"
            task.submission_error = "No model_url in training result"
            return

        if not task.miner_key:
            logger.error(f"Task {task.task_id}: Cannot submit - no miner_key")
            task.submission_status = "failed"
            task.submission_error = "No miner_key available"
            return

        model_metadata = {
            "training_steps": result.get("training_steps", 0),
            "final_loss": result.get("final_loss", 0.0),
            "model_path": result.get("model_path", ""),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

        test_results = result.get("test_results", [])
        if test_results:
            model_metadata["local_test_results"] = test_results
            model_metadata["local_test_passed"] = all(r.get("test_passed", False) for r in test_results)

        submission_data = {
            "task_id": task.task_id,
            "miner_key": task.miner_key,
            "training_mode": training_mode,
            "model_url": model_url,
            "model_metadata": model_metadata,
            "sample_images": None
        }

        task_center_url = settings.TASK_CENTER_URL
        try:
            from rediris.miner import shared
            if shared.yaml_config:
                task_center_url = shared.yaml_config.get_task_center_url() or task_center_url
        except Exception:
            pass

        await self._wait_for_submission_window(task, task_center_url)

        submit_url = f"{task_center_url}/v1/miners/submit"
        submit_endpoint = "/v1/miners/submit"

        last_error = None
        for attempt in range(1, SUBMISSION_MAX_RETRIES + 1):
            try:
                logger.info(f"Task {task.task_id}: Submitting to task center (attempt {attempt}/{SUBMISSION_MAX_RETRIES})")

                async with httpx.AsyncClient(timeout=60.0) as client:
                    headers = {"Content-Type": "application/json"}
                    try:
                        from rediris.miner import shared
                        if shared.wallet:
                            from rediris.common.crypto.signature import SignatureAuth
                            signature_auth = SignatureAuth(shared.wallet)
                            nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
                            auth_headers = signature_auth.create_auth_headers_with_nonce(submit_endpoint, nonce)
                            headers = {**auth_headers, "Content-Type": "application/json"}
                    except Exception:
                        pass
                    response = await client.post(
                        submit_url,
                        json=submission_data,
                        headers=headers
                    )

                    if response.status_code >= 400:
                        error_body = response.text[:500]
                        logger.warning(f"Task {task.task_id}: Submission failed with status {response.status_code}: {error_body}")
                        last_error = f"HTTP {response.status_code}: {error_body}"
                        if attempt < SUBMISSION_MAX_RETRIES:
                            await asyncio.sleep(SUBMISSION_RETRY_DELAY)
                            continue
                        else:
                            raise httpx.HTTPStatusError(
                                message=last_error,
                                request=response.request,
                                response=response
                            )

                    response.raise_for_status()
                    result_data = response.json()

                    task.submission_status = "submitted"
                    logger.info(f"Task {task.task_id}: Successfully submitted to task center. "
                               f"submission_id={result_data.get('submission_id')}, "
                               f"status={result_data.get('status')}, "
                               f"estimated_reward={result_data.get('estimated_reward')}")
                    return

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                logger.warning(f"Task {task.task_id}: Submission timeout (attempt {attempt}/{SUBMISSION_MAX_RETRIES})")
            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                logger.warning(f"Task {task.task_id}: Submission request error (attempt {attempt}/{SUBMISSION_MAX_RETRIES}): {e}")
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                logger.warning(f"Task {task.task_id}: Submission HTTP error (attempt {attempt}/{SUBMISSION_MAX_RETRIES}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Task {task.task_id}: Unexpected submission error (attempt {attempt}/{SUBMISSION_MAX_RETRIES}): {e}", exc_info=True)

            if attempt < SUBMISSION_MAX_RETRIES:
                await asyncio.sleep(SUBMISSION_RETRY_DELAY)

        task.submission_status = "failed"
        task.submission_error = last_error
        logger.error(f"Task {task.task_id}: Submission failed after {SUBMISSION_MAX_RETRIES} attempts. Last error: {last_error}")

    async def _wait_for_submission_window(self, task: QueuedTask, task_center_url: str):
        phase_url = f"{task_center_url}/v1/tasks/{task.task_id}/phase"
        phase_endpoint = f"/v1/tasks/{task.task_id}/phase"

        execution_start_str = task.task_data.get("execution_start")
        if execution_start_str:
            try:
                if isinstance(execution_start_str, str):
                    execution_start = datetime.fromisoformat(execution_start_str.replace('Z', '+00:00'))
                else:
                    execution_start = execution_start_str
            except Exception as e:
                logger.warning(f"Task {task.task_id}: Failed to parse execution_start '{execution_start_str}': {e}, using current time")
                execution_start = datetime.now(timezone.utc)
        else:
            logger.warning(f"Task {task.task_id}: No execution_start in task data, using current time")
            execution_start = datetime.now(timezone.utc)

        signature_auth = None
        try:
            from rediris.miner import shared
            if shared.wallet:
                from rediris.common.crypto.signature import SignatureAuth
                signature_auth = SignatureAuth(shared.wallet)
        except Exception as e:
            logger.warning(f"Task {task.task_id}: Failed to init signature auth: {e}")

        while True:
            elapsed = (datetime.now(timezone.utc) - execution_start).total_seconds()
            if elapsed >= PHASE_WAIT_MAX_TIME:
                logger.error(f"Task {task.task_id}: {elapsed}s elapsed since execution_start, exceeds max wait time {PHASE_WAIT_MAX_TIME}s, giving up")
                raise Exception("Submission window wait timeout - exceeded max time since execution start")

            try:
                headers = {}
                if signature_auth:
                    nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
                    headers = signature_auth.create_auth_headers_with_nonce(phase_endpoint, nonce)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(phase_url, headers=headers)

                    if response.status_code == 200:
                        phase_info = response.json()
                        phase = phase_info.get("phase", "")
                        can_submit = phase_info.get("can_submit", False)

                        if can_submit:
                            logger.info(f"Task {task.task_id}: Submission window open (phase={phase})")
                            return

                        if phase in ["review", "reward", "ended"]:
                            logger.error(f"Task {task.task_id}: Task phase is {phase}, submission window closed")
                            raise Exception(f"Submission window closed (phase={phase})")

                        time_remaining = phase_info.get("time_remaining", 0)
                        if time_remaining > 0 and time_remaining < PHASE_CHECK_INTERVAL:
                            wait_time = time_remaining + 5
                        else:
                            wait_time = PHASE_CHECK_INTERVAL

                        logger.info(f"Task {task.task_id}: Phase={phase}, waiting {wait_time}s for submission window")
                        await asyncio.sleep(wait_time)

                    elif response.status_code == 404:
                        logger.warning(f"Task {task.task_id}: Task not found, waiting...")
                        await asyncio.sleep(PHASE_CHECK_INTERVAL)
                    else:
                        logger.warning(f"Task {task.task_id}: Phase check failed with status {response.status_code}")
                        await asyncio.sleep(PHASE_CHECK_INTERVAL)

            except httpx.RequestError as e:
                logger.warning(f"Task {task.task_id}: Phase check request error: {e}")
                await asyncio.sleep(PHASE_CHECK_INTERVAL)
            except Exception as e:
                if "Submission window" in str(e):
                    raise
                logger.warning(f"Task {task.task_id}: Phase check error: {e}")
                await asyncio.sleep(PHASE_CHECK_INTERVAL)

    async def _test_model_locally(self, task: QueuedTask, training_result: Dict[str, Any], workflow_type: str) -> Dict[str, Any]:

        try:
            from rediris.miner.services.inference_service import InferenceService
            from rediris.miner.schemas.inference import InferenceTestRequest, TestCase

            logger.info(f"Task {task.task_id}: task_data keys: {list(task.task_data.keys())}")
            workflow_spec = task.task_data.get("workflow_spec") or {}
            logger.info(f"Task {task.task_id}: workflow_spec keys: {list(workflow_spec.keys()) if workflow_spec else 'None'}")
            test_spec = workflow_spec.get("test_spec") or {}
            logger.info(f"Task {task.task_id}: test_spec: {test_spec}")

            if not test_spec:
                logger.info(f"Task {task.task_id}: No test_spec configured, skipping local test")
                return {
                    "test_passed": True,
                    "test_results": [],
                    "skipped": True,
                    "reason": "No test_spec configured in task"
                }

            if not test_spec.get("enabled", True):
                logger.info(f"Task {task.task_id}: Local testing is disabled, skipping")
                return {
                    "test_passed": True,
                    "test_results": [],
                    "skipped": True,
                    "reason": "Testing disabled in task configuration"
                }

            configured_test_cases = test_spec.get("test_cases", [])
            if not configured_test_cases:
                logger.warning(f"Task {task.task_id}: No test cases configured in task center, skipping local test")
                return {
                    "test_passed": True,
                    "test_results": [],
                    "skipped": True,
                    "reason": "No test cases configured"
                }

            quality_threshold = test_spec.get("quality_threshold", 5.0)
            safety_threshold = test_spec.get("safety_threshold", 0.5)

            yaml_config = None
            try:
                from rediris.miner import shared
                yaml_config = shared.yaml_config
            except Exception:
                pass

            inference_service = InferenceService(config=yaml_config)
            inference_service.quality_threshold = quality_threshold
            inference_service.safety_threshold = safety_threshold

            model_path = training_result.get("model_path", "")

            test_cases = []
            for tc in configured_test_cases:
                test_cases.append(TestCase(
                    prompt=tc.get("prompt", ""),
                    seed=tc.get("seed", 42),
                    inference_steps=tc.get("inference_steps", 30),
                    guidance_scale=tc.get("guidance_scale", 7.0)
                ))

            test_request = InferenceTestRequest(
                model_url=model_path,
                test_cases=test_cases
            )

            logger.info(f"Task {task.task_id}: Running {len(test_cases)} test cases from task center config")
            logger.info(f"Task {task.task_id}: Quality threshold: {quality_threshold}, Safety threshold: {safety_threshold}")

            test_results = await inference_service.test_lora(test_request, workflow_type)

            all_passed = all(r.get("test_passed", False) for r in test_results)
            passed_count = sum(1 for r in test_results if r.get("test_passed", False))

            logger.info(f"Task {task.task_id}: Local testing completed - {passed_count}/{len(test_results)} tests passed")

            return {
                "test_passed": all_passed,
                "test_results": test_results,
                "passed_count": passed_count,
                "total_count": len(test_results),
                "quality_threshold": quality_threshold,
                "safety_threshold": safety_threshold
            }

        except Exception as e:
            logger.error(f"Task {task.task_id}: Local testing failed: {e}", exc_info=True)
            return {
                "test_passed": False,
                "error": str(e),
                "test_results": []
            }
    
    def get_queue_stats(self) -> Dict[str, Any]:
        submitted_count = len([t for t in self.completed_tasks.values() if t.submission_status == "submitted"])
        submission_failed_count = len([t for t in self.completed_tasks.values() if t.submission_status == "failed"])
        upload_skipped_count = len([t for t in self.completed_tasks.values() if t.submission_status == "upload_skipped"])

        return {
            "high_priority_queue_length": self.high_priority_queue.qsize(),
            "medium_priority_queue_length": self.medium_priority_queue.qsize(),
            "low_priority_queue_length": self.low_priority_queue.qsize(),
            "total_queue_length": self.get_total_queue_size(),
            "running_training_tasks": len([t for t in self.running_tasks.values() if t.task_type in ["text_lora_training", "image_lora_training"]]),
            "running_inference_tasks": len([t for t in self.running_tasks.values() if t.task_type == "inference"]),
            "completed_tasks": len(self.completed_tasks),
            "submitted_to_task_center": submitted_count,
            "submission_failed": submission_failed_count,
            "upload_skipped": upload_skipped_count,
            "available_workers": self.gpu_manager.get_available_gpu_count(),
            "gpu_utilization": self.gpu_manager.get_gpu_utilization()
        }

