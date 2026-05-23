from fastapi import APIRouter, HTTPException, Request
from rediris.miner.schemas.workflow import WorkflowReceive, WorkflowReceiveResponse, WorkflowSubmit, WorkflowSubmitResponse
from rediris.common.utils.logging import setup_logger
from rediris.miner import shared
import httpx
from rediris.common.config import settings
import traceback
import secrets
import time
from rediris.common.crypto.signature import SignatureAuth

router = APIRouter()
logger = setup_logger(__name__)

logger.debug(f"Workflows router initialized, routes will be at /v1/workflows/*")


@router.post("/receive", response_model=WorkflowReceiveResponse)
async def receive_workflow(request: WorkflowReceive):

    task_id = None
    try:
        request_data = request.dict() if hasattr(request, 'dict') else str(request)
        task_id = request_data.get('task_id') if isinstance(request_data, dict) else None
        logger.info(f"Received workflow request: task_id={task_id}, type={request.workflow_type}, has_spec={request.workflow_spec is not None}")

        if shared.queue_manager is None:
            error_msg = "Queue manager not initialized"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        if request.workflow_type and request.workflow_spec:
            logger.info(f"Task center pushed complete task data for task_id={task_id}, dataset_url={request.dataset_url}")
            task_data = {
                "task_id": request.task_id,
                "workflow_type": request.workflow_type,
                "workflow_spec": request.workflow_spec,
                "dataset_url": request.dataset_url,
                "announcement_start": request.announcement_start,
                "dataset_validation_start": request.dataset_validation_start,
                "execution_start": request.execution_start,
                "review_start": request.review_start,
                "reward_start": request.reward_start,
                "workflow_end": request.workflow_end,
            }
        else:
            logger.info(f"Fetching task data from task center for task_id={task_id}")
            task_center_url = f"{settings.TASK_CENTER_URL}/v1/miners/receive"

            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    if not shared.wallet:
                        raise HTTPException(status_code=500, detail="Wallet not initialized")
                    signature_auth = SignatureAuth(shared.wallet)
                    nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
                    auth_headers = signature_auth.create_auth_headers_with_nonce("/v1/miners/receive", nonce)
                    response = await client.post(
                        task_center_url,
                        json={"task_id": request.task_id, "miner_key": request.miner_key},
                        headers={**auth_headers, "Content-Type": "application/json"}
                    )

                    if response.status_code >= 400:
                        try:
                            error_body = response.text[:1000]
                            logger.error(f"Task center error response: status={response.status_code}, body={error_body}")
                        except Exception:
                            pass

                    response.raise_for_status()
                    task_data = response.json()

                    logger.info(f"Task center returned task data: task_id={task_data.get('task_id')}, "
                              f"workflow_type={task_data.get('workflow_type')}")

                except httpx.HTTPStatusError as e:
                    error_detail = f"Task center returned error status {e.response.status_code}"
                    try:
                        error_body = e.response.text[:1000]
                        error_detail += f": {error_body}"
                        logger.error(f"HTTP error from task center: {error_detail}")
                    except Exception:
                        logger.error(f"HTTP error from task center: {e}")
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Task center error: {error_detail}"
                    )
                except httpx.TimeoutException as e:
                    error_msg = f"Timeout connecting to task center at {task_center_url}"
                    logger.error(f"{error_msg}: {e}")
                    raise HTTPException(status_code=504, detail=error_msg)
                except httpx.RequestError as e:
                    error_msg = f"Request error connecting to task center at {task_center_url}"
                    logger.error(f"{error_msg}: {e}", exc_info=True)
                    raise HTTPException(status_code=503, detail=error_msg)

        try:
            logger.info(f"Enqueuing task: task_id={task_data.get('task_id')}")
            await shared.queue_manager.enqueue_task(task_data)
            logger.info(f"Task enqueued successfully: task_id={task_data.get('task_id')}")
        except Exception as e:
            error_msg = f"Failed to enqueue task: {str(e)}"
            logger.error(f"{error_msg}", exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)

        try:
            from datetime import datetime

            execution_start = task_data.get('execution_start')
            review_start = task_data.get('review_start')

            if isinstance(execution_start, str):
                deadline = datetime.fromisoformat(execution_start.replace('Z', '+00:00'))
            elif execution_start:
                deadline = execution_start
            else:
                deadline = datetime.now()

            if isinstance(review_start, str):
                review_deadline = datetime.fromisoformat(review_start.replace('Z', '+00:00'))
            elif review_start:
                review_deadline = review_start
            else:
                review_deadline = datetime.now()

            response_obj = WorkflowReceiveResponse(
                task_id=task_data.get('task_id'),
                workflow_type=task_data.get('workflow_type'),
                workflow_spec=task_data.get('workflow_spec', {}),
                deadline=deadline,
                review_deadline=review_deadline
            )
            logger.info(f"Workflow receive completed successfully: task_id={task_data.get('task_id')}")
            return response_obj
        except Exception as e:
            error_msg = f"Failed to create response object: {str(e)}"
            logger.error(f"{error_msg}, task_data keys: {list(task_data.keys()) if isinstance(task_data, dict) else 'N/A'}", exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error in receive_workflow: {str(e)}"
        logger.error(f"{error_msg}, task_id={task_id}", exc_info=True)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )


@router.post("/submit", response_model=WorkflowSubmitResponse)
async def submit_workflow(request: WorkflowSubmit):
    task_id = None
    try:
        request_data = request.dict() if hasattr(request, 'dict') else str(request)
        task_id = request_data.get('task_id') if isinstance(request_data, dict) else None
        logger.info(f"Submitting workflow: task_id={task_id}")
        
        task_center_url = f"{settings.TASK_CENTER_URL}/v1/miners/submit"
        logger.debug(f"Submitting to task center: {task_center_url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                if not shared.wallet:
                    raise HTTPException(status_code=500, detail="Wallet not initialized")
                signature_auth = SignatureAuth(shared.wallet)
                nonce = f"{int(time.time())}_{secrets.token_hex(8)}"
                auth_headers = signature_auth.create_auth_headers_with_nonce("/v1/miners/submit", nonce)
                response = await client.post(
                    task_center_url,
                    json=request.dict() if hasattr(request, 'dict') else request,
                    headers={**auth_headers, "Content-Type": "application/json"}
                )
                
                logger.debug(f"Task center response status: {response.status_code}")
                
                if response.status_code >= 400:
                    try:
                        error_body = response.text[:1000]
                        logger.error(f"Task center error response: status={response.status_code}, body={error_body}")
                    except Exception:
                        pass
                
                response.raise_for_status()
                result = response.json()

                logger.info(f"Workflow submitted successfully: task_id={task_id}")
                return WorkflowSubmitResponse(**result)

            except httpx.HTTPStatusError as e:
                error_detail = f"Task center returned error status {e.response.status_code}"
                try:
                    error_body = e.response.text[:1000]
                    error_detail += f": {error_body}"
                    logger.error(f"HTTP error from task center: {error_detail}")
                except Exception:
                    logger.error(f"HTTP error from task center: {e}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Task center error: {error_detail}"
                )
            except httpx.TimeoutException as e:
                error_msg = f"Timeout connecting to task center at {task_center_url}"
                logger.error(f"{error_msg}: {e}")
                raise HTTPException(status_code=504, detail=error_msg)
            except httpx.RequestError as e:
                error_msg = f"Request error connecting to task center at {task_center_url}"
                logger.error(f"{error_msg}: {e}", exc_info=True)
                raise HTTPException(status_code=503, detail=error_msg)
                
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error in submit_workflow: {str(e)}"
        logger.error(f"{error_msg}, task_id={task_id}", exc_info=True)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )

