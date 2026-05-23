from fastapi import APIRouter, HTTPException
from rediris.miner.services.inference_service import InferenceService
from rediris.miner.schemas.inference import InferenceTestRequest, InferenceTestResponse
from rediris.common.utils.logging import setup_logger

router = APIRouter()
logger = setup_logger(__name__)
inference_service = InferenceService()

@router.post("/test", response_model=InferenceTestResponse)
async def test_inference(request: InferenceTestRequest):
    try:
        results = await inference_service.test_lora(request)
        return InferenceTestResponse(results=results)
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

