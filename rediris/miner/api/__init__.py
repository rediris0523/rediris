from fastapi import APIRouter
from rediris.miner.api import workflows, inference, health, queue

router = APIRouter()

router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
router.include_router(inference.router, prefix="/inference", tags=["inference"])
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(queue.router, prefix="/queue", tags=["queue"])

