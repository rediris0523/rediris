from fastapi import APIRouter
from rediris.task_center.api import tasks, miners, validators, audit, scores

router = APIRouter()

router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
router.include_router(miners.router, prefix="/miners", tags=["miners"])
router.include_router(validators.router, prefix="/validators", tags=["validators"])
router.include_router(audit.router, prefix="/audit", tags=["audit"])
router.include_router(scores.router, prefix="/scores", tags=["scores"])

