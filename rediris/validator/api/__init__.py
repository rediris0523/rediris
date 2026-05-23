
from fastapi import APIRouter

from rediris.validator.api import audit, scores

router = APIRouter()

router.include_router(audit.router, prefix="/audit", tags=["audit"])

router.include_router(scores.router, prefix="/scores", tags=["scores"])
