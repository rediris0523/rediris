from fastapi import Header, HTTPException, Request, Depends
from rediris.common.crypto import SignatureAuth
from rediris.common.bittensor.client import BittensorClient
from rediris.common.utils.logging import setup_logger
from rediris.common.services.rate_limiter import RateLimiter
from rediris.common.services.nonce_manager import NonceManager
from rediris.common.services.hotkey_validator import HotkeyValidator
from rediris.common.services.auth_logger import AuthLogger

logger = setup_logger(__name__)

_nonce_manager = None
_rate_limiter = None
_hotkey_validator = None
_auth_logger = None
_bittensor_client = None


def get_nonce_manager() -> NonceManager:
    global _nonce_manager
    if _nonce_manager is None:
        _nonce_manager = NonceManager()
    return _nonce_manager


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_auth_logger() -> AuthLogger:
    global _auth_logger
    if _auth_logger is None:
        _auth_logger = AuthLogger()
    return _auth_logger


def get_bittensor_client() -> BittensorClient:
    global _bittensor_client
    if _bittensor_client is None:
        try:
            from rediris.task_center import shared as task_center_shared
            if task_center_shared.bittensor_client:
                _bittensor_client = task_center_shared.bittensor_client
        except Exception:
            pass
    
    if _bittensor_client is None:
        raise HTTPException(status_code=503, detail="Bittensor client not available")
    
    return _bittensor_client


def get_hotkey_validator() -> HotkeyValidator:
    global _hotkey_validator
    if _hotkey_validator is None:
        bittensor_client = get_bittensor_client()
        _hotkey_validator = HotkeyValidator(bittensor_client)
    return _hotkey_validator


async def verify_node_signature(
    request: Request,
    x_signature: str = Header(..., alias="X-Signature"),
    x_timestamp: str = Header(..., alias="X-Timestamp"),
    x_hotkey: str = Header(..., alias="X-Hotkey"),
    x_nonce: str = Header(..., alias="X-Nonce"),
    x_message: str = Header(..., alias="X-Message"),
    bittensor_client: BittensorClient = Depends(get_bittensor_client),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    nonce_manager: NonceManager = Depends(get_nonce_manager),
    hotkey_validator: HotkeyValidator = Depends(get_hotkey_validator),
    auth_logger: AuthLogger = Depends(get_auth_logger)
) -> str:
    try:
        signature_auth = SignatureAuth(None)
        is_valid = signature_auth.verify_signature(
            signature=x_signature,
            message=x_message,
            timestamp=x_timestamp,
            hotkey=x_hotkey
        )
        
        if not is_valid:
            await auth_logger.log_auth_failure(request, x_hotkey, "invalid_signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        if not await nonce_manager.verify_nonce(x_hotkey, x_nonce, int(x_timestamp)):
            await auth_logger.log_auth_failure(request, x_hotkey, "replay_attack")
            raise HTTPException(status_code=401, detail="Invalid or reused nonce")
        
        node_type = "miner" if "/miners/" in str(request.url.path) else "validator"
        
        if not await hotkey_validator.verify_hotkey_registered(x_hotkey, node_type):
            await auth_logger.log_auth_failure(request, x_hotkey, "unregistered_hotkey")
            raise HTTPException(
                status_code=403,
                detail=f"Hotkey {x_hotkey[:16]}... is not registered on subnet"
            )
        
        if not await rate_limiter.check_rate_limit(x_hotkey):
            await auth_logger.log_auth_failure(request, x_hotkey, "rate_limit_exceeded")
            rate_info = await rate_limiter.get_rate_limit_info(x_hotkey)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. "
                       f"Requests: {rate_info['requests_last_minute']}/{rate_info['limit_per_minute']} per minute, "
                       f"{rate_info['requests_last_hour']}/{rate_info['limit_per_hour']} per hour",
                headers={
                    "X-RateLimit-Limit-PerMinute": str(rate_info['limit_per_minute']),
                    "X-RateLimit-Remaining-PerMinute": str(rate_info['remaining_per_minute']),
                    "X-RateLimit-Limit-PerHour": str(rate_info['limit_per_hour']),
                    "X-RateLimit-Remaining-PerHour": str(rate_info['remaining_per_hour'])
                }
            )
        
        await auth_logger.log_auth_success(request, x_hotkey)
        
        return x_hotkey
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verify_node_signature: {e}", exc_info=True)
        await auth_logger.log_auth_failure(request, x_hotkey if 'x_hotkey' in locals() else "unknown", "internal_error")
        raise HTTPException(status_code=500, detail="Internal server error")
