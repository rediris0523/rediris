from fastapi import APIRouter, Request, HTTPException, Header
from rediris.common.utils.logging import setup_logger
import bittensor as bt
from rediris.common.crypto.signature import SignatureAuth
from rediris.common.config import settings, load_yaml_config
from datetime import datetime, timezone
import os

router = APIRouter()
logger = setup_logger(__name__)

config_path = os.getenv("MINER_CONFIG", "config.yml")
yaml_config = load_yaml_config(config_path)

if yaml_config:
    wallet_name = yaml_config.get_wallet_name()
    hotkey_name = yaml_config.get_hotkey_name()
else:
    wallet_name = "miner"
    hotkey_name = "default"

wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
signature_auth = SignatureAuth(wallet)


@router.post("/heartbeat")
async def heartbeat(
    request: Request,
    x_signature: str = Header(None, alias="X-Signature"),
    x_timestamp: str = Header(None, alias="X-Timestamp"),
    x_hotkey: str = Header(None, alias="X-Hotkey"),
    x_message: str = Header(None, alias="X-Message")
):
    try:
        if not all([x_signature, x_timestamp, x_hotkey, x_message]):
            raise HTTPException(status_code=401, detail="Missing authentication headers")
        
        endpoint = "/v1/health/heartbeat"
        is_valid = signature_auth.verify_signature(
            signature=x_signature,
            message=x_message,
            timestamp=x_timestamp,
            hotkey=x_hotkey
        )
        
        if not is_valid:
            logger.warning(f"Invalid signature from {x_hotkey}")
            raise HTTPException(status_code=403, detail="Invalid signature")
        
        body = await request.json()
        
        response_data = {
            "status": "online",
            "hotkey": wallet.hotkey.ss58_address,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        signed_response = signature_auth.sign_response(response_data)
        
        return signed_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heartbeat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status")
async def status():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
