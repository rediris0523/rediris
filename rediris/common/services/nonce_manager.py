from typing import Dict
from datetime import datetime
import asyncio
import secrets
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class NonceManager:
    def __init__(self):
        self._nonce_cache: Dict[str, Dict[str, int]] = {}
        self._used_nonces: Dict[str, set] = {}
        self._nonce_window = 300
        self._cleanup_interval = 60
        self._lock = asyncio.Lock()
    
    async def generate_nonce(self, hotkey: str) -> str:
        timestamp = int(datetime.now().timestamp())
        random_part = secrets.token_hex(8)
        nonce = f"{timestamp}_{random_part}"
        
        async with self._lock:
            if hotkey not in self._nonce_cache:
                self._nonce_cache[hotkey] = {}
            self._nonce_cache[hotkey][nonce] = timestamp
        
        return nonce
    
    async def verify_nonce(
        self,
        hotkey: str,
        nonce: str,
        request_timestamp: int
    ) -> bool:
        try:
            nonce_timestamp_str, random_part = nonce.split("_", 1)
            nonce_timestamp = int(nonce_timestamp_str)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid nonce format: {nonce}")
            return False
        
        current_time = int(datetime.now().timestamp())
        if abs(current_time - nonce_timestamp) > self._nonce_window:
            logger.warning(
                f"Nonce timestamp out of window: {nonce_timestamp}, "
                f"current: {current_time}, diff: {abs(current_time - nonce_timestamp)}s"
            )
            return False
        
        if abs(nonce_timestamp - request_timestamp) > 60:
            logger.warning(
                f"Nonce timestamp mismatch: nonce_ts={nonce_timestamp}, "
                f"request_ts={request_timestamp}"
            )
            return False
        
        async with self._lock:
            if hotkey not in self._used_nonces:
                self._used_nonces[hotkey] = set()
            
            if nonce in self._used_nonces[hotkey]:
                logger.warning(f"Nonce already used: {nonce[:20]}...")
                return False
            
            self._used_nonces[hotkey].add(nonce)
            logger.debug(f"Nonce accepted and marked as used: {nonce[:20]}...")
            return True
    
    async def cleanup_expired_nonces(self):
        current_time = int(datetime.now().timestamp())
        async with self._lock:
            for hotkey in list(self._used_nonces.keys()):
                expired_nonces = set()
                for nonce in self._used_nonces[hotkey]:
                    try:
                        nonce_timestamp_str, _ = nonce.split("_", 1)
                        nonce_timestamp = int(nonce_timestamp_str)
                        if abs(current_time - nonce_timestamp) > self._nonce_window:
                            expired_nonces.add(nonce)
                    except (ValueError, AttributeError):
                        expired_nonces.add(nonce)
                
                self._used_nonces[hotkey] -= expired_nonces
                
                if not self._used_nonces[hotkey]:
                    del self._used_nonces[hotkey]