import time
from typing import Dict, Optional

try:
    import pandas as pd
    if not hasattr(pd.io.json, 'json_normalize'):
        try:
            from pandas import json_normalize
            pd.io.json.json_normalize = json_normalize
        except ImportError:
            pass
except (ImportError, AttributeError):
    pass

import bittensor as bt
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)


class SignatureAuth:
    def __init__(self, wallet: bt.wallet):
        self.wallet = wallet
    
    def sign_message(self, message: str, timestamp: Optional[int] = None) -> Dict[str, str]:
        if timestamp is None:
            timestamp = int(time.time())
        
        message_with_timestamp = f"{message}:{timestamp}"
        
        signature = self.wallet.hotkey.sign(message_with_timestamp.encode())
        signature_hex = signature.hex()
        
        return {
            "signature": signature_hex,
            "timestamp": str(timestamp),
            "hotkey": self.wallet.hotkey.ss58_address,
            "message": message
        }
    
    def verify_signature(
        self,
        signature: str,
        message: str,
        timestamp: str,
        hotkey: str
    ) -> bool:
        try:
            message_with_timestamp = f"{message}:{timestamp}"
            
            hotkey_obj = bt.Keypair(ss58_address=hotkey)
            
            signature_bytes = bytes.fromhex(signature)
            
            is_valid = hotkey_obj.verify(
                message_with_timestamp.encode(),
                signature_bytes
            )
            
            if not is_valid:
                return False
            
            current_time = int(time.time())
            request_time = int(timestamp)
            
            max_age = 300
            if abs(current_time - request_time) > max_age:
                logger.warning(f"Signature timestamp expired: {current_time - request_time}s (max: {max_age}s)")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}", exc_info=True)
            return False
    
    def create_auth_headers(self, endpoint: str) -> Dict[str, str]:
        timestamp = int(time.time())
        message = f"{endpoint}:{timestamp}"
        
        signature_data = self.sign_message(message, timestamp)
        
        return {
            "X-Signature": signature_data["signature"],
            "X-Timestamp": signature_data["timestamp"],
            "X-Hotkey": signature_data["hotkey"],
            "X-Message": signature_data["message"]
        }
    
    def create_auth_headers_with_nonce(
        self,
        endpoint: str,
        nonce: str
    ) -> Dict[str, str]:
        timestamp = int(time.time())
        message = f"{endpoint}:{timestamp}:{nonce}"
        
        signature_data = self.sign_message(message, timestamp)
        
        return {
            "X-Signature": signature_data["signature"],
            "X-Timestamp": signature_data["timestamp"],
            "X-Hotkey": signature_data["hotkey"],
            "X-Message": message,
            "X-Nonce": nonce
        }
    
    def sign_response(self, response_data: Dict) -> Dict:
        timestamp = int(time.time())
        response_str = f"{response_data.get('status', '')}:{response_data.get('hotkey', '')}:{timestamp}"
        
        signature = self.wallet.hotkey.sign(response_str.encode())
        
        response_data["signature"] = signature.hex()
        response_data["timestamp"] = str(timestamp)
        response_data["hotkey"] = self.wallet.hotkey.ss58_address
        
        return response_data
    
    def verify_response(self, response_data: Dict) -> bool:
        try:
            if "signature" not in response_data or "hotkey" not in response_data or "timestamp" not in response_data:
                return False
            
            signature = response_data.get("signature")
            hotkey = response_data.get("hotkey")
            timestamp = response_data.get("timestamp")
            status = response_data.get("status", "")
            response_hotkey = response_data.get("hotkey", "")
            
            response_str = f"{status}:{response_hotkey}:{timestamp}"
            
            hotkey_obj = bt.Keypair(ss58_address=hotkey)
            signature_bytes = bytes.fromhex(signature)
            
            is_valid = hotkey_obj.verify(
                response_str.encode(),
                signature_bytes
            )
            
            if not is_valid:
                return False
            
            current_time = int(time.time())
            response_time = int(timestamp)
            
            max_age = 300
            if abs(current_time - response_time) > max_age:
                logger.warning(f"Response signature timestamp expired: {current_time - response_time}s (max: {max_age}s)")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Response signature verification failed: {e}", exc_info=True)
            return False
