import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)


class MinerCache:
    def __init__(self, heartbeat_timeout: int = 120):
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._last_update: Optional[datetime] = None
        self.heartbeat_timeout = heartbeat_timeout
    
    def update_miner(self, hotkey: str, miner_data: Dict):
        with self._lock:
            self._cache[hotkey] = {
                "hotkey": hotkey,
                "stake": miner_data.get("stake", 0.0),
                "reputation": miner_data.get("reputation", 0.0),
                "is_active": miner_data.get("is_active", False),
                "is_online": miner_data.get("is_online", False),
                "miner_url": miner_data.get("miner_url"),
                "last_heartbeat": miner_data.get("last_heartbeat"),
                "updated_at": datetime.now(timezone.utc)
            }
    
    def get_miner(self, hotkey: str) -> Optional[Dict]:
        with self._lock:
            return self._cache.get(hotkey)
    
    def get_online_miners(self) -> List[Dict]:
        with self._lock:
            now = datetime.now(timezone.utc)
            online_miners = []
            
            for hotkey, miner_data in self._cache.items():
                if not miner_data.get("is_online", False):
                    continue
                
                last_heartbeat = miner_data.get("last_heartbeat")
                if last_heartbeat:
                    time_since_heartbeat = (now - last_heartbeat).total_seconds()
                    if time_since_heartbeat > self.heartbeat_timeout :
                        continue
                
                online_miners.append(miner_data)
            
            return online_miners
    
    def get_online_miner_hotkeys(self) -> List[str]:
        online_miners = self.get_online_miners()
        return [m["hotkey"] for m in online_miners]
    
    def is_miner_online(self, hotkey: str) -> bool:
        miner = self.get_miner(hotkey)
        if not miner:
            return False
        
        if not miner.get("is_online", False):
            return False
        
        last_heartbeat = miner.get("last_heartbeat")
        if last_heartbeat:
            now = datetime.now(timezone.utc)
            time_since_heartbeat = (now - last_heartbeat).total_seconds()
            if time_since_heartbeat > self.heartbeat_timeout:
                return False
        
        return True
    
    def get_miner_url(self, hotkey: str) -> Optional[str]:
        miner = self.get_miner(hotkey)
        if miner:
            return miner.get("miner_url")
        return None
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._last_update = None
    
    def set_last_update(self, timestamp: datetime):
        with self._lock:
            self._last_update = timestamp
    
    def get_last_update(self) -> Optional[datetime]:
        with self._lock:
            return self._last_update
    
    def get_cache_size(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def get_online_count(self) -> int:
        return len(self.get_online_miners())

