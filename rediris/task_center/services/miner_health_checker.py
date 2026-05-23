import asyncio
import threading
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from rediris.common.models.miner import Miner
from rediris.common.bittensor.client import BittensorClient
import bittensor as bt
from rediris.common.config.yaml_config import YamlConfig
from rediris.common.crypto.signature import SignatureAuth
from rediris.common.utils.logging import setup_logger
from rediris.common.utils.thread_pool import get_thread_pool
from rediris.common.database import SessionLocal
from rediris.task_center.services.miner_cache import MinerCache
import httpx

logger = setup_logger(__name__)

class MinerHealthChecker:
    def __init__(
        self,
        db: Session,
        wallet: bt.wallet,
        wallet_name: str,
        hotkey_name: str,
        miner_cache: MinerCache,
        check_interval: int = 600,
        heartbeat_timeout: int = 120,
        yaml_config: Optional[YamlConfig] = None
    ):
        self.db = db
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.bittensor_client = BittensorClient(wallet_name, hotkey_name, yaml_config=yaml_config)
        self.signature_auth = SignatureAuth(wallet)
        self.miner_cache = miner_cache
        self.is_running = False
        self.check_interval = check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._check_task = None
        self.thread_pool = get_thread_pool()
    
    async def start(self):
        if self.is_running:
            logger.warning("Miner health checker is already running")
            return
        
        self.is_running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Miner health checker started")
    
    async def stop(self):
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Miner health checker stopped")
    
    async def _health_check_loop(self):
        while self.is_running:
            try:
                await self._check_all_miners_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_miners_health(self):
        db = SessionLocal()
        try:
            all_miners_data = self.bittensor_client.get_all_miners()
            logger.info(f"_check_all_miners_health all_miners_data  {all_miners_data}")
            for miner_data in all_miners_data:
                hotkey = miner_data["hotkey"]
                
                miner = db.query(Miner).filter(Miner.hotkey == hotkey).first()
                
                if not miner:
                    miner = Miner(
                        hotkey=hotkey,
                        stake=miner_data.get("stake", 0.0),
                        reputation=0.0,
                        is_active=miner_data.get("is_active", False),
                        is_online=False
                    )
                    db.add(miner)
                    db.commit()
                
                miner_url = self._get_miner_url(miner_data)
                if miner_url:
                    miner.miner_url = miner_url

                logger.info(f"_check_all_miners_health miner_url  {miner_url}")
                is_online = await self._check_miner_online(miner_url if miner_url else None, hotkey)
                logger.info(f"_check_all_miners_health is_online  {is_online}")

                if is_online:
                    miner.is_online = True
                    miner.last_heartbeat = datetime.now(timezone.utc)
                else:
                    if miner.last_heartbeat:
                        time_since_heartbeat = datetime.now(timezone.utc) - miner.last_heartbeat
                        if time_since_heartbeat.total_seconds() > self.heartbeat_timeout:
                            miner.is_online = False
                    else:
                        miner.is_online = False
                
                miner.stake = miner_data.get("stake", 0.0)
                miner.is_active = miner_data.get("is_active", False)
                
                db.commit()
                
                cache_data = {
                    "stake": miner.stake,
                    "reputation": miner.reputation,
                    "is_active": miner.is_active,
                    "is_online": miner.is_online,
                    "miner_url": miner.miner_url,
                    "last_heartbeat": miner.last_heartbeat
                }
                
                self.miner_cache.update_miner(hotkey, cache_data)
            
            self.miner_cache.set_last_update(datetime.now(timezone.utc))
            logger.debug(f"Health check completed for {len(all_miners_data)} miners, {self.miner_cache.get_online_count()} online")
        except Exception as e:
            logger.error(f"Error checking miner health: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()
    
    def _get_miner_url(self, miner_data: Dict) -> Optional[str]:
        try:
            if miner_data.get("uid") is not None:
                uid = miner_data.get("uid")
                if self.bittensor_client.metagraph and uid < len(self.bittensor_client.metagraph.axons):
                    axon = self.bittensor_client.metagraph.axons[uid]
                    ip = axon.ip
                    port = axon.port
                    if ip and ip != "0.0.0.0" and port:
                        return f"http://{ip}:{port}"
        except Exception as e:
            logger.debug(f"Failed to get miner URL: {e}")
        
        return None
    
    async def _check_miner_online(self, miner_url: Optional[str], hotkey: str) -> bool:
        if not miner_url:
            return False
        
        try:
            def check_health():
                try:
                    endpoint = "/v1/health/heartbeat"
                    auth_headers = self.signature_auth.create_auth_headers(endpoint)
                    
                    request_data = {
                        "hotkey": self.wallet.hotkey.ss58_address,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    logger.info(f"Health check miner url  {miner_url}{endpoint}")

                    response = httpx.post(
                        f"{miner_url}{endpoint}",
                        json=request_data,
                        headers=auth_headers,
                        timeout=25.0
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        if self.signature_auth.verify_response(response_data):
                            return response_data.get("status") == "online"
                    
                    return False
                except Exception as e:
                    logger.info(f"Health check request failed: {e}")
                    return False
            
            loop = asyncio.get_event_loop()
            future = self.thread_pool.submit(check_health)

            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=10.0
                )
                logger.info(f"Health result {result} ")
                return result
            except asyncio.TimeoutError:
                logger.info(f"Miner {hotkey} health check timed out")
                return False
        except Exception as e:
            logger.info(f"Failed to check miner {hotkey} health: {e}")
            return False
    
    def is_miner_online(self, hotkey: str) -> bool:
        return self.miner_cache.is_miner_online(hotkey)
    
    def get_online_miners(self) -> List[str]:
        return self.miner_cache.get_online_miner_hotkeys()
