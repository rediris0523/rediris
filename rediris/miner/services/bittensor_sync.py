import asyncio
import threading
from typing import Optional
import bittensor as bt
from rediris.common.bittensor.client import BittensorClient
from rediris.common.config.yaml_config import YamlConfig
from rediris.common.utils.logging import setup_logger
from rediris.common.utils.thread_pool import get_thread_pool
from rediris.common.utils.retry import retry_sync_with_backoff

logger = setup_logger(__name__)

class BittensorSyncService:
    def __init__(self, wallet: bt.wallet, wallet_name: str, hotkey_name: str, yaml_config: Optional[YamlConfig] = None):
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.client = BittensorClient(wallet_name, hotkey_name, yaml_config=yaml_config)
        self.is_running = False
        self.sync_interval = 60
        self._sync_task = None
        self._lock = threading.Lock()
        self.thread_pool = get_thread_pool()
    
    async def start_sync(self):
        if self.is_running:
            logger.warning("Bittensor sync is already running")
            return
        
        self.is_running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Bittensor sync service started")
    
    async def stop_sync(self):
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Bittensor sync service stopped")
    
    @retry_sync_with_backoff(max_retries=3, initial_delay=2.0, max_delay=30.0)
    def _sync_metagraph_safe(self):
        try:
            self.client.sync_metagraph()
            return True
        except Exception as e:
            logger.error(f"Metagraph sync error: {e}", exc_info=True)
            raise
    
    async def _sync_loop(self):
        while self.is_running:
            try:
                loop = asyncio.get_event_loop()
                future = self.thread_pool.submit(self._sync_metagraph_safe)
                
                try:
                    await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Metagraph sync timed out after 30 seconds")
                except Exception as e:
                    logger.error(f"Metagraph sync failed: {e}", exc_info=True)
                
                await asyncio.sleep(self.sync_interval)
                
            except asyncio.CancelledError:
                logger.info("Sync loop cancelled")
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}", exc_info=True)
                await asyncio.sleep(self.sync_interval)
