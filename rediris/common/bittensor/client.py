import sys
try:
    import asyncio
    loop = asyncio.get_event_loop()
    if hasattr(loop, '__class__') and 'uvloop' in str(type(loop)):
        raise RuntimeError(
            "uvloop is not compatible with bittensor. "
            "Please run uvicorn with --loop asyncio flag: "
            "uvicorn rediris.task_center.task_center_main:app --host 0.0.0.0 --port 8000 --loop asyncio"
        )
except RuntimeError:
    pass

try:
    import pandas as pd
    if not hasattr(pd.io.json, 'json_normalize'):
        try:
            from pandas import json_normalize
            pd.io.json.json_normalize = json_normalize
        except ImportError:
            try:
                from pandas import json_normalize as _json_normalize
                def _wrapper(*args, **kwargs):
                    return _json_normalize(*args, **kwargs)
                pd.io.json.json_normalize = _wrapper
            except ImportError:
                pass
except (ImportError, AttributeError):
    pass

import bittensor as bt
from typing import Dict, List, Optional
from rediris.common.config import settings, load_yaml_config
from rediris.common.config.yaml_config import YamlConfig
from rediris.common.utils.logging import setup_logger
from rediris.common.utils.retry import retry_sync_with_backoff

logger = setup_logger(__name__)


class BittensorClient:
    def __init__(
        self, 
        wallet_name: str, 
        hotkey_name: str, 
        yaml_config: Optional[YamlConfig] = None,
        config_path: Optional[str] = None
    ):

        try:
            if yaml_config is None:
                if config_path:
                    yaml_config = load_yaml_config(config_path)
                else:
                    yaml_config = load_yaml_config()
            
            if yaml_config:
                chain_endpoint = yaml_config.get('bittensor.chain_endpoint', settings.BITNETWORK_CHAIN_ENDPOINT)
                network = yaml_config.get('bittensor.network', settings.BITNETWORK_NETWORK)
                netuid = yaml_config.get('bittensor.netuid', settings.BITNETWORK_NETUID)
            else:
                chain_endpoint = settings.BITNETWORK_CHAIN_ENDPOINT
                netuid = settings.BITNETWORK_NETUID
                network = settings.BITNETWORK_NETWORK

            self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
            self.subtensor = bt.subtensor(network=network or "test")
            self.netuid = netuid
            self.metagraph = None
            self._sync_lock = False
            try:
                self.sync_metagraph()
            except Exception as e:
                logger.warning(f"Failed to sync metagraph on initialization (network may be unavailable): {e}")
        except Exception as e:
            logger.error(f"Failed to initialize BittensorClient: {e}", exc_info=True)

            self.wallet = None
            self.subtensor = None
            self.metagraph = None
            self._sync_lock = False
    
    @retry_sync_with_backoff(max_retries=3, initial_delay=2.0, max_delay=30.0)
    def sync_metagraph(self):
        if self.subtensor is None:
            logger.warning("Subtensor is not initialized; skipping metagraph sync")
            return

        if self._sync_lock:
            logger.warning("Metagraph sync already in progress")
            return
        
        try:
            self._sync_lock = True
            self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        except Exception as e:
            logger.error(f"Failed to sync metagraph: {e}", exc_info=True)
            raise
        finally:
            self._sync_lock = False
    
    def get_miner_stake(self, hotkey: str) -> float:
        if not self.metagraph:
            try:
                self.sync_metagraph()
            except Exception as e:
                logger.error(f"Failed to sync metagraph for stake query: {e}")
                return 0.0
        
        try:
            uid = self.metagraph.hotkeys.index(hotkey)
            stake = float(self.metagraph.S[uid])
            return stake
        except (ValueError, IndexError) as e:
            logger.warning(f"Hotkey {hotkey} not found in metagraph: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get miner stake: {e}", exc_info=True)
            return 0.0
    
    def get_validator_stake(self, hotkey: str) -> float:
        return self.get_miner_stake(hotkey)
    
    def get_all_miners(self) -> List[Dict]:
        if not self.metagraph:
            try:
                self.sync_metagraph()
            except Exception as e:
                logger.error(f"Failed to sync metagraph for miners query: {e}")
                return []
        
        try:
            miners = []
            for uid in range(len(self.metagraph.hotkeys)):
                try:
                    axon = self.metagraph.axons[uid] if uid < len(self.metagraph.axons) else None
                    logger.info(f"Start to process miner {uid}")
                    miners.append({
                        "uid": uid,
                        "hotkey": self.metagraph.hotkeys[uid],
                        "stake": float(self.metagraph.S[uid]),
                        "is_active": axon.ip != "0.0.0.0" if axon else False,
                        "axon": axon
                    })
                except Exception as e:
                    logger.warning(f"Failed to process miner {uid}: {e}")
                    continue
            
            return miners
        except Exception as e:
            logger.error(f"Failed to get all miners: {e}", exc_info=True)
            return []
    
    def set_weights(
        self,
        uids: List[int],
        weights: List[float]
    ):
        try:
            if not uids or not weights:
                logger.warning("Empty uids or weights provided")
                return
            
            if len(uids) != len(weights):
                raise ValueError(f"Uids length ({len(uids)}) != weights length ({len(weights)})")

            # Attempt to capture on-chain hash / info from bittensor.
            result = self.subtensor.set_weights(
                netuid=self.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                version_key=settings.VERSION_KEY
            )
            logger.info(f"set_weights result is true")

        except Exception as e:
            logger.error(f"Failed to set weights: {e}", exc_info=True)
            raise
    
    def get_emission(self) -> float:
        try:
            emission = self.subtensor.get_emission(netuid=self.netuid)
            return float(emission)
        except Exception as e:
            logger.error(f"Failed to get emission: {e}", exc_info=True)
            return 0.0

    def get_miners_by_uids(self, uids: List[int]) -> List[Dict]:
        if not self.metagraph:
            try:
                self.sync_metagraph()
            except Exception as e:
                logger.error(f"Failed to sync metagraph for miners query: {e}")
                return []

        try:
            miners = []
            for uid in uids:
                if uid < 0 or uid >= len(self.metagraph.hotkeys):
                    logger.warning(f"Invalid UID {uid}, skipping")
                    continue

                try:
                    axon = self.metagraph.axons[uid] if uid < len(self.metagraph.axons) else None
                    miners.append({
                        "uid": uid,
                        "hotkey": self.metagraph.hotkeys[uid],
                        "stake": float(self.metagraph.S[uid]),
                        "is_active": axon.ip != "0.0.0.0" if axon else False,
                        "axon": axon
                    })
                except Exception as e:
                    logger.warning(f"Failed to process miner {uid}: {e}")
                    continue

            return miners
        except Exception as e:
            logger.error(f"Failed to get miners by UIDs: {e}", exc_info=True)
            return []

    def get_miner_by_uid(self, uid: int) -> Optional[Dict]:
        miners = self.get_miners_by_uids([uid])
        return miners[0] if miners else None

    def get_alpha_price(self) -> Optional[float]:

        try:
            if self.subtensor is None:
                logger.warning("Subtensor not initialized, cannot get alpha price")
                return None

            try:
                if hasattr(self.subtensor, 'get_subnet_info'):
                    subnet_info = self.subtensor.get_subnet_info(netuid=self.netuid)
                    if subnet_info and hasattr(subnet_info, 'alpha_price'):
                        return float(subnet_info.alpha_price)
            except Exception as e:
                logger.debug(f"get_subnet_info not available: {e}")

            try:
                if hasattr(self.subtensor, 'get_alpha_per_block'):
                    alpha_per_block = self.subtensor.get_alpha_per_block(netuid=self.netuid)
                    if alpha_per_block:
                        return float(alpha_per_block)
            except Exception as e:
                logger.debug(f"get_alpha_per_block not available: {e}")

            try:
                if hasattr(self.subtensor, 'substrate'):
                    result = self.subtensor.substrate.query(
                        module='SubtensorModule',
                        storage_function='AlphaPrice',
                        params=[self.netuid]
                    )
                    if result:
                        return float(result.value) / 1e9
            except Exception as e:
                logger.debug(f"Substrate query for AlphaPrice failed: {e}")

            logger.warning(f"Could not retrieve alpha price for netuid {self.netuid}")
            return None

        except Exception as e:
            logger.error(f"Failed to get alpha price: {e}", exc_info=True)
            return None

    def get_subnet_emission_info(self) -> Dict:

        try:
            if self.subtensor is None:
                return {}

            emission_info = {
                "netuid": self.netuid,
                "alpha_price": self.get_alpha_price(),
            }

            try:
                emission = self.get_emission()
                emission_info["emission_per_block"] = emission
                emission_info["daily_emission"] = emission * 7200
            except Exception as e:
                logger.debug(f"Could not get emission info: {e}")

            return emission_info

        except Exception as e:
            logger.error(f"Failed to get subnet emission info: {e}", exc_info=True)
            return {}
