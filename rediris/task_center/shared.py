from typing import Optional, Any
from rediris.task_center.services.miner_cache import MinerCache
import bittensor as bt
from rediris.common.config.yaml_config import YamlConfig

miner_cache = MinerCache()

bittensor_client: Optional[Any] = None
wallet: Optional[bt.wallet] = None
wallet_name: Optional[str] = None
hotkey_name: Optional[str] = None
yaml_config: Optional[YamlConfig] = None

