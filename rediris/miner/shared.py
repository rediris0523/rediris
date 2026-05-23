from rediris.miner.services.queue_manager import QueueManager
import bittensor as bt
from rediris.common.config.yaml_config import YamlConfig
from typing import Optional

queue_manager: QueueManager = None
wallet: Optional[bt.wallet] = None
wallet_name: Optional[str] = None
hotkey_name: Optional[str] = None
yaml_config: Optional[YamlConfig] = None

