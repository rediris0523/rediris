import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class YamlConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        
        logger.info(f"Config loaded from {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def get_wallet_name(self) -> str:
        return self.get('wallet.name', 'default')
    
    def get_hotkey_name(self) -> str:
        return self.get('wallet.hotkey', 'default')
    
    def get_netuid(self) -> Optional[int]:
        return self.get('bittensor.netuid')
    
    def get_chain_endpoint(self) -> Optional[str]:
        return self.get('bittensor.chain_endpoint')
    
    def get_task_center_url(self) -> str:
        return self.get('task_center.url', 'http://localhost:8000')

    def get_task_center_api_key(self) -> Optional[str]:
        return self.get('task_center.api_key') or self.get('api.key')
    
    def get_auto_update_config(self) -> Dict[str, Any]:
        return self.get('auto_update', {})
    
    def get_github_repo(self) -> Optional[str]:
        return self.get('auto_update.github_repo')
    
    def get_auto_update_enabled(self) -> bool:
        return self.get('auto_update.enabled', False)
    
    def get_auto_update_interval(self) -> int:
        return self.get('auto_update.check_interval', 300)
    
    def get_min_stake(self) -> float:
        return self.get('miner.min_stake', 0.0)
    
    def get_gpu_count(self) -> int:
        return self.get('miner.gpu_count', 1)
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.get('training', {})
    
    def get_text_training_config(self) -> Dict[str, Any]:
        return self.get('training.text', {})
    
    def get_image_training_config(self) -> Dict[str, Any]:
        return self.get('training.image', {})
    
    def get_datasets_config(self) -> Dict[str, Any]:
        return self.get('datasets', {})

    def get_axon_enabled(self) -> bool:
        return self.get('axon.enabled', True)

    def get_axon_ip(self) -> str:
        return self.get('axon.ip', '0.0.0.0')

    def get_axon_port(self) -> int:
        return self.get('axon.port', 8001)

    def get_axon_external_ip(self) -> Optional[str]:
        return self.get('axon.external_ip')
