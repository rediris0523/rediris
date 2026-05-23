from pydantic_settings import BaseSettings
from typing import Optional
from rediris.common.config.yaml_config import YamlConfig
import os
import inspect
from pathlib import Path

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://rediris:rediris@localhost:5432/rediris"
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"
    
    BITNETWORK_NETUID: Optional[int] = None
    BITNETWORK_NETWORK: str = 'local_or_devnet_placeholder'
    BITNETWORK_CHAIN_ENDPOINT: Optional[str] = None
    VERSION_KEY: int = 0
    
    TASK_CENTER_URL: str = "http://localhost:8000"
    
    API_KEY: Optional[str] = None
    
    LOG_LEVEL: str = "INFO"
    
    GITHUB_REPO: Optional[str] = None
    AUTO_UPDATE_ENABLED: bool = False
    
    MINER_MIN_STAKE: float = 0.0

    IDLE_REWARD_UID: int = 1

    TREASURY_RATIO: float = 0.10
    MINER_POOL_RATIO: float = 0.90
    TEXT_POOL_RATIO: float = 0.30
    IMAGE_POOL_RATIO: float = 0.70

    CONSENSUS_MIN_VALIDATORS: int = 1
    CONSENSUS_MAX_VALIDATORS: int = 2

    MIN_REWARD_MINERS: int = 3
    MAX_REWARD_MINERS: int = 18

    CONFIG_FILE: Optional[str] = None
    
    class Config:
        env_file = None
        case_sensitive = True

def load_yaml_config(config_path: Optional[str] = None) -> Optional[YamlConfig]:

    caller_frame = inspect.currentframe()
    try:
        if caller_frame and caller_frame.f_back:
            caller_module_file = caller_frame.f_back.f_globals.get('__file__')
            if caller_module_file:
                caller_module_dir = Path(caller_module_file).parent
            else:
                caller_module_dir = None
        else:
            caller_module_dir = None
    finally:
        del caller_frame
    
    if config_path:
        config_path_obj = Path(config_path)
        
        if config_path_obj.is_absolute():
            if config_path_obj.exists():
                return YamlConfig(str(config_path_obj))
        else:
            if config_path_obj.exists():
                return YamlConfig(str(config_path_obj))
            
            if caller_module_dir:
                module_config_path = caller_module_dir / config_path
                if module_config_path.exists():
                    return YamlConfig(str(module_config_path))
            
            if caller_module_dir:
                current_dir = caller_module_dir
                for _ in range(5):
                    if current_dir.name == "rediris" or (current_dir / "rediris").exists():
                        root_config_path = current_dir / config_path
                        if root_config_path.exists():
                            return YamlConfig(str(root_config_path))
                        moirai_dir = current_dir if current_dir.name == "rediris" else current_dir / "rediris"
                        if moirai_dir.exists():
                            for subdir in ["miner", "validator", "task_center", "website_admin"]:
                                module_config = moirai_dir / subdir / config_path
                                if module_config.exists():
                                    return YamlConfig(str(module_config))
                        break
                    parent = current_dir.parent
                    if parent == current_dir:
                        break
                    current_dir = parent
    
    default_paths = [
        "config.yml",
        "config.yaml",
        "config/config.yml",
        "config/config.yaml"
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            return YamlConfig(path)
    
    if caller_module_dir:
        for path in default_paths:
            module_config_path = caller_module_dir / path
            if module_config_path.exists():
                return YamlConfig(str(module_config_path))
        
        current_dir = caller_module_dir
        for _ in range(5):
            if current_dir.name == "rediris" or (current_dir / "rediris").exists():
                moirai_dir = current_dir if current_dir.name == "rediris" else current_dir / "rediris"
                if moirai_dir.exists():
                    check_dir = caller_module_dir
                    for _ in range(3):
                        if check_dir.parent == moirai_dir:
                            module_name = check_dir.name
                            if module_name in ["miner", "validator", "task_center", "website_admin"]:
                                for path in default_paths:
                                    module_config = moirai_dir / module_name / path
                                    if module_config.exists():
                                        return YamlConfig(str(module_config))
                            break
                        parent = check_dir.parent
                        if parent == check_dir or parent == moirai_dir:
                            break
                        check_dir = parent
                break
            parent = current_dir.parent
            if parent == current_dir:
                break
            current_dir = parent
    
    return None
