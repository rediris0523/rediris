import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from rediris.common.utils.logging import set_module_prefix, set_global_log_level, setup_logger, reinitialize_all_loggers
from rediris.common.config import load_yaml_config

set_module_prefix("VALIDATOR")

_config_path = os.getenv("VALIDATOR_CONFIG")
if not _config_path:
    import rediris.validator
    _validator_dir = Path(rediris.validator.__file__).parent
    _config_path = str(_validator_dir / "config.yml")
_yaml_config = load_yaml_config(_config_path)

if _yaml_config:
    _log_level = _yaml_config.get('logging.level', 'INFO')
    set_global_log_level(_log_level)
    _db_url = _yaml_config.get('database.url')
    if _db_url:
        from rediris.common.database.session import recreate_engine_and_session
        recreate_engine_and_session(_db_url)

from rediris.validator.api import router
from rediris.validator.services.bittensor_sync import BittensorSyncService
from rediris.validator.services.task_processor import TaskProcessor
from rediris.validator.services.weight_sync_service import WeightSyncService
from rediris.validator.services.score_cache import ScoreCache
from rediris.common.services.auto_update import AutoUpdateService
import bittensor as bt
from rediris.common.config import settings
from rediris.common.utils.thread_pool import get_thread_pool
from rediris.common.database import SessionLocal

logger = setup_logger(__name__)

config_path = _config_path
yaml_config = _yaml_config

if yaml_config:
    wallet_name = yaml_config.get_wallet_name()
    hotkey_name = yaml_config.get_hotkey_name()
    task_center_url = yaml_config.get_task_center_url()
    auto_update_config = yaml_config.get_auto_update_config()
else:
    wallet_name = "validator"
    hotkey_name = "default"
    task_center_url = settings.TASK_CENTER_URL
    auto_update_config = {}

wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
bittensor_sync = BittensorSyncService(wallet, wallet_name, hotkey_name, yaml_config=yaml_config)

score_cache = ScoreCache()

task_processor = TaskProcessor(wallet, wallet_name, hotkey_name, score_cache=score_cache, yaml_config=yaml_config)

db_session = SessionLocal()

weight_sync_interval = yaml_config.get('validator.weight_sync_interval', 1800) if yaml_config else 1800
weight_sync_service = WeightSyncService(
    wallet=wallet,
    wallet_name=wallet_name,
    hotkey_name=hotkey_name,
    bittensor_sync=bittensor_sync,
    score_cache=score_cache,
    sync_interval=weight_sync_interval,
    yaml_config=yaml_config,
    db_session=db_session
)

if yaml_config:
    github_repo = yaml_config.get_github_repo()
    auto_update_enabled = yaml_config.get_auto_update_enabled()
    check_interval = yaml_config.get_auto_update_interval()
else:
    github_repo = settings.GITHUB_REPO
    auto_update_enabled = settings.AUTO_UPDATE_ENABLED
    check_interval = 300

auto_update = AutoUpdateService(
    github_repo=github_repo or "rediris/validator",
    branch=auto_update_config.get('branch', 'main'),
    check_interval=check_interval,
    restart_delay=auto_update_config.get('restart_delay', 20)
)

_cached_balance = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cached_balance
    log_file = yaml_config.get('logging.file') if yaml_config else None
    reinitialize_all_loggers(log_file)

    logger.info("Validator service starting up")
    logger.info(f"Validator hotkey: {wallet.hotkey.ss58_address}")

    from rediris.common.bittensor.axon_helper import get_balance, register_axon
    _cached_balance = get_balance(wallet, yaml_config)
    logger.info(f"Validator balance: {_cached_balance} TAO" if _cached_balance > 0 else "Validator balance: unavailable")
    logger.info(f"Config loaded from: {config_path if yaml_config else 'default'}")

    register_axon(wallet, yaml_config)

    try:
        await bittensor_sync.start_sync()
    except Exception as e:
        logger.error(f"Failed to start bittensor sync: {e}", exc_info=True)
    
    try:
        await task_processor.start()
    except Exception as e:
        logger.error(f"Failed to start task processor: {e}", exc_info=True)

    try:
        await weight_sync_service.start()
        logger.info("Weight sync service started")
    except Exception as e:
        logger.error(f"Failed to start weight sync service: {e}", exc_info=True)

    if auto_update_enabled:
        try:
            await auto_update.start()
        except Exception as e:
            logger.error(f"Failed to start auto-update: {e}", exc_info=True)

    try:
        yield
    finally:
        logger.info("Validator service shutting down")
        try:
            await bittensor_sync.stop_sync()
        except Exception as e:
            logger.error(f"Error stopping bittensor sync: {e}", exc_info=True)

        try:
            await task_processor.stop()
        except Exception as e:
            logger.error(f"Error stopping task processor: {e}", exc_info=True)

        try:
            await weight_sync_service.stop()
        except Exception as e:
            logger.error(f"Error stopping weight sync service: {e}", exc_info=True)

        try:
            await auto_update.stop()
        except Exception as e:
            logger.error(f"Error stopping auto-update: {e}", exc_info=True)

        try:
            thread_pool = get_thread_pool()
            thread_pool.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}", exc_info=True)

        try:
            db_session.close()
            logger.info("Database session closed")
        except Exception as e:
            logger.error(f"Error closing database session: {e}", exc_info=True)

app = FastAPI(title="Red Iris Validator", version="1.0.0", lifespan=lifespan)
app.include_router(router, prefix="/v1")

# Expose runtime singletons to debug endpoints (same process memory).
app.state.score_cache = score_cache
app.state.validator_hotkey = wallet.hotkey.ss58_address

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "ok",
            "hotkey": wallet.hotkey.ss58_address,
            "balance": _cached_balance
        }
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    if yaml_config:
        default_host = yaml_config.get_axon_ip()
        default_port = yaml_config.get_axon_port()
    else:
        default_host = "0.0.0.0"
        default_port = 8002

    host = os.getenv("VALIDATOR_HOST", default_host)
    port = int(os.getenv("VALIDATOR_PORT", str(default_port)))

    uvicorn_log_level = "debug" if _log_level and _log_level.upper() == "DEBUG" else "info"

    logger.info(f"Starting Validator service on {host}:{port}")
    logger.info("Using asyncio event loop (required for bittensor compatibility)")

    uvicorn.run(
        app,
        host=host,
        port=port,
        loop="asyncio",
        log_level=uvicorn_log_level,
        log_config=None
    )
