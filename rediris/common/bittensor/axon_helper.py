import bittensor as bt
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)


def register_axon(wallet, yaml_config) -> bool:
    """Register axon on Bittensor chain. Returns True if successful."""
    if not yaml_config or not yaml_config.get_axon_enabled():
        if yaml_config:
            logger.info("Axon registration disabled in config")
        return False

    axon_ip = yaml_config.get_axon_ip()
    axon_port = yaml_config.get_axon_port()
    axon_external_ip = yaml_config.get_axon_external_ip()
    netuid = yaml_config.get_netuid()
    chain_endpoint = yaml_config.get_chain_endpoint()

    logger.info(f"Registering axon on chain: ip={axon_ip}, port={axon_port}, netuid={netuid}")
    try:
        subtensor = bt.subtensor(network=chain_endpoint if chain_endpoint else "test")

        axon = bt.axon(
            wallet=wallet,
            ip=axon_ip,
            port=axon_port,
            external_ip=axon_external_ip
        )

        success = subtensor.serve_axon(netuid=netuid, axon=axon)
        if success:
            logger.info("Axon registration successful")
        else:
            logger.warning("Axon registration failed, continuing without chain registration")
        return success
    except Exception as e:
        logger.error(f"Failed to register axon: {e}", exc_info=True)
        return False


def get_balance(wallet, yaml_config) -> float:
    """Get wallet balance from chain. Returns 0.0 on failure."""
    try:
        chain_endpoint = yaml_config.get_chain_endpoint() if yaml_config and yaml_config.get_chain_endpoint() else None
        subtensor = bt.subtensor(network=chain_endpoint if chain_endpoint else "test")
        return float(subtensor.get_balance(wallet.coldkeypub.ss58_address))
    except Exception as e:
        logger.warning(f"Failed to get balance: {e}")
        return 0.0
