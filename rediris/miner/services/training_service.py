from typing import Dict, Any, Optional
from rediris.common.utils.logging import setup_logger
from rediris.common.config.yaml_config import YamlConfig
from rediris.common.models.workflow_type import WorkflowType
from rediris.miner.services.text_training_service import TextTrainingService
from rediris.miner.services.image_training_service import ImageTrainingService

logger = setup_logger(__name__)


class TrainingService:
    
    def __init__(self, config: Optional[YamlConfig] = None):
        self.config = config
        self.text_training_service = TextTrainingService(config)
        self.image_training_service = ImageTrainingService(config)
    
    async def train(self, task: Dict[str, Any]) -> Dict[str, Any]:
        workflow_type = task.get("workflow_type", "")
        task_id = task.get("task_id", "unknown")

        logger.info(f"[TrainingService] train() called - task_id: {task_id}, workflow_type: {workflow_type}")
        logger.debug(f"[TrainingService] Full task data: {task}")

        try:
            workflow_type_enum = WorkflowType(workflow_type)
            logger.info(f"[TrainingService] Parsed workflow_type_enum: {workflow_type_enum}")
        except ValueError as e:
            logger.error(f"[TrainingService] Unknown workflow type: {workflow_type}, error: {e}")
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        if workflow_type_enum == WorkflowType.TEXT_LORA_CREATION:
            logger.info(f"[TrainingService] Routing to TEXT_LORA training for {task_id}")
            return await self.text_training_service.train_lora(task)
        elif workflow_type_enum == WorkflowType.IMAGE_LORA_CREATION:
            logger.info(f"[TrainingService] Routing to IMAGE_LORA training for {task_id}")
            return await self.image_training_service.train_lora(task)
        else:
            logger.error(f"[TrainingService] Unsupported workflow type: {workflow_type}")
            raise ValueError(f"Unsupported workflow type: {workflow_type}")
    
    async def train_text_lora(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return await self.train(task)
    
    async def train_image_lora(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return await self.train(task)
