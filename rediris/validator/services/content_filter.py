from typing import Optional
from PIL import Image
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class ContentFilter:
    def __init__(self):
        self.nsfw_model = None
        self._load_nsfw_model()
    
    def _load_nsfw_model(self):
        try:
            from transformers import pipeline, AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(
                "Falconsai/nsfw_image_detection",
                use_fast=True
            )
            self.nsfw_model = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                image_processor=image_processor
            )
            logger.info("NSFW model loaded")
        except Exception as e:
            logger.warning(f"Failed to load NSFW model: {e}")
    
    async def detect_content(self, image: Optional[Image.Image]) -> float:
        if image is None:
            return 0.0
        
        if self.nsfw_model is None:
            return 0.1
        
        try:
            results = self.nsfw_model(image)
            
            for result in results:
                if result["label"].lower() in ["nsfw", "explicit", "porn"]:
                    return float(result["score"])
            
            return 0.1
        except Exception as e:
            logger.warning(f"Content detection failed: {e}")
            return 0.1
    
    def is_safe(self, safety_score: float) -> bool:
        return safety_score < 0.7

