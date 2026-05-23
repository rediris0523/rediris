from typing import Dict, Any, Optional, List
from PIL import Image
import torch
import numpy as np
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)


class QualityEvaluator:

    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

    COMPOSITION_PROMPTS = {
        "good": [
            "a well-composed photograph with balanced elements",
            "professional photography with rule of thirds composition",
            "harmonious visual balance and good framing",
            "excellent composition with clear focal point",
            "visually balanced image with good use of space"
        ],
        "bad": [
            "poorly composed image with unbalanced elements",
            "amateur photography with bad framing",
            "cluttered composition with no clear subject",
            "off-center composition with awkward cropping",
            "visually chaotic image with poor arrangement"
        ]
    }

    COLOR_PROMPTS = {
        "good": [
            "vibrant and harmonious color palette",
            "rich colors with good contrast and saturation",
            "beautiful color harmony and visual appeal",
            "well-balanced colors with pleasant tones",
            "professional color grading with depth"
        ],
        "bad": [
            "dull and washed out colors",
            "harsh and clashing color combinations",
            "oversaturated or undersaturated image",
            "poor color balance with unnatural tones",
            "muddy colors lacking vibrancy"
        ]
    }

    DETAIL_PROMPTS = {
        "good": [
            "sharp and detailed high resolution image",
            "crisp details with excellent clarity",
            "fine textures and intricate details visible",
            "high quality image with no blur or noise",
            "professionally captured with perfect focus"
        ],
        "bad": [
            "blurry and out of focus image",
            "low resolution with visible pixelation",
            "noisy image with poor quality",
            "soft focus lacking sharpness",
            "compressed image with artifacts"
        ]
    }

    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_clip_model()

    def _load_clip_model(self):
        try:
            from transformers import CLIPProcessor, CLIPModel
            logger.info(f"Loading CLIP model ({self.CLIP_MODEL_ID})...")
            self.clip_model = CLIPModel.from_pretrained(self.CLIP_MODEL_ID).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.CLIP_MODEL_ID,
                use_fast=True
            )
            self.clip_model.eval()
            logger.info("CLIP model loaded for quality evaluation")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    async def evaluate_quality(
        self,
        task_type: str,
        content: Any
    ) -> float:
        if task_type == "text_lora" or task_type == "text_lora_creation":
            return await self._evaluate_text_quality(content)
        elif task_type == "image_lora" or task_type == "image_lora_creation":
            return await self._evaluate_image_quality(content)
        else:
            return 5.0
    
    async def _evaluate_text_quality(self, text: Optional[str]) -> float:
        if text is None:
            return 5.0
        
        relevance_score = self._evaluate_relevance(text)
        accuracy_score = self._evaluate_accuracy(text)
        fluency_score = self._evaluate_fluency(text)
        cultural_accuracy = self._evaluate_cultural_accuracy(text)
        
        return (relevance_score + accuracy_score + fluency_score + cultural_accuracy) / 4.0
    
    def _evaluate_relevance(self, text: str) -> float:
        japanese_keywords = ["日本", "文化", "传统", "和", "茶道", "武士", "樱花", "神社"]
        score = 5.0
        for keyword in japanese_keywords:
            if keyword in text:
                score += 0.5
        return min(10.0, score)
    
    def _evaluate_accuracy(self, text: str) -> float:
        return 8.0
    
    def _evaluate_fluency(self, text: str) -> float:
        if len(text) < 10:
            return 5.0
        return 8.0
    
    def _evaluate_cultural_accuracy(self, text: str) -> float:
        return 8.0
    
    async def _evaluate_image_quality(self, image: Optional[Image.Image]) -> float:

        if image is None:
            return 5.0

        aesthetic_score = await self._calculate_aesthetic_score(image)
        composition_score = await self._evaluate_composition(image)
        color_score = await self._evaluate_color(image)
        detail_score = await self._evaluate_detail(image)

        quality_score = (
            aesthetic_score * 0.5 +
            composition_score * 0.2 +
            color_score * 0.2 +
            detail_score * 0.1
        )

        logger.info(f"Quality evaluation: aesthetic={aesthetic_score:.2f}, "
                   f"composition={composition_score:.2f}, color={color_score:.2f}, "
                   f"detail={detail_score:.2f}, total={quality_score:.2f}")

        return quality_score

    def _calculate_clip_similarity(
        self,
        image: Image.Image,
        text_prompts: List[str]
    ) -> float:

        if self.clip_model is None or self.clip_processor is None:
            return 0.5

        try:
            inputs = self.clip_processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)
                return float(probs.mean().item())

        except Exception as e:
            logger.warning(f"CLIP similarity calculation failed: {e}")
            return 0.5

    def _evaluate_with_prompts(
        self,
        image: Image.Image,
        good_prompts: List[str],
        bad_prompts: List[str]
    ) -> float:

        if self.clip_model is None or self.clip_processor is None:
            return 7.0

        try:
            all_prompts = good_prompts + bad_prompts
            num_good = len(good_prompts)

            inputs = self.clip_processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image.squeeze(0)

                good_logits = logits[:num_good]
                bad_logits = logits[num_good:]

                good_score = good_logits.mean().item()
                bad_score = bad_logits.mean().item()

                combined = torch.tensor([good_score, bad_score])
                probs = torch.softmax(combined, dim=0)
                good_prob = probs[0].item()

                score = 2.0 + good_prob * 8.0

                return min(10.0, max(0.0, score))

        except Exception as e:
            logger.warning(f"Prompt-based evaluation failed: {e}")
            return 7.0

    async def _calculate_aesthetic_score(self, image: Image.Image) -> float:

        if self.clip_model is None:
            return 7.0

        try:
            aesthetic_prompts = {
                "good": [
                    "a beautiful and aesthetically pleasing image",
                    "stunning visual artwork with artistic quality",
                    "professionally created high quality image",
                    "visually appealing masterpiece",
                    "gorgeous image with excellent visual design"
                ],
                "bad": [
                    "ugly and unappealing image",
                    "low quality amateur artwork",
                    "visually unattractive poor quality image",
                    "badly made unprofessional image",
                    "aesthetically unpleasant visual"
                ]
            }

            score = self._evaluate_with_prompts(
                image,
                aesthetic_prompts["good"],
                aesthetic_prompts["bad"]
            )
            return score

        except Exception as e:
            logger.warning(f"Aesthetic score calculation failed: {e}")
            return 7.0

    async def _evaluate_composition(self, image: Image.Image) -> float:

        if self.clip_model is None:
            return self._simple_composition_score(image)

        try:
            score = self._evaluate_with_prompts(
                image,
                self.COMPOSITION_PROMPTS["good"],
                self.COMPOSITION_PROMPTS["bad"]
            )
            return score

        except Exception as e:
            logger.warning(f"Composition evaluation failed: {e}")
            return self._simple_composition_score(image)

    async def _evaluate_color(self, image: Image.Image) -> float:

        if self.clip_model is None:
            return self._simple_color_score(image)

        try:
            score = self._evaluate_with_prompts(
                image,
                self.COLOR_PROMPTS["good"],
                self.COLOR_PROMPTS["bad"]
            )
            return score

        except Exception as e:
            logger.warning(f"Color evaluation failed: {e}")
            return self._simple_color_score(image)

    async def _evaluate_detail(self, image: Image.Image) -> float:

        if self.clip_model is None:
            return self._simple_detail_score(image)

        try:
            score = self._evaluate_with_prompts(
                image,
                self.DETAIL_PROMPTS["good"],
                self.DETAIL_PROMPTS["bad"]
            )
            return score

        except Exception as e:
            logger.warning(f"Detail evaluation failed: {e}")
            return self._simple_detail_score(image)

    def _simple_composition_score(self, image: Image.Image) -> float:
        width, height = image.size
        aspect_ratio = width / height

        good_ratios = [1.0, 4/3, 3/2, 16/9, 3/4, 2/3, 9/16]
        min_diff = min(abs(aspect_ratio - r) for r in good_ratios)

        if min_diff < 0.05:
            return 8.5
        elif min_diff < 0.15:
            return 7.5
        elif min_diff < 0.3:
            return 6.5
        else:
            return 5.5

    def _simple_color_score(self, image: Image.Image) -> float:
        try:
            img_rgb = image.convert("RGB")
            img_array = np.array(img_rgb)

            color_std = np.std(img_array)

            from PIL import ImageStat
            stat = ImageStat.Stat(img_rgb)
            mean_brightness = sum(stat.mean) / 3

            score = 5.0

            if color_std > 60:
                score += 2.0
            elif color_std > 40:
                score += 1.0

            if 80 < mean_brightness < 180:
                score += 1.5
            elif 60 < mean_brightness < 200:
                score += 0.5

            return min(10.0, max(0.0, score))

        except Exception as e:
            logger.warning(f"Simple color evaluation failed: {e}")
            return 6.0

    def _simple_detail_score(self, image: Image.Image) -> float:
        try:
            width, height = image.size
            total_pixels = width * height

            if total_pixels >= 1024 * 1024:  # >= 1MP
                base_score = 8.0
            elif total_pixels >= 512 * 512:  # >= 0.25MP
                base_score = 7.0
            elif total_pixels >= 256 * 256:  # >= 0.06MP
                base_score = 6.0
            else:
                base_score = 5.0

            try:
                import cv2
                img_array = np.array(image.convert("L"))
                laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()

                if laplacian_var > 500:
                    base_score += 1.5
                elif laplacian_var > 200:
                    base_score += 1.0
                elif laplacian_var > 100:
                    base_score += 0.5
            except ImportError:
                pass

            return min(10.0, max(0.0, base_score))

        except Exception as e:
            logger.warning(f"Simple detail evaluation failed: {e}")
            return 6.0

