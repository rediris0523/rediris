#!/usr/bin/env python3

import torch
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    from transformers import AutoImageProcessor, AutoModel
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False
    logger.warning("DINOv2 not available: pip install transformers")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available: pip install pillow")

try:
    import requests
    from io import BytesIO
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests not available: pip install requests")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available: pip install transformers")

try:
    from transformers import pipeline, AutoImageProcessor as NSFWImageProcessor
    NSFW_AVAILABLE = True
except ImportError:
    NSFW_AVAILABLE = False
    logger.warning("NSFW detection not available: pip install transformers")


class ImageSimilarityCalculator:

    DINOV2_MODEL_ID = "facebook/dinov2-large"
    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
    NSFW_MODEL_ID = "Falconsai/nsfw_image_detection"

    BASE_THRESHOLD = 0.35
    NSFW_THRESHOLD = 0.7

    SIMILARITY_WEIGHT = 0.95
    QUALITY_WEIGHT = 0.05

    SCORE_PIVOT_HIGH = 0.96
    SCORE_PIVOT_LOW = 0.85

    EXPONENT_HIGH = 0.5
    EXPONENT_MID = 2.5
    EXPONENT_LOW = 8

    BASE_TRAINING_TIME = 30
    TIME_WEIGHT = 0.15

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

    AESTHETIC_PROMPTS = {
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

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dinov2_model = None
        self.dinov2_processor = None
        self.clip_model = None
        self.clip_processor = None
        self.nsfw_model = None
        logger.info(f"Using device: {self.device}")

    def _load_dinov2(self):
        if self.dinov2_model is not None:
            return

        if not DINOV2_AVAILABLE:
            raise RuntimeError("DINOv2 not available: pip install transformers")

        logger.info(f"Loading DINOv2 model ({self.DINOV2_MODEL_ID})...")
        self.dinov2_processor = AutoImageProcessor.from_pretrained(self.DINOV2_MODEL_ID)
        self.dinov2_model = AutoModel.from_pretrained(self.DINOV2_MODEL_ID).to(self.device)
        self.dinov2_model.eval()
        logger.info("DINOv2 model loaded (1024 dimensions)")

    def _load_clip(self):
        if self.clip_model is not None:
            return

        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available, quality score will use default value")
            return

        logger.info(f"Loading CLIP model ({self.CLIP_MODEL_ID})...")
        self.clip_processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL_ID, use_fast=True)
        self.clip_model = CLIPModel.from_pretrained(self.CLIP_MODEL_ID).to(self.device)
        self.clip_model.eval()
        logger.info("CLIP model loaded")

    def _load_nsfw(self):
        if self.nsfw_model is not None:
            return

        if not NSFW_AVAILABLE:
            logger.warning("NSFW detection not available")
            return

        logger.info(f"Loading NSFW model ({self.NSFW_MODEL_ID})...")
        try:
            nsfw_processor = NSFWImageProcessor.from_pretrained(self.NSFW_MODEL_ID, use_fast=True)
            self.nsfw_model = pipeline(
                "image-classification",
                model=self.NSFW_MODEL_ID,
                image_processor=nsfw_processor,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("NSFW model loaded")
        except Exception as e:
            logger.warning(f"Failed to load NSFW model: {e}")
            self.nsfw_model = None

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        inputs = self.dinov2_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            features = outputs.last_hidden_state[:, 0]
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().float()

    def load_image_from_path(self, image_path: str) -> Image.Image:
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available: pip install pillow")

        logger.info(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return image

    def load_image_from_url(self, image_url: str) -> Image.Image:
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available: pip install pillow")
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("Requests not available: pip install requests")

        logger.info(f"Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image

    def extract_vector_from_image(self, image: Image.Image) -> List[float]:
        self._load_dinov2()

        logger.info("Extracting image features with DINOv2...")
        features = self._extract_features(image)
        return features.tolist()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        t1 = torch.tensor(vec1)
        t2 = torch.tensor(vec2)

        t1 = t1 / t1.norm()
        t2 = t2 / t2.norm()

        similarity = torch.dot(t1, t2).item()
        return similarity

    def evaluate_quality(self, image: Image.Image) -> float:

        self._load_clip()

        aesthetic_score = self._calculate_aesthetic_score(image)
        composition_score = self._evaluate_composition(image)
        color_score = self._evaluate_color(image)
        detail_score = self._evaluate_detail(image)

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
                logits = outputs.logits_per_image.squeeze(0)  # shape: [num_prompts]

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

    def _calculate_aesthetic_score(self, image: Image.Image) -> float:

        if self.clip_model is None:
            return 7.0

        try:
            score = self._evaluate_with_prompts(
                image,
                self.AESTHETIC_PROMPTS["good"],
                self.AESTHETIC_PROMPTS["bad"]
            )
            return score
        except Exception as e:
            logger.warning(f"Aesthetic score calculation failed: {e}")
            return 7.0

    def _evaluate_composition(self, image: Image.Image) -> float:

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

    def _evaluate_color(self, image: Image.Image) -> float:

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

    def _evaluate_detail(self, image: Image.Image) -> float:

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
            import numpy as np
            from PIL import ImageStat

            img_rgb = image.convert("RGB")
            img_array = np.array(img_rgb)

            color_std = np.std(img_array)
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
            import numpy as np

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

                # 锐度加分
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

    def detect_nsfw(self, image: Image.Image) -> float:

        self._load_nsfw()

        if self.nsfw_model is None:
            return 0.1

        try:
            results = self.nsfw_model(image)

            for result in results:
                if result["label"].lower() in ["nsfw", "explicit", "porn"]:
                    return float(result["score"])

            return 0.1
        except Exception as e:
            logger.warning(f"NSFW detection failed: {e}")
            return 0.1

    def calculate_time_coefficient(
        self,
        training_time_minutes: float,
        base_time_minutes: Optional[float] = None
    ) -> float:

        if base_time_minutes is None:
            base_time_minutes = self.BASE_TRAINING_TIME

        if base_time_minutes <= 0:
            return 1.0

        ratio = training_time_minutes / base_time_minutes

        if ratio <= 0.5:
            return 1.0 + self.TIME_WEIGHT
        elif ratio <= 1.0:
            bonus = self.TIME_WEIGHT * (1.0 - ratio) / 0.5
            return 1.0 + bonus
        elif ratio <= 2.0:
            penalty = self.TIME_WEIGHT * (ratio - 1.0)
            return 1.0 - penalty
        else:
            return 1.0 - self.TIME_WEIGHT

    def calculate_final_score(
        self,
        cosine_similarity: float,
        quality_score: float = 5.0,
        time_coefficient: float = 1.0,
        constraint_coefficient: float = 1.0
    ) -> float:

        if cosine_similarity < self.BASE_THRESHOLD:
            logger.warning(f"Cosine similarity below threshold: {cosine_similarity:.4f} < {self.BASE_THRESHOLD}")
            return 0.0

        normalized_quality = quality_score / 10.0

        combined_score = (
            self.SIMILARITY_WEIGHT * cosine_similarity +
            self.QUALITY_WEIGHT * normalized_quality
        )


        if combined_score >= self.SCORE_PIVOT_HIGH:
            normalized = (combined_score - self.SCORE_PIVOT_HIGH) / (1.0 - self.SCORE_PIVOT_HIGH)
            transformed = normalized ** self.EXPONENT_HIGH
            final_base = 0.85 + transformed * 0.15

        elif combined_score >= self.SCORE_PIVOT_LOW:

            normalized = (combined_score - self.SCORE_PIVOT_LOW) / (self.SCORE_PIVOT_HIGH - self.SCORE_PIVOT_LOW)
            transformed = normalized ** self.EXPONENT_MID
            final_base = 0.50 + transformed * 0.35

        else:
            normalized = (combined_score - self.BASE_THRESHOLD) / (self.SCORE_PIVOT_LOW - self.BASE_THRESHOLD)
            transformed = normalized ** self.EXPONENT_LOW
            final_base = transformed * 0.50

        final_weight = final_base * time_coefficient * constraint_coefficient

        final_score = final_weight * 10.0

        logger.debug(f"Score calculation: cosine={cosine_similarity:.4f}, quality={quality_score:.2f}, "
                    f"combined={combined_score:.4f}, zone={'high' if combined_score >= self.SCORE_PIVOT_HIGH else 'mid' if combined_score >= self.SCORE_PIVOT_LOW else 'low'}, "
                    f"final={final_score:.2f}")

        return max(0.0, min(10.0, final_score))

    def compare_image_with_target(
        self,
        image: Image.Image,
        target_vector: List[float],
        quality_score: Optional[float] = None,
        training_time_minutes: Optional[float] = None,
        skip_nsfw: bool = False
    ) -> dict:

        image_vector = self.extract_vector_from_image(image)

        if len(image_vector) != len(target_vector):
            raise ValueError(
                f"Vector dimension mismatch: image={len(image_vector)}, target={len(target_vector)}"
            )

        similarity = self.cosine_similarity(image_vector, target_vector)

        nsfw_score = 0.0
        rejected = False
        reject_reason = None

        if not skip_nsfw:
            nsfw_score = self.detect_nsfw(image)
            if nsfw_score >= self.NSFW_THRESHOLD:
                rejected = True
                reject_reason = f"Content safety violation (NSFW score: {nsfw_score:.2f})"
                logger.warning(reject_reason)

        if quality_score is None:
            quality_score = self.evaluate_quality(image)

        time_coefficient = 1.0
        if training_time_minutes is not None:
            time_coefficient = self.calculate_time_coefficient(training_time_minutes)

        if rejected:
            final_score = 0.0
        else:
            final_score = self.calculate_final_score(
                cosine_similarity=similarity,
                quality_score=quality_score,
                time_coefficient=time_coefficient
            )

        combined_score = (
            self.SIMILARITY_WEIGHT * similarity +
            self.QUALITY_WEIGHT * (quality_score / 10.0)
        )

        if combined_score >= self.SCORE_PIVOT_HIGH:
            score_zone = "high"
        elif combined_score >= self.SCORE_PIVOT_LOW:
            score_zone = "mid"
        else:
            score_zone = "low"

        return {
            "cosine_similarity": similarity,
            "quality_score": quality_score,
            "combined_score": combined_score,
            "score_zone": score_zone,
            "time_coefficient": time_coefficient,
            "training_time_minutes": training_time_minutes,
            "base_training_time": self.BASE_TRAINING_TIME,
            "final_score": final_score,
            "nsfw_score": nsfw_score,
            "rejected": rejected,
            "reject_reason": reject_reason,
            "image_vector_dim": len(image_vector),
            "target_vector_dim": len(target_vector),
            "threshold": self.BASE_THRESHOLD,
            "pivot_high": self.SCORE_PIVOT_HIGH,
            "pivot_low": self.SCORE_PIVOT_LOW,
            "nsfw_threshold": self.NSFW_THRESHOLD
        }

    def compare_two_images(
        self,
        image1: Image.Image,
        image2: Image.Image,
        quality_score: Optional[float] = None,
        training_time_minutes: Optional[float] = None,
        skip_nsfw: bool = False
    ) -> dict:

        vector1 = self.extract_vector_from_image(image1)
        vector2 = self.extract_vector_from_image(image2)

        similarity = self.cosine_similarity(vector1, vector2)

        nsfw_score = 0.0
        rejected = False
        reject_reason = None

        if not skip_nsfw:
            nsfw_score = self.detect_nsfw(image1)
            if nsfw_score >= self.NSFW_THRESHOLD:
                rejected = True
                reject_reason = f"Content safety violation (NSFW score: {nsfw_score:.2f})"
                logger.warning(reject_reason)

        if quality_score is None:
            quality_score = self.evaluate_quality(image1)

        time_coefficient = 1.0
        if training_time_minutes is not None:
            time_coefficient = self.calculate_time_coefficient(training_time_minutes)

        if rejected:
            final_score = 0.0
        else:
            final_score = self.calculate_final_score(
                cosine_similarity=similarity,
                quality_score=quality_score,
                time_coefficient=time_coefficient
            )

        combined_score = (
            self.SIMILARITY_WEIGHT * similarity +
            self.QUALITY_WEIGHT * (quality_score / 10.0)
        )

        if combined_score >= self.SCORE_PIVOT_HIGH:
            score_zone = "high"
        elif combined_score >= self.SCORE_PIVOT_LOW:
            score_zone = "mid"
        else:
            score_zone = "low"

        return {
            "cosine_similarity": similarity,
            "quality_score": quality_score,
            "combined_score": combined_score,
            "score_zone": score_zone,
            "time_coefficient": time_coefficient,
            "training_time_minutes": training_time_minutes,
            "base_training_time": self.BASE_TRAINING_TIME,
            "final_score": final_score,
            "nsfw_score": nsfw_score,
            "rejected": rejected,
            "reject_reason": reject_reason,
            "vector1_dim": len(vector1),
            "vector2_dim": len(vector2),
            "threshold": self.BASE_THRESHOLD,
            "pivot_high": self.SCORE_PIVOT_HIGH,
            "pivot_low": self.SCORE_PIVOT_LOW,
            "nsfw_threshold": self.NSFW_THRESHOLD
        }


def load_target_vector_from_json(json_path: str) -> List[float]:
    logger.info(f"Loading target vector from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "target_vector" in data:
        return data["target_vector"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON file must contain 'target_vector' field or be a vector array")


def main():
    parser = argparse.ArgumentParser(
        description="Compare image similarity with target vector using DINOv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare image with target vector from JSON file
  python compare_image_similarity.py --image ./generated.png --target-json ./target.json

  # Compare image with target vector string
  python compare_image_similarity.py --image ./generated.png --target-vector "[0.1, 0.2, ...]"

  # Compare two images directly
  python compare_image_similarity.py --image ./image1.png --image2 ./image2.png

  # Compare image from URL with target vector
  python compare_image_similarity.py --image-url https://example.com/image.png --target-json ./target.json
        """
    )

    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", help="Path to local image file")
    image_group.add_argument("--image-url", help="URL of image to download")

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target-json", help="Path to JSON file containing target_vector")
    target_group.add_argument("--target-vector", help="Target vector as JSON string")
    target_group.add_argument("--image2", help="Path to second image for direct comparison")

    parser.add_argument("--quality-score", type=float, default=None,
                        help="Quality score (0-10). If not specified, auto-evaluate using CLIP model")
    parser.add_argument("--training-time", type=float, default=None,
                        help="Training time in minutes. Affects final score by ±15%%")
    parser.add_argument("--base-time", type=float, default=30,
                        help="Base training time in minutes (default: 30)")
    parser.add_argument("--skip-nsfw", action="store_true",
                        help="Skip NSFW content detection")

    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output the final score")

    args = parser.parse_args()

    calculator = ImageSimilarityCalculator()

    if args.image:
        image1 = calculator.load_image_from_path(args.image)
        image1_source = args.image
    else:
        image1 = calculator.load_image_from_url(args.image_url)
        image1_source = args.image_url

    quality_score = args.quality_score
    training_time = args.training_time
    skip_nsfw = args.skip_nsfw

    if args.base_time != 30:
        calculator.BASE_TRAINING_TIME = args.base_time

    if args.image2:
        image2 = calculator.load_image_from_path(args.image2)
        result = calculator.compare_two_images(
            image1, image2,
            quality_score=quality_score,
            training_time_minutes=training_time,
            skip_nsfw=skip_nsfw
        )
        result["image1"] = image1_source
        result["image2"] = args.image2
        result["comparison_type"] = "image_to_image"

    elif args.target_json:
        target_vector = load_target_vector_from_json(args.target_json)
        result = calculator.compare_image_with_target(
            image1, target_vector,
            quality_score=quality_score,
            training_time_minutes=training_time,
            skip_nsfw=skip_nsfw
        )
        result["image"] = image1_source
        result["target_source"] = args.target_json
        result["comparison_type"] = "image_to_target_vector"

    else:
        target_vector = json.loads(args.target_vector)
        result = calculator.compare_image_with_target(
            image1, target_vector,
            quality_score=quality_score,
            training_time_minutes=training_time,
            skip_nsfw=skip_nsfw
        )
        result["image"] = image1_source
        result["target_source"] = "command_line"
        result["comparison_type"] = "image_to_target_vector"

    if args.quiet:
        print(f"{result['final_score']:.6f}")
    else:
        print("")
        print("=" * 60)
        print("                  SCORING RESULT")
        print("=" * 60)
        print(f"  (Comparison Type): {result['comparison_type']}")
        print("-" * 60)

        # 相似度分数
        print(f"  余弦相似度 (Cosine Similarity): {result['cosine_similarity']:.6f}")
        print(f"  质量分数 (Quality Score):       {result['quality_score']:.2f} / 10")
        print(f"  组合分数 (Combined Score):      {result['combined_score']:.6f}")
        print(f"  分数区间 (Score Zone):          {result['score_zone']}")
        print(f"  相似度阈值 (Threshold):         {result['threshold']:.2f}")
        print(f"  高分区分界点 (Pivot High):      {result['pivot_high']:.2f}")
        print(f"  低分区分界点 (Pivot Low):       {result['pivot_low']:.2f}")

        # 训练时间
        print("-" * 60)
        if result.get('training_time_minutes') is not None:
            time_effect = (result['time_coefficient'] - 1.0) * 100
            time_effect_str = f"+{time_effect:.0f}%" if time_effect > 0 else f"{time_effect:.0f}%"
            print(f"  训练时间 (Training Time):       {result['training_time_minutes']:.1f} 分钟")
            print(f"  基准时间 (Base Time):           {result['base_training_time']:.1f} 分钟")
            print(f"  时间系数 (Time Coefficient):    {result['time_coefficient']:.2f} ({time_effect_str})")
        else:
            print(f"  训练时间 (Training Time):       未指定")
            print(f"  时间系数 (Time Coefficient):    1.00 (默认)")

        # NSFW 检测结果
        print("-" * 60)
        nsfw_status = "安全 (Safe)" if result['nsfw_score'] < result['nsfw_threshold'] else "不安全 (Unsafe)"
        print(f"  NSFW 分数 (NSFW Score):         {result['nsfw_score']:.4f}")
        print(f"  NSFW 阈值 (NSFW Threshold):     {result['nsfw_threshold']:.2f}")
        print(f"  内容安全 (Content Safety):      {nsfw_status}")

        # 最终得分
        print("-" * 60)
        if result['rejected']:
            print(f"  ✗ 已拒绝 (Rejected): {result['reject_reason']}")
            print(f"  ★ 最终得分 (Final Score):       0.0000 / 10")
        else:
            print(f"  ★ 最终得分 (Final Score):       {result['final_score']:.4f} / 10")
        print("=" * 60)

        # 评级说明
        final = result['final_score']
        if result['rejected']:
            grade = "已拒绝 (Rejected)"
        elif final == 0:
            grade = "不合格 (低于阈值)"
        elif final < 3:
            grade = "较差"
        elif final < 5:
            grade = "一般"
        elif final < 7:
            grade = "良好"
        elif final < 9:
            grade = "优秀"
        else:
            grade = "完美"
        print(f"  评级 (Grade): {grade}")
        print("=" * 60)
        print("")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to: {args.output}")
        else:
            print(" (Detail):")
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
