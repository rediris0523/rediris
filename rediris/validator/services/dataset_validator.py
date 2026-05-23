from typing import Dict, Any, Optional, List, Union
from rediris.common.utils.logging import setup_logger
import random
import tempfile
import shutil
from pathlib import Path
import asyncio

logger = setup_logger(__name__)

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not available, dataset validation will be limited")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available for text quality evaluation")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available for image-text alignment check")

try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    IMAGE_CLASSIFICATION_AVAILABLE = True
except ImportError:
    IMAGE_CLASSIFICATION_AVAILABLE = False
    logger.warning("Image classification not available")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available for image processing")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DatasetValidator:

    SAMPLE_COUNT = 10
    MIN_SAMPLES_REQUIRED = 5

    TEXT_QUALITY_MODEL = "textattack/distilbert-base-uncased-CoLA"
    NSFW_IMAGE_MODEL = "Falconsai/nsfw_image_detection"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 10000
    MIN_IMAGE_SIZE = 64
    MAX_IMAGE_SIZE = 4096

    QUALITY_THRESHOLD = 0.6
    NSFW_THRESHOLD = 0.7

    def __init__(self):
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._text_quality_pipeline = None
        self._nsfw_pipeline = None
        self._clip_model = None
        self._clip_processor = None
        self._cache_dir = Path(tempfile.gettempdir()) / "moirai_dataset_cache"
        self._cache_dir.mkdir(exist_ok=True)

    def _load_text_quality_model(self):
        if self._text_quality_pipeline is None and TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading text quality model: {self.TEXT_QUALITY_MODEL}")
                self._text_quality_pipeline = pipeline(
                    "text-classification",
                    model=self.TEXT_QUALITY_MODEL,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Text quality model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load text quality model: {e}")
                self._text_quality_pipeline = None

    def _load_nsfw_model(self):
        if self._nsfw_pipeline is None and IMAGE_CLASSIFICATION_AVAILABLE:
            try:
                logger.info(f"Loading NSFW detection model: {self.NSFW_IMAGE_MODEL}")
                self._nsfw_pipeline = pipeline(
                    "image-classification",
                    model=self.NSFW_IMAGE_MODEL,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("NSFW detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NSFW model: {e}")
                self._nsfw_pipeline = None

    def _load_clip_model(self):
        if self._clip_model is None and CLIP_AVAILABLE:
            try:
                logger.info(f"Loading CLIP model: {self.CLIP_MODEL}")
                self._clip_model = CLIPModel.from_pretrained(self.CLIP_MODEL).to(self.device)
                self._clip_processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL)
                self._clip_model.eval()
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                self._clip_model = None
                self._clip_processor = None

    async def validate_dataset(
        self,
        dataset_url: str,
        task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = {
            "is_valid": False,
            "format_check": {"passed": False, "details": ""},
            "quality_check": {"passed": False, "details": ""},
            "safety_check": {"passed": False, "details": ""},
            "sample_results": [],
            "details": "",
            "rejection_reason": None
        }

        try:
            dataset_id = self._parse_dataset_url(dataset_url)
            if not dataset_id:
                result["rejection_reason"] = "Invalid HuggingFace dataset URL format"
                logger.warning(f"Dataset validation failed early: {result['rejection_reason']} (url={dataset_url})")
                return result

            logger.info(f"Starting dataset validation for: {dataset_id}")

            dataset, format_result = await self._load_and_check_format(dataset_id)
            result["format_check"] = format_result
            if not format_result["passed"]:
                result["rejection_reason"] = f"Format check failed: {format_result['details']}"
                logger.warning(
                    f"Dataset validation failed at format_check: dataset={dataset_id}, "
                    f"details={format_result.get('details')}"
                )
                return result

            min_sample_count = self._get_min_sample_count(task_info)
            actual_sample_count = format_result.get("num_rows", 0)
            if min_sample_count > 0 and actual_sample_count < min_sample_count:
                result["rejection_reason"] = f"Insufficient dataset samples: {actual_sample_count} < {min_sample_count} (required by dataset_spec.sample_count)"
                result["format_check"]["details"] = f"Dataset has {actual_sample_count} samples, but minimum {min_sample_count} required"
                result["format_check"]["passed"] = False
                logger.warning(
                    "Dataset validation failed at sample_count check: "
                    f"dataset={dataset_id}, actual={actual_sample_count}, required={min_sample_count}"
                )
                return result

            columns_result = self._validate_required_columns(task_info, format_result.get("columns", []))
            if not columns_result["passed"]:
                result["rejection_reason"] = columns_result["details"]
                result["format_check"]["details"] = columns_result["details"]
                result["format_check"]["passed"] = False
                logger.warning(
                    f"Dataset validation failed at columns check: dataset={dataset_id}, "
                    f"details={columns_result.get('details')}"
                )
                return result

            task_type = self._determine_task_type(task_info)
            logger.info(f"Detected task type: {task_type}")

            samples = self._random_sample(dataset, self.SAMPLE_COUNT)
            if len(samples) < self.MIN_SAMPLES_REQUIRED:
                result["rejection_reason"] = f"Insufficient samples: {len(samples)} < {self.MIN_SAMPLES_REQUIRED}"
                logger.warning(
                    "Dataset validation failed at random sampling: "
                    f"dataset={dataset_id}, sampled={len(samples)}, required={self.MIN_SAMPLES_REQUIRED}"
                )
                return result

            logger.info(f"Sampled {len(samples)} records for validation")

            if task_type == "text":
                quality_result, sample_results = await self._validate_text_samples(samples, task_info)
            else:
                quality_result, sample_results = await self._validate_image_samples(samples, task_info)

            result["quality_check"] = quality_result
            result["sample_results"] = sample_results
            if not quality_result["passed"]:
                result["rejection_reason"] = f"Quality check failed: {quality_result['details']}"
                failed_samples = [s for s in sample_results if not s.get("valid", False)]
                failed_preview = []
                for s in failed_samples[:3]:
                    failed_preview.append({
                        "index": s.get("index"),
                        "issues": s.get("issues", []),
                        "quality_score": s.get("quality_score"),
                    })
                logger.warning(
                    "Dataset validation failed at quality_check: "
                    f"dataset={dataset_id}, details={quality_result.get('details')}, "
                    f"pass_rate={quality_result.get('pass_rate')}, avg_score={quality_result.get('average_score')}, "
                    f"failed_preview={failed_preview}"
                )
                return result

            if task_type == "text":
                safety_result = await self._check_text_safety(samples)
            else:
                safety_result = await self._check_image_safety(samples)

            result["safety_check"] = safety_result
            if not safety_result["passed"]:
                result["rejection_reason"] = f"Safety check failed: {safety_result['details']}"
                logger.warning(
                    "Dataset validation failed at safety_check: "
                    f"dataset={dataset_id}, details={safety_result.get('details')}"
                )
                return result

            result["is_valid"] = True
            result["details"] = f"All validation checks passed for {len(samples)} samples"
            logger.info(f"Dataset {dataset_id} passed all validation checks")

        except Exception as e:
            logger.error(f"Dataset validation error: {e}", exc_info=True)
            result["rejection_reason"] = f"Validation error: {str(e)}"

        if not result.get("is_valid", False):
            logger.warning(
                "Dataset validation final result: invalid. "
                f"rejection_reason={result.get('rejection_reason')}, "
                f"format_check={result.get('format_check')}, "
                f"quality_check={result.get('quality_check')}, "
                f"safety_check={result.get('safety_check')}"
            )

        return result

    def _parse_dataset_url(self, url: str) -> Optional[str]:
        from rediris.common.utils.huggingface import parse_dataset_url, is_valid_hf_repository_id
        try:
            dataset_id = parse_dataset_url(url)
            if dataset_id and is_valid_hf_repository_id(dataset_id):
                return dataset_id
            return None
        except Exception as e:
            logger.error(f"Error parsing dataset URL: {e}")
            return None

    def _get_min_sample_count(self, task_info: Dict[str, Any]) -> int:
        try:
            workflow_spec = task_info.get("workflow_spec", {})
            if not workflow_spec:
                return 0

            dataset_spec = workflow_spec.get("dataset_spec", {})
            if not dataset_spec:
                return 0

            sample_count = dataset_spec.get("sample_count", 0)
            if isinstance(sample_count, int) and sample_count > 0:
                logger.info(f"Minimum sample count requirement: {sample_count}")
                return sample_count

            return 0
        except Exception as e:
            logger.warning(f"Error getting min sample count: {e}")
            return 0

    def _validate_required_columns(self, task_info: Dict[str, Any], dataset_columns: List[str]) -> Dict[str, Any]:
        result = {"passed": True, "details": ""}

        try:
            workflow_spec = task_info.get("workflow_spec", {})
            if not workflow_spec:
                return result

            dataset_spec = workflow_spec.get("dataset_spec", {})
            if not dataset_spec:
                return result

            task_type = self._determine_task_type(task_info)
            logger.info(f"Validating columns for task type: {task_type}")

            missing_columns = []
            configured_columns = []

            if task_type == "image":
                image_column = dataset_spec.get("image_column")
                if image_column:
                    if image_column not in dataset_columns:
                        missing_columns.append(f"image_column '{image_column}'")
                    else:
                        configured_columns.append(f"image_column='{image_column}'")

                caption_column = dataset_spec.get("caption_column")
                if caption_column:
                    if caption_column not in dataset_columns:
                        missing_columns.append(f"caption_column '{caption_column}'")
                    else:
                        configured_columns.append(f"caption_column='{caption_column}'")
            else:
                question_column = dataset_spec.get("question_column")
                if question_column:
                    if question_column not in dataset_columns:
                        missing_columns.append(f"question_column '{question_column}'")
                    else:
                        configured_columns.append(f"question_column='{question_column}'")

                answer_column = dataset_spec.get("answer_column")
                if answer_column:
                    if answer_column not in dataset_columns:
                        missing_columns.append(f"answer_column '{answer_column}'")
                    else:
                        configured_columns.append(f"answer_column='{answer_column}'")

            if missing_columns:
                result["passed"] = False
                result["details"] = f"Required columns not found in dataset: {', '.join(missing_columns)}. Available columns: {dataset_columns}"
                logger.warning(f"Column validation failed: {result['details']}")
            elif configured_columns:
                logger.info(f"Column validation passed: {', '.join(configured_columns)}")

            return result

        except Exception as e:
            logger.warning(f"Error validating required columns: {e}")
            return result

    def _determine_task_type(self, task_info: Dict[str, Any]) -> str:
        workflow_spec = task_info.get("workflow_spec", {})
        task_type = workflow_spec.get("task_type", "") or task_info.get("task_type", "")

        if task_type:
            task_type_lower = task_type.lower()
            if any(keyword in task_type_lower for keyword in ["image", "vision", "photo", "picture"]):
                return "image"
            elif any(keyword in task_type_lower for keyword in ["text", "nlp", "language", "chat"]):
                return "text"

        training_spec = workflow_spec.get("training_spec", {})
        base_model = training_spec.get("base_model", "") or task_info.get("base_model", "")
        base_model_lower = base_model.lower()

        if any(keyword in base_model_lower for keyword in ["flux", "sdxl", "stable-diffusion", "diffusion", "dalle"]):
            return "image"
        elif any(keyword in base_model_lower for keyword in ["qwen", "llama", "gpt", "bert", "t5", "mistral"]):
            return "text"

        return "text"

    async def _load_and_check_format(self, dataset_id: str) -> tuple:
        format_result = {"passed": False, "details": "", "columns": [], "num_rows": 0}

        if not DATASETS_AVAILABLE:
            format_result["details"] = "datasets library not available"
            return None, format_result

        try:
            logger.info(f"Loading dataset: {dataset_id}")

            loop = asyncio.get_event_loop()
            dataset = await loop.run_in_executor(
                None,
                lambda: load_dataset(dataset_id, split="train", trust_remote_code=True)
            )

            if dataset is None or len(dataset) == 0:
                format_result["details"] = "Dataset is empty"
                return None, format_result

            columns = list(dataset.column_names) if hasattr(dataset, 'column_names') else []
            num_rows = len(dataset)

            logger.info(f"Dataset loaded: {num_rows} rows, columns: {columns}")

            format_result["passed"] = True
            format_result["details"] = f"Dataset loaded successfully: {num_rows} rows"
            format_result["columns"] = columns
            format_result["num_rows"] = num_rows

            return dataset, format_result

        except Exception as e:
            error_msg = str(e)
            if "DatasetGenerationError" in type(e).__name__ or "An error occurred while generating the dataset" in error_msg:
                friendly_msg = f"Dataset '{dataset_id}' uses an incompatible format and cannot be loaded. Please use a standard HuggingFace dataset format."
            elif "FileNotFoundError" in type(e).__name__ or "doesn't exist" in error_msg.lower():
                friendly_msg = f"Dataset '{dataset_id}' not found on HuggingFace."
            elif "ConnectionError" in type(e).__name__ or "connection" in error_msg.lower():
                friendly_msg = f"Failed to connect to HuggingFace to load dataset '{dataset_id}'."
            elif "PermissionError" in type(e).__name__ or "permission" in error_msg.lower() or "gated" in error_msg.lower():
                friendly_msg = f"Dataset '{dataset_id}' requires authentication or is gated."
            else:
                friendly_msg = f"Failed to load dataset '{dataset_id}': {error_msg[:200]}"

            logger.warning(f"Dataset load failed for {dataset_id}: {friendly_msg}")
            format_result["details"] = friendly_msg
            return None, format_result

    def _random_sample(self, dataset, count: int) -> List[Dict[str, Any]]:
        if dataset is None:
            return []

        try:
            total = len(dataset)
            sample_count = min(count, total)

            if sample_count <= 0:
                return []

            indices = random.sample(range(total), sample_count)
            samples = [dataset[i] for i in indices]

            return samples

        except Exception as e:
            logger.error(f"Error sampling dataset: {e}")
            return []

    async def _validate_text_samples(
        self,
        samples: List[Dict[str, Any]],
        task_info: Dict[str, Any]
    ) -> tuple:
        quality_result = {"passed": False, "details": "", "scores": []}
        sample_results = []

        try:
            self._load_text_quality_model()

            text_column = self._get_configured_text_column(task_info)
            if not text_column:
                text_column = self._find_text_column(samples[0] if samples else {})
            if not text_column:
                quality_result["details"] = "No text column found in dataset"
                return quality_result, sample_results

            valid_count = 0
            total_score = 0.0

            for i, sample in enumerate(samples):
                text = sample.get(text_column, "")
                sample_result = {
                    "index": i,
                    "valid": False,
                    "length": len(text) if text else 0,
                    "quality_score": 0.0,
                    "issues": []
                }

                if not text or not isinstance(text, str):
                    sample_result["issues"].append("Empty or invalid text")
                    sample_results.append(sample_result)
                    continue

                if len(text) < self.MIN_TEXT_LENGTH:
                    sample_result["issues"].append(f"Text too short: {len(text)} < {self.MIN_TEXT_LENGTH}")
                    sample_results.append(sample_result)
                    continue

                if len(text) > self.MAX_TEXT_LENGTH:
                    sample_result["issues"].append(f"Text too long: {len(text)} > {self.MAX_TEXT_LENGTH}")
                    text = text[:self.MAX_TEXT_LENGTH]

                quality_score = await self._evaluate_text_quality(text)
                sample_result["quality_score"] = quality_score
                total_score += quality_score

                if quality_score >= self.QUALITY_THRESHOLD:
                    sample_result["valid"] = True
                    valid_count += 1
                else:
                    sample_result["issues"].append(f"Low quality score: {quality_score:.2f}")

                sample_results.append(sample_result)

            pass_rate = valid_count / len(samples) if samples else 0
            avg_score = total_score / len(samples) if samples else 0

            quality_result["scores"] = [r["quality_score"] for r in sample_results]
            quality_result["pass_rate"] = pass_rate
            quality_result["average_score"] = avg_score

            if pass_rate >= 0.7:
                quality_result["passed"] = True
                quality_result["details"] = f"Quality check passed: {valid_count}/{len(samples)} samples valid (avg score: {avg_score:.2f})"
            else:
                quality_result["details"] = f"Quality check failed: only {valid_count}/{len(samples)} samples valid (avg score: {avg_score:.2f})"

        except Exception as e:
            logger.error(f"Text validation error: {e}", exc_info=True)
            quality_result["details"] = f"Text validation error: {str(e)}"

        return quality_result, sample_results

    async def _validate_image_samples(
        self,
        samples: List[Dict[str, Any]],
        task_info: Dict[str, Any]
    ) -> tuple:
        quality_result = {"passed": False, "details": "", "scores": []}
        sample_results = []

        try:
            configured_image_col, configured_caption_col = self._get_configured_image_columns(task_info)

            image_column = configured_image_col or self._find_image_column(samples[0] if samples else {})
            if not image_column:
                quality_result["details"] = "No image column found in dataset"
                return quality_result, sample_results

            caption_column = configured_caption_col or self._find_caption_column(samples[0] if samples else {})

            self._load_clip_model()

            valid_count = 0
            total_score = 0.0

            for i, sample in enumerate(samples):
                image_data = sample.get(image_column)
                caption = sample.get(caption_column, "") if caption_column else ""

                sample_result = {
                    "index": i,
                    "valid": False,
                    "has_image": False,
                    "has_caption": bool(caption),
                    "image_size": None,
                    "quality_score": 0.0,
                    "issues": []
                }

                image = self._load_image(image_data)
                if image is None:
                    sample_result["issues"].append("Failed to load image")
                    sample_results.append(sample_result)
                    continue

                sample_result["has_image"] = True
                sample_result["image_size"] = image.size

                width, height = image.size
                if width < self.MIN_IMAGE_SIZE or height < self.MIN_IMAGE_SIZE:
                    sample_result["issues"].append(f"Image too small: {width}x{height}")
                    sample_results.append(sample_result)
                    continue

                if width > self.MAX_IMAGE_SIZE or height > self.MAX_IMAGE_SIZE:
                    sample_result["issues"].append(f"Image too large: {width}x{height}")

                quality_score = await self._evaluate_image_quality(image, caption)
                sample_result["quality_score"] = quality_score
                total_score += quality_score

                if quality_score >= self.QUALITY_THRESHOLD:
                    sample_result["valid"] = True
                    valid_count += 1
                else:
                    sample_result["issues"].append(f"Low quality score: {quality_score:.2f}")

                sample_results.append(sample_result)

            pass_rate = valid_count / len(samples) if samples else 0
            avg_score = total_score / len(samples) if samples else 0

            quality_result["scores"] = [r["quality_score"] for r in sample_results]
            quality_result["pass_rate"] = pass_rate
            quality_result["average_score"] = avg_score

            if pass_rate >= 0.7:
                quality_result["passed"] = True
                quality_result["details"] = f"Quality check passed: {valid_count}/{len(samples)} samples valid (avg score: {avg_score:.2f})"
            else:
                quality_result["details"] = f"Quality check failed: only {valid_count}/{len(samples)} samples valid (avg score: {avg_score:.2f})"

        except Exception as e:
            logger.error(f"Image validation error: {e}", exc_info=True)
            quality_result["details"] = f"Image validation error: {str(e)}"

        return quality_result, sample_results

    def _get_configured_text_column(self, task_info: Dict[str, Any]) -> Optional[str]:
        try:
            workflow_spec = task_info.get("workflow_spec", {})
            dataset_spec = workflow_spec.get("dataset_spec", {})

            question_column = dataset_spec.get("question_column")
            if question_column:
                return question_column

            answer_column = dataset_spec.get("answer_column")
            if answer_column:
                return answer_column

            return None
        except Exception:
            return None

    def _get_configured_image_columns(self, task_info: Dict[str, Any]) -> tuple:
        try:
            workflow_spec = task_info.get("workflow_spec", {})
            dataset_spec = workflow_spec.get("dataset_spec", {})

            image_column = dataset_spec.get("image_column")
            caption_column = dataset_spec.get("caption_column")

            return image_column, caption_column
        except Exception:
            return None, None

    def _find_text_column(self, sample: Dict[str, Any]) -> Optional[str]:
        text_columns = ["text", "content", "sentence", "input", "prompt", "question", "instruction"]

        for col in text_columns:
            if col in sample and isinstance(sample[col], str):
                return col

        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 10:
                return key

        return None

    def _find_image_column(self, sample: Dict[str, Any]) -> Optional[str]:
        image_columns = ["image", "img", "photo", "picture", "file"]

        for col in image_columns:
            if col in sample:
                return col

        for key, value in sample.items():
            if PIL_AVAILABLE and isinstance(value, Image.Image):
                return key
            if isinstance(value, dict) and "bytes" in value:
                return key

        return None

    def _find_caption_column(self, sample: Dict[str, Any]) -> Optional[str]:
        caption_columns = ["caption", "text", "description", "label", "prompt"]

        for col in caption_columns:
            if col in sample and isinstance(sample[col], str):
                return col

        return None

    def _load_image(self, image_data: Any) -> Optional["Image.Image"]:
        if not PIL_AVAILABLE:
            return None

        try:
            if isinstance(image_data, Image.Image):
                return image_data

            if isinstance(image_data, dict):
                if "bytes" in image_data:
                    return Image.open(io.BytesIO(image_data["bytes"]))
                if "path" in image_data:
                    return Image.open(image_data["path"])

            if isinstance(image_data, bytes):
                return Image.open(io.BytesIO(image_data))

            if isinstance(image_data, str):
                return Image.open(image_data)

            return None

        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            return None

    async def _evaluate_text_quality(self, text: str) -> float:
        if self._text_quality_pipeline is None:
            word_count = len(text.split())
            has_punctuation = any(c in text for c in ".!?")
            length_score = min(1.0, word_count / 20)
            structure_score = 0.3 if has_punctuation else 0.0
            return min(1.0, length_score * 0.7 + structure_score)

        try:
            truncated_text = text[:512]
            result = self._text_quality_pipeline(truncated_text)

            if result and len(result) > 0:
                label = result[0].get("label", "")
                score = result[0].get("score", 0.5)

                if label.lower() in ["acceptable", "positive", "1", "label_1"]:
                    return score
                else:
                    return 1.0 - score

            return 0.5

        except Exception as e:
            logger.warning(f"Text quality evaluation error: {e}")
            return 0.5

    async def _evaluate_image_quality(self, image: "Image.Image", caption: str = "") -> float:
        base_score = 0.7

        width, height = image.size
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3:
            base_score -= 0.2

        if width >= 512 and height >= 512:
            base_score += 0.1

        if self._clip_model is not None and self._clip_processor is not None and caption:
            try:
                alignment_score = await self._calculate_clip_alignment(image, caption)
                base_score = base_score * 0.5 + alignment_score * 0.5
            except Exception as e:
                logger.warning(f"CLIP alignment error: {e}")

        return min(1.0, max(0.0, base_score))

    async def _calculate_clip_alignment(self, image: "Image.Image", caption: str) -> float:
        if self._clip_model is None or self._clip_processor is None:
            return 0.5

        try:
            inputs = self._clip_processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = logits_per_image.softmax(dim=1)[0][0].item()

            return similarity

        except Exception as e:
            logger.warning(f"CLIP alignment calculation error: {e}")
            return 0.5

    async def _check_text_safety(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        safety_result = {"passed": True, "details": "", "flagged_samples": []}

        try:
            text_column = self._find_text_column(samples[0] if samples else {})
            if not text_column:
                return safety_result

            sensitive_patterns = [
                "password", "secret", "api_key", "token", "credential",
                "ssn", "social security", "credit card"
            ]

            flagged = []
            for i, sample in enumerate(samples):
                text = sample.get(text_column, "").lower()
                for pattern in sensitive_patterns:
                    if pattern in text:
                        flagged.append({"index": i, "reason": f"Contains sensitive pattern: {pattern}"})
                        break

            if flagged:
                safety_result["flagged_samples"] = flagged
                if len(flagged) > len(samples) * 0.3:
                    safety_result["passed"] = False
                    safety_result["details"] = f"Too many samples contain sensitive content: {len(flagged)}/{len(samples)}"
                else:
                    safety_result["details"] = f"Some samples flagged but within acceptable range: {len(flagged)}/{len(samples)}"
            else:
                safety_result["details"] = "No sensitive content detected"

        except Exception as e:
            logger.warning(f"Text safety check error: {e}")
            safety_result["details"] = f"Safety check error: {str(e)}"

        return safety_result

    async def _check_image_safety(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        safety_result = {"passed": True, "details": "", "flagged_samples": []}

        try:
            self._load_nsfw_model()

            image_column = self._find_image_column(samples[0] if samples else {})
            if not image_column:
                safety_result["details"] = "No image column found, skipping safety check"
                return safety_result

            if self._nsfw_pipeline is None:
                safety_result["details"] = "NSFW model not available, skipping safety check"
                return safety_result

            flagged = []
            for i, sample in enumerate(samples):
                image = self._load_image(sample.get(image_column))
                if image is None:
                    continue

                try:
                    result = self._nsfw_pipeline(image)
                    for item in result:
                        label = item.get("label", "").lower()
                        score = item.get("score", 0)
                        if "nsfw" in label and score > self.NSFW_THRESHOLD:
                            flagged.append({
                                "index": i,
                                "reason": f"NSFW content detected (score: {score:.2f})"
                            })
                            break
                except Exception as e:
                    logger.warning(f"NSFW check failed for sample {i}: {e}")

            if flagged:
                safety_result["flagged_samples"] = flagged
                safety_result["passed"] = False
                safety_result["details"] = f"NSFW content detected in {len(flagged)} samples"
            else:
                safety_result["details"] = "No NSFW content detected"

        except Exception as e:
            logger.warning(f"Image safety check error: {e}")
            safety_result["details"] = f"Safety check error: {str(e)}"

        return safety_result


    def cleanup_cache(self):
        try:
            if self._cache_dir.exists():
                shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(exist_ok=True)
                logger.info("Dataset cache cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup cache: {e}")
