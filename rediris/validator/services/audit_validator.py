
from typing import Dict, Any, Optional, Union
import torch
import os
import tempfile
import shutil
import gc
import random
import warnings
from pathlib import Path
from PIL import Image
from rediris.validator.schemas.audit import AuditTaskRequest
from rediris.validator.services.quality_evaluator import QualityEvaluator
from rediris.validator.services.content_filter import ContentFilter
from rediris.validator.services.dataset_validator import DatasetValidator
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    from transformers import AutoImageProcessor, AutoModel
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False
    logger.warning("DINOv2 not available, image feature extraction will use fallback")

try:
    from diffusers import DiffusionPipeline, StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers not available, image LoRA validation will be limited")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, text LoRA validation will be limited")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available, text feature extraction will use fallback")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available, LoRA download will be limited")


class AuditValidator:

    DINOV2_MODEL_ID = "facebook/dinov2-large"

    BASE_THRESHOLD = 0.35
    NSFW_THRESHOLD = 0.7

    SIMILARITY_WEIGHT = 0.95
    QUALITY_WEIGHT = 0.05

    SCORE_PIVOT_HIGH = 0.94
    SCORE_PIVOT_LOW = 0.85

    EXPONENT_HIGH = 0.5
    EXPONENT_MID = 2.5
    EXPONENT_LOW = 8

    BASE_TRAINING_TIME = 30
    TIME_WEIGHT = 0.15

    def __init__(self):
        self.quality_evaluator = QualityEvaluator()
        self.content_filter = ContentFilter()
        self.dataset_validator = DatasetValidator()
        self.dinov2_model = None
        self.dinov2_processor = None
        self.text_encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lora_cache_dir = Path(tempfile.gettempdir()) / "moirai_lora_cache"
        self._lora_cache_dir.mkdir(exist_ok=True)
        # Cache for image generation pipelines (base_model -> pipeline)
        self._image_pipelines = {}
        # Track currently loaded LoRA for each pipeline
        self._current_lora = {}
        self._load_models()

    def _load_models(self):
        try:
            if DINOV2_AVAILABLE:
                logger.info(f"Loading DINOv2 model ({self.DINOV2_MODEL_ID})...")
                self.dinov2_processor = AutoImageProcessor.from_pretrained(self.DINOV2_MODEL_ID)
                self.dinov2_model = AutoModel.from_pretrained(self.DINOV2_MODEL_ID).to(self.device)
                self.dinov2_model.eval()
                logger.info("DINOv2 model loaded successfully (1024 dimensions)")
            else:
                self.dinov2_model = None
                self.dinov2_processor = None

            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("Loading Sentence-BERT model...")
                self.text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                logger.info("Sentence-BERT model loaded successfully")
            else:
                self.text_encoder = None

            logger.info("Feature extraction models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            self.dinov2_model = None
            self.dinov2_processor = None
            self.text_encoder = None

    def _cleanup_gpu_memory(self, *objects_to_delete):
        try:
            # Delete objects
            for obj in objects_to_delete:
                if obj is not None:
                    try:
                        del obj
                    except Exception:
                        pass
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {e}")

    def _load_peft_lora_to_diffusers(self, pipe, lora_path: str) -> bool:
        try:
            import safetensors.torch
            from pathlib import Path

            lora_path = Path(lora_path)

            safetensor_files = list(lora_path.glob("*.safetensors"))
            if not safetensor_files:
                logger.debug(f"No safetensors files found in {lora_path}")
                return False

            lora_file = None
            for f in safetensor_files:
                if f.name == "adapter_model.safetensors":
                    lora_file = f
                    break
            if lora_file is None:
                lora_file = safetensor_files[0]

            logger.info(f"Loading PEFT LoRA from: {lora_file}")
            state_dict = safetensors.torch.load_file(str(lora_file))

            is_peft_format = any(k.startswith("base_model.model.") for k in state_dict.keys())

            if not is_peft_format:
                logger.debug("Not PEFT format, trying direct load")
                return False

            model_component = None
            if hasattr(pipe, 'transformer'):
                model_component = pipe.transformer
                component_name = "transformer"
            elif hasattr(pipe, 'unet'):
                model_component = pipe.unet
                component_name = "unet"
            else:
                logger.warning("Pipeline has neither transformer nor unet")
                return False

            converted_state_dict = {}
            prefix_to_remove = "base_model.model."

            for key, value in state_dict.items():
                if key.startswith(prefix_to_remove):
                    new_key = key[len(prefix_to_remove):]
                    converted_state_dict[new_key] = value
                else:
                    converted_state_dict[key] = value

            logger.info(f"Converted {len(converted_state_dict)} LoRA weights from PEFT format")

            try:
                from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict

                adapter_config_path = lora_path / "adapter_config.json"
                if adapter_config_path.exists():
                    import json
                    with open(adapter_config_path, "r") as f:
                        adapter_config = json.load(f)

                    lora_config = LoraConfig(
                        r=adapter_config.get("r", 16),
                        lora_alpha=adapter_config.get("lora_alpha", 32),
                        target_modules=adapter_config.get("target_modules", []),
                        lora_dropout=adapter_config.get("lora_dropout", 0.0),
                    )

                    model_component = get_peft_model(model_component, lora_config)
                    set_peft_model_state_dict(model_component, state_dict)

                    if component_name == "transformer":
                        pipe.transformer = model_component
                    else:
                        pipe.unet = model_component

                    logger.info(f"PEFT LoRA weights loaded successfully to {component_name}")
                    return True

            except Exception as e:
                logger.debug(f"PEFT direct load failed: {e}, trying manual injection")

            try:
                model_state = model_component.state_dict()
                matched_keys = 0

                for key, value in converted_state_dict.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            model_state[key] = value.to(model_state[key].device, dtype=model_state[key].dtype)
                            matched_keys += 1

                if matched_keys > 0:
                    model_component.load_state_dict(model_state, strict=False)
                    logger.info(f"Manually injected {matched_keys} LoRA weights")
                    return True
                else:
                    logger.warning("No matching keys found for manual injection")
                    return False

            except Exception as e:
                logger.error(f"Manual weight injection failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to load PEFT LoRA: {e}", exc_info=True)
            return False

    async def process_audit_task(self, request: Union[AuditTaskRequest, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(request, dict):
            audit_task_id = request.get("audit_task_id", "")
            miner_hotkey = request.get("miner_hotkey", "")
            lora_url = request.get("lora_url", "")
            dataset_url = request.get("dataset_url", "")
            audit_type = request.get("audit_type", "lora")
            task_info = request.get("task_info", {})
            training_time_minutes = request.get("training_time_minutes")
        else:
            audit_task_id = request.audit_task_id
            miner_hotkey = request.miner_hotkey
            lora_url = getattr(request, 'lora_url', "")
            dataset_url = getattr(request, 'dataset_url', "")
            audit_type = getattr(request, 'audit_type', "lora")
            task_info = request.task_info
            training_time_minutes = getattr(request, 'training_time_minutes', None)

        if audit_type == "dataset":
            return await self._process_dataset_audit(audit_task_id, miner_hotkey, dataset_url, task_info)

        task_type = self._determine_task_type(task_info)

        logger.info(f"Processing audit task {audit_task_id}: type={task_type}, miner={miner_hotkey[:20]}...")

        try:
            lora_path = await self._download_lora(lora_url)
            if not lora_path:
                return self._create_error_result(
                    audit_task_id, miner_hotkey,
                    reason="Failed to download LoRA model"
                )

            generated_content = await self._generate_content(task_info, lora_path, task_type)
            if generated_content is None:
                return self._create_error_result(
                    audit_task_id, miner_hotkey,
                    reason="Failed to generate content with LoRA"
                )

            cosine_similarity = await self._calculate_cosine_similarity(
                task_info, generated_content, task_type
            )

            quality_score = await self.quality_evaluator.evaluate_quality(
                task_type, generated_content
            )

            content_safety_score = 0.0
            if task_type in ["image_lora", "image_lora_creation"]:
                content_safety_score = await self.content_filter.detect_content(generated_content)
                if content_safety_score >= self.NSFW_THRESHOLD:
                    logger.warning(f"Content safety violation: score={content_safety_score:.2f}")
                    return {
                        "audit_task_id": audit_task_id,
                        "miner_hotkey": miner_hotkey,
                        "cosine_similarity": cosine_similarity,
                        "quality_score": 0.0,
                        "final_score": 0.0,
                        "content_safety_score": content_safety_score,
                        "time_coefficient": 1.0,
                        "training_time_minutes": training_time_minutes,
                        "rejected": True,
                        "reason": "Content safety violation (NSFW detected)"
                    }

            time_coefficient = 1.0
            if training_time_minutes is not None:
                time_coefficient = self.calculate_time_coefficient(training_time_minutes)

            final_score = self._calculate_final_score(
                cosine_similarity, quality_score, time_coefficient
            )

            logger.info(f"Audit task {audit_task_id} completed: "
                       f"cosine_sim={cosine_similarity:.4f}, "
                       f"quality={quality_score:.2f}, "
                       f"time_coef={time_coefficient:.2f}, "
                       f"final={final_score:.2f}")

            return {
                "audit_task_id": audit_task_id,
                "miner_hotkey": miner_hotkey,
                "cosine_similarity": cosine_similarity,
                "quality_score": quality_score,
                "final_score": final_score,
                "content_safety_score": content_safety_score,
                "time_coefficient": time_coefficient,
                "training_time_minutes": training_time_minutes,
                "rejected": False,
                "reason": None
            }

        except Exception as e:
            logger.error(f"Audit task {audit_task_id} failed: {e}", exc_info=True)
            return self._create_error_result(
                audit_task_id, miner_hotkey,
                reason=f"Validation error: {str(e)}"
            )

    def _determine_task_type(self, task_info: Dict[str, Any]) -> str:
        task_type = task_info.get("task_type", "")

        if task_type:
            task_type_lower = task_type.lower()
            if "text" in task_type_lower:
                return "text_lora"
            elif "image" in task_type_lower:
                return "image_lora"
            return task_type

        base_model = task_info.get("base_model", "").lower()
        if any(keyword in base_model for keyword in ["flux", "sdxl", "stable-diffusion", "diffusion"]):
            return "image_lora"
        elif any(keyword in base_model for keyword in ["qwen", "llama", "gpt", "bert", "t5"]):
            return "text_lora"

        return "image_lora"

    async def _download_lora(self, lora_url: str) -> Optional[str]:

        if not lora_url:
            logger.error("Empty LoRA URL provided")
            return None

        try:
            import os
            from urllib.parse import unquote

            # Support local filesystem paths (commonly produced when skipping HF upload).
            # Example: "file:///data/code/models/TASK-0330"
            if lora_url.startswith("file://"):
                local_path = unquote(lora_url[len("file://"):])
                if os.path.isdir(local_path) or os.path.isfile(local_path):
                    logger.info(f"Using local LoRA path: {local_path}")
                    return local_path
                logger.error(f"Local LoRA path not found: {local_path}")
                return None

            # If it's already a filesystem path without scheme, support it too.
            if os.path.exists(lora_url):
                logger.info(f"Using local LoRA path: {lora_url}")
                return lora_url

            if lora_url.startswith("https://huggingface.co/"):
                repo_id = lora_url.replace("https://huggingface.co/", "")
            elif lora_url.startswith("http"):
                logger.warning(f"Non-HuggingFace URL: {lora_url}")
                repo_id = lora_url
            else:
                repo_id = lora_url

            repo_id = repo_id.rstrip("/")

            cache_path = self._lora_cache_dir / repo_id.replace("/", "_")
            if cache_path.exists():
                logger.info(f"Using cached LoRA: {cache_path}")
                return str(cache_path)

            if not HF_HUB_AVAILABLE:
                logger.error("huggingface_hub not available for download")
                return None

            logger.info(f"Downloading LoRA from {repo_id}...")

            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(cache_path),
                local_dir_use_symlinks=False
            )

            logger.info(f"LoRA downloaded to: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download LoRA from {lora_url}: {e}", exc_info=True)
            return None

    async def _generate_content(
        self,
        task_info: Dict[str, Any],
        lora_path: str,
        task_type: str
    ) -> Optional[Union[Image.Image, str]]:
        prompt = task_info.get("prompt", "")
        seed = task_info.get("seed", 42)
        base_model = task_info.get("base_model", "")

        if not prompt:
            logger.error("Empty prompt provided")
            return None

        if not base_model:
            logger.error("Empty base_model provided")
            return None

        logger.info(f"Generating content: type={task_type}, model={base_model}, seed={seed}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if task_type in ["image_lora", "image_lora_creation"]:
            return await self._generate_image(base_model, lora_path, prompt, generator, task_info)
        else:
            return await self._generate_text(base_model, lora_path, prompt, seed, task_info)

    async def _generate_image(
        self,
        base_model: str,
        lora_path: str,
        prompt: str,
        generator: torch.Generator,
        task_info: Dict[str, Any]
    ) -> Optional[Image.Image]:
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available for image generation")
            return None

        try:
            num_inference_steps = task_info.get("num_inference_steps", 30)
            guidance_scale = task_info.get("guidance_scale", 7.5)

            if base_model not in self._image_pipelines:
                logger.info(f"Loading base model (first time): {base_model}")
                if "flux" in base_model.lower():
                    pipe = DiffusionPipeline.from_pretrained(
                        base_model,
                        torch_dtype=torch.float16
                    )
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        base_model,
                        torch_dtype=torch.float16
                    )

                pipe = pipe.to(self.device)
                
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                except Exception as e:
                    logger.warning(f"Failed to enable memory optimizations: {e}")

                self._image_pipelines[base_model] = pipe
                self._current_lora[base_model] = None
                logger.info(f"Base model {base_model} loaded and cached")
            else:
                pipe = self._image_pipelines[base_model]
                logger.debug(f"Using cached pipeline for base model: {base_model}")

            # Always try to unload previous LoRA weights before loading new ones
            # This prevents the "multiple adapters" warning
            try:
                if self._current_lora.get(base_model) is not None:
                    logger.debug(f"Unloading previous LoRA weights")
                # Unload even if not tracked (in case of previous incomplete cleanup)
                pipe.unload_lora_weights()
                self._current_lora[base_model] = None
            except Exception as e:
                # Ignore errors if no LoRA was loaded
                logger.debug(f"No previous LoRA to unload or unload failed: {e}")

            try:
                logger.info(f"Loading LoRA weights from: {lora_path}")
                lora_loaded = False

                try:
                    # Suppress the "multiple adapters" warning as we've already unloaded
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*peft_config.*")
                        warnings.filterwarnings("ignore", message=".*multiple adapters.*")
                        pipe.load_lora_weights(lora_path)
                    lora_loaded = True
                    self._current_lora[base_model] = lora_path
                    logger.info("LoRA weights loaded successfully (diffusers format)")
                except Exception as e1:
                    logger.debug(f"Diffusers load failed: {e1}, trying PEFT format...")

                if not lora_loaded:
                    try:
                        lora_loaded = self._load_peft_lora_to_diffusers(pipe, lora_path)
                        if lora_loaded:
                            self._current_lora[base_model] = lora_path
                            logger.info("LoRA weights loaded successfully (PEFT format converted)")
                    except Exception as e2:
                        logger.debug(f"PEFT conversion failed: {e2}")

                if not lora_loaded:
                    logger.warning(f"Failed to load LoRA weights from {lora_path}")

            except Exception as e:
                logger.warning(f"Failed to load LoRA weights: {e}")

            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            
            # Use torch.inference_mode() to reduce memory usage
            with torch.inference_mode():
                result = pipe(
                    prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )

            image = result.images[0]
            logger.info("Image generated successfully")

            # Clean up only result and generator (pipeline stays in cache)
            self._cleanup_gpu_memory(result, generator)

            return image

        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            return None

    async def _generate_text(
        self,
        base_model: str,
        lora_path: str,
        prompt: str,
        seed: int,
        task_info: Dict[str, Any]
    ) -> Optional[str]:
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available for text generation")
            return None

        try:
            max_length = task_info.get("max_length", 512)
            temperature = task_info.get("temperature", 0.7)

            logger.info(f"Loading base model: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            try:
                from peft import PeftModel
                logger.info(f"Loading LoRA weights from: {lora_path}")
                model = PeftModel.from_pretrained(model, lora_path)
                logger.info("LoRA weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights: {e}")

            logger.info(f"Generating text with prompt: {prompt[:50]}...")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            torch.manual_seed(seed)

            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Text generated: {len(text)} characters")

            # Clean up GPU memory
            self._cleanup_gpu_memory(model, tokenizer, inputs, outputs)

            return text

        except Exception as e:
            logger.error(f"Text generation failed: {e}", exc_info=True)
            # Clean up GPU memory even on error
            try:
                if 'model' in locals():
                    self._cleanup_gpu_memory(model)
                if 'tokenizer' in locals():
                    self._cleanup_gpu_memory(tokenizer)
            except Exception:
                pass
            return None

    async def _calculate_cosine_similarity(
        self,
        task_info: Dict[str, Any],
        generated_content: Union[Image.Image, str],
        task_type: str
    ) -> float:

        target_vector = task_info.get("target_vector", [])

        if not target_vector:
            logger.warning("No target vector provided, using default similarity 0.85")
            return 0.85

        try:
            if task_type in ["image_lora", "image_lora_creation"]:
                current_vector = self._extract_image_features(generated_content)
            else:
                current_vector = self._extract_text_features(generated_content)

            target_tensor = torch.tensor(target_vector, dtype=torch.float32, device=self.device)

            if current_vector.shape[-1] != target_tensor.shape[-1]:
                logger.warning(f"Vector dimension mismatch: {current_vector.shape} vs {target_tensor.shape}")
                if current_vector.numel() > target_tensor.numel():
                    current_vector = current_vector[:target_tensor.shape[-1]]
                else:
                    return 0.0

            current_vector = current_vector / current_vector.norm(dim=-1, keepdim=True)
            target_tensor = target_tensor / target_tensor.norm(dim=-1, keepdim=True)

            cosine_sim = torch.cosine_similarity(
                current_vector.unsqueeze(0),
                target_tensor.unsqueeze(0)
            ).item()

            cosine_sim = max(0.0, min(1.0, cosine_sim))

            logger.info(f"Cosine similarity: {cosine_sim:.4f}")
            return cosine_sim

        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}", exc_info=True)
            return 0.0

    def _extract_image_features(self, image: Image.Image) -> torch.Tensor:

        if not DINOV2_AVAILABLE or self.dinov2_model is None:
            logger.warning("DINOv2 not available, using random features")
            return torch.randn(1024, device=self.device)

        try:
            inputs = self.dinov2_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.dinov2_model(**inputs)
                features = outputs.last_hidden_state[:, 0]
                features = features / features.norm(dim=-1, keepdim=True)

            return features.squeeze(0).float()

        except Exception as e:
            logger.error(f"Image feature extraction failed: {e}", exc_info=True)
            return torch.randn(1024, device=self.device)

    def _extract_text_features(self, text: str) -> torch.Tensor:
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.text_encoder is None:
            logger.warning("Sentence-BERT not available, using random features")
            return torch.randn(384, device=self.device)

        try:
            features = self.text_encoder.encode(text, convert_to_tensor=True)
            features = features.to(self.device)

            features = features / features.norm(dim=-1, keepdim=True)

            return features.float()

        except Exception as e:
            logger.error(f"Text feature extraction failed: {e}", exc_info=True)
            return torch.randn(384, device=self.device)

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

    def _calculate_final_score(
        self,
        cosine_similarity: float,
        quality_score: float,
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

        # final_score = random.uniform(9.1, 9.3)

        score_zone = 'high' if combined_score >= self.SCORE_PIVOT_HIGH else 'mid' if combined_score >= self.SCORE_PIVOT_LOW else 'low'
        logger.debug(f"Score calculation: cosine={cosine_similarity:.4f}, quality={quality_score:.2f}, "
                    f"combined={combined_score:.4f}, zone={score_zone}, "
                    f"time_coef={time_coefficient:.2f}, final={final_score:.2f}")

        return max(0.0, min(10.0, final_score))

    async def _process_dataset_audit(
        self,
        audit_task_id: str,
        miner_hotkey: str,
        dataset_url: str,
        task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Processing dataset audit task {audit_task_id}: miner={miner_hotkey[:20]}...")

        try:
            validation_result = await self.dataset_validator.validate_dataset(
                dataset_url=dataset_url,
                task_info=task_info
            )

            is_valid = validation_result.get("is_valid", False)

            logger.info(
                f"Dataset audit {audit_task_id} completed: is_valid={is_valid}, "
                f"rejection_reason={validation_result.get('rejection_reason')}, "
                f"format_check={validation_result.get('format_check')}, "
                f"quality_check={validation_result.get('quality_check')}, "
                f"safety_check={validation_result.get('safety_check')}"
            )

            return {
                "audit_task_id": audit_task_id,
                "miner_hotkey": miner_hotkey,
                "audit_type": "dataset",
                "is_valid": is_valid,
                "format_check": validation_result.get("format_check", {}),
                "quality_check": validation_result.get("quality_check", {}),
                "safety_check": validation_result.get("safety_check", {}),
                "relevance_check": validation_result.get("relevance_check", {}),
                "rejection_reason": validation_result.get("rejection_reason"),
                "rejected": not is_valid,
                "reason": validation_result.get("rejection_reason") if not is_valid else None
            }

        except Exception as e:
            logger.error(f"Dataset audit task {audit_task_id} failed: {e}", exc_info=True)
            return {
                "audit_task_id": audit_task_id,
                "miner_hotkey": miner_hotkey,
                "audit_type": "dataset",
                "is_valid": False,
                "format_check": {"passed": False},
                "quality_check": {"passed": False},
                "safety_check": {"passed": False},
                "relevance_check": {"passed": False},
                "rejection_reason": f"Validation error: {str(e)}",
                "rejected": True,
                "reason": f"Validation error: {str(e)}"
            }

    def _create_error_result(
        self,
        audit_task_id: str,
        miner_hotkey: str,
        reason: str
    ) -> Dict[str, Any]:
        return {
            "audit_task_id": audit_task_id,
            "miner_hotkey": miner_hotkey,
            "cosine_similarity": 0.0,
            "quality_score": 0.0,
            "final_score": 0.0,
            "content_safety_score": 0.0,
            "time_coefficient": 1.0,
            "training_time_minutes": None,
            "rejected": True,
            "reason": reason
        }

    def cleanup_cache(self, max_age_hours: int = 24):
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for cache_dir in self._lora_cache_dir.iterdir():
            if cache_dir.is_dir():
                dir_age = current_time - cache_dir.stat().st_mtime
                if dir_age > max_age_seconds:
                    try:
                        shutil.rmtree(cache_dir)
                        logger.info(f"Cleaned up expired cache: {cache_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to clean cache {cache_dir}: {e}")
