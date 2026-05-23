from typing import Dict, Any, Optional
from rediris.common.utils.logging import setup_logger
from rediris.common.config.yaml_config import YamlConfig
import asyncio
import os
import torch
from pathlib import Path

logger = setup_logger(__name__)

try:
    from diffusers import DiffusionPipeline, StableDiffusionPipeline
    from datasets import load_dataset
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    logger.warning(f"Diffusers not available: {e}", exc_info=True)

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError as e:
    PEFT_AVAILABLE = False
    logger.warning(f"PEFT not available: {e}", exc_info=True)


class ImageTrainingService:

    def __init__(self, config: Optional[YamlConfig] = None):
        self.config = config
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)

        if config:
            hf_token = config.get('huggingface.token')
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
                logger.info("[ImageTrainingService] HuggingFace token configured")

        logger.info(f"[ImageTrainingService] Initialized. DIFFUSERS_AVAILABLE={DIFFUSERS_AVAILABLE}, PEFT_AVAILABLE={PEFT_AVAILABLE}")

    def _is_flux_model(self, model_name: str) -> bool:
        return "flux" in model_name.lower()

    def _get_trainable_model(self, pipe):
        if hasattr(pipe, 'transformer'):
            return pipe.transformer, 'transformer'
        elif hasattr(pipe, 'unet'):
            return pipe.unet, 'unet'
        else:
            raise ValueError("Pipeline has neither 'transformer' nor 'unet' attribute")

    def _get_target_modules(self, model_type: str, is_flux: bool) -> list:
        if is_flux:
            return ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"]
        else:
            return ["to_k", "to_q", "to_v", "to_out.0"]

    def _is_model_complete(self, model_path: Path) -> bool:
        if not model_path.exists() or not model_path.is_dir():
            return False
        
        has_adapter = (
            (model_path / "adapter_model.safetensors").exists() or
            (model_path / "adapter_model.bin").exists()
        )
        
        has_config = (model_path / "adapter_config.json").exists()
        
        return has_adapter and has_config

    async def train_lora(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task.get('task_id', 'unknown')
        logger.info(f"[ImageTrainingService] train_lora() called - task_id: {task_id}")
        logger.info(f"[ImageTrainingService] DIFFUSERS_AVAILABLE={DIFFUSERS_AVAILABLE}, PEFT_AVAILABLE={PEFT_AVAILABLE}")

        if not DIFFUSERS_AVAILABLE:
            logger.error(f"[ImageTrainingService] Diffusers library not available, cannot proceed with training")
            raise RuntimeError("Diffusers library not available")

        logger.info(f"[ImageTrainingService] Starting image LoRA training for task {task_id}")

        model_path = self.models_dir / task_id
        if self._is_model_complete(model_path):
            logger.info(f"Model already exists for task {task_id} at {model_path}, skipping training")
            
            workflow_spec = task.get("workflow_spec", {})
            training_spec = workflow_spec.get("training_spec", {})
            training_mode = workflow_spec.get("training_mode", "new")
            iteration_count = training_spec.get("iteration_count", 1000)
            
            return {
                "status": "completed",
                "model_url": None,
                "training_steps": iteration_count,
                "model_path": str(model_path),
                "training_mode": training_mode,
                "final_loss": 0.0
            }
        elif model_path.exists():
            logger.warning(f"Model directory exists for task {task_id} but files are incomplete, will retrain")

        workflow_spec = task.get("workflow_spec", {})
        training_spec = workflow_spec.get("training_spec", {})
        dataset_spec = workflow_spec.get("dataset_spec", {})

        image_config = self.config.get_image_training_config() if self.config else {}

        base_model = training_spec.get("base_model", image_config.get("base_model", "stabilityai/stable-diffusion-2-1"))
        lora_rank = training_spec.get("lora_rank", image_config.get("default_lora_rank", 16))
        lora_alpha = training_spec.get("lora_alpha", image_config.get("default_lora_alpha", 32))
        iteration_count = training_spec.get("iteration_count", image_config.get("default_iteration_count", 1000))
        batch_size = training_spec.get("batch_size", image_config.get("default_batch_size", 2))
        learning_rate = training_spec.get("learning_rate", image_config.get("default_learning_rate", 1e-4))
        resolution = training_spec.get("resolution", image_config.get("default_resolution", [512, 768]))

        training_mode = workflow_spec.get("training_mode", "new")
        base_lora_url = workflow_spec.get("base_lora_url")

        datasets_config = self.config.get_datasets_config() if self.config else {}
        image_dataset_config = datasets_config.get("image", {})

        from rediris.common.utils.huggingface import parse_dataset_url
        dataset_url = task.get("dataset_url")
        if dataset_url:
            dataset_repo = parse_dataset_url(dataset_url)
            logger.info(f"[ImageTrainingService] Using miner's validated dataset: {dataset_repo}")
        else:
            dataset_repo = dataset_spec.get("repository_id", image_dataset_config.get("repository_id", "rediris/manga-style-dataset"))
            logger.info(f"[ImageTrainingService] Using task spec dataset: {dataset_repo}")

        image_column = dataset_spec.get("image_column", image_dataset_config.get("image_column", "image"))
        caption_column = dataset_spec.get("caption_column", image_dataset_config.get("caption_column", "text"))
        sample_count = dataset_spec.get("sample_count") or image_config.get("default_sample_count", 300)
        logger.info(f"[ImageTrainingService] Will use {sample_count} samples for training")

        is_flux = self._is_flux_model(base_model)
        logger.info(f"[ImageTrainingService] Model type: {'FLUX' if is_flux else 'Stable Diffusion'}")

        try:
            logger.info(f"Loading base model: {base_model}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[ImageTrainingService] Using device: {device}")

            hf_token = self.config.get('huggingface.token') if self.config else os.environ.get('HF_TOKEN')

            if device == "cuda":
                torch.cuda.empty_cache()
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            if device == "cuda":
                if is_flux:
                    model_dtype = torch.bfloat16
                else:
                    model_dtype = torch.float16
            else:
                model_dtype = torch.float32

            pipe = DiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=model_dtype,
                token=hf_token
            )

            self._current_dtype = model_dtype

            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing(1)
                logger.info("[ImageTrainingService] Attention slicing enabled")


            pipe = pipe.to(device)
            logger.info(f"[ImageTrainingService] Pipeline moved to {device}")

            if hasattr(pipe, 'vae'):
                pipe.vae.requires_grad_(False)
            if hasattr(pipe, 'text_encoder'):
                pipe.text_encoder.requires_grad_(False)
            if hasattr(pipe, 'text_encoder_2'):
                pipe.text_encoder_2.requires_grad_(False)
            logger.info("[ImageTrainingService] VAE and text encoders frozen")

            logger.info(f"[ImageTrainingService] Pipeline loaded: {type(pipe).__name__}")

            trainable_model, model_type = self._get_trainable_model(pipe)
            logger.info(f"[ImageTrainingService] Trainable model type: {model_type}")

            if PEFT_AVAILABLE:
                target_modules = self._get_target_modules(model_type, is_flux)
                logger.info(f"[ImageTrainingService] LoRA target modules: {target_modules}")

                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=target_modules
                )

                trainable_model = get_peft_model(trainable_model, lora_config)

                if model_type == 'transformer':
                    pipe.transformer = trainable_model
                else:
                    pipe.unet = trainable_model

                logger.info("[ImageTrainingService] LoRA applied successfully")

            if training_mode == "incremental" and base_lora_url:
                logger.info(f"Incremental training: Loading base LoRA weights from {base_lora_url}")
                lora_repo_or_path = base_lora_url
                if base_lora_url.startswith("https://huggingface.co/"):
                    lora_repo_or_path = base_lora_url.replace("https://huggingface.co/", "")
                    lora_repo_or_path = lora_repo_or_path.rstrip("/")
                    logger.info(f"Converted HF URL to repo ID: {lora_repo_or_path}")

                base_lora_filename = workflow_spec.get("base_lora_filename")

                from huggingface_hub import hf_hub_download, list_repo_files
                import safetensors.torch

                try:
                    if base_lora_filename:
                        lora_filename = base_lora_filename
                        logger.info(f"Using configured LoRA filename: {lora_filename}")
                    else:
                        try:
                            repo_files = list_repo_files(lora_repo_or_path, token=hf_token)
                            safetensor_files = [f for f in repo_files if f.endswith('.safetensors')]

                            if 'adapter_model.safetensors' in safetensor_files:
                                lora_filename = 'adapter_model.safetensors'
                            elif safetensor_files:
                                lora_filename = safetensor_files[0]
                                logger.info(f"Auto-detected LoRA filename: {lora_filename}")
                            else:
                                raise ValueError("No .safetensors file found in repository")
                        except Exception as e:
                            logger.warning(f"Failed to list repo files: {e}, trying default filename")
                            lora_filename = 'adapter_model.safetensors'

                    lora_path = hf_hub_download(
                        repo_id=lora_repo_or_path,
                        filename=lora_filename,
                        token=hf_token
                    )
                    state_dict = safetensors.torch.load_file(lora_path)

                    is_peft_format = any('lora_A' in k or 'lora_B' in k for k in state_dict.keys())
                    is_diffusers_format = any('transformer' in k or 'unet' in k for k in state_dict.keys())

                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*peft_config.*")
                        warnings.filterwarnings("ignore", message=".*multiple adapters.*")

                        if is_peft_format:
                            from peft import set_peft_model_state_dict
                            set_peft_model_state_dict(trainable_model, state_dict)
                            logger.info("[ImageTrainingService] Base LoRA weights loaded (PEFT format)")
                        elif is_diffusers_format:
                            pipe.load_lora_weights(lora_path)
                            logger.info("[ImageTrainingService] Base LoRA weights loaded (diffusers format)")
                        else:
                            converted_state_dict = {}
                            for key, value in state_dict.items():
                                if 'lora' in key.lower():
                                    converted_state_dict[key] = value
                            if converted_state_dict:
                                from peft import set_peft_model_state_dict
                                set_peft_model_state_dict(trainable_model, converted_state_dict)
                                logger.info("[ImageTrainingService] Base LoRA weights loaded (converted format)")
                            else:
                                logger.warning("[ImageTrainingService] Unknown LoRA format, skipping weight loading")

                except Exception as e:
                    logger.warning(f"Failed to load base LoRA weights: {e}, starting from scratch")
            else:
                logger.info("New training: Starting from base model")

            logger.info(f"Loading dataset: {dataset_repo}")
            hf_token_for_dataset = hf_token
            try:
                dataset = load_dataset(dataset_repo, split=f"train[:{sample_count}]", token=hf_token_for_dataset)
            except Exception as e:
                logger.warning(f"Failed to load dataset with token, trying without: {e}")
                dataset = load_dataset(dataset_repo, split=f"train[:{sample_count}]")

            logger.info(f"[ImageTrainingService] Dataset loaded with {len(dataset)} samples")

            model_path = self.models_dir / task_id
            model_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Starting {'incremental' if training_mode == 'incremental' else 'new'} training...")
            logger.info(f"[ImageTrainingService] Training config: batch_size={batch_size}, iteration_count={iteration_count}, lr={learning_rate}")

            gradient_accumulation_steps = training_spec.get("gradient_accumulation_steps", image_config.get("gradient_accumulation_steps", 2))
            effective_batch_size = batch_size * gradient_accumulation_steps
            logger.info(f"[ImageTrainingService] Gradient accumulation steps: {gradient_accumulation_steps}, Effective batch size: {effective_batch_size}")

            from PIL import Image as PILImage
            import torchvision.transforms as transforms

            target_size = (resolution[0], resolution[1]) if isinstance(resolution, list) else (resolution, resolution)

            transform = transforms.Compose([
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            def collate_fn(examples):
                images = []
                captions = []
                for example in examples:
                    image = example.get(image_column) or example.get("image")
                    caption = example.get(caption_column) or example.get("text") or ""

                    if not isinstance(caption, str):
                        caption = str(caption) if caption is not None else ""

                    if image is None:
                        continue

                    if isinstance(image, str):
                        import requests
                        try:
                            image = PILImage.open(requests.get(image, stream=True).raw)
                        except:
                            continue

                    if not isinstance(image, PILImage.Image):
                        try:
                            image = PILImage.fromarray(image)
                        except:
                            continue

                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    images.append(transform(image))
                    captions.append(caption)

                if not images:
                    return None

                return {
                    "images": torch.stack(images),
                    "captions": captions
                }

            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                drop_last=True
            )

            optimizer = torch.optim.AdamW(
                trainable_model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                eps=1e-8
            )

            total_steps = 0
            trainable_model.train()
            accumulated_loss = 0.0
            log_interval = 100

            if hasattr(trainable_model, 'enable_gradient_checkpointing'):
                trainable_model.enable_gradient_checkpointing()
                logger.info("[ImageTrainingService] Gradient checkpointing enabled")

            num_epochs = (iteration_count * gradient_accumulation_steps) // len(dataloader) + 1
            logger.info(f"[ImageTrainingService] Will train for approximately {num_epochs} epochs")

            first_batch_logged = False

            for epoch in range(num_epochs):
                for batch_idx, batch_data in enumerate(dataloader):
                    if total_steps >= iteration_count:
                        break

                    if batch_data is None:
                        continue

                    try:
                        if batch_idx % 50 == 0 and device == "cuda":
                            torch.cuda.empty_cache()

                        image_tensors = batch_data["images"].to(device, dtype=self._current_dtype)
                        captions = batch_data["captions"]
                        current_batch_size = image_tensors.shape[0]

                        with torch.no_grad():
                            latents = pipe.vae.encode(image_tensors).latent_dist.sample()
                            latents = latents * pipe.vae.config.scaling_factor
                            latents = latents.to(dtype=self._current_dtype)

                        pooled_prompt_embeds = None
                        text_ids = None

                        with torch.no_grad():
                            if is_flux:
                                if hasattr(pipe, 'encode_prompt'):
                                    prompt_embeds_list = []
                                    pooled_embeds_list = []
                                    text_ids_list = []

                                    for caption in captions:
                                        pe, ppe, tid = pipe.encode_prompt(
                                            prompt=caption,
                                            prompt_2=caption,
                                            device=device,
                                            num_images_per_prompt=1,
                                        )
                                        prompt_embeds_list.append(pe)
                                        pooled_embeds_list.append(ppe)
                                        if tid is not None:
                                            text_ids_list.append(tid)

                                    prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(dtype=self._current_dtype)
                                    pooled_prompt_embeds = torch.cat(pooled_embeds_list, dim=0).to(dtype=self._current_dtype)
                                    if text_ids_list:
                                        text_ids = text_ids_list[0]
                                else:
                                    tokens = pipe.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                                    prompt_embeds = pipe.text_encoder(tokens)[0].to(dtype=self._current_dtype)
                                    pooled_prompt_embeds = torch.zeros(current_batch_size, 768, device=device, dtype=self._current_dtype)
                            else:
                                text_inputs = pipe.tokenizer(
                                    captions,
                                    padding="max_length",
                                    max_length=pipe.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt"
                                )
                                prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
                                prompt_embeds = prompt_embeds.to(dtype=self._current_dtype)

                        noise = torch.randn_like(latents, dtype=self._current_dtype)

                        if is_flux:
                            t = torch.rand(current_batch_size, device=device, dtype=self._current_dtype)
                            t_expanded = t.view(-1, 1, 1, 1)
                            noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
                            timesteps = (t * 1000).long()
                        else:
                            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (current_batch_size,), device=device).long()
                            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                        noisy_latents = noisy_latents.to(dtype=self._current_dtype)

                        if is_flux:
                            channels = noisy_latents.shape[1]
                            height = noisy_latents.shape[2]
                            width = noisy_latents.shape[3]

                            if not first_batch_logged:
                                logger.info(f"[FLUX Batch] noisy_latents shape: {noisy_latents.shape}")
                                logger.info(f"[FLUX Batch] batch_size={current_batch_size}, channels={channels}, height={height}, width={width}")

                            if height % 2 != 0 or width % 2 != 0:
                                pad_h = height % 2
                                pad_w = width % 2
                                noisy_latents = torch.nn.functional.pad(noisy_latents, (0, pad_w, 0, pad_h))
                                noise = torch.nn.functional.pad(noise, (0, pad_w, 0, pad_h))
                                latents = torch.nn.functional.pad(latents, (0, pad_w, 0, pad_h))
                                height = noisy_latents.shape[2]
                                width = noisy_latents.shape[3]

                            noisy_latents = noisy_latents.contiguous()
                            packed_latents = noisy_latents.view(current_batch_size, channels, height // 2, 2, width // 2, 2)
                            packed_latents = packed_latents.permute(0, 2, 4, 1, 3, 5).contiguous()
                            packed_latents = packed_latents.reshape(current_batch_size, (height // 2) * (width // 2), channels * 4)

                            if hasattr(pipe, '_prepare_latent_image_ids'):
                                latent_image_ids = pipe._prepare_latent_image_ids(
                                    current_batch_size, height // 2, width // 2, device, self._current_dtype
                                )
                            else:
                                latent_image_ids = torch.zeros(current_batch_size, (height // 2) * (width // 2), 3, device=device, dtype=self._current_dtype)

                            timestep_value = (timesteps.float() / 1000.0)

                            packed_latents = packed_latents.to(device=device, dtype=self._current_dtype)
                            prompt_embeds = prompt_embeds.to(device=device, dtype=self._current_dtype)
                            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=self._current_dtype)
                            if text_ids is not None:
                                text_ids = text_ids.to(device=device, dtype=self._current_dtype)
                            latent_image_ids = latent_image_ids.to(device=device, dtype=self._current_dtype)

                            guidance = torch.ones(current_batch_size, device=device, dtype=self._current_dtype)

                            if not first_batch_logged:
                                first_batch_logged = True

                            model_pred = trainable_model(
                                hidden_states=packed_latents,
                                timestep=timestep_value,
                                guidance=guidance,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                return_dict=False
                            )[0]

                            model_pred = model_pred.reshape(current_batch_size, height // 2, width // 2, channels, 2, 2)
                            model_pred = model_pred.permute(0, 3, 1, 4, 2, 5).reshape(current_batch_size, channels, height, width)

                            target = noise - latents
                            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        else:
                            model_pred = trainable_model(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=prompt_embeds
                            ).sample

                            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                        loss = loss / gradient_accumulation_steps
                        accumulated_loss += loss.item()

                        loss.backward()

                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            total_steps += 1

                            if device == "cuda" and total_steps % 10 == 0:
                                torch.cuda.empty_cache()

                            if total_steps % log_interval == 0:
                                avg_loss = accumulated_loss / log_interval
                                samples_processed = total_steps * effective_batch_size
                                logger.info(f"Step {total_steps}/{iteration_count}, Avg Loss: {avg_loss:.4f}, Samples: {samples_processed}")
                                accumulated_loss = 0.0

                    except Exception as e:
                        logger.warning(f"Error processing batch {batch_idx}: {e}", exc_info=True)
                        optimizer.zero_grad()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    if total_steps >= iteration_count:
                        break

                if total_steps >= iteration_count:
                    break

                logger.info(f"Epoch {epoch + 1}/{num_epochs} completed, total steps: {total_steps}")

            logger.info("Saving model...")

            if PEFT_AVAILABLE:
                trainable_model.save_pretrained(str(model_path))
            else:
                pipe.save_pretrained(str(model_path))

            try:
                from rediris.common.utils.lora_metadata import fix_lora_base_model
                fix_lora_base_model(
                    model_path=str(model_path),
                    base_model_id=base_model,
                    lora_type="image"
                )
            except Exception as e:
                logger.warning(f"Failed to fix LoRA metadata: {e}")

            logger.info(f"Image LoRA training completed for task {task_id}, saved to {model_path}")

            final_loss_value = 0.0
            if 'loss' in dir() and loss is not None:
                try:
                    final_loss_value = loss.item()
                except:
                    pass

            return {
                "status": "completed",
                "model_url": None,
                "training_steps": total_steps,
                "model_path": str(model_path),
                "training_mode": training_mode,
                "final_loss": final_loss_value
            }
        except Exception as e:
            logger.error(f"Image LoRA training failed: {e}", exc_info=True)
            raise
