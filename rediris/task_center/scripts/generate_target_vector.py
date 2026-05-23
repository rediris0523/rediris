#!/usr/bin/env python3

import torch
import json
import argparse
from pathlib import Path
from typing import Optional, List
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    from transformers import AutoImageProcessor, AutoModel
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False
    logger.warning("DINOv2 not available: pip install transformers")

try:
    from diffusers import DiffusionPipeline, FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers not available: pip install diffusers")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available: pip install sentence-transformers")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers/PEFT not available: pip install transformers peft")


class TargetVectorGenerator:

    DINOV2_MODEL_ID = "facebook/dinov2-large"

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dinov2_model = None
        self.dinov2_processor = None
        self.text_encoder = None
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

    def _load_sentence_transformer(self):
        if self.text_encoder is not None:
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Sentence transformers not available")

        logger.info("Loading Sentence-BERT model...")
        self.text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("Sentence-BERT model loaded")

    def _extract_dinov2_features(self, image) -> torch.Tensor:
        inputs = self.dinov2_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            features = outputs.last_hidden_state[:, 0]
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().float()

    def generate_image_target_vector(
        self,
        base_model: str,
        teacher_lora_path: str,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        save_image_path: Optional[str] = None
    ) -> List[float]:
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers not available")

        self._load_dinov2()

        logger.info(f"Loading base model: {base_model}")

        if "flux" in base_model.lower():
            pipe = FluxPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16
            )

        pipe = pipe.to(self.device)

        if teacher_lora_path:
            logger.info(f"Loading Teacher LoRA: {teacher_lora_path}")
            try:
                lora_loaded = False

                try:
                    pipe.load_lora_weights(teacher_lora_path)
                    lora_loaded = True
                    logger.info("Teacher LoRA loaded (diffusers format)")
                except Exception as e1:
                    logger.warning(f"Failed to load as diffusers format: {e1}")

                if not lora_loaded:
                    from pathlib import Path
                    lora_path = Path(teacher_lora_path)

                    if lora_path.is_dir():
                        safetensor_files = list(lora_path.glob("*.safetensors"))
                        if safetensor_files:
                            try:
                                pipe.load_lora_weights(
                                    teacher_lora_path,
                                    weight_name=safetensor_files[0].name
                                )
                                lora_loaded = True
                                logger.info(f"Teacher LoRA loaded from: {safetensor_files[0].name}")
                            except Exception as e2:
                                logger.warning(f"Failed to load safetensors: {e2}")

                if not lora_loaded:
                    try:
                        from peft import PeftModel, LoraConfig

                        if hasattr(pipe, 'transformer'):
                            pipe.transformer = PeftModel.from_pretrained(
                                pipe.transformer,
                                teacher_lora_path
                            )
                            lora_loaded = True
                            logger.info("Teacher LoRA loaded via PEFT (transformer)")
                        elif hasattr(pipe, 'unet'):
                            pipe.unet = PeftModel.from_pretrained(
                                pipe.unet,
                                teacher_lora_path
                            )
                            lora_loaded = True
                            logger.info("Teacher LoRA loaded via PEFT (unet)")
                    except Exception as e3:
                        logger.warning(f"Failed to load via PEFT: {e3}")

                if not lora_loaded:
                    try:
                        from safetensors.torch import load_file
                        from pathlib import Path

                        lora_path = Path(teacher_lora_path)
                        if lora_path.is_dir():
                            safetensor_file = lora_path / "adapter_model.safetensors"
                            if not safetensor_file.exists():
                                safetensor_files = list(lora_path.glob("*.safetensors"))
                                if safetensor_files:
                                    safetensor_file = safetensor_files[0]
                        else:
                            safetensor_file = lora_path

                        if safetensor_file.exists():
                            state_dict = load_file(str(safetensor_file))

                            converted_state_dict = {}
                            for key, value in state_dict.items():
                                new_key = key.replace("base_model.model.", "")
                                converted_state_dict[new_key] = value

                            if hasattr(pipe, 'transformer'):
                                pipe.transformer.load_state_dict(converted_state_dict, strict=False)
                                lora_loaded = True
                                logger.info("Teacher LoRA loaded via manual state_dict (transformer)")
                            elif hasattr(pipe, 'unet'):
                                pipe.unet.load_state_dict(converted_state_dict, strict=False)
                                lora_loaded = True
                                logger.info("Teacher LoRA loaded via manual state_dict (unet)")
                    except Exception as e4:
                        logger.warning(f"Failed to load via manual state_dict: {e4}")

                if not lora_loaded:
                    raise RuntimeError(
                        f"Could not load LoRA from {teacher_lora_path}. "
                        "Tried: diffusers format, safetensors, PEFT, manual state_dict"
                    )

            except Exception as e:
                logger.error(f"Failed to load Teacher LoRA: {e}")
                raise

        logger.info(f"Generating image with prompt: {prompt[:50]}...")
        logger.info(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if "flux" in base_model.lower():
            result = pipe(
                prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024
            )
        else:
            result = pipe(
                prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

        image = result.images[0]
        logger.info("Image generated successfully")

        if save_image_path:
            image.save(save_image_path)
            logger.info(f"Image saved to: {save_image_path}")

        logger.info("Extracting image features with DINOv2...")
        features = self._extract_dinov2_features(image)
        target_vector = features.tolist()

        logger.info(f"Target vector generated: {len(target_vector)} dimensions")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return target_vector

    def generate_text_target_vector(
        self,
        base_model: str,
        teacher_lora_path: str,
        prompt: str,
        seed: int = 42,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> List[float]:
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers/PEFT not available")

        self._load_sentence_transformer()

        logger.info(f"Loading base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if teacher_lora_path:
            logger.info(f"Loading Teacher LoRA: {teacher_lora_path}")
            try:
                model = PeftModel.from_pretrained(model, teacher_lora_path)
                logger.info("Teacher LoRA loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Teacher LoRA: {e}")
                raise

        logger.info(f"Generating text with prompt: {prompt[:50]}...")
        logger.info(f"Seed: {seed}, Max length: {max_length}, Temperature: {temperature}")

        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Text generated: {len(generated_text)} characters")

        logger.info("Extracting text features with Sentence-BERT...")
        features = self.text_encoder.encode(generated_text, convert_to_tensor=True)
        features = features / features.norm(dim=-1, keepdim=True)

        target_vector = features.cpu().float().tolist()

        logger.info(f"Target vector generated: {len(target_vector)} dimensions")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return target_vector

    def extract_image_vector_from_file(self, image_path: str) -> List[float]:
        from PIL import Image

        self._load_dinov2()

        logger.info(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")

        logger.info("Extracting image features with DINOv2...")
        features = self._extract_dinov2_features(image)
        target_vector = features.tolist()

        logger.info(f"Target vector extracted: {len(target_vector)} dimensions")
        return target_vector

    def extract_text_vector(self, text: str) -> List[float]:
        self._load_sentence_transformer()

        logger.info(f"Extracting features from text: {text[:50]}...")

        features = self.text_encoder.encode(text, convert_to_tensor=True)
        features = features / features.norm(dim=-1, keepdim=True)

        target_vector = features.cpu().float().tolist()

        logger.info(f"Target vector extracted: {len(target_vector)} dimensions")
        return target_vector


def main():
    parser = argparse.ArgumentParser(
        description="Generate target_vector for Teacher Model Projection"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    image_parser = subparsers.add_parser("image", help="Generate image target vector")
    image_parser.add_argument("--base-model", required=True, help="Base model ID (e.g., black-forest-labs/FLUX.1-dev)")
    image_parser.add_argument("--teacher-lora", required=True, help="Teacher LoRA path or HF repo ID")
    image_parser.add_argument("--prompt", required=True, help="Prompt for image generation")
    image_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    image_parser.add_argument("--steps", type=int, default=30, help="Inference steps (default: 30)")
    image_parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    image_parser.add_argument("--save-image", help="Save generated image to path")
    image_parser.add_argument("--output", "-o", help="Output JSON file path")

    text_parser = subparsers.add_parser("text", help="Generate text target vector")
    text_parser.add_argument("--base-model", required=True, help="Base model ID (e.g., Qwen/Qwen2.5-7B)")
    text_parser.add_argument("--teacher-lora", required=True, help="Teacher LoRA path or HF repo ID")
    text_parser.add_argument("--prompt", required=True, help="Prompt for text generation")
    text_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    text_parser.add_argument("--max-length", type=int, default=512, help="Max generation length (default: 512)")
    text_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    text_parser.add_argument("--output", "-o", help="Output JSON file path")

    extract_image_parser = subparsers.add_parser("extract-image", help="Extract vector from existing image")
    extract_image_parser.add_argument("--image", required=True, help="Path to image file")
    extract_image_parser.add_argument("--output", "-o", help="Output JSON file path")

    extract_text_parser = subparsers.add_parser("extract-text", help="Extract vector from text")
    extract_text_parser.add_argument("--text", required=True, help="Text to extract vector from")
    extract_text_parser.add_argument("--output", "-o", help="Output JSON file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    generator = TargetVectorGenerator()

    if args.command == "image":
        target_vector = generator.generate_image_target_vector(
            base_model=args.base_model,
            teacher_lora_path=args.teacher_lora,
            prompt=args.prompt,
            seed=args.seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            save_image_path=args.save_image
        )
        result = {
            "type": "image",
            "base_model": args.base_model,
            "teacher_lora": args.teacher_lora,
            "prompt": args.prompt,
            "seed": args.seed,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "vector_dimensions": len(target_vector),
            "target_vector": target_vector
        }

    elif args.command == "text":
        target_vector = generator.generate_text_target_vector(
            base_model=args.base_model,
            teacher_lora_path=args.teacher_lora,
            prompt=args.prompt,
            seed=args.seed,
            max_length=args.max_length,
            temperature=args.temperature
        )
        result = {
            "type": "text",
            "base_model": args.base_model,
            "teacher_lora": args.teacher_lora,
            "prompt": args.prompt,
            "seed": args.seed,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "vector_dimensions": len(target_vector),
            "target_vector": target_vector
        }

    elif args.command == "extract-image":
        target_vector = generator.extract_image_vector_from_file(args.image)
        result = {
            "type": "image_extraction",
            "source_image": args.image,
            "vector_dimensions": len(target_vector),
            "target_vector": target_vector
        }

    elif args.command == "extract-text":
        target_vector = generator.extract_text_vector(args.text)
        result = {
            "type": "text_extraction",
            "source_text": args.text,
            "vector_dimensions": len(target_vector),
            "target_vector": target_vector
        }

    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
