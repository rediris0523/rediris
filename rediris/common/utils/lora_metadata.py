import json
import os
import re
from pathlib import Path
from typing import Optional
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)


def fix_lora_base_model(
    model_path: str,
    base_model_id: str,
    lora_type: str = "image"
) -> bool:

    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    success = True

    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            old_base_model = config.get("base_model_name_or_path", "")
            if old_base_model != base_model_id:
                config["base_model_name_or_path"] = base_model_id
                with open(adapter_config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                logger.info(f"Fixed adapter_config.json: base_model '{old_base_model}' -> '{base_model_id}'")
            else:
                logger.debug(f"adapter_config.json already has correct base_model: {base_model_id}")
        except Exception as e:
            logger.error(f"Failed to fix adapter_config.json: {e}")
            success = False

    readme_path = model_path / "README.md"
    try:
        readme_content = generate_readme(
            base_model_id=base_model_id,
            lora_type=lora_type,
            model_path=model_path
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        logger.info(f"Generated README.md with correct base_model: {base_model_id}")
    except Exception as e:
        logger.error(f"Failed to generate README.md: {e}")
        success = False

    return success


def generate_readme(
    base_model_id: str,
    lora_type: str = "image",
    model_path: Optional[Path] = None,
    additional_tags: Optional[list] = None
) -> str:

    if lora_type == "image":
        library_name = "diffusers"
        pipeline_tag = "text-to-image"
        tags = ["lora", "diffusers", "text-to-image", "stable-diffusion"]
        if "flux" in base_model_id.lower():
            tags.append("flux")
    else:
        library_name = "peft"
        pipeline_tag = "text-generation"
        tags = ["lora", "peft", "text-generation"]
        if "qwen" in base_model_id.lower():
            tags.append("qwen")
        elif "llama" in base_model_id.lower():
            tags.append("llama")

    if additional_tags:
        tags.extend(additional_tags)

    lora_info = ""
    if model_path:
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                lora_rank = config.get("r", "N/A")
                lora_alpha = config.get("lora_alpha", "N/A")
                target_modules = config.get("target_modules", [])
                lora_info = f"""
                ## LoRA Configuration
                
                - **Rank (r)**: {lora_rank}
                - **Alpha**: {lora_alpha}
                - **Target Modules**: {', '.join(target_modules) if target_modules else 'N/A'}
                """
            except Exception:
                pass

    tags_yaml = "\n".join([f"  - {tag}" for tag in tags])

    readme = f"""---
        license: apache-2.0
        base_model: {base_model_id}
        library_name: {library_name}
        pipeline_tag: {pipeline_tag}
        tags:
        {tags_yaml}
        ---

        # LoRA Model
        
        This is a LoRA (Low-Rank Adaptation) model fine-tuned from [{base_model_id}](https://huggingface.co/{base_model_id}).
        {lora_info}
        ## Usage
        
        """

    if lora_type == "image":
        readme += f"""### With Diffusers

        ```python
        from diffusers import DiffusionPipeline
        import torch
        
        # Load base model
        pipe = DiffusionPipeline.from_pretrained(
            "{base_model_id}",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        
        # Load LoRA weights
        pipe.load_lora_weights("YOUR_USERNAME/YOUR_REPO_NAME")
        
        # Generate image
        image = pipe(
            prompt="your prompt here",
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        image.save("output.png")
        ```
        """
    else:
        readme += f"""### With PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model_id}",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{base_model_id}")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/YOUR_REPO_NAME")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

    readme += """
## Training

This model was trained using the Moirai network.

## License

This model is released under the Apache 2.0 license.
"""

    return readme


def extract_model_id_from_path(cache_path: str) -> Optional[str]:

    match = re.search(r'models--([^/]+)--([^/]+)', cache_path)
    if match:
        org = match.group(1)
        model = match.group(2)
        return f"{org}/{model}"

    if "/" in cache_path and not cache_path.startswith("/"):
        return cache_path

    return None
