import os
from typing import Dict, Any, Optional, List
from rediris.common.utils.logging import setup_logger
from rediris.common.config.yaml_config import YamlConfig
import asyncio
import torch
import time
from pathlib import Path

logger = setup_logger(__name__)


def retry_on_connection_error(max_retries=3, delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionResetError, ConnectionError, OSError) as e:
                    last_exception = e
                    logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                except Exception as e:
                    if "Connection" in str(e) or "reset" in str(e).lower():
                        last_exception = e
                        logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {delay} seconds...")
                            time.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        TrainerCallback,
    )
    from datasets import load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

try:
    from huggingface_hub import HfApi
    HF_API_AVAILABLE = True
except ImportError:
    HF_API_AVAILABLE = False
    logger.warning("huggingface_hub not available")


# Default LoRA target modules for Qwen-based models (attention + MLP layers)
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TextTrainingService:

    def __init__(self, config: Optional[YamlConfig] = None):
        self.config = config
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)

    def _is_model_complete(self, model_path: Path) -> bool:
        if not model_path.exists() or not model_path.is_dir():
            return False

        has_adapter = (
            (model_path / "adapter_model.safetensors").exists() or
            (model_path / "adapter_model.bin").exists()
        )

        has_config = (model_path / "adapter_config.json").exists()

        return has_adapter and has_config

    def _detect_dataset_format(self, dataset) -> str:
        """Auto-detect dataset format: 'sharegpt', 'prompt_story', 'prompt_response', 'raw_text', or 'qa'."""
        columns = dataset.column_names
        if "conversations" in columns:
            return "sharegpt"
        if "prompt" in columns and "story" in columns:
            return "prompt_story"
        if "prompt" in columns and "response" in columns:
            return "prompt_response"
        # Many text datasets expose a single free-form field like `text`/`content`.
        if any(c in columns for c in ("text", "content", "article", "paragraph", "body", "document")):
            return "raw_text"
        return "qa"

    def _build_sharegpt_texts(self, examples) -> List[str]:
        """Build training texts from ShareGPT multi-turn conversations."""
        texts = []
        conversations_list = examples.get("conversations", [])
        for conversations in conversations_list:
            parts = []
            for turn in conversations:
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if role in ("human", "user"):
                    parts.append(f"### User:\n{value}")
                elif role in ("gpt", "assistant"):
                    parts.append(f"### Assistant:\n{value}")
            if parts:
                texts.append("\n\n".join(parts))
        return texts

    def _build_prompt_story_texts(self, examples, prompt_col="prompt", story_col="story") -> List[str]:
        """Build training texts from prompt/story pairs."""
        texts = []
        prompts = examples.get(prompt_col, [])
        stories = examples.get(story_col, [])
        for p, s in zip(prompts, stories):
            texts.append(f"### Writing Task:\n{p}\n\n### Story:\n{s}")
        return texts

    def _build_qa_texts(self, examples, q_col="question", a_col="answer") -> List[str]:
        """Build training texts from question/answer pairs (backward compatible)."""
        texts = []
        questions = examples.get(q_col, examples.get("question", []))
        answers = examples.get(a_col, examples.get("answer", []))
        for q, a in zip(questions, answers):
            texts.append(f"### Question:\n{q}\n\n### Answer:\n{a}")
        return texts

    def _pick_raw_text_column(self, dataset_columns: List[str]) -> str:
        # Prefer well-known free-form columns first.
        for c in ("text", "content", "article", "paragraph", "body", "document"):
            if c in dataset_columns:
                return c
        # Fallback: choose the first string-like column name.
        return dataset_columns[0] if dataset_columns else "text"

    def _build_raw_texts(self, examples, text_col: str) -> List[str]:
        """Build training texts from a single raw text column."""
        raw = examples.get(text_col, [])
        texts: List[str] = []
        for t in raw:
            if t is None:
                continue
            # Some datasets may return nested types; stringify best-effort.
            texts.append(str(t))
        return texts

    async def train_lora(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available")
        if not HF_API_AVAILABLE:
            raise RuntimeError("huggingface_hub library not available")

        task_id = task.get('task_id')
        logger.info(f"Starting text LoRA training for task {task_id}")

        model_path = self.models_dir / task_id
        if self._is_model_complete(model_path):
            logger.info(f"Model already exists for task {task_id} at {model_path}, skipping training")

            workflow_spec = task.get("workflow_spec", {})
            training_spec = workflow_spec.get("training_spec", {})
            training_mode = workflow_spec.get("training_mode", "new")
            iteration_count = training_spec.get("iteration_count", 500)

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

        text_config = self.config.get_text_training_config() if self.config else {}

        # Read parameters from config with sensible defaults for 9B model on 48GB VRAM
        base_model = training_spec.get("base_model", text_config.get("base_model", "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"))
        lora_rank = training_spec.get("lora_rank", text_config.get("default_lora_rank", 32))
        lora_alpha = training_spec.get("lora_alpha", text_config.get("default_lora_alpha", 64))
        iteration_count = training_spec.get("iteration_count", text_config.get("default_iteration_count", 800))
        num_train_epochs = training_spec.get("num_train_epochs", text_config.get("default_num_train_epochs", 1))
        batch_size = training_spec.get("batch_size", text_config.get("default_batch_size", 2))
        gradient_accumulation_steps = training_spec.get("gradient_accumulation_steps", text_config.get("default_gradient_accumulation_steps", 4))
        learning_rate = training_spec.get("learning_rate", text_config.get("default_learning_rate", 2e-5))
        weight_decay = training_spec.get("weight_decay", text_config.get("default_weight_decay", 0.01))
        warmup_ratio = training_spec.get("warmup_ratio", text_config.get("default_warmup_ratio", 0.1))
        max_grad_norm = training_spec.get("max_grad_norm", text_config.get("default_max_grad_norm", 1.0))
        max_length = training_spec.get("max_length", text_config.get("default_max_length", 2048))
        save_steps = training_spec.get("save_steps", text_config.get("default_save_steps", 200))
        save_total_limit = training_spec.get("save_total_limit", text_config.get("default_save_total_limit", 3))
        lora_dropout = training_spec.get("lora_dropout", text_config.get("default_lora_dropout", 0.05))

        # Read target modules from config or use defaults
        target_modules = text_config.get("lora_target_modules", DEFAULT_TARGET_MODULES)
        if isinstance(target_modules, str):
            target_modules = [m.strip() for m in target_modules.split(",")]

        training_mode = workflow_spec.get("training_mode", "new")
        base_lora_url = workflow_spec.get("base_lora_url")

        datasets_config = self.config.get_datasets_config() if self.config else {}
        text_dataset_config = datasets_config.get("text", {})

        from rediris.common.utils.huggingface import parse_dataset_url
        dataset_url = task.get("dataset_url")
        if dataset_url:
            dataset_repo = parse_dataset_url(dataset_url)
            logger.info(f"[TextTrainingService] Using miner's validated dataset: {dataset_repo}")
        else:
            dataset_repo = dataset_spec.get("repository_id", text_dataset_config.get("repository_id", "Nitral-AI/Creative_Writing-ShareGPT"))
            logger.info(f"[TextTrainingService] Using task spec dataset: {dataset_repo}")

        question_column = dataset_spec.get("question_column", text_dataset_config.get("question_column", "question"))
        answer_column = dataset_spec.get("answer_column", text_dataset_config.get("answer_column", "answer"))
        sample_count = dataset_spec.get("sample_count", 2000)

        logger.info("=" * 60)
        logger.info(f"Task: {task_id}, Training Mode: {training_mode}")
        logger.info(f"Model: base_model={base_model}, lora_rank={lora_rank}, lora_alpha={lora_alpha}")
        logger.info(f"LoRA target_modules: {target_modules}")
        logger.info(f"Training: lr={learning_rate}, batch={batch_size}, grad_accum={gradient_accumulation_steps}, "
                   f"effective_batch={batch_size * gradient_accumulation_steps}")
        logger.info(f"Training: epochs={num_train_epochs}, max_steps={iteration_count}, warmup={warmup_ratio}, "
                   f"weight_decay={weight_decay}, max_grad_norm={max_grad_norm}")
        logger.info(f"Training: max_length={max_length}, save_steps={save_steps}")
        logger.info(f"Dataset: repo={dataset_repo}, samples={sample_count}")
        logger.info("=" * 60)

        try:
            @retry_on_connection_error(max_retries=3, delay=10)
            def load_tokenizer():
                logger.info(f"Loading tokenizer: {base_model}")
                return AutoTokenizer.from_pretrained(base_model)

            @retry_on_connection_error(max_retries=3, delay=10)
            def load_model():
                logger.info(f"Loading base model: {base_model}")
                return AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto"
                )

            tokenizer = load_tokenizer()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = load_model()

            if training_mode == "incremental" and base_lora_url:
                logger.info(f"Incremental training: Loading base LoRA from {base_lora_url}")
                model = PeftModel.from_pretrained(model, base_lora_url)
            else:
                logger.info("New training: Starting from base model")

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            model = get_peft_model(model, lora_config)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
            if trainable_params <= 0:
                logger.error(
                    "No trainable parameters after LoRA injection. "
                    f"base_model={base_model}, target_modules={target_modules}"
                )
                raise RuntimeError(
                    "LoRA initialization produced zero trainable parameters. "
                    "Please verify target_modules and base model architecture."
                )

            # Gradient checkpointing + PEFT requires enabling input grads,
            # otherwise loss can become detached and backward will fail.
            if hasattr(model, "config"):
                model.config.use_cache = False
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.train()

            @retry_on_connection_error(max_retries=3, delay=10)
            def load_training_dataset():
                logger.info(f"Loading dataset: {dataset_repo} (sample_count={sample_count})")
                return load_dataset(dataset_repo, split=f"train[:{sample_count}]")

            dataset = load_training_dataset()

            dataset_columns = list(getattr(dataset, "column_names", []) or [])
            detected_format = self._detect_dataset_format(dataset)

            # If the miner config forces a format (e.g. sharegpt) but the dataset
            # doesn't have the required columns, fall back to auto-detection so
            # training doesn't silently become empty.
            configured_format = text_dataset_config.get("format")
            dataset_format = configured_format or detected_format

            if configured_format:
                if configured_format == "sharegpt" and "conversations" not in dataset_columns:
                    dataset_format = detected_format
                elif configured_format in ("prompt_story", "prompt_response"):
                    if "prompt" not in dataset_columns:
                        dataset_format = detected_format
                    else:
                        if configured_format == "prompt_story" and "story" not in dataset_columns:
                            dataset_format = detected_format
                        if configured_format == "prompt_response" and "response" not in dataset_columns:
                            dataset_format = detected_format
                elif configured_format == "qa":
                    q_col = question_column if question_column is not None else "question"
                    a_col = answer_column if answer_column is not None else "answer"
                    if q_col not in dataset_columns or a_col not in dataset_columns:
                        dataset_format = detected_format

            logger.info(f"Dataset format detected: {dataset_format} (configured={configured_format}, auto={detected_format})")

            raw_text_col = self._pick_raw_text_column(dataset_columns)

            def preprocess_function(examples):
                # datasets.map(batched=True) passes column->list batches.
                # We must always feed tokenizer a non-empty list, otherwise
                # fast tokenizers can throw IndexError(list index out of range).
                batch_size = 0
                if examples:
                    first_key = next(iter(examples.keys()), None)
                    if first_key is not None and isinstance(examples.get(first_key), list):
                        batch_size = len(examples.get(first_key, []))

                if dataset_format == "sharegpt":
                    texts = self._build_sharegpt_texts(examples)
                elif dataset_format in ("prompt_story", "prompt_response"):
                    story_col = "story" if "story" in dataset.column_names else "response"
                    texts = self._build_prompt_story_texts(examples, "prompt", story_col)
                elif dataset_format == "raw_text":
                    texts = self._build_raw_texts(examples, raw_text_col)
                else:
                    texts = self._build_qa_texts(examples, question_column, answer_column)

                # Fallback: if configured format produced zero samples for this batch,
                # try raw text extraction from common columns.
                if not texts:
                    fallback_col = self._pick_raw_text_column(list(examples.keys()))
                    texts = self._build_raw_texts(examples, fallback_col)
                    if texts:
                        logger.warning(
                            f"Preprocess fallback triggered: format={dataset_format}, "
                            f"fallback_col={fallback_col}, batch_size={batch_size}"
                        )

                # Last resort: keep map pipeline alive and visible in logs.
                # Use placeholders matching batch size so downstream shapes stay valid.
                if not texts:
                    if batch_size <= 0:
                        batch_size = 1
                    texts = [" " for _ in range(batch_size)]
                    logger.warning(
                        f"Preprocess produced empty texts; using placeholders. "
                        f"format={dataset_format}, batch_size={batch_size}, example_keys={list(examples.keys())}"
                    )

                model_inputs = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors=None
                )

                model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
                return model_inputs

            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset.column_names
            )

            logger.info(f"Dataset size: {len(tokenized_dataset)} samples")
            if len(tokenized_dataset) > 0:
                sample = tokenized_dataset[0]
                logger.debug(f"Sample input_ids length: {len(sample['input_ids'])}, labels length: {len(sample['labels'])}")

            from dataclasses import dataclass

            @dataclass
            class DataCollatorForCausalLM:
                tokenizer: any
                padding: bool = True
                max_length: int = None
                pad_to_multiple_of: int = None

                def __call__(self, features):
                    labels = [f.pop("labels") for f in features] if "labels" in features[0] else None

                    batch = self.tokenizer.pad(
                        features,
                        padding=self.padding,
                        max_length=self.max_length,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors="pt",
                    )

                    if labels is not None:
                        max_label_length = max(len(l) for l in labels)
                        padding_side = self.tokenizer.padding_side
                        padded_labels = []
                        for l in labels:
                            remainder = [-100] * (max_label_length - len(l))
                            if padding_side == "right":
                                padded_labels.append(l + remainder)
                            else:
                                padded_labels.append(remainder + l)
                        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

                    return batch

            data_collator = DataCollatorForCausalLM(
                tokenizer=tokenizer,
                padding="max_length",
                max_length=max_length
            )

            model_path = self.models_dir / task_id
            model_path.mkdir(parents=True, exist_ok=True)

            log_steps = max(1, iteration_count // 20)

            bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

            class LoggerCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not logs:
                        return
                    logger.info(
                        f"[step {state.global_step}/{state.max_steps}] "
                        f"loss={logs.get('loss')}, "
                        f"lr={logs.get('learning_rate')}, "
                        f"grad_norm={logs.get('grad_norm')}"
                    )

            training_args = TrainingArguments(
                output_dir=str(model_path),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type="cosine",
                logging_steps=log_steps,
                logging_first_step=True,
                save_steps=save_steps,
                max_steps=iteration_count,
                save_total_limit=save_total_limit,
                load_best_model_at_end=False,
                report_to="none",
                bf16=bf16_available,
                fp16=not bf16_available,
                gradient_checkpointing=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                callbacks=[LoggerCallback()],
            )

            logger.info(f"Starting {'incremental' if training_mode == 'incremental' else 'new'} training...")
            trainer.train()

            logger.info("Saving model...")
            model.save_pretrained(str(model_path))
            tokenizer.save_pretrained(str(model_path))

            try:
                from rediris.common.utils.lora_metadata import fix_lora_base_model
                fix_lora_base_model(
                    model_path=str(model_path),
                    base_model_id=base_model,
                    lora_type="text"
                )
            except Exception as e:
                logger.warning(f"Failed to fix LoRA metadata: {e}")

            logger.info(f"Text LoRA training completed for task {task_id}, saved to {model_path}")

            return {
                "status": "completed",
                "model_url": None,
                "training_steps": iteration_count,
                "model_path": str(model_path),
                "training_mode": training_mode,
                "final_loss": trainer.state.log_history[-1].get("loss", 0.0) if trainer.state.log_history else 0.0
            }
        except Exception as e:
            logger.error(f"Text LoRA training failed: {e}", exc_info=True)
            raise
