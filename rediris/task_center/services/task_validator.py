from typing import Dict, Any, List, Tuple
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class TaskValidationError(Exception):
    pass

class TaskValidator:

    @staticmethod
    def validate_workflow_spec(workflow_spec: Dict[str, Any]) -> Tuple[bool, List[str]]:

        errors = []
        
        required_fields = ['theme', 'target_platform', 'deployment_target', 'training_mode', 'dataset_spec', 'training_spec']
        for field in required_fields:
            if field not in workflow_spec:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        training_mode = workflow_spec.get('training_mode')
        if training_mode not in ['new', 'incremental']:
            errors.append(f"Invalid training_mode: {training_mode}. Must be 'new' or 'incremental'")
        
        if training_mode == 'incremental':
            base_lora_url = workflow_spec.get('base_lora_url')
            if not base_lora_url:
                errors.append("base_lora_url is required for incremental training")
            elif not isinstance(base_lora_url, str):
                errors.append("base_lora_url must be a string")
            else:

                is_url = base_lora_url.startswith(('http://', 'https://'))
                is_hf_repo_id = '/' in base_lora_url and not base_lora_url.startswith(('.', '/'))
                is_local_path = base_lora_url.startswith(('.', '/'))
                if not (is_url or is_hf_repo_id or is_local_path):
                    errors.append("base_lora_url must be a valid HuggingFace URL, repo ID (namespace/repo_name), or local path")
        
        training_spec = workflow_spec.get('training_spec', {})
        training_errors = TaskValidator._validate_training_spec(training_spec, workflow_spec.get('target_platform'))
        errors.extend(training_errors)
        
        dataset_spec = workflow_spec.get('dataset_spec', {})
        dataset_errors = TaskValidator._validate_dataset_spec(dataset_spec, workflow_spec.get('target_platform'))
        errors.extend(dataset_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_training_spec(training_spec: Dict[str, Any], target_platform: str) -> List[str]:
        errors = []
        
        required_fields = ['base_model', 'lora_rank', 'lora_alpha', 'num_train_epochs', 'iteration_count', 'batch_size', 'save_steps', 'save_total_limit']
        for field in required_fields:
            if field not in training_spec:
                errors.append(f"Missing required training_spec field: {field}")
        
        if errors:
            return errors
        
        base_model = training_spec.get('base_model')
        if not isinstance(base_model, str) or not base_model:
            errors.append("base_model must be a non-empty string")
        
        lora_rank = training_spec.get('lora_rank')
        if not isinstance(lora_rank, int) or lora_rank < 1 or lora_rank > 128:
            errors.append("lora_rank must be an integer between 1 and 128")
        
        lora_alpha = training_spec.get('lora_alpha')
        if not isinstance(lora_alpha, int) or lora_alpha < 1 or lora_alpha > 256:
            errors.append("lora_alpha must be an integer between 1 and 256")
        
        num_train_epochs = training_spec.get('num_train_epochs')
        if not isinstance(num_train_epochs, int) or num_train_epochs < 1 or num_train_epochs > 20:
            errors.append("num_train_epochs must be an integer between 1 and 20")
        
        save_steps = training_spec.get('save_steps')
        if not isinstance(save_steps, int) or save_steps < 50 or save_steps > 5000:
            errors.append("save_steps must be an integer between 50 and 5000")

        save_total_limit = training_spec.get('save_total_limit')
        if not isinstance(save_total_limit, int) or save_total_limit < 1 or save_total_limit > 20:
            errors.append("save_total_limit must be an integer between 1 and 20")
        
        iteration_count = training_spec.get('iteration_count')
        if not isinstance(iteration_count, int) or iteration_count < 100 or iteration_count > 100000:
            errors.append("iteration_count must be an integer between 100 and 100000")

        batch_size = training_spec.get('batch_size')
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append("batch_size must be a positive integer")

        if target_platform == 'mobile':
            if batch_size > 16:
                errors.append("batch_size for mobile (text) tasks should not exceed 16")
        elif target_platform == 'executor':
            if batch_size > 8:
                errors.append("batch_size for executor (image) tasks should not exceed 8")
        
        learning_rate = training_spec.get('learning_rate')
        if learning_rate is not None:
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 0.01:
                errors.append("learning_rate must be a positive number between 0 and 0.01")
        
        if target_platform == 'executor':
            resolution = training_spec.get('resolution')
            if resolution is not None:
                if not isinstance(resolution, list) or len(resolution) != 2:
                    errors.append("resolution must be a list of two integers [width, height]")
                else:
                    width, height = resolution[0], resolution[1]
                    if not isinstance(width, int) or not isinstance(height, int):
                        errors.append("resolution width and height must be integers")
                    elif width < 256 or width > 1024 or height < 256 or height > 1024:
                        errors.append("resolution dimensions must be between 256 and 1024")
        
        return errors
    
    @staticmethod
    def _validate_dataset_spec(dataset_spec: Dict[str, Any], target_platform: str) -> List[str]:
        errors = []
        
        required_fields = ['source', 'repository_id']
        for field in required_fields:
            if field not in dataset_spec:
                errors.append(f"Missing required dataset_spec field: {field}")
        
        if errors:
            return errors
        
        source = dataset_spec.get('source')
        if source != 'huggingface':
            errors.append("dataset source must be 'huggingface'")
        
        repository_id = dataset_spec.get('repository_id')
        if not isinstance(repository_id, str) or not repository_id:
            errors.append("repository_id must be a non-empty string")
        
        if target_platform == 'mobile':
            data_format = dataset_spec.get('data_format')
            if data_format != 'jsonl':
                errors.append("data_format for text tasks must be 'jsonl'")
            
            if 'question_column' not in dataset_spec or 'answer_column' not in dataset_spec:
                errors.append("text tasks require question_column and answer_column in dataset_spec")
        
        elif target_platform == 'executor':
            if 'image_column' not in dataset_spec or 'caption_column' not in dataset_spec:
                errors.append("image tasks require image_column and caption_column in dataset_spec")
        
        sample_count = dataset_spec.get('sample_count')
        if sample_count is not None:
            if not isinstance(sample_count, int) or sample_count < 1:
                errors.append("sample_count must be a positive integer")
        
        return errors
    
    @staticmethod
    def validate_task_create(task_data: Dict[str, Any]) -> Tuple[bool, List[str]]:

        errors = []
        
        task_id = task_data.get('task_id')
        if not task_id or not isinstance(task_id, str):
            errors.append("task_id is required and must be a string")
        
        workflow_type = task_data.get('workflow_type')
        if workflow_type not in ['text_lora_creation', 'image_lora_creation']:
            errors.append(f"Invalid workflow_type: {workflow_type}. Must be 'text_lora_creation' or 'image_lora_creation'")
        
        workflow_spec = task_data.get('workflow_spec')
        if not workflow_spec:
            errors.append("workflow_spec is required")
        else:
            spec_valid, spec_errors = TaskValidator.validate_workflow_spec(workflow_spec)
            if not spec_valid:
                errors.extend(spec_errors)
        
        for duration_field in ['announcement_duration', 'execution_duration', 'review_duration']:
            duration = task_data.get(duration_field)
            if duration is not None:
                if not isinstance(duration, (int, float)) or duration < 0:
                    errors.append(f"{duration_field} must be a non-negative number")
        
        return len(errors) == 0, errors

