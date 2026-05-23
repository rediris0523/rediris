from typing import Dict, Optional
from datetime import datetime, timezone
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class GPUManager:
    def __init__(self, gpu_count: int = 2):
        self.gpus = [
            {
                "id": i,
                "available": True,
                "current_task": None,
                "task_type": None,
                "allocated_at": None
            }
            for i in range(gpu_count)
        ]
    
    def allocate_gpu(self, task_type: str) -> Optional[int]:
        for gpu in self.gpus:
            if gpu["available"]:
                gpu["available"] = False
                gpu["current_task"] = task_type
                gpu["allocated_at"] = datetime.now(timezone.utc)
                return gpu["id"]
        return None
    
    def release_gpu(self, gpu_id: int):
        gpu = self.gpus[gpu_id]
        gpu["available"] = True
        gpu["current_task"] = None
        gpu["allocated_at"] = None
    
    def get_available_gpu_count(self) -> int:
        return sum(1 for gpu in self.gpus if gpu["available"])
    
    def get_gpu_utilization(self) -> float:
        total_gpus = len(self.gpus)
        if total_gpus == 0:
            return 0.0
        used_gpus = sum(1 for gpu in self.gpus if not gpu["available"])
        return used_gpus / total_gpus