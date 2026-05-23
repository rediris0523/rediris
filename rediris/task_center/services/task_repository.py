from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import Optional, Tuple, List
from rediris.common.models.task import Task, TaskStatus
from rediris.common.models.miner_submission import MinerSubmission
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class TaskRepository:
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_task_id(self, task_id: str) -> Optional[Task]:
        return self.db.query(Task).filter(Task.task_id == task_id).first()
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Task], int]:
        query = self.db.query(Task)
        
        if status:
            query = query.filter(Task.status == status)
        
        total = query.count()
        
        tasks = query.offset((page - 1) * page_size).limit(page_size).all()
        
        return tasks, total
    
    def update_status(self, task_id: str, status: TaskStatus):
        task = self.get_by_task_id(task_id)
        if task:
            task.status = status
            self.db.commit()
    
    def get_submissions_by_task(self, task_id: str) -> List[MinerSubmission]:
        return self.db.query(MinerSubmission).filter(
            MinerSubmission.task_id == task_id
        ).all()
    
    def get_submission_by_id(self, submission_id: str) -> Optional[MinerSubmission]:
        return self.db.query(MinerSubmission).filter(
            MinerSubmission.id == submission_id
        ).first()

