"""
Asynchronous background task queue and progress streaming engine.

Enables non-blocking execution of long-running analytics (Leave-Cell-Out cross-validation,
PyBaMM parameter fits, physics calibration, large fleet batch simulations) with real-time
Server-Sent Events (SSE) and WebSocket progress reporting.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple, Any, Generator


class TaskRecord:
    """Represents a tracked asynchronous task."""
    
    def __init__(self, task_id: str, name: str, org_id: int = 1):
        self.task_id = task_id
        self.name = name
        self.org_id = org_id
        self.status = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
        self.progress_pct = 0
        self.current_stage = "Initialized"
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.updated_at = self.created_at
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def update(self, progress_pct: int, stage: str, result: Optional[Any] = None):
        """Update task progress and state."""
        with self._lock:
            self.progress_pct = int(min(100, max(0, progress_pct)))
            self.current_stage = stage
            self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if self.progress_pct >= 100:
                self.status = "COMPLETED"
                self.result = result
            elif self.status == "PENDING":
                self.status = "RUNNING"
                
            event = {
                "task_id": self.task_id,
                "status": self.status,
                "progress_pct": self.progress_pct,
                "stage": self.current_stage,
                "timestamp": self.updated_at,
            }
            self._events.append(event)
            
    def fail(self, error_msg: str):
        """Mark task as failed."""
        with self._lock:
            self.status = "FAILED"
            self.error = error_msg
            self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._events.append({
                "task_id": self.task_id,
                "status": "FAILED",
                "error": error_msg,
                "timestamp": self.updated_at,
            })
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert task record to dict."""
        with self._lock:
            return {
                "task_id": self.task_id,
                "name": self.name,
                "org_id": self.org_id,
                "status": self.status,
                "progress_pct": self.progress_pct,
                "current_stage": self.current_stage,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "result": self.result,
                "error": self.error,
            }


class TaskQueue:
    """Thread pool task queue manager."""
    
    def __init__(self, max_workers: int = 4):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.Lock()
        
    def submit_task(
        self,
        name: str,
        fn: Callable[[TaskRecord, Any], Any],
        *args,
        org_id: int = 1,
        **kwargs,
    ) -> str:
        """Submit a new background job."""
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        task = TaskRecord(task_id=task_id, name=name, org_id=org_id)
        
        with self._lock:
            self._tasks[task_id] = task
            
        def _wrapper():
            try:
                task.update(5, f"Starting {name}...")
                res = fn(task, *args, **kwargs)
                task.update(100, "Completed successfully.", result=res)
            except Exception as e:
                task.fail(str(e))
                
        self._executor.submit(_wrapper)
        return task_id
        
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None
            
    def list_tasks(self, org_id: int = 1) -> List[Dict[str, Any]]:
        """List all tasks for an organization."""
        with self._lock:
            return [t.to_dict() for t in self._tasks.values() if t.org_id == org_id]
            
    def stream_task_events(self, task_id: str) -> Generator[str, None, None]:
        """Generator yielding SSE event strings for a running task."""
        last_idx = 0
        while True:
            with self._lock:
                task = self._tasks.get(task_id)
                if not task:
                    yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                    break
                events = list(task._events[last_idx:])
                status = task.status
                
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"
                last_idx += 1
                
            if status in ("COMPLETED", "FAILED"):
                break
                
            time.sleep(0.5)


# Global task queue instance
task_queue = TaskQueue()
