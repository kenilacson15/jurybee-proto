from pydantic import BaseModel
from typing import Any, Dict, Optional

class Task(BaseModel):
    data: Dict[str, Any]  # Main input payload (e.g., {'clause': ...})
    output: Optional[Dict[str, Any]] = None  # Agent's result, can be set after execution
    meta: Optional[Dict[str, Any]] = None    # Optional: for traceability, e.g., timestamps, agent info

    class Config:
        arbitrary_types_allowed = True
