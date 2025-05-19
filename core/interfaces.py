# core/interfaces.py
from abc import ABC, abstractmethod
from core.schemas import AgentResponse

class IPlanner(ABC):
    @abstractmethod
    def generate_plan(self, clause: str) -> AgentResponse:
        pass