# agents/crew_risk_analyst.py
from crewai import Agent
from .risk_analyst import RiskAnalystAgent as BaseRiskAnalyst

class CrewRiskAnalyst(Agent):
    def __init__(self, llm, verbose=True):
        super().__init__(
            role="Risk Analyst",
            goal="Generate risk scores for NDA clauses",
            backstory="Specializes in identifying high-risk clauses in NDAs.",
            verbose=verbose,
            llm=llm,
            allow_delegation=False
        )

    def _execute_task(self, task):
        clause = task.input.get("clause", "")
        base_agent = BaseRiskAnalyst()  # Instantiate only when needed
        result = base_agent.analyze_clause(clause)
        return result