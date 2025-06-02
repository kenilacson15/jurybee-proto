# agents/crew_risk_analyst.py
from crewai import Agent
from .risk_analyst import RiskAnalystAgent as BaseRiskAnalyst

class CrewRiskAnalyst(Agent):
    """
    CrewRiskAnalyst is an agent specialized in analyzing NDA clauses
    to generate risk scores, identifying potential high-risk areas.
    """

    def __init__(self, llm, verbose=True):
        """
        Initializes the CrewRiskAnalyst agent.

        :param llm: Language model to be used by the agent.
        :param verbose: If True, enables verbose output.
        """
        super().__init__(
            role="Risk Analyst",
            goal="Generate risk scores for NDA clauses",
            backstory="Specializes in identifying high-risk clauses in NDAs.",
            verbose=verbose,
            llm=llm,
            allow_delegation=False
        )

    def _execute_task(self, task):
        """
        Accepts a clause for risk analysis. Returns risk assessment or error.

        :param task: The task containing the clause to be analyzed.
        :return: Risk assessment result or error message.
        """
        clause = task.data.get("clause", "")
        base_agent = BaseRiskAnalyst()  # Instantiate only when needed
        try:
            result = base_agent.analyze_clause(clause)
            return result
        except Exception as e:
            return {"error": str(e)}
