# agents/crew_negotiation_strategist.py
from crewai import Agent
from .negotiation_strategist import NegotiationStrategistAgent as BaseNegotiationStrategist

class CrewNegotiationStrategist(Agent):
    """
    Agent specialized in negotiating contract clauses with a focus on risk management and compliance.
    """

    def __init__(self, llm, verbose=True):
        """
        Initializes the negotiation strategist agent.

        Args:
            llm: The language model to be used by the agent.
            verbose (bool): If True, enables verbose output for debugging.
        """
        super().__init__(
            role="Negotiation Strategist",
            goal="Propose alternative wording for high-risk clauses",
            backstory="Specializes in suggesting compliant and balanced NDA language.",
            verbose=verbose,
            llm=llm,
            allow_delegation=False
        )

    def _execute_task(self, task):
        """
        Accepts a clause and risk assessment for negotiation strategy.

        Args:
            task: An object containing the clause and its risk assessment.

        Returns:
            A dictionary with the proposed changes or an error message.
        """
        clause = task.data.get("clause", "")
        risk_assessment = task.data.get("risk_assessment", {})
        base_agent = BaseNegotiationStrategist()
        try:
            result = base_agent.run(clause, risk_assessment)
            return result
        except Exception as e:
            return {"error": str(e)}
