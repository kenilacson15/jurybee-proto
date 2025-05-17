# agents/crew_negotiation_strategist.py
from crewai import Agent
from .negotiation_strategist import NegotiationStrategistAgent as BaseNegotiationStrategist

class CrewNegotiationStrategist(Agent):
    def __init__(self, llm, verbose=True):
        super().__init__(
            role="Negotiation Strategist",
            goal="Propose alternative wording for high-risk clauses",
            backstory="Specializes in suggesting compliant and balanced NDA language.",
            verbose=verbose,
            llm=llm,
            allow_delegation=False
        )

    def _execute_task(self, task):
        clause = task.input.get("clause", "")
        risk_assessment = task.input.get("risk_assessment", {})
        base_agent = BaseNegotiationStrategist()
        result = base_agent.run(clause, risk_assessment)
        return result