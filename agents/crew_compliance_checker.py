# agents/crew_compliance_checker.py
from crewai import Agent
from .compliance_checker import ComplianceCheckerAgent as BaseComplianceChecker

class CrewComplianceChecker(Agent):
    def __init__(self, llm, verbose=True):
        super().__init__(
            role="Compliance Checker",
            goal="Verify clause compliance with legal standards",
            backstory="Specializes in checking legal compliance of NDA clauses.",
            verbose=verbose,
            llm=llm,
            allow_delegation=False
        )

    def _execute_task(self, task):
        clause = task.input.get("clause", "")
        base_agent = BaseComplianceChecker()
        result = base_agent.check_compliance(clause)
        return result