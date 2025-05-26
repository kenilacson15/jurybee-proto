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
        """
        Accepts either a text clause or a file path (PDF/image) for compliance checking.
        """
        base_agent = BaseComplianceChecker()
        clause = task.input.get("clause", "")
        file_path = task.input.get("file_path", None)
        try:
            if file_path:
                result = base_agent.check_compliance_from_file(file_path)
            else:
                result = base_agent.check_compliance(clause)
            return result
        except Exception as e:
            return {"error": str(e)}