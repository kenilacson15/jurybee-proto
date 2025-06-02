# tasks/nda_tasks.py
from core.task import Task

class NDATaskFactory:
    @staticmethod
    def create_risk_analysis_task(agent, clause):
        return Task(
            data={"clause": clause},
            meta={"description": f"Analyze risk for clause: {clause[:100]}...", "agent": str(agent), "expected_output": "Risk score (High/Medium/Low) with statute references"}
        )

    @staticmethod
    def create_compliance_check_task(agent, clause):
        return Task(
            data={"clause": clause},
            meta={"description": f"Check compliance for clause: {clause[:100]}...", "agent": str(agent), "expected_output": "Compliance status (Compliant/Non-compliant) with issues"}
        )

    @staticmethod
    def create_negotiation_task(agent, clause, risk_assessment):
        return Task(
            data={"clause": clause, "risk_assessment": risk_assessment},
            meta={"description": f"Propose alternatives for clause: {clause[:100]}...", "agent": str(agent), "expected_output": "Alternative clause wording with rationale"}
        )