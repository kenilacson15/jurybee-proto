# main_crew.py
import logging
from crewai import Crew, LLM  # Import LLM from crewai
from agents.crew_risk_analyst import CrewRiskAnalyst
from agents.crew_compliance_checker import CrewComplianceChecker
from agents.crew_negotiation_strategist import CrewNegotiationStrategist
from tasks.nda_tasks import NDATaskFactory

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

class NDACrew:
    def __init__(self, llm):
        self.llm = llm
        self.risk_agent = CrewRiskAnalyst(llm)
        self.compliance_agent = CrewComplianceChecker(llm)
        self.negotiation_agent = CrewNegotiationStrategist(llm)

    def analyze_clause(self, clause):
        try:
            # Create tasks
            risk_task = NDATaskFactory.create_risk_analysis_task(self.risk_agent, clause)
            compliance_task = NDATaskFactory.create_compliance_check_task(self.compliance_agent, clause)
            negotiation_task = NDATaskFactory.create_negotiation_task(
                self.negotiation_agent, clause, risk_task.output
            )

            # Orchestrate crew
            crew = Crew(
                agents=[self.risk_agent, self.compliance_agent, self.negotiation_agent],
                tasks=[risk_task, compliance_task, negotiation_task],
                verbose=True  # Show detailed execution steps
            )

            result = crew.kickoff()
            return result
        except Exception as e:
            logging.error("Crew execution failed: %s", e, exc_info=True)
            return {"error": str(e)}

if __name__ == "__main__":
    # ✅ Use CrewAI's LLM wrapper (LiteLLM-based)
    from crewai import LLM

    # Verify API key is set
    import os
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found in environment")

    # Initialize LLM with valid LiteLLM format
    llm = LLM(
        model="gemini/gemini-1.5-flash",  # ✅ Valid format
        temperature=0.3
    )

    # Run crew
    nda_crew = NDACrew(llm)
    clause = "Confidentiality lasts indefinitely for all information disclosed."
    result = nda_crew.analyze_clause(clause)
    print("✅ Final Result:", result)