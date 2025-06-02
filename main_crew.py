# main_crew.py
import logging
from crewai import Crew, LLM  # Import LLM from crewai
from tools.reflexion_loop import ReflexionAgentMixin  # Import the reflexion mixin
from agents.crew_risk_analyst import CrewRiskAnalyst
from agents.crew_compliance_checker import CrewComplianceChecker
from agents.crew_negotiation_strategist import CrewNegotiationStrategist
from tasks.nda_tasks import NDATaskFactory

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Wrap all CrewAI agents with ReflexionAgentMixin for self-reflection
class ReflexiveCrewRiskAnalyst(ReflexionAgentMixin, CrewRiskAnalyst):
    pass

class ReflexiveCrewComplianceChecker(ReflexionAgentMixin, CrewComplianceChecker):
    pass

class ReflexiveCrewNegotiationStrategist(ReflexionAgentMixin, CrewNegotiationStrategist):
    pass

class NDACrew:
    def __init__(self, llm):
        self.llm = llm
        self.risk_agent = ReflexiveCrewRiskAnalyst(llm)
        self.compliance_agent = ReflexiveCrewComplianceChecker(llm)
        self.negotiation_agent = ReflexiveCrewNegotiationStrategist(llm)

    def analyze_clause(self, clause):
        try:
            # Create tasks
            risk_task = NDATaskFactory.create_risk_analysis_task(self.risk_agent, clause)
            compliance_task = NDATaskFactory.create_compliance_check_task(self.compliance_agent, clause)
            negotiation_task = NDATaskFactory.create_negotiation_task(
                self.negotiation_agent, clause, getattr(risk_task, 'output', None)
            )

            # Orchestrate crew
            crew = Crew(
                agents=[self.risk_agent, self.compliance_agent, self.negotiation_agent],
                tasks=[risk_task, compliance_task, negotiation_task],
                verbose=True  # Show detailed execution steps
            )

            # Use run_with_reflexion for each agent's task execution
            for task in crew.tasks:
                agent = task.agent
                if hasattr(agent, 'run_with_reflexion'):
                    # Replace the agent's output with reflexion-audited output
                    task.output = agent.run_with_reflexion(task)
                else:
                    task.output = agent._execute_task(task)

            # Optionally, aggregate or return all outputs
            return {
                'risk_analysis': getattr(risk_task, 'output', None),
                'compliance_check': getattr(compliance_task, 'output', None),
                'negotiation': getattr(negotiation_task, 'output', None)
            }
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