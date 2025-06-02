"""
reflexion_loop.py
Self-reflection mechanism for agentic AI systems.
Audits agent outputs, detects vague/insufficient explanations, and triggers retries/refinements.
Compatible with Jurybee agentic AI architecture.
"""
import logging
from typing import Any, Dict, Callable, Optional
import re
from crewai import Agent, Task  # Assumes CrewAI is installed and used
from pydantic import PrivateAttr  # Add this import

class ReflexionLoop:
    """
    Self-reflection loop for agent outputs.
    - Audits explanations for vagueness or insufficiency.
    - Triggers retry/refinement if issues are found.
    - Supports continuous improvement and transparency.
    """
    def __init__(self, agent_invoke_fn: Callable[[str], Any], max_retries: int = 2):
        """
        Args:
            agent_invoke_fn: Function to call the agent (e.g., agent.check_compliance).
            max_retries: Maximum number of refinement attempts.
        """
        self.agent_invoke_fn = agent_invoke_fn
        self.max_retries = max_retries
        self.logger = logging.getLogger("ReflexionLoop")

    def audit_explanation(self, explanation: str) -> Optional[str]:
        """
        Analyze the explanation for vagueness or insufficiency.
        Returns a string describing the issue if found, else None.
        """
        vague_patterns = [
            r"unclear|not specified|not enough information|insufficient|vague|cannot determine|unknown",
            r"no explanation|no details|not explained|not provided|not stated",
            r"as needed|as appropriate|as required|may vary|depends on context"
        ]
        for pattern in vague_patterns:
            if re.search(pattern, explanation, re.IGNORECASE):
                return f"Detected vague or insufficient explanation: '{pattern}'"
        # Heuristic: very short explanations are likely insufficient
        if len(explanation.strip()) < 30:
            return "Explanation too short to be meaningful."
        return None

    def run(self, clause: str, initial_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main reflexion loop. Audits and refines agent output if needed.
        Args:
            clause: The input clause analyzed by the agent.
            initial_output: The agent's initial output (should include 'issues' or 'explanation').
        Returns:
            Final output after possible refinement(s).
        """
        output = initial_output
        explanation = self._extract_explanation(output)
        for attempt in range(self.max_retries):
            issue = self.audit_explanation(explanation)
            if issue:
                self.logger.info(f"Reflexion triggered (attempt {attempt+1}): {issue}")
                # Trigger retry/refinement: ask agent to clarify or expand
                prompt = self._refinement_prompt(clause, explanation, issue)
                output = self.agent_invoke_fn(prompt)
                explanation = self._extract_explanation(output)
            else:
                break
        output['reflexion_attempts'] = attempt + 1
        output['reflexion_final'] = explanation
        return output

    def _extract_explanation(self, output: Dict[str, Any]) -> str:
        # Try to extract the explanation or issues field
        if 'explanation' in output:
            return output['explanation']
        if 'issues' in output and isinstance(output['issues'], list) and output['issues']:
            return ' '.join(str(issue) for issue in output['issues'])
        return str(output)

    def _refinement_prompt(self, clause: str, prev_explanation: str, issue: str) -> str:
        """
        Compose a prompt to ask the agent for a more detailed/clear explanation.
        """
        return (
            f"The previous explanation was flagged as insufficient: {issue}\n"
            f"Clause: {clause}\n"
            f"Previous explanation: {prev_explanation}\n"
            "Please provide a more detailed, clear, and specific explanation."
        )

class ReflexionAgentMixin:
    """
    Mixin for CrewAI agents to enable self-reflection on outputs.
    Integrates the ReflexionLoop for output auditing and retry logic.
    """
    # Use a Pydantic-compatible private attribute for reflexion
    _reflexion: any = PrivateAttr()

    def __init__(self, *args, reflexion_max_retries=2, **kwargs):
        super().__init__(*args, **kwargs)
        # ReflexionLoop expects a callable agent function
        self._reflexion = ReflexionLoop(self._invoke_for_reflexion, max_retries=reflexion_max_retries)

    def _invoke_for_reflexion(self, prompt_or_clause):
        """
        Internal method to invoke the agent for reflexion retries.
        Override this in your agent if input signature differs.
        """
        # By default, assume the agent's main method is _execute_task or check_compliance
        if hasattr(self, 'check_compliance'):
            return self.check_compliance(prompt_or_clause)
        elif hasattr(self, '_execute_task'):
            from crewai import Task
            return self._execute_task(Task(data={'clause': prompt_or_clause}))
        else:
            raise NotImplementedError("Agent must implement check_compliance or _execute_task.")

    def run_with_reflexion(self, task):
        """
        Run the agent on a CrewAI Task, then audit and refine output if needed.
        Returns the final (possibly refined) output.
        """
        # Run the agent as usual
        initial_output = self._execute_task(task)
        clause = getattr(task, 'data', {}).get('clause', '')
        # Reflexion loop audits and retries if needed
        final_output = self._reflexion.run(clause, initial_output)
        return final_output

# Example usage (to be integrated with agentic pipeline):
# from agents.compliance_checker import ComplianceCheckerAgent
# agent = ComplianceCheckerAgent()
# reflexion = ReflexionLoop(agent.check_compliance)
# clause = "Example NDA clause..."
# initial_output = agent.check_compliance(clause)
# final_output = reflexion.run(clause, initial_output)
# print(final_output)

# Example integration for a CrewAI agent:
# from agents.crew_compliance_checker import CrewComplianceChecker
# class ReflexiveCrewComplianceChecker(ReflexionAgentMixin, CrewComplianceChecker):
#     pass
#
# Usage in CrewAI pipeline:
# agent = ReflexiveCrewComplianceChecker(llm=None)
# result = agent.run_with_reflexion(task)
#
# This ensures the agent's output is audited and refined before returning to the CrewAI task flow.
#
# Assumptions:
# - CrewAI agents implement _execute_task(Task) or check_compliance(str)
# - Task.data is a dict with at least a 'clause' key
# - ReflexionLoop is imported from this module
#
# Constraints:
# - For custom agent input/output, override _invoke_for_reflexion
# - Reflexion adds some latency due to retries
#
# Dependencies:
# - CrewAI (crewai)
# - This file should be placed in the tools/ folder and imported by agents as needed
