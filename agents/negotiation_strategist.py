import os
import logging
from typing import Any, Dict, List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STRATEGIST] %(message)s',
    handlers=[
        logging.FileHandler("negotiation_strategist.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Placeholder for legal-specific data models
class NegotiationStrategy(BaseModel):
    """Pydantic model for structured negotiation output"""
    proposed_clause: str = "Proposed counter-clause text"
    rationale: str = "Legal reasoning for proposed changes"
    alternatives: List[str] = ["Alternative formulation 1", "Alternative formulation 2"]
    compliance_status: str = "Compliant/Non-compliant"
    risk_impact: str = "Risk impact assessment"

class NegotiationStrategistAgent:
    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize negotiation strategist with required components
        
        Args:
            llm: Optional pre-configured LLM instance
        """
        self.llm = llm or self._initialize_default_llm()
        self.parser = PydanticOutputParser(pydantic_object=NegotiationStrategy)
        self.prompt = self._load_prompt_template()
        self.chain = self._setup_chain()

    def _initialize_default_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize default LLM with basic configuration"""
        # TODO: Add API key validation and error handling
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.4,
            max_output_tokens=1024
        )

    def _load_prompt_template(self) -> ChatPromptTemplate:
        """Load system prompt from designated directory"""
        prompt_path = "prompts/system_prompts/negotiation_strategist.txt"
        
        # TODO: Add proper error handling for file operations
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                system_prompt = f.read()
        else:
            logger.warning(f"Prompt file not found at {prompt_path}. Using default template.")
            system_prompt = self._default_prompt_template()
            
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Clause: {clause}\n\nRisk Assessment:\n{risk_assessment}")
        ])

    def _default_prompt_template(self) -> str:
        """Fallback prompt template for negotiation strategy"""
        return """You are a Legal Negotiation Strategist specializing in NDAs.
Current Date: {current_date}
Your task:
1. Propose a negotiated clause based on risk assessment
2. Provide legal rationale for changes
3. Suggest alternative formulations
4. Evaluate compliance status and risk impact

{format_instructions}"""

    def _setup_chain(self) -> RunnableSequence:
        """Configure the processing chain with prompt, LLM, and parser"""
        return self.prompt | self.llm | self.parser

    def propose_counter_clause(self, clause: str) -> Dict[str, Any]:
        """
        Generate counter-clause proposals based on NDA analysis
        
        Args:
            clause: Original NDA clause text
            
        Returns:
            Dictionary containing negotiation strategy
        """
        # TODO: Implement actual negotiation logic
        logger.info(f"Proposing counter-clause for: {clause[:100]}...")
        return {
            "proposed_clause": f"Revised {clause}",
            "rationale": "Placeholder legal rationale",
            "alternatives": ["Alternative 1", "Alternative 2"],
            "compliance_status": "Compliant",
            "risk_impact": "Reduced"
        }

    def evaluate_clause_risk(self, clause: str) -> Dict[str, Any]:
        """
        Assess risk level of proposed clauses
        
        Args:
            clause: Clause to evaluate
            
        Returns:
            Risk assessment dictionary
        """
        # TODO: Integrate with RiskAnalystAgent
        logger.info(f"Evaluating risk for: {clause[:100]}...")
        return {
            "risk_level": "Medium",
            "confidence": 0.8,
            "legal_basis": "Standard NDA guidelines"
        }

    def run(self, clause: str, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute negotiation strategy generation workflow
        
        Args:
            clause: Original NDA clause
            risk_assessment: Risk assessment from RiskAnalystAgent
            
        Returns:
            Structured negotiation strategy
        """
        try:
            # Format risk assessment for prompt
            risk_str = "\n".join([f"{k}: {v}" for k, v in risk_assessment.items()])
            
            # Generate strategy using chain
            result = self.chain.invoke({
                "clause": clause,
                "risk_assessment": risk_str
            })
            
            return result.model_dump()
            
        except OutputParserException as pe:
            logger.error(f"Output parsing failed: {pe}")
            # TODO: Add fallback strategy generation
            return {
                "proposed_clause": "Manual review required",
                "rationale": "Failed to parse model output",
                "alternatives": ["Consult legal counsel", "Use default template"],
                "compliance_status": "Unknown",
                "risk_impact": "Unknown"
            }
        except Exception as e:
            logger.error(f"Negotiation strategy generation failed: {e}", exc_info=True)
            raise

# Example usage (for testing purposes)
if __name__ == "__main__":
    strategist = NegotiationStrategistAgent()
    test_clause = "Confidentiality lasts indefinitely for all information disclosed."
    test_risk = {"risk_level": "High", "reason": "Indefinite duration"}
    
    strategy = strategist.run(test_clause, test_risk)
    print("\nNEGOTIATION STRATEGY:")
    for k, v in strategy.items():
        print(f"{k.upper()}: {v}")