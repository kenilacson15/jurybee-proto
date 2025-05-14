import os
import logging
from typing import Any, Dict, List, Optional
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pydantic import BaseModel, ValidationError, Field, model_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableSequence
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    before_sleep_log, RetryError
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [RISK] %(message)s',
    handlers=[
        logging.FileHandler("risk_analyst.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class MissingAPIKeyError(RuntimeError):
    """Raised when required API keys are missing"""
    pass

class LLMInvocationError(RuntimeError):
    """Raised for LLM service failures"""
    pass

class ResponseParseError(RuntimeError):
    """Raised for malformed LLM outputs"""
    pass

class EmptyClauseError(ValueError):
    """Raised when input clause is empty"""
    pass

class RiskAnalysisError(RuntimeError):
    """Raised for any unhandled failure during risk analysis"""
    pass

# --- Response Schema ---
class RiskAssessment(BaseModel):
    risk_level: str = Field(..., description="Risk level: Low/Medium/High/Critical/Unknown")
    compliance_status: str = Field(..., description="Compliance status: Compliant/Non-compliant")
    explanation: str = Field(..., description="Detailed explanation of the assessment")
    issues: List[str] = Field(default_factory=list, description="List of specific issues")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    suggestion: Optional[str] = Field(default=None, description="Alternative wording suggestion")

    @model_validator(mode="after")
    def validate_fields(self):
        valid_levels = {"Low", "Medium", "High", "Critical", "Unknown"}
        if self.risk_level not in valid_levels:
            raise ValueError(f"risk_level must be one of {valid_levels}")
        
        valid_compliance = {"Compliant", "Non-compliant"}
        if self.compliance_status not in valid_compliance:
            raise ValueError(f"compliance_status must be one of {valid_compliance}")
            
        return self

class RiskAnalystAgent:
    def __init__(self):
        """Initialize agent with validated dependencies"""
        self.api_key = self._validate_api_key()
        self.llm = self._create_llm()
        self.parser = PydanticOutputParser(pydantic_object=RiskAssessment)
        
        # Build the prompt template with dynamic variables
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Clause: {clause}")
        ])
        
        # Create the processing chain with explicit variable injection
        self.chain: RunnableSequence = (
            self._prepare_input
            | self.prompt
            | self.llm
            | self._parse_response
        )

    def _validate_api_key(self) -> str:
        """Ensure required API key exists"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Missing GEMINI_API_KEY in environment")
            raise MissingAPIKeyError("GEMINI_API_KEY not found in environment variables")
        return api_key

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create LLM with validated config"""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                max_output_tokens=512,
                google_api_key=self.api_key,
                timeout=15
            )
        except Exception as e:
            logger.error("LLM initialization failed: %s", e, exc_info=True)
            raise LLMInvocationError(f"LLM initialization failed: {e}")

    def _get_system_prompt(self) -> str:
        """Get parameterized system prompt template"""
        return """You are a Risk Analyst specializing in NDAs.
Current Date: {current_date}
Your task:
1. Score the clause's risk level: Low/Medium/High/Critical/Unknown
2. Determine compliance status: Compliant/Non-compliant
3. List specific issues with the clause
4. Provide actionable recommendations
5. Suggest alternative wording if needed

Return a valid JSON object with keys: 
{format_instructions}

Important: 
- Return ONLY the JSON object without any additional text
- Use empty lists for issues and recommendations if none apply
- If unsure, use "Unknown" for risk_level and "Non-compliant" for compliance_status"""

    def _prepare_input(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Inject required variables into the input dictionary"""
        return {
            "clause": input_dict["clause"],
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "format_instructions": self.parser.get_format_instructions()
        }

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        raw = response.content.strip()
        logger.debug("Raw LLM response: %s", raw)

        try:
            parsed = self.parser.invoke(raw)
            return parsed.model_dump()  # Use Pydantic's built-in method

        except ValidationError as ve:
            logger.warning("Validation failed, attempting repair: %s", ve)
            return self._repair_output(raw)
        except OutputParserException as pe:
            logger.warning("Parsing failed, attempting repair: %s", pe)
            return self._repair_output(pe.message)
        except Exception as e:
            logger.error("Unexpected parsing error: %s", e, exc_info=True)
            raise ResponseParseError(f"Failed to parse response: {e}") from e

    @lru_cache(maxsize=128)
    def _repair_output(self, raw_text: str) -> Dict[str, Any]:
        from langchain.output_parsers import OutputFixingParser
        fixer = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

        try:
            fixed = fixer.parse(raw_text)
            logger.info("Successfully repaired response")
            return fixed.model_dump()  # Use Pydantic's method
        except Exception as e:
            logger.error("Response repair failed: %s", e, exc_info=True)
            return {
                "risk_level": "Critical",
                "compliance_status": "Non-compliant",
                "explanation": "Unable to analyze due to model error",
                "issues": ["Failed to parse model output"],
                "recommendations": ["Try rephrasing the clause or try again later"],
                "suggestion": None
            }

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMInvocationError, ResponseParseError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def analyze_clause(self, clause: str) -> Dict[str, Any]:
        """
        Analyze an NDA clause with robust error handling
        
        Args:
            clause: The NDA clause to analyze
            
        Returns:
            Dictionary containing risk assessment results
            
        Raises:
            EmptyClauseError: If input is empty or invalid
            LLMInvocationError: If LLM service fails
            ResponseParseError: If response parsing fails
            RiskAnalysisError: For unexpected errors
        """
        try:
            if not clause or not isinstance(clause, str):
                raise EmptyClauseError("Clause must be a non-empty string")

            logger.info("Analyzing clause: %s", clause[:100] + ("..." if len(clause) > 100 else ""))
            return self.chain.invoke({"clause": clause})

        except RetryError as re:
            logger.critical("All retry attempts failed: %s", re)
            raise LLMInvocationError("All retry attempts failed") from re

        except (MissingAPIKeyError, LLMInvocationError, ResponseParseError, EmptyClauseError):
            raise

        except Exception as e:
            logger.error("Unexpected error during analysis: %s", e, exc_info=True)
            raise RiskAnalysisError("Unexpected failure in risk analysis") from e

# --- Usage Example ---
if __name__ == "__main__":
    try:
        agent = RiskAnalystAgent()
        clause = "Confidentiality lasts indefinitely for all information disclosed."
        result = agent.analyze_clause(clause)
        print("\n Risk Assessment:")
        for k, v in result.items():
            print(f"{k.title()}: {v}")

    except Exception as e:
        print(" Error:", e)