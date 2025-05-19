import os
import sys
import logging
from typing import Any, Dict, List
from pathlib import Path
from logging.handlers import RotatingFileHandler
import httpx
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException, LangChainException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from .env
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')

# ------------------------
# Custom Exceptions
# ------------------------
class ConfigurationError(Exception):
    pass

class LLMOperationError(Exception):
    pass

class ResourceLoadError(Exception):
    pass

# ------------------------
# Logging Setup
# ------------------------
def setup_logging() -> None:
    log_format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    max_bytes = 10 * 1024 * 1024  # 10 MB
    backup_count = 5
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            RotatingFileHandler("negotiation_strategist.log", maxBytes=max_bytes, backupCount=backup_count),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("NegotiationStrategist")
    logger.setLevel(logging.DEBUG)

setup_logging()
logger = logging.getLogger("NegotiationStrategist")

# ------------------------
# Environment Validation
# ------------------------
def validate_environment() -> None:
    required_vars = ['GEMINI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.critical("Missing required environment variables: %s", ', '.join(missing))
        raise ConfigurationError(f"Missing required environment variables: {', '.join(missing)}")

# ------------------------
# Output Schema
# ------------------------
class NegotiationStrategy(BaseModel):
    proposed_clause: str
    rationale: str
    alternatives: List[str]
    compliance_status: str
    risk_impact: str

# ------------------------
# Main Agent Class
# ------------------------
class NegotiationStrategistAgent:
    def __init__(self, llm: Any = None):
        validate_environment()
        self.llm = llm or self._initialize_default_llm()
        self.parser = PydanticOutputParser(pydantic_object=NegotiationStrategy)
        self.prompt = self._load_prompt_template()
        self.chain = self._setup_chain()
        self._validate_components()

    def _validate_components(self) -> None:
        if not all([self.llm, self.prompt, self.parser]):
            raise ConfigurationError("Failed to initialize required components")

    def _initialize_default_llm(self) -> ChatGoogleGenerativeAI:
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.4,
                max_output_tokens=1024,
                timeout=30,  # ← Changed from request_timeout to timeout
                api_key=api_key
            )
        except ValidationError as ve:
            logger.critical("Invalid LLM configuration: %s", ve)
            raise ConfigurationError("Invalid LLM parameters") from ve
        except Exception as e:
            logger.critical("LLM initialization failed: %s", e)
            raise LLMOperationError("LLM setup failed") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException, OSError))
    )
    def _load_prompt_template(self) -> ChatPromptTemplate:
        prompt_path = Path(__file__).parents[1] / "prompts/system_prompts/negotiation_strategist.txt"
        try:
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
            system_prompt = prompt_path.read_text(encoding='utf-8')
            if not system_prompt.strip():
                raise ValueError("Empty prompt template file")
            return ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Clause: {clause}\n\nRisk Assessment:\n{risk_assessment}")
            ])
        except (FileNotFoundError, IOError) as fe:
            logger.warning("Prompt file error: %s. Using default template.", fe)
            return self._create_fallback_prompt()
        except Exception as e:
            logger.error("Unexpected prompt loading error: %s", e)
            raise ResourceLoadError("Prompt template loading failed") from e

    def _create_fallback_prompt(self) -> ChatPromptTemplate:
        logger.info("Using fallback prompt template")
        return ChatPromptTemplate.from_messages([
            ("system", self._default_prompt_template()),
            ("human", "Clause: {clause}\n\nRisk Assessment:\n{risk_assessment}")
        ])

    def _default_prompt_template(self):
        return (
            "You are a Legal Negotiation Strategist specializing in NDAs. "
            "Respond with a JSON object using the following schema:\n"
            "{schema}\n"
            "Example:\n"
            "{example}\n"
            "Clause: {clause}\n"
            "Risk Assessment:\n{risk_assessment}"
        ).format(
            schema=NegotiationStrategy.model_json_schema(),
            example=NegotiationStrategy(
                proposed_clause="Recipient agrees to maintain the confidentiality...",
                rationale="Clarifies proprietary definition and remedies",
                alternatives=["Add geographic scope", "Include breach penalties"],
                compliance_status="Compliant",
                risk_impact="Medium"
            ).model_dump_json()
        )

    def _setup_chain(self) -> RunnableSequence:
        try:
            chain = self.prompt | self.llm | self.parser
            if not isinstance(chain, RunnableSequence):
                raise TypeError("Invalid chain composition")
            return chain
        except LangChainException as lce:
            logger.error("Chain configuration error: %s", lce)
            raise ConfigurationError("Invalid processing chain") from lce

    def _validate_inputs(self, clause: str, risk_assessment: Dict[str, Any]) -> None:
        if not clause.strip():
            raise ValueError("Empty clause input")
        if not isinstance(risk_assessment, dict):
            raise TypeError("risk_assessment must be a dictionary")
        required_risk_keys = {'risk_level', 'confidence'}
        missing = required_risk_keys - set(risk_assessment.keys())
        if missing:
            raise ValueError(f"Missing required risk assessment keys: {missing}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException))
    )
    def run(self, clause: str, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_inputs(clause, risk_assessment)
        risk_str = "\n".join([f"{k}: {v}" for k, v in risk_assessment.items()])
        try:
            result = self.chain.invoke({...})
            return result.model_dump()
        except OutputParserException as pe:
            logger.warning("First attempt failed. Retrying with strict format instructions.")
        # Modify prompt to enforce strict JSON
            self.prompt = self._create_strict_prompt()
            result = self.chain.invoke({...})  # Retry
            return result.model_dump()
            
            return result.model_dump()
        except (OutputParserException, ValidationError) as pe:
            logger.error("Parsing/validation failed: %s", pe, exc_info=True)
            return self._handle_parsing_failure(pe)
        except httpx.HTTPStatusError as he:
            logger.error("HTTP error %d: %s", he.response.status_code, he)
            return self._handle_http_error(he)
        except LangChainException as lce:
            logger.error("LangChain operation failed: %s", lce)
            return self._handle_parsing_failure(lce)
        except Exception as e:
            logger.critical("Unexpected error: %s", e, exc_info=True)
            raise LLMOperationError("Critical system failure") from e

    def _handle_parsing_failure(self, error: Exception, raw_output: str = "") -> Dict[str, Any]:
        logger.warning("Raw LLM output: %s", raw_output)
        return {
            "proposed_clause": "Manual review required",
            "rationale": f"Error: {error}. Raw output: {raw_output[:200]}...",
            "alternatives": ["Consult legal counsel", "Use default template"],
            "compliance_status": "Unknown",
            "risk_impact": "Unknown"
        }

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> Dict[str, Any]:
        if error.response.status_code in [429, 503]:
            return {
                "proposed_clause": "System overloaded - retry later",
                "rationale": "Service temporarily unavailable",
                "alternatives": [],
                "compliance_status": "Unknown",
                "risk_impact": "Unknown"
            }
        return self._handle_parsing_failure(error)

# ------------------------
# Main Entry Point
# ------------------------
if __name__ == "__main__":
    logger.info("Starting Negotiation Strategist Agent example")
    try:
        agent = NegotiationStrategistAgent()
        example_clause = "The party agrees to non-disclosure of proprietary information."
        example_risk_assessment = {
            "risk_level": "high",
            "confidence": 0.85,
            "details": "Potential for significant competitive harm"
        }
        result = agent.run(example_clause, example_risk_assessment)
        logger.info("Result: %s", result)
        print("Negotiation Result:", result)
    except Exception as e:
        logger.critical("Application failed: %s", e)
        print(f"Error: {str(e)}")