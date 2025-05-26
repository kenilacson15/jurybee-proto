import os
import json
import time
import logging
from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from core.schemas import ComplianceResult  # Instead of from tools
# OCR and PDF/image processing imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [AGENT] %(message)s',
    handlers=[logging.FileHandler("compliance_agent.log"), logging.StreamHandler()]
)

load_dotenv()



class ComplianceError(Exception):
    pass


class ComplianceResult(BaseModel):
    compliance_status: str
    issues: list


class ComplianceCheckerAgent:
    def __init__(self):
        """LangChain-powered agent with advanced capabilities"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Parser + model
        self.parser = PydanticOutputParser(pydantic_object=ComplianceResult)
        self.model = self._initialize_model()

        # —— Initialize state before building the chain ——
        self.parse_success_rate = 1.0
        self.consecutive_errors = 0
        self.last_updated = time.time()

        # Now build the agentic chain (which uses parse_success_rate & last_updated)
        self.chain = self._build_agentic_chain()

        self.ocr_lang = 'eng'  # Default OCR language


    def _initialize_model(self):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.2,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )

    def _adaptive_prompt(self) -> ChatPromptTemplate:
        """Self-modifying prompt using LangChain templates"""
        system_template = """
        You are a Constitutional Legal Agent specializing in global NDA compliance.
        Current Date: {date}
        Parse Success Rate: {success_rate:.2f}
        Jurisdictions: Global coverage with local legal nuance
        
        {dynamic_instructions}
        
        {format_instructions}
        """
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                system_template,
                partial_variables={
                    "date": time.strftime("%Y-%m-%d"),
                    "success_rate": self.parse_success_rate,
                    "dynamic_instructions": self._get_dynamic_instructions(),
                    "format_instructions": self.parser.get_format_instructions()
                }
            ),
            ("human", "Analyze clause: {clause}")
        ])

    def _get_dynamic_instructions(self):
        instructions = []
        if self.parse_success_rate < 0.9:
            instructions.append("CRITICAL: Enforce JSON syntax validation before responding")
        if time.time() - self.last_updated > 86400:  # 24 hours
            instructions.append("SYSTEM UPDATE: Check for recent legal changes past 24 hours")
        return "\n".join(instructions) or "No dynamic instructions."

    def _build_agentic_chain(self):
        return (
            RunnablePassthrough.assign(clause=lambda args: args["clause"])  # Explicitly pass clause
            | RunnableLambda(lambda inputs: 
                self._adaptive_prompt().format_prompt(clause=inputs["clause"]).to_string()
            )
            | self.model
            | RunnableLambda(lambda resp: resp.content)
            | self.parser
            | RunnableLambda(self._update_agent_state)
        ) 


    def _handle_response(self, response):
        """Process response with error handling"""
        try:
            result = self.parser.invoke(response)
            return result
        except ValidationError as e:
            logging.warning(f"Validation error: {e}")
            return self._repair_output(response.content, e)

    def _repair_output(self, text, error):
        """LangChain-powered self-repair"""
        from langchain.output_parsers import OutputFixingParser
        
        fix_parser = OutputFixingParser.from_llm(
            parser=self.parser,
            llm=self.model
        )
        return fix_parser.parse(text)

    def _update_agent_state(self, result: ComplianceResult):
        """Update agent's performance metrics"""
        if result.compliance_status == "Error":
            self.consecutive_errors += 1
            self.parse_success_rate *= 0.95
        else:
            self.consecutive_errors = 0
            self.parse_success_rate = min(1.0, self.parse_success_rate * 1.05)
            
        self.last_updated = time.time()
        return result

    def check_compliance(self, clause: str):
        """Enhanced LangChain-powered entry point"""
        return self.chain.invoke({"clause": clause})

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF or image file using OCR."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.pdf']:
            return self._extract_text_from_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self._extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Convert PDF pages to images and extract text via OCR."""
        text = []
        with tempfile.TemporaryDirectory() as path:
            images = convert_from_path(pdf_path, output_folder=path)
            for img in images:
                text.append(pytesseract.image_to_string(img, lang=self.ocr_lang))
        return '\n'.join(text)

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image file using OCR."""
        img = Image.open(image_path)
        return pytesseract.image_to_string(img, lang=self.ocr_lang)

    def check_compliance_from_file(self, file_path: str):
        """Extracts text from a file (PDF/image), then checks compliance."""
        try:
            clause = self.extract_text_from_file(file_path)
            if not clause.strip():
                raise ComplianceError("No text extracted from file.")
            return self.check_compliance(clause)
        except Exception as e:
            logging.error(f"OCR/File processing error: {e}")
            raise ComplianceError(f"Failed to process file: {e}")

# Usage example
if __name__ == "__main__":
    agent = ComplianceCheckerAgent()
    test_clause = "Perpetual confidentiality with unlimited liability"
    result = agent.check_compliance(test_clause)
    print("⚖️ 2025 Agentic Compliance Report:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    # Example for file upload (uncomment to use):
    # file_result = agent.check_compliance_from_file('path_to_pdf_or_image.pdf')
    # print(json.dumps(file_result.model_dump(), indent=2, ensure_ascii=False))