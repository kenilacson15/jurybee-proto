# core/schemas.py

from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel, Field, validator, model_validator

# --- Custom Exceptions for Schema Errors ---
class SchemaValidationError(ValueError):
    """Raised when a schema validation fails."""
    pass

class ComplianceResult(Enum):
    PASS = "pass"
    FAIL = "fail"

class AgentResponse(BaseModel):
    """
    Standardized response from any agent:
      – content: the raw payload (must be a non-empty dict)
      – compliance_status: PASS / FAIL
      – confidence: float in [0.0, 1.0]
    """

    content: Dict[str, Any] = Field(
        ...,
        description="The JSON-serializable payload returned by the agent."
    )
    compliance_status: ComplianceResult = Field(
        ...,
        description="Whether the response passes compliance checks."
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Model confidence score, between 0 and 1."
    )

    class Config:
        extra = "forbid"            # no unexpected fields
        validate_assignment = True  # re-validate if you ever set attributes

    @validator("content", pre=True, always=True)
    def content_must_be_dict_and_not_empty(cls, v):
        if not isinstance(v, dict):
            raise SchemaValidationError("`content` must be a dict.")
        if not v:
            raise SchemaValidationError("`content` must not be empty.")
        return v

    @validator("compliance_status", pre=True, always=True)
    def valid_compliance_status(cls, v):
        try:
            return ComplianceResult(v)
        except ValueError:
            allowed = [e.value for e in ComplianceResult]
            raise SchemaValidationError(
                f"`compliance_status` must be one of {allowed}, got {v!r}"
            )

    @model_validator(mode="after")
    def check_confidence_matches_content(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-field check: if content itself
        has a 'confidence' field, they should match.
        """
        content = values.get("content")
        confidence = values.get("confidence")
        if isinstance(content, dict) and "confidence" in content:
            inner_conf = content["confidence"]
            if abs(inner_conf - confidence) > 1e-6:
                raise SchemaValidationError(
                    f"Top‐level `confidence` ({confidence}) "
                    f"does not match content['confidence'] ({inner_conf})."
                )
        return values
