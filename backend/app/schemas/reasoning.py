from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class CourseReasoningRequest(BaseModel):
    """Request model for course reasoning."""
    course_id: str

class CourseReasoningResponse(BaseModel):
    """Response model for course reasoning."""
    success: bool
    reasoning: str
    course_name: str
    model_used: str
    prompt_length: int
    response_length: int
    error: Optional[str] = None

class LLMHealthResponse(BaseModel):
    """Response model for LLM health check."""
    connected: bool
    model: str
    url: str
    error: Optional[str] = None 