from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from uuid import UUID
from datetime import datetime

class RecommendationBase(BaseModel):
    recommendation_type: str
    top_k_requested: int
    eligible_matches: Optional[List[Any]] = None
    ineligible_matches: Optional[List[Any]] = None
    total_matches: Optional[int] = None
    processing_metrics: Optional[Dict[str, Any]] = None
    system_health: Optional[Dict[str, Any]] = None
    prerequisite_analysis: Optional[Dict[str, Any]] = None
    user_completed_courses: Optional[List[str]] = None
    resume_info: Optional[Dict[str, Any]] = None
    keyword_analysis: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None

class RecommendationCreate(RecommendationBase):
    pass

class RecommendationOut(RecommendationBase):
    id: UUID
    user_id: UUID
    created_at: datetime

    class Config:
        orm_mode = True 