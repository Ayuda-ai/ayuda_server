from sqlalchemy import Column, String, Text, TIMESTAMP, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import uuid
from app.db.session import Base

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    recommendation_type = Column(String(50), nullable=False)  # "hybrid", "semantic", "keyword"
    top_k_requested = Column(Integer, nullable=False)
    
    # Store the complete recommendation response
    eligible_matches = Column(JSONB, nullable=True)  # List of eligible course matches
    ineligible_matches = Column(JSONB, nullable=True)  # List of ineligible course matches
    total_matches = Column(Integer, nullable=True)
    
    # Store processing metrics and system health
    processing_metrics = Column(JSONB, nullable=True)  # Timing and performance data
    system_health = Column(JSONB, nullable=True)  # System status information
    prerequisite_analysis = Column(JSONB, nullable=True)  # Prerequisite checking results
    
    # Store user context at time of recommendation
    user_completed_courses = Column(JSONB, nullable=True)  # User's completed courses
    resume_info = Column(JSONB, nullable=True)  # Resume information
    keyword_analysis = Column(JSONB, nullable=True)  # Keyword matching analysis
    
    # Store error information if recommendation failed
    error_info = Column(JSONB, nullable=True)  # Error details if any
    
    # Metadata
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationship to User model
    user = relationship("User", back_populates="recommendations")
    
    def __repr__(self):
        return f"<Recommendation(id={self.id}, user_id={self.user_id}, type={self.recommendation_type})>" 