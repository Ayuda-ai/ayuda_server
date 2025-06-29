from sqlalchemy import Column, String, Text, TIMESTAMP, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.db.session import Base

class Course(Base):
    __tablename__ = "courses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(String(100), nullable=False, unique=True)
    course_name = Column(Text, nullable=False)
    course_description = Column(Text, nullable=False)
    prerequisites = Column(ARRAY(Text), nullable=True)
    major = Column(String(100), nullable=False)
    domains = Column(ARRAY(Text), nullable=True)
    skills_associated = Column(ARRAY(Text), nullable=True)
    combined_text = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
