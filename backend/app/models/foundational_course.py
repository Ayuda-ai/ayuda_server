from sqlalchemy import Column, String, TIMESTAMP, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.db.session import Base

class FoundationalCourse(Base):
    __tablename__ = "foundational_courses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    major = Column(String(100), nullable=False, unique=True)
    foundational_course_ids = Column(ARRAY(String), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())