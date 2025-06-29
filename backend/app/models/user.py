from sqlalchemy import Column, String, Date, Text, TIMESTAMP, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.db.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    university = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    dob = Column(Date, nullable=False)
    major = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default="USER")
    resume_embedding = Column(JSON, nullable=True)  # Store embedding as JSON array
    resume_text = Column(Text, nullable=True)  # Store extracted resume text for keyword matching
    created_at = Column(TIMESTAMP, server_default=func.now())
