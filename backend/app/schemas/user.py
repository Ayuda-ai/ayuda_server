from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import Optional, List
from uuid import UUID

class UserCreate(BaseModel):
    first_name: str
    last_name: str
    university: str
    email: EmailStr
    password: str
    dob: date
    major: str
    access_code: str

class UserOut(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    university: str
    email: EmailStr
    dob: date
    major: str
    role: str
    profile_enhanced: Optional[bool] = None
    completed_courses: Optional[List[str]] = None
    additional_skills: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
