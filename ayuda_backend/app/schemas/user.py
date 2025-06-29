from pydantic import BaseModel, EmailStr
from datetime import date

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
    first_name: str
    last_name: str
    university: str
    email: EmailStr
    dob: date
    major: str
    role: str

    class Config:
        from_attributes = True
