"""
Course Schemas

This module contains Pydantic schemas for course-related operations
including creation, updates, and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class CourseCreate(BaseModel):
    """
    Schema for creating a new course.
    
    This schema defines the required and optional fields for creating
    a new course in the system.
    """
    course_id: str = Field(..., description="Unique course identifier")
    course_name: str = Field(..., description="Name of the course")
    domains: Optional[List[str]] = Field(None, description="List of domains the course belongs to")
    major: str = Field(..., description="Major code the course belongs to")
    skills_associated: Optional[List[str]] = Field(None, description="List of skills associated with the course")
    prerequisites: Optional[str] = Field(None, description="Prerequisites for the course")
    description: Optional[str] = Field(None, description="Course description")


class CourseUpdate(BaseModel):
    """
    Schema for updating an existing course.
    
    This schema defines the optional fields that can be updated
    for an existing course. All fields are optional to allow
    partial updates.
    """
    course_name: Optional[str] = Field(None, description="Name of the course")
    domains: Optional[List[str]] = Field(None, description="List of domains the course belongs to")
    major: Optional[str] = Field(None, description="Major code the course belongs to")
    skills_associated: Optional[List[str]] = Field(None, description="List of skills associated with the course")
    prerequisites: Optional[str] = Field(None, description="Prerequisites for the course")
    description: Optional[str] = Field(None, description="Course description")


class CourseOut(BaseModel):
    """
    Schema for course response data.
    
    This schema defines the structure of course data returned
    by the API, excluding sensitive or internal fields.
    """
    id: str = Field(..., description="Database ID of the course")
    course_id: str = Field(..., description="Unique course identifier")
    course_name: str = Field(..., description="Name of the course")
    domains: List[str] = Field(default_factory=list, description="List of domains the course belongs to")
    major: str = Field(..., description="Major code the course belongs to")
    skills_associated: List[str] = Field(default_factory=list, description="List of skills associated with the course")
    prerequisites: Optional[str] = Field(None, description="Prerequisites for the course")
    description: Optional[str] = Field(None, description="Course description")
    created_at: Optional[datetime] = Field(None, description="When the course was created")
    updated_at: Optional[datetime] = Field(None, description="When the course was last updated")

    class Config:
        from_attributes = True 