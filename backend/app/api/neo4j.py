from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.neo4j_service import Neo4jService
from app.services.course_service import CourseService
from pydantic import BaseModel
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize services
neo4j_service = Neo4jService()
course_service = CourseService()

# Pydantic models for request/response
class PrerequisiteRequest(BaseModel):
    course_id: str
    prerequisite_id: str
    relationship_type: str = "OR"  # "OR" or "AND"

class PrerequisiteResponse(BaseModel):
    course_id: str
    prerequisite_id: str
    relationship_type: str
    success: bool
    message: str

class PrerequisitesCheckRequest(BaseModel):
    course_id: str
    completed_courses: List[str]

class CoursePrerequisitesResponse(BaseModel):
    course_id: str
    has_prerequisites: bool
    prerequisites_met: bool
    total_prerequisites: int
    completed_count: int
    missing_count: int
    missing_prerequisites: List[Dict[str, Any]]
    completed_prerequisites: List[Dict[str, Any]]
    prerequisite_groups: List[Dict[str, Any]]

class AvailableCoursesRequest(BaseModel):
    completed_courses: List[str]

class CourseAnalyticsResponse(BaseModel):
    total_courses: int
    courses_with_prerequisites: int
    courses_without_prerequisites: int
    total_prerequisite_relationships: int
    average_prerequisites_per_course: float
    courses_by_major: List[Dict[str, Any]]

@router.get("/health")
async def neo4j_health_check():
    """Check Neo4j connection health."""
    try:
        is_healthy = neo4j_service.test_connection()
        if is_healthy:
            return {"status": "healthy", "message": "Neo4j connection is working"}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j connection failed"
            )
    except Exception as e:
        logger.error(f"Neo4j health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Neo4j service unavailable: {str(e)}"
        )

@router.post("/sync-courses", response_model=Dict[str, Any])
async def sync_courses_to_neo4j(
    clear_first: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Sync all courses from PostgreSQL to Neo4j.
    This endpoint requires authentication and admin privileges.
    
    Args:
        clear_first: Whether to clear existing Neo4j data before syncing
    """
    try:
        # Get all courses from PostgreSQL
        courses = course_service.get_all_courses(db)
        
        if not courses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No courses found in PostgreSQL"
            )
        
        # Convert to list of dictionaries
        courses_data = []
        for course in courses:
            courses_data.append({
                "id": course.id,
                "course_id": course.course_id,  # The actual course identifier
                "course_name": course.course_name,
                "course_description": course.course_description,
                "major": course.major,
                "domains": course.domains,
                "skills_associated": course.skills_associated,
                "prerequisites": course.prerequisites  # Include prerequisites field
            })
        
        # Sync to Neo4j
        success = neo4j_service.sync_courses_from_postgresql(courses_data, clear_first=clear_first)
        
        if success:
            return {
                "message": "Courses synced successfully",
                "total_courses": len(courses_data),
                "clear_first": clear_first,
                "status": "success"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to sync courses to Neo4j"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing courses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/prerequisites", response_model=PrerequisiteResponse)
async def add_prerequisite(
    request: PrerequisiteRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Add a prerequisite relationship between two courses.
    """
    try:
        success = neo4j_service.add_prerequisite_relationship(
            request.course_id,
            request.prerequisite_id,
            request.relationship_type
        )
        
        if success:
            return PrerequisiteResponse(
                course_id=request.course_id,
                prerequisite_id=request.prerequisite_id,
                relationship_type=request.relationship_type,
                success=True,
                message="Prerequisite relationship added successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to add prerequisite relationship"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding prerequisite: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.delete("/prerequisites/{course_id}/{prerequisite_id}")
async def remove_prerequisite(
    course_id: str,
    prerequisite_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Remove a prerequisite relationship between two courses.
    """
    try:
        success = neo4j_service.remove_prerequisite_relationship(course_id, prerequisite_id)
        
        if success:
            return {
                "message": "Prerequisite relationship removed successfully",
                "course_id": course_id,
                "prerequisite_id": prerequisite_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prerequisite relationship not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing prerequisite: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/prerequisites/{course_id}")
async def get_course_prerequisites(course_id: str):
    """
    Get all prerequisites for a specific course.
    """
    try:
        prerequisites = neo4j_service.get_course_prerequisites(course_id)
        
        return {
            "course_id": course_id,
            "prerequisites": prerequisites,
            "total_prerequisites": len(prerequisites)
        }
        
    except Exception as e:
        logger.error(f"Error getting course prerequisites: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/prerequisites/{course_id}/path")
async def get_prerequisite_path(
    course_id: str,
    max_depth: int = 3
):
    """
    Get the complete prerequisite path for a course.
    """
    try:
        path = neo4j_service.get_prerequisite_path(course_id, max_depth)
        
        return {
            "course_id": course_id,
            "max_depth": max_depth,
            "prerequisite_path": path,
            "total_courses_in_path": len(path)
        }
        
    except Exception as e:
        logger.error(f"Error getting prerequisite path: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/prerequisites/check", response_model=CoursePrerequisitesResponse)
async def check_prerequisites_completion(
    request: PrerequisitesCheckRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Check if a user has completed the prerequisites for a course.
    """
    try:
        completion_status = neo4j_service.check_prerequisites_completion(
            request.course_id,
            request.completed_courses
        )
        
        return CoursePrerequisitesResponse(**completion_status)
        
    except Exception as e:
        logger.error(f"Error checking prerequisites completion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/courses/available")
async def get_available_courses(
    request: AvailableCoursesRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get courses that the user can take based on their completed prerequisites.
    """
    try:
        available_courses = neo4j_service.get_courses_by_prerequisites(request.completed_courses)
        
        # Separate courses by availability
        courses_available = [c for c in available_courses if c["prerequisites_met"]]
        courses_not_available = [c for c in available_courses if not c["prerequisites_met"]]
        
        return {
            "total_courses": len(available_courses),
            "courses_available": len(courses_available),
            "courses_not_available": len(courses_not_available),
            "available_courses": courses_available,
            "unavailable_courses": courses_not_available
        }
        
    except Exception as e:
        logger.error(f"Error getting available courses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/analytics", response_model=CourseAnalyticsResponse)
async def get_course_analytics(
    current_user: User = Depends(get_current_user)
):
    """
    Get analytics about courses and prerequisites.
    """
    try:
        analytics = neo4j_service.get_course_analytics()
        
        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No analytics data available"
            )
        
        return CourseAnalyticsResponse(**analytics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting course analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/courses/{course_id}/recommendations")
async def get_course_recommendations(
    course_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get course recommendations based on prerequisites and user profile.
    This endpoint combines Neo4j graph analysis with user profile data.
    """
    try:
        # Get user's completed courses from profile
        completed_courses = current_user.completed_courses or []
        
        # Check prerequisites for the specific course
        completion_status = neo4j_service.check_prerequisites_completion(
            course_id,
            completed_courses
        )
        
        # Get prerequisite path
        prerequisite_path = neo4j_service.get_prerequisite_path(course_id)
        
        # Get all available courses for the user
        all_available_courses = neo4j_service.get_courses_by_prerequisites(completed_courses)
        
        # Filter courses by user's major if available
        user_major = current_user.major
        if user_major:
            major_courses = [c for c in all_available_courses if c["major"] == user_major]
        else:
            major_courses = all_available_courses
        
        return {
            "course_id": course_id,
            "prerequisites_status": completion_status,
            "prerequisite_path": prerequisite_path,
            "user_completed_courses": completed_courses,
            "user_major": user_major,
            "total_available_courses": len(all_available_courses),
            "major_specific_courses": len(major_courses),
            "recommended_courses": major_courses[:10]  # Top 10 recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting course recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        ) 