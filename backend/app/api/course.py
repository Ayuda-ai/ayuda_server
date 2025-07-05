from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
import logging
from app.db.session import get_db
from app.services.course_search_service import CourseSearchService
from app.core.security import oauth2_scheme, get_current_user
from app.models.user import User

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/search")
def search_courses(
    q: str = Query(..., description="Search query string", min_length=1, max_length=200),
    limit: int = Query(10, description="Maximum number of results to return", ge=1, le=100),
    offset: int = Query(0, description="Number of results to skip for pagination", ge=0),
    major: Optional[str] = Query(None, description="Filter results by major (CSYE, INFO, DAMG)"),
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search courses using fuzzy matching across course_id, course_name, and course_description.
    
    This endpoint provides comprehensive course search functionality that allows users to:
    - Search by course ID (e.g., "CSYE6200")
    - Search by course name (e.g., "Software Engineering")
    - Search by course description keywords (e.g., "database", "machine learning")
    - Filter results by major
    - Get paginated results with relevance scoring
    
    The search uses multiple strategies:
    1. Exact and partial matching in course_id (highest priority)
    2. Partial matching in course_name (medium priority)
    3. Partial matching in course_description (lower priority)
    4. Word boundary matching for better precision
    
    Results are ranked by relevance score and include detailed course information.
    
    Args:
        q (str): Search query string (required, 1-200 characters)
        limit (int): Maximum number of results (default: 10, max: 100)
        offset (int): Number of results to skip for pagination (default: 0)
        major (str, optional): Filter results by major (CSYE, INFO, DAMG)
        token (str): JWT access token for authentication
        current_user (User): Authenticated user object
        db (Session): Database session dependency
        
    Returns:
        dict: Search results containing:
            - courses: List of matching courses with detailed information
            - total_count: Total number of matching courses
            - has_more: Boolean indicating if more results exist
            - search_metadata: Information about the search parameters
            
    Raises:
        HTTPException: If search query is invalid (status_code=400) or search fails (status_code=500)
    """
    try:
        logger.info(f"Course search request from user {current_user.id} with query: '{q}'")
        
        # Initialize search service
        search_service = CourseSearchService(db)
        
        # Perform search
        search_results = search_service.search_courses(
            query=q,
            limit=limit,
            offset=offset,
            major_filter=major
        )
        
        logger.info(f"Course search completed successfully. Found {search_results['total_count']} matches")
        
        return search_results
        
    except ValueError as ve:
        logger.warning(f"Invalid search parameters: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid search parameters: {str(ve)}")
        
    except Exception as e:
        logger.error(f"Error in course search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search/{course_id}")
def get_course_by_id(
    course_id: str,
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific course by its course_id.
    
    This endpoint retrieves complete course information including:
    - Basic course details (ID, name, description)
    - Prerequisites and requirements
    - Major and domain information
    - Associated skills and competencies
    
    Args:
        course_id (str): The course ID to retrieve (e.g., "CSYE6200")
        token (str): JWT access token for authentication
        current_user (User): Authenticated user object
        db (Session): Database session dependency
        
    Returns:
        dict: Complete course information
        
    Raises:
        HTTPException: If course not found (status_code=404) or retrieval fails (status_code=500)
    """
    try:
        logger.info(f"Course detail request from user {current_user.id} for course: {course_id}")
        
        # Initialize search service
        search_service = CourseSearchService(db)
        
        # Get course by ID
        course = search_service.get_course_by_id(course_id)
        
        if not course:
            logger.warning(f"Course not found: {course_id}")
            raise HTTPException(status_code=404, detail=f"Course with ID '{course_id}' not found")
        
        # Format course response
        course_data = {
            "id": str(course.id),
            "course_id": course.course_id,
            "course_name": course.course_name,
            "course_description": course.course_description,
            "major": course.major,
            "domains": course.domains or [],
            "skills_associated": course.skills_associated or [],
            "prerequisites": course.prerequisites or [],
            "created_at": course.created_at.isoformat() if course.created_at else None
        }
        
        logger.info(f"Course detail retrieved successfully for course: {course_id}")
        return course_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving course {course_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve course: {str(e)}")

@router.get("/major/{major}")
def get_courses_by_major(
    major: str,
    limit: int = Query(50, description="Maximum number of courses to return", ge=1, le=200),
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all courses for a specific major.
    
    This endpoint retrieves all courses belonging to a particular major,
    useful for browsing courses by academic program.
    
    Args:
        major (str): The major to filter by (CSYE, INFO, DAMG)
        limit (int): Maximum number of courses to return (default: 50, max: 200)
        token (str): JWT access token for authentication
        current_user (User): Authenticated user object
        db (Session): Database session dependency
        
    Returns:
        dict: List of courses for the specified major
        
    Raises:
        HTTPException: If major is invalid (status_code=400) or retrieval fails (status_code=500)
    """
    try:
        # Validate major
        valid_majors = ["CSYE", "INFO", "DAMG"]
        if major.upper() not in valid_majors:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid major. Must be one of: {', '.join(valid_majors)}"
            )
        
        logger.info(f"Major courses request from user {current_user.id} for major: {major}")
        
        # Initialize search service
        search_service = CourseSearchService(db)
        
        # Get courses by major
        courses = search_service.get_courses_by_major(major.upper(), limit)
        
        # Format courses
        formatted_courses = []
        for course in courses:
            formatted_course = {
                "id": str(course.id),
                "course_id": course.course_id,
                "course_name": course.course_name,
                "course_description": course.course_description,
                "major": course.major,
                "domains": course.domains or [],
                "skills_associated": course.skills_associated or [],
                "prerequisites": course.prerequisites or [],
                "created_at": course.created_at.isoformat() if course.created_at else None
            }
            formatted_courses.append(formatted_course)
        
        response = {
            "major": major.upper(),
            "courses": formatted_courses,
            "total_count": len(formatted_courses),
            "limit": limit
        }
        
        logger.info(f"Retrieved {len(formatted_courses)} courses for major: {major}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving courses for major {major}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve courses: {str(e)}") 