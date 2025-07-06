"""
Admin API Endpoints

This module contains all admin-only endpoints for database operations,
system management, and administrative functions. All endpoints require
admin role authentication.

Endpoints:
- GET /admin/users - Get all users (admin only)
- GET /admin/users/{user_id} - Get specific user details (admin only)
- PUT /admin/users/{user_id} - Update user information (admin only)
- DELETE /admin/users/{user_id} - Delete user (admin only)
- GET /admin/courses - Get all courses (admin only)
- POST /admin/courses - Create new course (admin only)
- PUT /admin/courses/{course_id} - Update course (admin only)
- DELETE /admin/courses/{course_id} - Delete course (admin only)
- GET /admin/system/health - Get system health status (admin only)
- GET /admin/system/stats - Get system statistics (admin only)
- POST /admin/system/sync/neo4j - Sync data to Neo4j (admin only)
- POST /admin/system/sync/pinecone - Sync data to Pinecone (admin only)
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from typing import List, Optional
import logging
import pandas as pd
import uuid

from app.db.session import get_db
from app.services.user_service import UserService
from app.services.admin_service import AdminService
from app.services.neo4j_service import Neo4jService
from app.models.course import Course
from app.schemas.user import UserOut
from app.schemas.course import CourseCreate, CourseUpdate, CourseOut
from app.core.security import oauth2_scheme
from app.core.dependencies import admin_required
from app.services.course_service import embed_and_index_courses

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/users", response_model=List[UserOut], dependencies=[Depends(admin_required)])
def get_all_users(
    skip: int = 0,
    limit: int = 100,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get all users in the system (admin only).
    
    This endpoint retrieves all users from the database with pagination support.
    Only users with admin role can access this endpoint.
    
    Args:
        skip (int): Number of records to skip for pagination (default: 0)
        limit (int): Maximum number of records to return (default: 100, max: 1000)
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        List[UserOut]: List of user information (password_hash excluded)
        
    Raises:
        HTTPException: If user not found (404), not admin (403), 
                      or retrieval fails (500)
    """
    logger.info(f"Admin request to get all users: skip={skip}, limit={limit}")
    
    try:
        admin_service = AdminService(db)
        users = admin_service.get_all_users(skip=skip, limit=limit)
        
        logger.info(f"Retrieved {len(users)} users for admin")
        return users
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")


@router.get("/users/{user_id}", response_model=UserOut, dependencies=[Depends(admin_required)])
def get_user_by_id(
    user_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get specific user details by ID (admin only).
    
    This endpoint retrieves detailed information about a specific user
    including their profile, resume status, and completed courses.
    Only users with admin role can access this endpoint.
    
    Args:
        user_id (str): UUID of the user to retrieve
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        UserOut: Detailed user information (password_hash excluded)
        
    Raises:
        HTTPException: If user not found (404), not admin (403), 
                      or retrieval fails (500)
    """
    logger.info(f"Admin request to get user: {user_id}")
    
    try:
        admin_service = AdminService(db)
        user = admin_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"Retrieved user {user_id} for admin")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving user: {str(e)}")


@router.put("/users/{user_id}", response_model=UserOut, dependencies=[Depends(admin_required)])
def update_user(
    user_id: str,
    user_update: dict,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Update user information (admin only).
    
    This endpoint allows admins to update user information including
    profile details, role, and other attributes. Only users with admin
    role can access this endpoint.
    
    Args:
        user_id (str): UUID of the user to update
        user_update (dict): User data to update
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        UserOut: Updated user information
        
    Raises:
        HTTPException: If user not found (404), not admin (403), 
                      or update fails (500)
    """
    logger.info(f"Admin request to update user: {user_id}")
    
    try:
        admin_service = AdminService(db)
        updated_user = admin_service.update_user(user_id, user_update)
        
        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"Updated user {user_id} for admin")
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")


@router.delete("/users/{user_id}", dependencies=[Depends(admin_required)])
def delete_user(
    user_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Delete a user from the system (admin only).
    
    This endpoint permanently deletes a user and all associated data
    including resume embeddings, course completions, and profile information.
    Only users with admin role can access this endpoint.
    
    Args:
        user_id (str): UUID of the user to delete
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Deletion confirmation message
        
    Raises:
        HTTPException: If user not found (404), not admin (403), 
                      or deletion fails (500)
    """
    logger.info(f"Admin request to delete user: {user_id}")
    
    try:
        admin_service = AdminService(db)
        success = admin_service.delete_user(user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"Deleted user {user_id} for admin")
        return {"message": f"User {user_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")


# ============================================================================
# COURSE MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/courses", response_model=List[CourseOut], dependencies=[Depends(admin_required)])
def get_all_courses(
    skip: int = 0,
    limit: int = 100,
    major: Optional[str] = None,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get all courses in the system (admin only).
    
    This endpoint retrieves all courses from the database with optional
    filtering by major and pagination support. Only users with admin
    role can access this endpoint.
    
    Args:
        skip (int): Number of records to skip for pagination (default: 0)
        limit (int): Maximum number of records to return (default: 100, max: 1000)
        major (Optional[str]): Major code to filter courses by (optional)
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        List[CourseOut]: List of course information
        
    Raises:
        HTTPException: If not admin (403) or retrieval fails (500)
    """
    logger.info(f"Admin request to get all courses: skip={skip}, limit={limit}, major={major}")
    
    try:
        admin_service = AdminService(db)
        courses = admin_service.get_all_courses(skip=skip, limit=limit, major=major)
        
        logger.info(f"Retrieved {len(courses)} courses for admin")
        return courses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve courses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving courses: {str(e)}")


@router.get("/courses/{course_id}", response_model=CourseOut, dependencies=[Depends(admin_required)])
def get_course_by_id(
    course_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get specific course details by ID (admin only).
    
    This endpoint retrieves detailed information about a specific course
    including prerequisites, skills, and metadata. Only users with admin
    role can access this endpoint.
    
    Args:
        course_id (str): Course ID to retrieve
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        CourseOut: Detailed course information
        
    Raises:
        HTTPException: If course not found (404), not admin (403), 
                      or retrieval fails (500)
    """
    logger.info(f"Admin request to get course: {course_id}")
    
    try:
        admin_service = AdminService(db)
        course = admin_service.get_course_by_id(course_id)
        
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        logger.info(f"Retrieved course {course_id} for admin")
        return course
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve course {course_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving course: {str(e)}")


@router.post("/courses", response_model=CourseOut, dependencies=[Depends(admin_required)])
def create_course(
    course_data: CourseCreate,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Create a new course (admin only).
    
    This endpoint allows admins to create new courses in the system.
    The course will be automatically indexed in Pinecone and Neo4j
    for recommendation purposes. Only users with admin role can access this endpoint.
    
    Args:
        course_data (CourseCreate): Course creation data
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        CourseOut: Created course information
        
    Raises:
        HTTPException: If course creation fails (400), not admin (403), 
                      or database error (500)
    """
    logger.info(f"Admin request to create course: {course_data.course_id}")
    
    try:
        admin_service = AdminService(db)
        created_course = admin_service.create_course(course_data)
        
        logger.info(f"Created course {course_data.course_id} for admin")
        return created_course
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create course: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating course: {str(e)}")


@router.put("/courses/{course_id}", response_model=CourseOut, dependencies=[Depends(admin_required)])
def update_course(
    course_id: str,
    course_update: CourseUpdate,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Update course information (admin only).
    
    This endpoint allows admins to update course information including
    prerequisites, skills, and metadata. The course will be re-indexed
    in Pinecone and Neo4j after update. Only users with admin role can access this endpoint.
    
    Args:
        course_id (str): Course ID to update
        course_update (CourseUpdate): Course update data
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        CourseOut: Updated course information
        
    Raises:
        HTTPException: If course not found (404), not admin (403), 
                      or update fails (500)
    """
    logger.info(f"Admin request to update course: {course_id}")
    
    try:
        admin_service = AdminService(db)
        updated_course = admin_service.update_course(course_id, course_update)
        
        if not updated_course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        logger.info(f"Updated course {course_id} for admin")
        return updated_course
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update course {course_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating course: {str(e)}")


@router.delete("/courses/{course_id}", dependencies=[Depends(admin_required)])
def delete_course(
    course_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Delete a course from the system (admin only).
    
    This endpoint permanently deletes a course and removes it from
    Pinecone and Neo4j indexes. Only users with admin role can access this endpoint.
    
    Args:
        course_id (str): Course ID to delete
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Deletion confirmation message
        
    Raises:
        HTTPException: If course not found (404), not admin (403), 
                      or deletion fails (500)
    """
    logger.info(f"Admin request to delete course: {course_id}")
    
    try:
        admin_service = AdminService(db)
        success = admin_service.delete_course(course_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Course not found")
        
        logger.info(f"Deleted course {course_id} for admin")
        return {"message": f"Course {course_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete course {course_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting course: {str(e)}")


# ============================================================================
# SYSTEM MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/system/health", dependencies=[Depends(admin_required)])
def get_system_health(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive system health status (admin only).
    
    This endpoint provides detailed health information about all system
    components including database, Redis, Pinecone, and Neo4j connections.
    Only users with admin role can access this endpoint.
    
    Args:
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Comprehensive system health information
        
    Raises:
        HTTPException: If not admin (403) or health check fails (500)
    """
    logger.info("Admin request for system health check")
    
    try:
        admin_service = AdminService(db)
        health_status = admin_service.get_system_health()
        
        logger.info("System health check completed for admin")
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking system health: {str(e)}")


@router.get("/system/stats", dependencies=[Depends(admin_required)])
def get_system_statistics(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get system statistics and metrics (admin only).
    
    This endpoint provides comprehensive statistics about the system
    including user counts, course counts, recommendation metrics,
    and performance statistics. Only users with admin role can access this endpoint.
    
    Args:
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: System statistics and metrics
        
    Raises:
        HTTPException: If not admin (403) or statistics retrieval fails (500)
    """
    logger.info("Admin request for system statistics")
    
    try:
        admin_service = AdminService(db)
        stats = admin_service.get_system_statistics()
        
        logger.info("System statistics retrieved for admin")
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving system statistics: {str(e)}")


@router.post("/system/sync/neo4j", dependencies=[Depends(admin_required)])
def sync_data_to_neo4j(
    clear_first: bool = False,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Sync data to Neo4j for prerequisite checking (admin only).
    
    This endpoint synchronizes course and prerequisite data to Neo4j
    for graph-based prerequisite checking. Only users with admin role can access this endpoint.
    
    Args:
        clear_first (bool): Whether to clear existing data before syncing (default: False)
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Sync operation results
        
    Raises:
        HTTPException: If not admin (403) or sync fails (500)
    """
    logger.info(f"Admin request to sync data to Neo4j: clear_first={clear_first}")
    
    try:
        admin_service = AdminService(db)
        sync_results = admin_service.sync_data_to_neo4j(clear_first=clear_first)
        
        logger.info("Neo4j sync completed for admin")
        return sync_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync data to Neo4j: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error syncing data to Neo4j: {str(e)}")


@router.post("/system/sync/pinecone", dependencies=[Depends(admin_required)])
def sync_data_to_pinecone(
    clear_first: bool = False,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Sync data to Pinecone for semantic matching (admin only).
    
    This endpoint synchronizes course embeddings to Pinecone for
    semantic similarity search. Only users with admin role can access this endpoint.
    
    Args:
        clear_first (bool): Whether to clear existing data before syncing (default: False)
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Sync operation results
        
    Raises:
        HTTPException: If not admin (403) or sync fails (500)
    """
    logger.info(f"Admin request to sync data to Pinecone: clear_first={clear_first}")
    
    try:
        admin_service = AdminService(db)
        sync_results = admin_service.sync_data_to_pinecone(clear_first=clear_first)
        
        logger.info("Pinecone sync completed for admin")
        return sync_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync data to Pinecone: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error syncing data to Pinecone: {str(e)}")


@router.post("/system/backup", dependencies=[Depends(admin_required)])
def create_system_backup(
    backup_name: Optional[str] = None,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Create a system backup (admin only).
    
    This endpoint creates a comprehensive backup of the system including
    database, user data, and configuration. Only users with admin role can access this endpoint.
    
    Args:
        backup_name (Optional[str]): Custom name for the backup (optional)
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Backup operation results
        
    Raises:
        HTTPException: If not admin (403) or backup fails (500)
    """
    logger.info(f"Admin request to create system backup: {backup_name}")
    
    try:
        admin_service = AdminService(db)
        backup_results = admin_service.create_system_backup(backup_name=backup_name)
        
        logger.info("System backup completed for admin")
        return backup_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create system backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating system backup: {str(e)}")


@router.post("/system/restore", dependencies=[Depends(admin_required)])
def restore_system_backup(
    backup_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Restore system from backup (admin only).
    
    This endpoint restores the system from a previously created backup.
    This operation will overwrite current data. Only users with admin role can access this endpoint.
    
    Args:
        backup_id (str): ID of the backup to restore from
        token (str): JWT access token for admin authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Restore operation results
        
    Raises:
        HTTPException: If not admin (403), backup not found (404), 
                      or restore fails (500)
    """
    logger.info(f"Admin request to restore system from backup: {backup_id}")
    
    try:
        admin_service = AdminService(db)
        restore_results = admin_service.restore_system_backup(backup_id)
        
        logger.info("System restore completed for admin")
        return restore_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore system from backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error restoring system from backup: {str(e)}")

@router.post("/parse_courses")
def parse_courses(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(admin_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Parse and import courses from uploaded Excel or CSV file with background embedding generation.
    
    This endpoint allows administrators to bulk import courses from spreadsheet files.
    The function processes the uploaded file, validates course data, and stores courses
    in the database. After successful import, it triggers background embedding generation
    and indexing in Pinecone for course matching functionality.
    
    Supported file formats:
    - application/vnd.openxmlformats-officedocument.spreadsheetml.sheet (.xlsx)
    - text/csv (.csv)
    
    Required columns in the file:
    - course_id: Unique identifier for the course
    - course_name: Name of the course
    - course_description: Detailed description of the course
    - major: Academic major/field the course belongs to
    
    Optional columns:
    - prerequisite_1, prerequisite_2, prerequisite_3: Course prerequisites
    - domain_1, domain_2: Course domains/categories
    - skills_associated: Comma-separated list of skills covered
    
    The function automatically creates a combined_text field that concatenates
    all course information for embedding generation.
    
    Args:
        file (UploadFile): Excel or CSV file containing course data
        db (Session): Database session dependency
        current_user: Authenticated admin user (from admin_required dependency)
        background_tasks (BackgroundTasks): FastAPI background tasks for async processing
        
    Returns:
        dict: Import results:
            - message: Success message with number of courses processed
            
    Raises:
        HTTPException: If file type is unsupported (status_code=400), file is empty (status_code=400),
                      validation fails (status_code=422), or processing fails (status_code=500)
    """
    # Check file content type
    if file.content_type not in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .xlsx and .csv files are accepted."
        )

    try:
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file extension. Only .xlsx and .csv allowed."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file. Ensure it's a valid .xlsx or .csv. Error: {str(e)}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    inserted_courses = []

    for _, row in df.iterrows():
        try:
            # Extract fields with safety checks
            course_id = str(row.get("course_id", "")).strip()
            course_name = str(row.get("course_name", "")).strip()
            course_description = str(row.get("course_description", "")).strip()

            prerequisites = [
                str(row.get("prerequisite_1", "")).strip(),
                str(row.get("prerequisite_2", "")).strip(),
                str(row.get("prerequisite_3", "")).strip()
            ]
            prerequisites = [p for p in prerequisites if p] or None

            major = str(row.get("major", "")).strip()

            domains = [
                str(row.get("domain_1", "")).strip(),
                str(row.get("domain_2", "")).strip()
            ]
            domains = [d for d in domains if d] or None

            skills = str(row.get("skills_associated", "")).strip()
            skills_list = [skill.strip() for skill in skills.split(',')] if skills else None

            # Validate required fields
            if not course_id or not course_name or not course_description or not major:
                raise ValueError(f"Missing mandatory fields for a course at row {_+2}")  # +2 considering header and 0-based index

            # Create combined text
            combined_text_parts = [
                f"Course Name: {course_name}",
                f"Description: {course_description}",
                f"Prerequisites: {', '.join(prerequisites) if prerequisites else 'None'}",
                f"Major: {major}",
                f"Domains: {', '.join(domains) if domains else 'None'}",
                f"Skills: {', '.join(skills_list) if skills_list else 'None'}"
            ]
            combined_text = ". ".join(combined_text_parts)

            existing_course = db.query(Course).filter_by(course_id=course_id).first()
            course_id_to_use = existing_course.id if existing_course else uuid.uuid4()
            # Create Course object
            course = Course(
                id=course_id_to_use,
                course_id=course_id,
                course_name=course_name,
                course_description=course_description,
                prerequisites=prerequisites,
                major=major,
                domains=domains,
                skills_associated=skills_list,
                combined_text=combined_text
            )
            # inserted_courses.append(course)
            db.merge(course);

        except ValueError as ve:
            raise HTTPException(status_code=422, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error processing course at row {_+2}: {str(e)}")

    try:
        # db.add_all(inserted_courses)
        db.commit()
    except Exception as db_error:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

    # 🔁 Kick off the background embedding/indexing task
    background_tasks.add_task(embed_and_index_courses)

    return {"message": f"Successfully inserted {len(inserted_courses)} courses."}
