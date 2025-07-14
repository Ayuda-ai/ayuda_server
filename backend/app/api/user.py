"""
User API Endpoints

This module contains all user-related endpoints including authentication,
profile management, resume handling, and user-specific operations.

Endpoints:
- POST /users/signup - User registration
- GET /users/me - Get current user profile
- POST /users/resume/parse - Parse resume text from file
- POST /users/resume/embed - Upload/update resume with embedding
- GET /users/resume/embedding - Get resume embedding
- DELETE /users/resume/embedding - Delete resume embedding
- POST /users/resume - Upload/update resume
- GET /users/resume - Get resume information
- DELETE /users/resume - Delete resume
- GET /users/profile/completed-courses - Get completed courses
- POST /users/profile/completed-courses - Add completed courses
- PUT /users/profile/completed-courses - Update completed courses
- DELETE /users/profile/completed-courses/{course_id} - Remove completed course
- GET /users/profile/additional-skills - Get additional skills
- PUT /users/profile/additional-skills - Update additional skills
- GET /users/profile/enhancement-status - Get profile enhancement status
- POST /users/profile/enhance - Complete profile enhancement
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from typing import List
import logging
import time

from app.schemas.user import UserCreate, UserOut
from app.db.session import get_db
from app.services.user_service import UserService
from app.core.security import create_access_token, oauth2_scheme
from app.core.dependencies import admin_required

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


@router.post("/signup", response_model=UserOut)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user account with validation and access code verification.
    
    This endpoint handles user registration with the following validations:
    - Email domain must be in the allowed domains list
    - Access code must be valid and have remaining uses
    - Password is hashed before storage
    - Access code count is decremented upon successful registration
    
    Args:
        user_data (UserCreate): User registration data including:
            - first_name: User's first name
            - last_name: User's last name
            - university: User's university
            - email: User's email address
            - password: User's password (will be hashed)
            - dob: User's date of birth
            - major: User's major/field of study
            - access_code: Valid access code for registration
        db (Session): Database session dependency
        
    Returns:
        UserOut: User information without sensitive data (password_hash excluded)
        
    Raises:
        HTTPException: If validation fails (status_code=400)
    """
    logger.info(f"User registration attempt for email: {user_data.email}")
    start_time = time.time()
    
    try:
        user_service = UserService(db)
        user = user_service.create_user(user_data)
        
        # Generate access token
        access_token = create_access_token(data={"sub": user.email})
        
        registration_time = time.time() - start_time
        logger.info(f"User registration completed successfully for {user.email}. Time: {registration_time:.2f}s")
        
        return {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "university": user.university,
            "email": user.email,
            "dob": user.dob,
            "major": user.major,
            "role": user.role,
            "access_token": access_token,
            "token_type": "bearer"
        }
    except HTTPException as e:
        logger.error(f"User registration failed with HTTP error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"User registration failed with unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating user: {str(e)}"
        )


@router.get("/me", response_model=UserOut)
def get_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Retrieve current user information from JWT token.
    
    This endpoint decodes the JWT token to identify the user and returns
    their profile information. The token must be valid and the user must
    exist in the database.
    
    Args:
        token (str): JWT access token from Authorization header
        db (Session): Database session dependency
        
    Returns:
        UserOut: Current user's profile information
        
    Raises:
        HTTPException: If token is invalid (status_code=403) or user not found (status_code=404)
    """
    logger.debug("User profile retrieval request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        logger.debug(f"User profile retrieved successfully for: {user.email}")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User profile retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving user profile: {str(e)}")


@router.post("/resume/embed")
async def embed_resume(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Extract text from resume, generate embedding, and store in PostgreSQL and Pinecone.
    
    This endpoint performs a complete resume processing pipeline:
    1. Extracts text from uploaded PDF or DOCX file
    2. Generates 384-dimensional embedding using SentenceTransformer
    3. Stores embedding in both PostgreSQL (for quick retrieval) and Pinecone (for similarity search)
    4. Stores resume text in PostgreSQL for keyword matching
    5. Returns processing status and embedding dimensions
    
    The embedding is stored with user metadata and can be used for course matching.
    The resume text is stored for keyword-based matching using Redis.
    If a user already has a resume embedding, it will be overwritten with the new one.
    
    Args:
        file (UploadFile): Resume file to process
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Processing results including:
            - message: Success message
            - embedding_dim: Dimensions of generated embedding (384)
            - storage: Dictionary with success status for PostgreSQL and Pinecone storage
            
    Raises:
        HTTPException: If file type is unsupported (status_code=400), user not found (status_code=404), or processing fails (status_code=500)
    """
    logger.info(f"Resume embedding request for file: {file.filename}")
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        logger.warning(f"Unsupported file type rejected for embedding: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only PDF and DOCX files are allowed."
        )
    
    user_service = UserService(db)
    try:
        user = user_service.get_user_by_token(token)
        logger.debug(f"User authenticated for embedding: {user.id}")
        
        # Extract text and generate embedding
        resume_text = user_service.extract_text_from_document(file)
        embedding = user_service.create_resume_embedding(resume_text)
        
        # Store both embedding and text
        store_results = user_service.store_resume_embedding_hybrid(user.id, embedding, user, resume_text)
        
        # Update profile enhancement status
        user_service.update_profile_enhancement_status(user.id, True)
        
        total_time = time.time() - start_time
        logger.info(f"Resume embedding completed successfully for user {user.id}. Time: {total_time:.2f}s")
        
        return {
            "message": "Resume embedding generated and stored.",
            "embedding_dim": len(embedding),
            "storage": store_results,
            "profile_enhanced": True
        }
    except HTTPException as e:
        logger.error(f"Resume embedding failed with HTTP error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Resume embedding failed with unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error embedding resume: {str(e)}"
        )


@router.get("/resume/embedding")
def get_resume_embedding(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Retrieve the current user's resume embedding from PostgreSQL.
    
    This endpoint fetches the stored resume embedding for the authenticated user.
    The embedding is retrieved from PostgreSQL for quick access and can be used
    for analysis or debugging purposes.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Contains the embedding vector and its dimensions:
            - embedding: 384-dimensional embedding vector as list
            - dim: Number of dimensions (384)
            
    Raises:
        HTTPException: If user not found (status_code=404) or no embedding exists (status_code=404)
    """
    logger.debug("Resume embedding retrieval request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        if embedding:
            logger.debug(f"Resume embedding retrieved successfully for user: {user.id}")
            return {"embedding": embedding, "dim": len(embedding)}
        else:
            logger.warning(f"No resume embedding found for user: {user.id}")
            raise HTTPException(status_code=404, detail="No resume embedding found for user.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume embedding retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving resume embedding: {str(e)}")


@router.delete("/resume/embedding")
def delete_resume_embedding(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Delete the current user's resume embedding from both PostgreSQL and Pinecone.
    
    This endpoint removes the user's resume embedding from both storage systems:
    - PostgreSQL: Sets resume_embedding field to None
    - Pinecone: Deletes the vector with ID 'resume:{user_id}'
    
    This is useful when a user wants to remove their resume data or start fresh.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Deletion results for both storage systems:
            - postgresql_deleted: True if PostgreSQL deletion succeeded
            - pinecone_deleted: True if Pinecone deletion succeeded
            
    Raises:
        HTTPException: If user not found (status_code=404) or deletion fails (status_code=500)
    """
    logger.info("Resume embedding deletion request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        pg_result = user_service.delete_resume_embedding_postgresql(user.id)
        pinecone_result = user_service.delete_resume_embedding_from_pinecone(user.id)
        
        logger.info(f"Resume embedding deletion completed for user {user.id}. PostgreSQL: {pg_result}, Pinecone: {pinecone_result}")
        
        return {
            "postgresql_deleted": pg_result,
            "pinecone_deleted": pinecone_result
        }
    except Exception as e:
        logger.error(f"Resume embedding deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting resume embedding: {str(e)}")


# ============================================================================
# SMART RESUME MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/resume")
async def upload_or_update_resume(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Smart resume upload/update endpoint.
    
    This endpoint handles both initial resume upload and resume updates:
    - If user has no resume: Creates new resume embedding
    - If user has existing resume: Updates with new resume (overwrites)
    
    The endpoint performs complete resume processing:
    1. Extracts text from PDF/DOCX file
    2. Generates 384-dimensional embedding
    3. Stores in PostgreSQL and Pinecone
    
    Args:
        file (UploadFile): Resume file (PDF or DOCX)
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Processing results with operation type and storage status
        
    Raises:
        HTTPException: If file type is unsupported (400), user not found (404), or processing fails (500)
    """
    logger.info(f"Smart resume upload/update request for file: {file.filename}")
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        logger.warning(f"Unsupported file type rejected: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only PDF and DOCX files are allowed."
        )
    
    user_service = UserService(db)
    try:
        user = user_service.get_user_by_token(token)
        logger.debug(f"User authenticated for resume upload: {user.id}")
        
        # Check if user already has a resume
        existing_embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        operation_type = "update" if existing_embedding else "create"
        
        # Extract text and generate embedding
        resume_text = user_service.extract_text_from_document(file)
        embedding = user_service.create_resume_embedding(resume_text)
        
        # Store both embedding and text
        store_results = user_service.store_resume_embedding_hybrid(user.id, embedding, user, resume_text)
        
        total_time = time.time() - start_time
        logger.info(f"Resume {operation_type} completed successfully for user {user.id}. Time: {total_time:.2f}s")
        
        return {
            "message": f"Resume {operation_type}d successfully.",
            "operation_type": operation_type,
            "embedding_dim": len(embedding),
            "storage": store_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload/update failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )


@router.get("/resume")
def get_resume_info(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get current user's resume information.
    
    This endpoint returns metadata about the user's current resume,
    including whether they have one uploaded and basic processing info.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Resume information including:
            - has_resume: Boolean indicating if resume exists
            - embedding_dim: Dimensions of embedding (if exists)
            - profile_enhanced: Whether profile is enhanced
            - created_at: When resume was last updated
            
    Raises:
        HTTPException: If user not found (404)
    """
    logger.debug("Resume info retrieval request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Check if user has resume
        embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        resume_text = user_service.get_resume_text_from_postgresql(user.id)
        
        has_resume = embedding is not None and resume_text is not None
        
        response = {
            "has_resume": has_resume,
            "profile_enhanced": user.profile_enhanced,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        
        if has_resume:
            response.update({
                "embedding_dim": len(embedding),
                "text_length": len(resume_text)
            })
        
        logger.debug(f"Resume info retrieved successfully for user: {user.id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume info retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving resume info: {str(e)}")


@router.delete("/resume")
def delete_resume(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Delete the current user's resume completely.
    
    This endpoint removes all resume-related data:
    - Resume embedding from PostgreSQL
    - Resume text from PostgreSQL
    - Resume vector from Pinecone
    - Updates profile_enhanced status to False
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Deletion results including:
            - message: Success message
            - postgresql_deleted: True if PostgreSQL deletion succeeded
            - pinecone_deleted: True if Pinecone deletion succeeded
            - profile_enhanced: Updated status (False)
            
    Raises:
        HTTPException: If user not found (404) or deletion fails (500)
    """
    logger.info("Resume deletion request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Delete from both storage systems
        pg_result = user_service.delete_resume_embedding_postgresql(user.id)
        pinecone_result = user_service.delete_resume_embedding_from_pinecone(user.id)
        
        # Update profile enhancement status
        user_service.update_profile_enhancement_status(user.id, False)
        
        logger.info(f"Resume deletion completed for user {user.id}. PostgreSQL: {pg_result}, Pinecone: {pinecone_result}")
        
        return {
            "message": "Resume deleted successfully.",
            "postgresql_deleted": pg_result,
            "pinecone_deleted": pinecone_result,
            "profile_enhanced": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting resume: {str(e)}")


# ============================================================================
# COMPLETED COURSES MANAGEMENT
# ============================================================================

@router.get("/profile/completed-courses")
def get_completed_courses(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get the current user's completed courses.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Completed courses information:
            - completed_courses: List of course IDs
            - count: Number of completed courses
            
    Raises:
        HTTPException: If user not found (404)
    """
    logger.debug("Completed courses retrieval request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        completed_courses = user.completed_courses or []
        
        logger.debug(f"Completed courses retrieved for user {user.id}: {len(completed_courses)} courses")
        
        return {
            "completed_courses": completed_courses,
            "count": len(completed_courses)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Completed courses retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving completed courses: {str(e)}")


@router.post("/profile/completed-courses")
def add_completed_courses(
    course_ids: List[str],
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Add completed courses to the user's profile.
    
    Args:
        course_ids (List[str]): List of course IDs to add
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Updated completed courses information
        
    Raises:
        HTTPException: If user not found (404) or update fails (500)
    """
    logger.info(f"Adding completed courses request: {len(course_ids)} courses")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Add new course IDs to existing ones
        current_courses = user.completed_courses or []
        new_courses = list(set(current_courses + course_ids))  # Remove duplicates
        
        # Update user's completed courses
        user.completed_courses = new_courses
        user_service.update_profile_enhancement_status(user.id, True)
        
        db.commit()
        
        logger.info(f"Completed courses updated for user {user.id}: {len(new_courses)} total courses")
        
        return {
            "message": f"Added {len(course_ids)} completed courses.",
            "completed_courses": new_courses,
            "count": len(new_courses),
            "profile_enhanced": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Adding completed courses failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding completed courses: {str(e)}")


@router.put("/profile/completed-courses")
def update_completed_courses(
    course_ids: List[str],
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Replace the user's completed courses with a new list.
    
    Args:
        course_ids (List[str]): New list of course IDs
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Updated completed courses information
        
    Raises:
        HTTPException: If user not found (404) or update fails (500)
    """
    logger.info(f"Updating completed courses request: {len(course_ids)} courses")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Replace with new course IDs
        user.completed_courses = course_ids
        user_service.update_profile_enhancement_status(user.id, True)
        
        db.commit()
        
        logger.info(f"Completed courses replaced for user {user.id}: {len(course_ids)} courses")
        
        return {
            "message": f"Updated completed courses to {len(course_ids)} courses.",
            "completed_courses": course_ids,
            "count": len(course_ids),
            "profile_enhanced": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Updating completed courses failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating completed courses: {str(e)}")


@router.delete("/profile/completed-courses/{course_id}")
def remove_completed_course(
    course_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Remove a specific course from the user's completed courses.
    
    Args:
        course_id (str): Course ID to remove
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Updated completed courses information
        
    Raises:
        HTTPException: If user not found (404) or update fails (500)
    """
    logger.info(f"Removing completed course request: {course_id}")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        current_courses = user.completed_courses or []
        
        if course_id in current_courses:
            # Create a new list to ensure SQLAlchemy detects the change
            updated_courses = [c for c in current_courses if c != course_id]
            user.completed_courses = updated_courses
            
            # Force SQLAlchemy to mark the field as modified
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(user, "completed_courses")
            
            db.commit()
            
            logger.info(f"Removed course {course_id} from user {user.id}")
            
            return {
                "message": f"Removed course {course_id} from completed courses.",
                "completed_courses": updated_courses,
                "count": len(updated_courses)
            }
        else:
            logger.warning(f"Course {course_id} not found in user {user.id}'s completed courses")
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found in completed courses")
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Removing completed course failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing completed course: {str(e)}")


# ============================================================================
# ADDITIONAL SKILLS MANAGEMENT
# ============================================================================

@router.get("/profile/additional-skills")
def get_additional_skills(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get the current user's additional skills.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Additional skills information:
            - additional_skills: User's additional skills text
            - has_skills: Boolean indicating if skills are provided
            
    Raises:
        HTTPException: If user not found (404)
    """
    logger.debug("Additional skills retrieval request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        additional_skills = user.additional_skills or ""
        has_skills = bool(additional_skills.strip())
        
        logger.debug(f"Additional skills retrieved for user {user.id}: {len(additional_skills)} characters")
        
        return {
            "additional_skills": additional_skills,
            "has_skills": has_skills
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Additional skills retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving additional skills: {str(e)}")


@router.put("/profile/additional-skills")
def update_additional_skills(
    skills_data: dict,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Update the current user's additional skills.
    
    Args:
        skills_data (dict): Skills data containing:
            - additional_skills (str): New additional skills text
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Updated skills information:
            - message: Success message
            - additional_skills: Updated skills text
            - has_skills: Boolean indicating if skills are provided
            - profile_enhanced: Updated profile status
            
    Raises:
        HTTPException: If user not found (404) or update fails (500)
    """
    logger.info("Additional skills update request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        additional_skills = skills_data.get("additional_skills", "")
        
        # Update user's additional skills
        user.additional_skills = additional_skills
        user_service.update_profile_enhancement_status(user.id, True)
        
        db.commit()
        
        has_skills = bool(additional_skills.strip())
        logger.info(f"Additional skills updated for user {user.id}: {len(additional_skills)} characters")
        
        return {
            "message": "Additional skills updated successfully.",
            "additional_skills": additional_skills,
            "has_skills": has_skills,
            "profile_enhanced": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Additional skills update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating additional skills: {str(e)}")


# ============================================================================
# PROFILE ENHANCEMENT STATUS
# ============================================================================

@router.get("/profile/enhancement-status")
def get_profile_enhancement_status(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get the current user's profile enhancement status.
    
    This endpoint returns information about what profile enhancement
    features the user has completed.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Profile enhancement status:
            - profile_enhanced: Overall enhancement status
            - has_resume: Whether user has uploaded resume
            - has_completed_courses: Whether user has added completed courses
            - has_additional_skills: Whether user has provided additional skills
            - completion_percentage: Percentage of enhancement features completed
            
    Raises:
        HTTPException: If user not found (404)
    """
    logger.debug("Profile enhancement status request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Check each enhancement feature
        has_resume = user.resume_embedding is not None and user.resume_text is not None
        has_completed_courses = bool(user.completed_courses and len(user.completed_courses) > 0)
        has_additional_skills = bool(user.additional_skills and user.additional_skills.strip())
        
        # Calculate completion percentage
        features_completed = sum([has_resume, has_completed_courses, has_additional_skills])
        completion_percentage = (features_completed / 3) * 100
        
        logger.debug(f"Profile enhancement status retrieved for user {user.id}: {completion_percentage:.1f}% complete")
        
        return {
            "profile_enhanced": user.profile_enhanced,
            "has_resume": has_resume,
            "has_completed_courses": has_completed_courses,
            "has_additional_skills": has_additional_skills,
            "completion_percentage": round(completion_percentage, 1),
            "features_completed": features_completed,
            "total_features": 3
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile enhancement status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving profile enhancement status: {str(e)}")


@router.post("/profile/enhance")
def complete_profile_enhancement(
    enhancement_data: dict,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Complete profile enhancement with all available data.
    
    This endpoint allows users to provide all profile enhancement
    data in a single request.
    
    Args:
        enhancement_data (dict): Enhancement data containing:
            - completed_courses (List[str], optional): List of completed course IDs
            - additional_skills (str, optional): Additional skills text
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Profile enhancement results:
            - message: Success message
            - profile_enhanced: Updated profile status
            - completion_percentage: Updated completion percentage
            
    Raises:
        HTTPException: If user not found (404) or update fails (500)
    """
    logger.info("Profile enhancement completion request")
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Update completed courses if provided
        if "completed_courses" in enhancement_data:
            user.completed_courses = enhancement_data["completed_courses"]
        
        # Update additional skills if provided
        if "additional_skills" in enhancement_data:
            user.additional_skills = enhancement_data["additional_skills"]
        
        # Mark profile as enhanced
        user_service.update_profile_enhancement_status(user.id, True)
        
        db.commit()
        
        # Get updated status
        has_resume = user.resume_embedding is not None and user.resume_text is not None
        has_completed_courses = bool(user.completed_courses and len(user.completed_courses) > 0)
        has_additional_skills = bool(user.additional_skills and user.additional_skills.strip())
        
        features_completed = sum([has_resume, has_completed_courses, has_additional_skills])
        completion_percentage = (features_completed / 3) * 100
        
        logger.info(f"Profile enhancement completed for user {user.id}: {completion_percentage:.1f}% complete")
        
        return {
            "message": "Profile enhancement completed successfully.",
            "profile_enhanced": True,
            "completion_percentage": round(completion_percentage, 1),
            "features_completed": features_completed,
            "total_features": 3
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Profile enhancement completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error completing profile enhancement: {str(e)}")


@router.post("/resume/parse")
async def parse_resume(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Extract text content from uploaded resume file.
    
    This endpoint performs text extraction from uploaded PDF or DOCX files.
    It only extracts the text content without generating embeddings or storing
    the data. This is useful for previewing the extracted text before
    proceeding with embedding generation.
    
    Args:
        file (UploadFile): Resume file to process (PDF or DOCX)
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Processing results including:
            - message: Success message
            - extracted_text: The extracted text content
            - text_length: Length of extracted text in characters
            - file_type: Type of processed file
            
    Raises:
        HTTPException: If file type is unsupported (status_code=400), user not found (status_code=404), or processing fails (status_code=500)
    """
    logger.info(f"Resume parsing request for file: {file.filename}")
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        logger.warning(f"Unsupported file type rejected for parsing: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only PDF and DOCX files are allowed."
        )
    
    user_service = UserService(db)
    try:
        user = user_service.get_user_by_token(token)
        logger.debug(f"User authenticated for parsing: {user.id}")
        
        # Extract text from document
        extracted_text = user_service.extract_text_from_document(file)
        
        total_time = time.time() - start_time
        logger.info(f"Resume parsing completed successfully for user {user.id}. Time: {total_time:.2f}s")
        
        return {
            "message": "Resume text extracted successfully.",
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
            "file_type": file.content_type
        }
    except HTTPException as e:
        logger.error(f"Resume parsing failed with HTTP error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Resume parsing failed with unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing resume: {str(e)}"
        )
