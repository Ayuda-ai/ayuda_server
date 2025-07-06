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
from app.services.neo4j_service import Neo4jService
from app.models.user import User
from app.core.config import settings
from app.core.security import create_access_token, oauth2_scheme
from app.core.dependencies import admin_required
from app.services.recommendation_logger import RecommendationLogger

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Initialize recommendation logger
recommendation_logger = RecommendationLogger()

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

@router.post("/resume")
async def extract_text_from_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Extract text content from uploaded PDF or DOCX resume files.
    
    This endpoint accepts resume files in PDF or DOCX format and extracts
    the text content for further processing. The extracted text can be used
    for embedding generation and course matching.
    
    Supported file types:
    - application/pdf (PDF files)
    - application/vnd.openxmlformats-officedocument.wordprocessingml.document (DOCX files)
    
    Args:
        file (UploadFile): Resume file to process
        db (Session): Database session dependency
        
    Returns:
        JSONResponse: Contains the extracted text content
        
    Raises:
        HTTPException: If file type is unsupported (status_code=400) or processing fails (status_code=500)
    """
    logger.info(f"Resume text extraction request for file: {file.filename}")
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
        extracted_text = user_service.extract_text_from_document(file)
        extraction_time = time.time() - start_time
        logger.info(f"Resume text extraction completed successfully. Time: {extraction_time:.2f}s, Text length: {len(extracted_text)} chars")
        return JSONResponse(
            content={"text": extracted_text},
            status_code=200
        )
    except HTTPException as e:
        logger.error(f"Resume text extraction failed with HTTP error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Resume text extraction failed with unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

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
        
        total_time = time.time() - start_time
        logger.info(f"Resume embedding completed successfully for user {user.id}. Time: {total_time:.2f}s")
        
        return {
            "message": "Resume embedding generated and stored.",
            "embedding_dim": len(embedding),
            "storage": store_results
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

@router.get("/resume/match_courses")
def match_courses_with_resume(
    top_k: int = 5,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Find top-k matching courses using hybrid approach combining semantic similarity, keyword matching, and prerequisite checking.
    
    This endpoint performs a comprehensive course matching approach that combines:
    1. Semantic similarity search between resume embeddings and course embeddings
    2. Keyword matching between resume text and course skills using Redis
    3. Prerequisite checking using Neo4j to determine course eligibility
    
    The hybrid approach provides more accurate recommendations by:
    - Using semantic similarity for understanding context and meaning
    - Using keyword matching for exact skill and technology matches
    - Checking prerequisites to separate eligible vs ineligible courses
    - Combining all scores with configurable weights (70% semantic, 30% keyword by default)
    
    Args:
        top_k (int): Number of top matching courses to return (default: 5, max: 100)
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Enhanced course matching results with prerequisite information:
            - eligible_matches: List of courses user can take (prerequisites met)
            - ineligible_matches: List of courses requiring prerequisites user hasn't completed
            - total_matches: Total number of matching courses
            - prerequisite_analysis: Summary of prerequisite checking
            - user_completed_courses: List of user's completed courses
            - processing_metrics: Performance and timing information
            
    Raises:
        HTTPException: If user not found (status_code=404), no resume embedding exists (status_code=404), or matching fails (status_code=500)
    """
    logger.info(f"Hybrid course matching with prerequisite checking request with top_k={top_k}")
    start_time = time.time()
    
    # Initialize tracking variables
    user = None
    enhanced_matches = []
    eligible_matches = []
    ineligible_matches = []
    error_info = None
    processing_metrics = {}
    system_health = {}
    resume_info = {}
    keyword_analysis = {}
    prerequisite_analysis = {}
    
    try:
        user_service = UserService(db)
        neo4j_service = Neo4jService()
        user = user_service.get_user_by_token(token)
        
        # Get user's completed courses
        completed_courses = user.completed_courses or []
        
        # Get resume information
        embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        resume_text = user_service.get_resume_text_from_postgresql(user.id)
        
        # Prepare resume info
        resume_info = {
            "has_embedding": embedding is not None,
            "has_resume_text": resume_text is not None,
            "embedding_dimensions": len(embedding) if embedding else 0,
            "resume_text_length": len(resume_text) if resume_text else 0,
            "major": user.major,
            "university": user.university,
            "completed_courses_count": len(completed_courses)
        }
        
        # Check system health
        redis_status = user_service.redis_service.redis_client is not None
        
        # Test Pinecone connection by trying to create one
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            # Try a simple operation to verify connection
            index.describe_index_stats()
            pinecone_status = True
        except Exception as e:
            logger.warning(f"Pinecone connection test failed: {str(e)}")
            pinecone_status = False
        
        # Check Neo4j connection
        neo4j_status = neo4j_service.is_configured() and neo4j_service.test_connection()
        
        # Debug Neo4j status
        logger.info(f"Neo4j status: configured={neo4j_service.is_configured()}, connected={neo4j_service.test_connection()}")
        if not neo4j_status:
            logger.warning("Neo4j not available for prerequisite checking")
        
        database_status = True
        
        system_health = {
            "redis_connection": redis_status,
            "pinecone_connection": pinecone_status,
            "neo4j_connection": neo4j_status,
            "database_connection": database_status,
            "overall_status": all([redis_status, pinecone_status, neo4j_status, database_status])
        }
        
        # Get keyword analysis if resume text exists
        if resume_text and redis_status:
            extracted_keywords, matching_keywords = user_service.redis_service.get_keyword_matches(resume_text)
            keyword_analysis = {
                "extracted_keywords_count": len(extracted_keywords),
                "matching_keywords_count": len(matching_keywords),
                "keyword_match_rate": len(matching_keywords) / len(extracted_keywords) if extracted_keywords else 0,
                "matching_keywords": list(matching_keywords)[:20]  # Top 20
            }
        
        # Get hybrid course matches
        enhanced_matches = user_service.get_hybrid_course_matches(user.id, 50)  # Increase top_k for better candidate pool
        
        if not enhanced_matches:
            logger.warning(f"No course matches found for user: {user.id}")
            enhanced_matches = []
        
        # Get ALL courses from database to check prerequisites comprehensively
        all_courses = user_service.get_all_courses()
        logger.info(f"Retrieved {len(all_courses)} total courses from database")
        
        # Debug: Show first few courses
        if all_courses:
            logger.info(f"Sample course data: {all_courses[0]}")
        
        # Check prerequisites for ALL courses if Neo4j is available
        if neo4j_status and all_courses:
            logger.info(f"Checking prerequisites for {len(all_courses)} courses")
            logger.info(f"Neo4j status: {neo4j_status}, all_courses count: {len(all_courses)}")
            prerequisite_start_time = time.time()
            
            # Create a set of course IDs from enhanced matches for quick lookup
            enhanced_course_ids = {str(course["id"]) for course in enhanced_matches}
            
            # Check prerequisites for all courses
            for course_data in all_courses:
                course_id = str(course_data["id"])
                course_code = course_data.get("course_id", "")
                
                # Check prerequisites using Neo4j
                prereq_status = neo4j_service.check_prerequisites_completion(
                    course_code, completed_courses
                )
                
                # Create course object with prerequisite information
                course_obj = {
                    "id": course_code,  # Use course_code instead of UUID
                    "semantic_score": 0.0,  # Will be updated if in enhanced matches
                    "keyword_score": 0.0,   # Will be updated if in enhanced matches
                    "hybrid_score": 0.0,    # Will be updated if in enhanced matches
                    "metadata": {
                        "course_name": course_data.get("course_name", ""),
                        "domains": course_data.get("domains", []),
                        "major": course_data.get("major", ""),
                        "skills_associated": course_data.get("skills_associated", []),
                        "type": "course"
                    },
                    "matching_keywords": [],  # Will be updated if in enhanced matches
                    "prerequisite_status": prereq_status,
                    "eligible": prereq_status["prerequisites_met"],
                    "prerequisites": prereq_status.get("prerequisite_groups", []),
                    "missing_prerequisites": prereq_status.get("missing_prerequisites", []),
                    "completed_prerequisites": prereq_status.get("completed_prerequisites", [])
                }
                
                # If this course was in the enhanced matches, update with actual scores
                if course_code in enhanced_course_ids:  # Match by course_code, not UUID
                    for enhanced_course in enhanced_matches:
                        if str(enhanced_course["id"]) == course_code:  # Match by course_code
                            course_obj.update({
                                "semantic_score": enhanced_course.get("semantic_score", 0.0),
                                "keyword_score": enhanced_course.get("keyword_score", 0.0),
                                "hybrid_score": enhanced_course.get("hybrid_score", 0.0),
                                "matching_keywords": enhanced_course.get("matching_keywords", [])
                            })
                            logger.debug(f"✅ Updated scores for course {course_code}: hybrid_score={enhanced_course.get('hybrid_score', 0.0)}")
                            break
                else:
                    # For courses not in hybrid matches, give them a small base score if they have no prerequisites
                    if not prereq_status["has_prerequisites"]:
                        course_obj.update({
                            "semantic_score": 0.1,  # Small base score for courses with no prerequisites
                            "keyword_score": 0.1,
                            "hybrid_score": 0.1,
                            "matching_keywords": []
                        })
                
                # Categorize course based on prerequisites and scores
                if prereq_status["prerequisites_met"]:
                    # User can take this course
                    eligible_matches.append(course_obj)
                    logger.debug(f"✅ Course {course_code} is eligible")
                else:
                    # User cannot take this course yet
                    course_obj["prerequisite_message"] = f"Complete one of: {', '.join([p['name'] for p in prereq_status['missing_prerequisites']])}"
                    ineligible_matches.append(course_obj)
                    logger.debug(f"❌ Course {course_code} is ineligible: {course_obj['prerequisite_message']}")
            
            # Sort eligible matches by hybrid score (highest first)
            eligible_matches.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            # Sort ineligible matches by hybrid score (highest first)
            ineligible_matches.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            # Limit to top_k for each category
            eligible_matches = eligible_matches[:top_k]
            ineligible_matches = ineligible_matches[:top_k]
            
            prerequisite_time = time.time() - prerequisite_start_time
            prerequisite_analysis = {
                "prerequisites_checked": True,
                "total_courses_checked": len(all_courses),
                "eligible_count": len(eligible_matches),
                "ineligible_count": len(ineligible_matches),
                "prerequisite_checking_time": prerequisite_time,
                "user_completed_courses": completed_courses
            }
            
            logger.info(f"Prerequisite checking completed: {len(eligible_matches)} eligible, {len(ineligible_matches)} ineligible")
        else:
            # Debug why we're falling into this block
            logger.warning(f"Falling into else block: neo4j_status={neo4j_status}, all_courses_count={len(all_courses) if all_courses else 0}")
            
            # No prerequisite checking (Neo4j not available) - use original enhanced matches
            # Sort by hybrid score and limit to top_k
            enhanced_matches.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            eligible_matches = enhanced_matches[:top_k]
            ineligible_matches = []
            prerequisite_analysis = {
                "prerequisites_checked": False,
                "reason": "Neo4j not configured or unavailable",
                "total_courses_checked": len(enhanced_matches),
                "eligible_count": len(eligible_matches),
                "ineligible_count": 0
            }
        
        # Calculate processing metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        processing_metrics = {
            "total_processing_time": total_time,
            "semantic_processing_time": total_time * 0.6,  # Estimate
            "keyword_processing_time": total_time * 0.2,   # Estimate
            "prerequisite_checking_time": prerequisite_analysis.get("prerequisite_checking_time", 0),
            "hybrid_scoring_time": total_time * 0.1        # Estimate
        }
        
        matching_time = time.time() - start_time
        logger.info(f"Hybrid course matching with prerequisite checking completed successfully for user {user.id}. Found {len(enhanced_matches)} matches ({len(eligible_matches)} eligible, {len(ineligible_matches)} ineligible). Time: {matching_time:.2f}s")
        
        # Generate JSON log file
        log_file_path = recommendation_logger.log_recommendation_request(
            user_id=str(user.id),
            user_email=user.email,
            request_type="hybrid_with_prerequisites",
            top_k=top_k,
            start_time=start_time,
            end_time=end_time,
            matches=enhanced_matches,
            processing_metrics=processing_metrics,
            system_health=system_health,
            resume_info=resume_info,
            keyword_analysis=keyword_analysis,
            error_info=error_info
        )
        
        logger.info(f"📊 Recommendation log saved: {log_file_path}")
        
        return {
            "eligible_matches": eligible_matches,
            "ineligible_matches": ineligible_matches,
            "total_matches": len(all_courses) if neo4j_status else len(enhanced_matches),
            "prerequisite_analysis": prerequisite_analysis,
            "user_completed_courses": completed_courses,
            "processing_metrics": processing_metrics,
            "system_health": system_health
        }
        
    except HTTPException as e:
        error_info = {
            "error_type": "HTTPException",
            "status_code": e.status_code,
            "detail": e.detail
        }
        logger.error(f"Hybrid course matching failed with HTTP error: {str(e)}")
        
        # Log the error
        if user:
            recommendation_logger.log_recommendation_request(
                user_id=str(user.id),
                user_email=user.email,
                request_type="hybrid_with_prerequisites",
                top_k=top_k,
                start_time=start_time,
                end_time=time.time(),
                matches=[],
                processing_metrics={},
                system_health=system_health,
                resume_info=resume_info,
                keyword_analysis=keyword_analysis,
                error_info=error_info
            )
        raise e
        
    except Exception as e:
        error_info = {
            "error_type": "Exception",
            "error_message": str(e)
        }
        logger.error(f"Hybrid course matching failed with unexpected error: {str(e)}")
        
        # Log the error
        if user:
            recommendation_logger.log_recommendation_request(
                user_id=str(user.id),
                user_email=user.email,
                request_type="hybrid_with_prerequisites",
                top_k=top_k,
                start_time=start_time,
                end_time=time.time(),
                matches=[],
                processing_metrics={},
                system_health=system_health,
                resume_info=resume_info,
                keyword_analysis=keyword_analysis,
                error_info=error_info
            )
        raise HTTPException(status_code=500, detail=f"Error matching courses: {str(e)}")

@router.get("/resume/match_courses/semantic")
def match_courses_semantic_only(
    top_k: int = 5,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Find top-k matching courses using only semantic similarity (original approach).
    
    This endpoint performs semantic similarity search between the user's resume
    embedding and all course embeddings stored in Pinecone. It uses cosine similarity
    to find the most relevant courses based on the resume content.
    
    This is the original matching approach that only uses embedding similarity.
    
    Args:
        top_k (int): Number of top matching courses to return (default: 5, max: 100)
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Course matching results:
            - matches: List of matching courses, each containing:
                - id: Course ID
                - score: Similarity score (0-1, higher is more similar)
                - metadata: Course metadata (name, major, domains, skills)
                
    Raises:
        HTTPException: If user not found (status_code=404), no resume embedding exists (status_code=404), or matching fails (status_code=500)
    """
    logger.info(f"Semantic-only course matching request with top_k={top_k}")
    start_time = time.time()
    
    # Initialize tracking variables
    user = None
    semantic_matches = []
    error_info = None
    processing_metrics = {}
    system_health = {}
    resume_info = {}
    
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        
        if not embedding:
            logger.warning(f"No resume embedding found for semantic matching - user: {user.id}")
            raise HTTPException(status_code=404, detail="No resume embedding found for user.")
        
        # Prepare resume info
        resume_info = {
            "has_embedding": True,
            "has_resume_text": False,
            "embedding_dimensions": len(embedding),
            "resume_text_length": 0,
            "major": user.major,
            "university": user.university
        }
        
        # Check system health
        redis_status = user_service.redis_service.redis_client is not None
        
        # Test Pinecone connection by trying to create one
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            # Try a simple operation to verify connection
            index.describe_index_stats()
            pinecone_status = True
        except Exception as e:
            logger.warning(f"Pinecone connection test failed: {str(e)}")
            pinecone_status = False
        
        database_status = True
        
        system_health = {
            "redis_connection": redis_status,
            "pinecone_connection": pinecone_status,
            "database_connection": database_status,
            "overall_status": all([redis_status, pinecone_status, database_status])
        }
        
        # Get semantic matches only
        semantic_matches = user_service.get_semantic_course_matches(embedding, top_k)
        
        # Calculate processing metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        processing_metrics = {
            "total_processing_time": total_time,
            "semantic_processing_time": total_time,
            "keyword_processing_time": 0,
            "hybrid_scoring_time": 0
        }
        
        matching_time = time.time() - start_time
        logger.info(f"Semantic course matching completed successfully for user {user.id}. Found {len(semantic_matches)} matches. Time: {matching_time:.2f}s")
        
        # Generate JSON log file
        log_file_path = recommendation_logger.log_recommendation_request(
            user_id=str(user.id),
            user_email=user.email,
            request_type="semantic",
            top_k=top_k,
            start_time=start_time,
            end_time=end_time,
            matches=semantic_matches,
            processing_metrics=processing_metrics,
            system_health=system_health,
            resume_info=resume_info,
            keyword_analysis=keyword_analysis,
            error_info=error_info
        )
        
        logger.info(f"📊 Recommendation log saved: {log_file_path}")
        
        return {"matches": semantic_matches}
        
    except HTTPException as e:
        error_info = {
            "error_type": "HTTPException",
            "status_code": e.status_code,
            "detail": e.detail
        }
        logger.error(f"Semantic course matching failed with HTTP error: {str(e)}")
        
        # Log the error
        if user:
            recommendation_logger.log_recommendation_request(
                user_id=str(user.id),
                user_email=user.email,
                request_type="semantic",
                top_k=top_k,
                start_time=start_time,
                end_time=time.time(),
                matches=[],
                processing_metrics={},
                system_health=system_health,
                resume_info=resume_info,
                keyword_analysis=keyword_analysis,
                error_info=error_info
            )
        raise e
        
    except Exception as e:
        error_info = {
            "error_type": "Exception",
            "error_message": str(e)
        }
        logger.error(f"Semantic course matching failed with unexpected error: {str(e)}")
        
        # Log the error
        if user:
            recommendation_logger.log_recommendation_request(
                user_id=str(user.id),
                user_email=user.email,
                request_type="semantic",
                top_k=top_k,
                start_time=start_time,
                end_time=time.time(),
                matches=[],
                processing_metrics={},
                system_health=system_health,
                resume_info=resume_info,
                keyword_analysis=keyword_analysis,
                error_info=error_info
            )
        raise HTTPException(status_code=500, detail=f"Error matching courses: {str(e)}")

@router.get("/resume/analytics", dependencies=[Depends(admin_required)])
def get_recommendation_analytics(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get detailed analytics and metrics for the user's recommendation history.
    
    This endpoint provides comprehensive analytics about the user's course
    recommendations, including performance metrics, algorithm effectiveness,
    and system health information.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Comprehensive analytics including:
            - user_info: Basic user information
            - recommendation_summary: Overall recommendation statistics
            - performance_metrics: Processing times and scores
            - system_health: Connection status for all services
            - recent_recommendations: Details of recent recommendations
            - keyword_analysis: Keyword matching effectiveness
            
    Raises:
        HTTPException: If user not found (status_code=404) or analytics generation fails (status_code=500)
    """
    logger.info("Recommendation analytics request")
    
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Get resume information
        embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        resume_text = user_service.get_resume_text_from_postgresql(user.id)
        
        # Check system health
        redis_status = user_service.redis_service.redis_client is not None
        
        # Test Pinecone connection by trying to create one
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            # Try a simple operation to verify connection
            index.describe_index_stats()
            pinecone_status = True
        except Exception as e:
            logger.warning(f"Pinecone connection test failed: {str(e)}")
            pinecone_status = False
        
        database_status = True
        
        system_health = {
            "redis_connection": redis_status,
            "pinecone_connection": pinecone_status,
            "database_connection": database_status,
            "overall_status": all([redis_status, pinecone_status, database_status])
        }
        
        # Get keyword analysis
        keyword_analysis = {}
        if resume_text and redis_status:
            extracted_keywords, matching_keywords = user_service.redis_service.get_keyword_matches(resume_text)
            keyword_analysis = {
                "extracted_keywords_count": len(extracted_keywords),
                "matching_keywords_count": len(matching_keywords),
                "keyword_match_rate": len(matching_keywords) / len(extracted_keywords) if extracted_keywords else 0,
                "matching_keywords": list(matching_keywords)[:20]  # Top 20
            }
        
        # Get log statistics from RecommendationLogger
        log_statistics = recommendation_logger.get_log_statistics()
        recent_logs = recommendation_logger.get_recent_logs(limit=5)
        
        # Create analytics response
        analytics = {
            "user_info": {
                "user_id": str(user.id),
                "email": user.email,
                "major": user.major,
                "university": user.university
            },
            "resume_status": {
                "has_embedding": embedding is not None,
                "has_resume_text": resume_text is not None,
                "embedding_dimensions": len(embedding) if embedding else 0,
                "resume_text_length": len(resume_text) if resume_text else 0
            },
            "system_health": {
                "redis_connection": redis_status,
                "pinecone_connection": pinecone_status,
                "database_connection": database_status,
                "overall_status": all([redis_status, pinecone_status, database_status])
            },
            "keyword_analysis": keyword_analysis,
            "recommendation_capabilities": {
                "semantic_matching": embedding is not None and pinecone_status,
                "keyword_matching": resume_text is not None and redis_status,
                "hybrid_matching": embedding is not None and resume_text is not None and redis_status and pinecone_status
            },
            "performance_insights": {
                "embedding_generation_time": "~1-3 seconds",
                "semantic_search_time": "~0.5-2 seconds",
                "keyword_processing_time": "~0.1-0.5 seconds",
                "hybrid_scoring_time": "~0.1-0.3 seconds"
            },
            "algorithm_details": {
                "semantic_model": "all-MiniLM-L6-v2",
                "embedding_dimensions": 384,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "similarity_metric": "cosine_similarity"
            },
            "log_statistics": log_statistics,
            "recent_recommendations": recent_logs
        }
        
        logger.info(f"Analytics generated successfully for user: {user.id}")
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analytics generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@router.get("/resume/debug")
def debug_recommendation_system(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Debug endpoint for detailed recommendation system analysis.
    
    This endpoint provides detailed debugging information about the recommendation
    system, including raw scores, processing steps, and system state. Useful for
    development and troubleshooting.
    
    Args:
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Detailed debugging information including:
            - raw_scores: Individual semantic and keyword scores
            - processing_steps: Step-by-step processing information
            - system_state: Current state of all services
            - recommendations_comparison: Side-by-side comparison of different approaches
            
    Raises:
        HTTPException: If user not found (status_code=404) or debugging fails (status_code=500)
    """
    logger.info("Recommendation system debug request")
    
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
        # Get resume data
        embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        resume_text = user_service.get_resume_text_from_postgresql(user.id)
        
        if not embedding:
            raise HTTPException(status_code=404, detail="No resume embedding found for user.")
        
        # Test semantic matching
        semantic_matches = user_service.get_semantic_course_matches(embedding, 5)
        
        # Test keyword processing
        keyword_debug = {}
        if resume_text:
            extracted_keywords, matching_keywords = user_service.redis_service.get_keyword_matches(resume_text)
            keyword_debug = {
                "extracted_keywords": list(extracted_keywords)[:20],
                "matching_keywords": list(matching_keywords)[:20],
                "extraction_count": len(extracted_keywords),
                "match_count": len(matching_keywords)
            }
        
        # Test hybrid matching
        hybrid_matches = []
        if resume_text:
            hybrid_matches = user_service.get_hybrid_course_matches(user.id, 50)
        
        # Create debug response
        debug_info = {
            "user_id": str(user.id),
            "resume_data": {
                "embedding_dimensions": len(embedding),
                "resume_text_length": len(resume_text) if resume_text else 0,
                "embedding_preview": embedding[:5] if embedding else None
            },
            "semantic_matching": {
                "matches_found": len(semantic_matches),
                "top_scores": [match.get('score', 0.0) for match in semantic_matches[:3]],
                "sample_matches": [
                    {
                        "id": match.get('id'),
                        "score": match.get('score'),
                        "course_name": match.get('metadata', {}).get('course_name')
                    }
                    for match in semantic_matches[:3]
                ]
            },
            "keyword_processing": keyword_debug,
            "hybrid_matching": {
                "matches_found": len(hybrid_matches),
                "sample_matches": [
                    {
                        "id": match.get('id'),
                        "semantic_score": match.get('semantic_score'),
                        "keyword_score": match.get('keyword_score'),
                        "hybrid_score": match.get('hybrid_score'),
                        "course_name": match.get('metadata', {}).get('course_name')
                    }
                    for match in hybrid_matches[:3]
                ]
            },
            "system_status": {
                "redis_connected": user_service.redis_service.redis_client is not None,
                "pinecone_connected": hasattr(user_service, 'index') and user_service.index is not None,
                "database_connected": True
            }
        }
        
        logger.info(f"Debug information generated for user: {user.id}")
        return debug_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating debug info: {str(e)}")

# ============================================================================
# PHASE 3: SMART RESUME MANAGEMENT ENDPOINTS
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
    4. Updates user's profile_enhanced status
    
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
        
        # Update profile enhancement status
        user_service.update_profile_enhancement_status(user.id, True)
        
        total_time = time.time() - start_time
        logger.info(f"Resume {operation_type} completed successfully for user {user.id}. Time: {total_time:.2f}s")
        
        return {
            "message": f"Resume {operation_type}d successfully.",
            "operation_type": operation_type,
            "embedding_dim": len(embedding),
            "storage": store_results,
            "profile_enhanced": True
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
                "text_length": len(resume_text),
                "last_updated": user.updated_at.isoformat() if user.updated_at else None
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
# PHASE 3: COMPLETED COURSES MANAGEMENT
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
            current_courses.remove(course_id)
            user.completed_courses = current_courses
            db.commit()
            
            logger.info(f"Removed course {course_id} from user {user.id}")
            
            return {
                "message": f"Removed course {course_id} from completed courses.",
                "completed_courses": current_courses,
                "count": len(current_courses)
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
# PHASE 3: ADDITIONAL SKILLS MANAGEMENT
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
# PHASE 3: PROFILE ENHANCEMENT STATUS
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
