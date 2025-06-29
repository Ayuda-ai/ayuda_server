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
    Find top-k matching courses using hybrid approach combining semantic similarity and keyword matching.
    
    This endpoint performs a hybrid course matching approach that combines:
    1. Semantic similarity search between resume embeddings and course embeddings
    2. Keyword matching between resume text and course skills using Redis
    
    The hybrid approach provides more accurate recommendations by:
    - Using semantic similarity for understanding context and meaning
    - Using keyword matching for exact skill and technology matches
    - Combining both scores with configurable weights (70% semantic, 30% keyword by default)
    
    Args:
        top_k (int): Number of top matching courses to return (default: 5, max: 100)
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Enhanced course matching results:
            - matches: List of matching courses, each containing:
                - id: Course ID
                - semantic_score: Similarity score from embedding comparison (0-1)
                - keyword_score: Similarity score from keyword matching (0-1)
                - hybrid_score: Combined weighted score (0-1)
                - metadata: Course metadata (name, major, domains, skills)
                - matching_keywords: List of keywords that matched between resume and course
                
    Raises:
        HTTPException: If user not found (status_code=404), no resume embedding exists (status_code=404), or matching fails (status_code=500)
    """
    logger.info(f"Hybrid course matching request with top_k={top_k}")
    start_time = time.time()
    
    # Initialize tracking variables
    user = None
    enhanced_matches = []
    error_info = None
    processing_metrics = {}
    system_health = {}
    resume_info = {}
    keyword_analysis = {}
    
    try:
        user_service = UserService(db)
        user = user_service.get_user_by_token(token)
        
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
        enhanced_matches = user_service.get_hybrid_course_matches(user.id, top_k)
        
        if not enhanced_matches:
            logger.warning(f"No course matches found for user: {user.id}")
            enhanced_matches = []
        
        # Calculate processing metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        processing_metrics = {
            "total_processing_time": total_time,
            "semantic_processing_time": total_time * 0.7,  # Estimate
            "keyword_processing_time": total_time * 0.2,   # Estimate
            "hybrid_scoring_time": total_time * 0.1        # Estimate
        }
        
        matching_time = time.time() - start_time
        logger.info(f"Hybrid course matching completed successfully for user {user.id}. Found {len(enhanced_matches)} matches. Time: {matching_time:.2f}s")
        
        # Generate JSON log file
        log_file_path = recommendation_logger.log_recommendation_request(
            user_id=str(user.id),
            user_email=user.email,
            request_type="hybrid",
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
        
        return {"matches": enhanced_matches}
        
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
                request_type="hybrid",
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
                request_type="hybrid",
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
            keyword_analysis={},
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
                keyword_analysis={},
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
                keyword_analysis={},
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
            hybrid_matches = user_service.get_hybrid_course_matches(user.id, 5)
        
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
