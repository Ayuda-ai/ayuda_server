"""
Course Recommendation API Endpoints

This module contains all course matching and recommendation endpoints.
It provides semantic, keyword, and hybrid matching capabilities with
prerequisite checking using Neo4j.

Endpoints:
- GET /recommendations/match_courses - Hybrid course matching with prerequisites
- GET /recommendations/semantic - Semantic-only course matching
- GET /recommendations/analytics - Recommendation analytics (admin only)
- GET /recommendations/debug - Debug recommendation system
- POST /recommendations/explain - Get reasoning for course recommendation
- GET /recommendations/llm/health - Check LLM service health
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
import logging
import time

from app.db.session import get_db
from app.services.user_service import UserService
from app.services.neo4j_service import Neo4jService
from app.services.recommendation_service import RecommendationService
from app.services.llm_service import LLMService
from app.schemas.reasoning import CourseReasoningRequest, CourseReasoningResponse, LLMHealthResponse
from app.core.config import settings
from app.core.dependencies import admin_required
from app.services.recommendation_logger import RecommendationLogger

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Initialize services
recommendation_logger = RecommendationLogger()
llm_service = LLMService()

# Add cleanup function for LLM service
async def cleanup_llm_service():
    """Cleanup LLM service connection pool."""
    await llm_service.close()

@router.get("/match_courses")
async def get_hybrid_course_recommendations(
    top_k: int = 5,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get hybrid course recommendations with prerequisite checking.
    
    This endpoint performs comprehensive course matching using:
    1. Semantic similarity between resume embeddings and course embeddings
    2. Keyword matching between resume text and course skills
    3. Prerequisite checking using Neo4j to determine eligibility
    
    The hybrid approach combines semantic and keyword scores with configurable weights
    (70% semantic, 30% keyword by default) and separates courses into eligible
    and ineligible based on prerequisite requirements.
    
    Args:
        top_k (int): Number of top matching courses to return (default: 5, max: 100)
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: Comprehensive recommendation results including:
            - eligible_matches: List of courses user can take (prerequisites met)
            - ineligible_matches: List of courses requiring prerequisites user hasn't completed
            - total_matches: Total number of courses processed
            - prerequisite_analysis: Summary of prerequisite checking
            - user_completed_courses: List of user's completed courses
            - processing_metrics: Performance and timing information
            - system_health: Status of all external services
            
    Raises:
        HTTPException: If user not found (404), no resume embedding exists (404), 
                      or matching fails (500)
    """
    logger.info(f"Hybrid course recommendation request with top_k={top_k}")
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
        recommendation_service = RecommendationService(db)
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
        
        # Test Pinecone connection
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
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
        enhanced_matches = recommendation_service.get_hybrid_course_matches(user.id, 50)
        
        if not enhanced_matches:
            logger.warning(f"No course matches found for user: {user.id}")
            enhanced_matches = []
        
        # Get ALL courses from database to check prerequisites comprehensively
        all_courses = recommendation_service.get_all_courses()
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
                "user_completed_courses": completed_courses,
                "note": "Semantic matching scores are base values due to Pinecone configuration. For optimal recommendations, configure valid Pinecone API key and index courses."
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
        logger.info(f"Hybrid course recommendation completed successfully for user {user.id}. Found {len(enhanced_matches)} matches ({len(eligible_matches)} eligible, {len(ineligible_matches)} ineligible). Time: {matching_time:.2f}s")
        
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
        logger.error(f"Hybrid course recommendation failed with HTTP error: {str(e)}")
        
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
        logger.error(f"Hybrid course recommendation failed with unexpected error: {str(e)}")
        
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


@router.get("/semantic")
async def get_semantic_course_recommendations(
    top_k: int = 5,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get semantic-only course recommendations.
    
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
        HTTPException: If user not found (404), no resume embedding exists (404), 
                      or matching fails (500)
    """
    logger.info(f"Semantic-only course recommendation request with top_k={top_k}")
    start_time = time.time()
    
    # Initialize tracking variables
    user = None
    semantic_matches = []
    error_info = None
    processing_metrics = {}
    system_health = {}
    resume_info = {}
    keyword_analysis = {}
    
    try:
        user_service = UserService(db)
        recommendation_service = RecommendationService(db)
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
        
        # Test Pinecone connection
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
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
        semantic_matches = recommendation_service.get_semantic_course_matches(embedding, top_k)
        
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
        logger.info(f"Semantic course recommendation completed successfully for user {user.id}. Found {len(semantic_matches)} matches. Time: {matching_time:.2f}s")
        
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
        logger.error(f"Semantic course recommendation failed with HTTP error: {str(e)}")
        
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
        logger.error(f"Semantic course recommendation failed with unexpected error: {str(e)}")
        
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


@router.get("/analytics", dependencies=[Depends(admin_required)])
async def get_recommendation_analytics(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get detailed analytics and metrics for recommendation system.
    
    This endpoint provides comprehensive analytics about the recommendation
    system, including performance metrics, algorithm effectiveness,
    and system health information. Only accessible to admin users.
    
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
        HTTPException: If user not found (404), not admin (403), 
                      or analytics generation fails (500)
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
        
        # Test Pinecone connection
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
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


@router.get("/debug")
async def debug_recommendation_system(
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
        HTTPException: If user not found (404) or debugging fails (500)
    """
    logger.info("Recommendation system debug request")
    
    try:
        user_service = UserService(db)
        recommendation_service = RecommendationService(db)
        user = user_service.get_user_by_token(token)
        
        # Get resume data
        embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        resume_text = user_service.get_resume_text_from_postgresql(user.id)
        
        if not embedding:
            raise HTTPException(status_code=404, detail="No resume embedding found for user.")
        
        # Test semantic matching
        semantic_matches = recommendation_service.get_semantic_course_matches(embedding, 5)
        
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
            hybrid_matches = recommendation_service.get_hybrid_course_matches(user.id, 50)
        
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


@router.post("/explain", response_model=CourseReasoningResponse)
async def explain_course_recommendation(
    request: CourseReasoningRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get detailed reasoning for why a specific course was recommended to the user.
    
    This endpoint provides personalized explanations for course recommendations
    based on the user's complete profile including resume, work experience,
    completed courses, and additional skills.
    
    Args:
        request (CourseReasoningRequest): Request containing course ID
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        CourseReasoningResponse: Detailed reasoning explanation
        
    Raises:
        HTTPException: If user not found (404), course not found (404),
                      or reasoning generation fails (500)
    """
    logger.info(f"Course reasoning request for course ID: {request.course_id}")
    
    try:
        # Get user information
        user_service = UserService(db)
        recommendation_service = RecommendationService(db)
        neo4j_service = Neo4jService()
        
        user = user_service.get_user_by_token(token)
        
        # Get user's completed courses
        completed_courses = user.completed_courses or []
        
        # Get user profile data
        resume_text = user_service.get_resume_text_from_postgresql(user.id)
        resume_embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        
        # Build user profile
        user_profile = {
            "resume_text": resume_text or "",
            "work_experience": "",  # Could be extracted from resume or stored separately
            "completed_courses": completed_courses,
            "additional_skills": user.additional_skills or "",
            "major": user.major,
            "university": user.university
        }
        
        # Get course information
        course_info = recommendation_service.get_course_by_id(request.course_id)
        if not course_info:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Log course info for debugging
        logger.debug(f"Course info retrieved: {course_info.get('course_name', 'Unknown')}")
        logger.debug(f"Course prerequisites: {course_info.get('prerequisites', [])}")
        logger.debug(f"Course domains: {course_info.get('domains', [])}")
        logger.debug(f"Course skills: {course_info.get('skills_associated', [])}")
        
        # Get recommendation context
        recommendation_context = {}
        
        # Calculate semantic score if resume embedding exists
        if resume_embedding:
            try:
                semantic_matches = recommendation_service.get_semantic_course_matches(
                    resume_embedding, 50
                )
                # Find this specific course in semantic matches
                for match in semantic_matches:
                    if match.get("id") == request.course_id:
                        recommendation_context["semantic_score"] = match.get("score", 0)
                        break
            except Exception as e:
                logger.warning(f"Could not calculate semantic score: {str(e)}")
                recommendation_context["semantic_score"] = 0
        
        # Get keyword matches if resume text exists
        if resume_text:
            try:
                extracted_keywords, matching_keywords = user_service.redis_service.get_keyword_matches(resume_text)
                recommendation_context["keyword_matches"] = list(matching_keywords)
            except Exception as e:
                logger.warning(f"Could not get keyword matches: {str(e)}")
                recommendation_context["keyword_matches"] = []
        
        # Check prerequisites
        if neo4j_service.is_configured():
            try:
                prereq_status = neo4j_service.check_prerequisites_completion(
                    course_info.get("course_id", ""), completed_courses
                )
                recommendation_context["prerequisite_status"] = "eligible" if prereq_status["prerequisites_met"] else "ineligible"
                recommendation_context["missing_prerequisites"] = [p.get("name", "") for p in prereq_status.get("missing_prerequisites", [])]
            except Exception as e:
                logger.warning(f"Could not check prerequisites: {str(e)}")
                recommendation_context["prerequisite_status"] = "unknown"
                recommendation_context["missing_prerequisites"] = []
        else:
            recommendation_context["prerequisite_status"] = "unknown"
            recommendation_context["missing_prerequisites"] = []
        
        # Get reasoning from LLM
        reasoning_result = await llm_service.get_course_reasoning(
            user_profile, course_info, recommendation_context
        )
        
        if not reasoning_result["success"]:
            logger.error(f"Failed to generate reasoning: {reasoning_result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate reasoning: {reasoning_result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"Successfully generated reasoning for course: {course_info.get('course_name', 'Unknown')}")
        
        return CourseReasoningResponse(
            success=True,
            reasoning=reasoning_result["reasoning"],
            course_name=course_info.get("course_name", ""),
            model_used=reasoning_result["model_used"],
            prompt_length=reasoning_result["prompt_length"],
            response_length=reasoning_result["response_length"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in course reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/llm/health", response_model=LLMHealthResponse)
async def check_llm_health():
    """
    Check the health and connectivity of the LLM service.
    
    Returns:
        LLMHealthResponse: Health status of the LLM service
    """
    health_status = llm_service.test_connection()
    return LLMHealthResponse(**health_status) 