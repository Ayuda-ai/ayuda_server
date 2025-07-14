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
        
        # Log additional skills usage
        if user.additional_skills and user.additional_skills.strip():
            logger.info(f"User {user.id} has additional skills: {len(user.additional_skills)} characters")
            logger.info(f"Additional skills will be included in enhanced embedding for better course matching")
        else:
            logger.info(f"User {user.id} has no additional skills - using resume embedding only")
        
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
            "completed_courses_count": len(completed_courses),
            "additional_skills": user.additional_skills or "",
            "has_additional_skills": bool(user.additional_skills and user.additional_skills.strip())
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
        
        # Get hybrid course matches using the new prerequisite-based algorithm
        enhanced_matches = recommendation_service.get_hybrid_course_matches(user.id, top_k * 2)  # Get more for better selection
        
        if not enhanced_matches:
            logger.warning(f"No course matches found for user: {user.id}")
            enhanced_matches = []
        
        # Log the diversity of recommended courses
        majors_in_recommendations = set()
        for match in enhanced_matches:
            major = match.get('metadata', {}).get('major', '')
            if major:
                majors_in_recommendations.add(major)
        
        logger.info(f"Course recommendations span {len(majors_in_recommendations)} majors: {majors_in_recommendations}")
        
        # Separate eligible and ineligible courses from the combined results
        eligible_matches = []
        ineligible_matches = []
        
        logger.info(f"Processing {len(enhanced_matches)} total course matches from algorithm")
        
        for match in enhanced_matches:
            course_id = match.get('id', '')
            prereq_status = match.get('prerequisite_status', {})
            is_eligible = match.get('eligible', True)
            
            if is_eligible and prereq_status.get('prerequisites_met', True):
                # User can take this course
                eligible_matches.append(match)
                logger.debug(f"✅ Course {course_id} is eligible")
            else:
                # User cannot take this course yet
                missing_prereqs = prereq_status.get('missing_prerequisites', [])
                if missing_prereqs:
                    match["prerequisite_message"] = f"Complete one of: {', '.join([p.get('name', '') for p in missing_prereqs])}"
                else:
                    match["prerequisite_message"] = "Prerequisites not met"
                ineligible_matches.append(match)
                logger.debug(f"❌ Course {course_id} is ineligible: {match.get('prerequisite_message', '')}")
        
        logger.info(f"Separated courses: {len(eligible_matches)} eligible, {len(ineligible_matches)} ineligible")
        
        # Sort by hybrid score and limit to top_k
        eligible_matches.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        ineligible_matches.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        eligible_matches = eligible_matches[:top_k]
        ineligible_matches = ineligible_matches[:top_k]
        
        logger.info(f"Final results: {len(eligible_matches)} eligible, {len(ineligible_matches)} ineligible (limited to top_k={top_k})")
        
        # Get all courses for analysis
        all_courses = recommendation_service.get_all_courses()
        logger.info(f"Retrieved {len(all_courses)} total courses from database")
        
        # Log the diversity of courses in the database
        majors_in_database = set()
        for course in all_courses:
            major = course.get('major', '')
            if major:
                majors_in_database.add(major)
        
        logger.info(f"Database contains courses from {len(majors_in_database)} majors: {majors_in_database}")
        
        prerequisite_analysis = {
            "prerequisites_checked": True,
            "total_courses_checked": len(all_courses),
            "eligible_count": len(eligible_matches),
            "ineligible_count": len(ineligible_matches),
            "prerequisite_checking_time": 0,  # Will be calculated below
            "user_completed_courses": completed_courses,
            "additional_skills_used": bool(user.additional_skills and user.additional_skills.strip()),
            "majors_in_recommendations": list(majors_in_recommendations),
            "majors_in_database": list(majors_in_database),
            "background_filtering_applied": True,
            "prerequisites_included": True,
            "enhanced_scoring": True,
            "algorithm": "prerequisite_based_recommendation",
            "note": "Using new prerequisite-based recommendation algorithm that prioritizes logical academic progression"
        }
        
        # Calculate processing metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate prerequisite checking time
        prerequisite_time = total_time * 0.7  # Most time is spent on prerequisite checking
        
        processing_metrics = {
            "total_processing_time": total_time,
            "prerequisite_checking_time": prerequisite_time,
            "semantic_scoring_time": total_time * 0.15,  # 15% for semantic scoring
            "keyword_scoring_time": total_time * 0.1,    # 10% for keyword scoring
            "progression_analysis_time": total_time * 0.05,  # 5% for progression analysis
            "algorithm": "prerequisite_based_recommendation"
        }
        
        # Update prerequisite analysis with timing
        prerequisite_analysis["prerequisite_checking_time"] = prerequisite_time
        
        matching_time = time.time() - start_time
        logger.info(f"Prerequisite-based recommendation completed successfully for user {user.id}. Found {len(enhanced_matches)} matches ({len(eligible_matches)} eligible, {len(ineligible_matches)} ineligible). Time: {matching_time:.2f}s")
        
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
        
        # Store recommendation in database
        try:
            from app.models.recommendation import Recommendation
            
            # Create recommendation response object
            recommendation_response = {
                "eligible_matches": eligible_matches,
                "ineligible_matches": ineligible_matches,
                "total_matches": len(all_courses) if neo4j_status else len(enhanced_matches),
                "prerequisite_analysis": prerequisite_analysis,
                "user_completed_courses": completed_courses,
                "processing_metrics": processing_metrics,
                "system_health": system_health
            }
            
            # Create new recommendation record
            recommendation = Recommendation(
                user_id=user.id,
                recommendation_type="hybrid_with_prerequisites",
                top_k_requested=top_k,
                eligible_matches=eligible_matches,
                ineligible_matches=ineligible_matches,
                total_matches=recommendation_response["total_matches"],
                processing_metrics=processing_metrics,
                system_health=system_health,
                prerequisite_analysis=prerequisite_analysis,
                user_completed_courses=completed_courses,
                resume_info=resume_info,
                keyword_analysis=keyword_analysis,
                error_info=error_info
            )
            
            db.add(recommendation)
            db.commit()
            
            logger.info(f"💾 Recommendation stored in database for user {user.id}")
            
        except Exception as e:
            logger.error(f"Failed to store recommendation in database: {str(e)}")
            # Don't fail the request if storage fails
            db.rollback()
        
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
        
        # Store failed recommendation in database
        if user:
            try:
                from app.models.recommendation import Recommendation
                
                recommendation = Recommendation(
                    user_id=user.id,
                    recommendation_type="hybrid_with_prerequisites",
                    top_k_requested=top_k,
                    eligible_matches=[],
                    ineligible_matches=[],
                    total_matches=0,
                    processing_metrics={},
                    system_health=system_health,
                    prerequisite_analysis={},
                    user_completed_courses=user.completed_courses or [],
                    resume_info=resume_info,
                    keyword_analysis=keyword_analysis,
                    error_info=error_info
                )
                
                db.add(recommendation)
                db.commit()
                logger.info(f"💾 Failed recommendation stored in database for user {user.id}")
                
            except Exception as db_error:
                logger.error(f"Failed to store failed recommendation in database: {str(db_error)}")
                db.rollback()
        
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
        
        # Store failed recommendation in database
        if user:
            try:
                from app.models.recommendation import Recommendation
                
                recommendation = Recommendation(
                    user_id=user.id,
                    recommendation_type="hybrid_with_prerequisites",
                    top_k_requested=top_k,
                    eligible_matches=[],
                    ineligible_matches=[],
                    total_matches=0,
                    processing_metrics={},
                    system_health=system_health,
                    prerequisite_analysis={},
                    user_completed_courses=user.completed_courses or [],
                    resume_info=resume_info,
                    keyword_analysis=keyword_analysis,
                    error_info=error_info
                )
                
                db.add(recommendation)
                db.commit()
                logger.info(f"💾 Failed recommendation stored in database for user {user.id}")
                
            except Exception as db_error:
                logger.error(f"Failed to store failed recommendation in database: {str(db_error)}")
                db.rollback()
        
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
    start_time = time.time()
    
    try:
        # Get user information
        user_service = UserService(db)
        recommendation_service = RecommendationService(db)
        
        # Initialize Neo4j service with proper error handling
        neo4j_service = None
        try:
            neo4j_service = Neo4jService()
            # Test connection immediately to catch issues early
            if neo4j_service.is_configured():
                neo4j_service.test_connection()
        except Exception as e:
            logger.warning(f"Neo4j service initialization failed: {str(e)}")
            neo4j_service = None
        
        user = user_service.get_user_by_token(token)
        
        # Get user's completed courses
        completed_courses = user.completed_courses or []
        
        # Get user profile data with timeout protection
        resume_text = None
        resume_embedding = None
        
        try:
            resume_text = user_service.get_resume_text_from_postgresql(user.id)
            resume_embedding = user_service.get_resume_embedding_from_postgresql(user.id)
        except Exception as e:
            logger.warning(f"Failed to get resume data: {str(e)}")
        
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
        
        # Get recommendation context with timeout protection
        recommendation_context = {}
        
        # Calculate semantic score if resume embedding exists (with timeout)
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
        
        # Get keyword matches if resume text exists (with timeout)
        if resume_text:
            try:
                extracted_keywords, matching_keywords = user_service.redis_service.get_keyword_matches(resume_text)
                recommendation_context["keyword_matches"] = list(matching_keywords)
            except Exception as e:
                logger.warning(f"Could not get keyword matches: {str(e)}")
                recommendation_context["keyword_matches"] = []
        
        # Check prerequisites with proper error handling
        if neo4j_service and neo4j_service.is_configured():
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
        
        # Try to get reasoning from LLM with improved error handling and timeout
        try:
            # Set a longer timeout for LLM calls to handle slower responses
            llm_service.timeout = 30  # Increased timeout to 30 seconds
            
            # Test LLM connection first
            llm_health = await llm_service.test_connection()
            if not llm_health.get("connected", False):
                logger.warning(f"LLM service not available: {llm_health.get('error', 'Unknown error')}")
                # Provide fallback reasoning when LLM is not available
                fallback_reasoning = _generate_fallback_reasoning(course_info, recommendation_context)
                
                processing_time = time.time() - start_time
                logger.info(f"Course reasoning completed with fallback in {processing_time:.2f}s")
                
                return CourseReasoningResponse(
                    success=True,
                    reasoning=fallback_reasoning,
                    course_name=course_info.get("course_name", ""),
                    model_used="fallback",
                    prompt_length=0,
                    response_length=len(fallback_reasoning)
                )
            
            reasoning_result = await llm_service.get_course_reasoning(
                user_profile, course_info, recommendation_context
            )
            
            if reasoning_result["success"]:
                logger.info(f"Successfully generated reasoning for course: {course_info.get('course_name', 'Unknown')}")
                
                processing_time = time.time() - start_time
                logger.info(f"Course reasoning completed in {processing_time:.2f}s")
                
                return CourseReasoningResponse(
                    success=True,
                    reasoning=reasoning_result["reasoning"],
                    course_name=course_info.get("course_name", ""),
                    model_used=reasoning_result["model_used"],
                    prompt_length=reasoning_result["prompt_length"],
                    response_length=reasoning_result["response_length"]
                )
            else:
                # LLM failed, provide fallback reasoning
                logger.warning(f"LLM reasoning failed: {reasoning_result.get('error', 'Unknown error')}")
                fallback_reasoning = _generate_fallback_reasoning(course_info, recommendation_context)
                
                processing_time = time.time() - start_time
                logger.info(f"Course reasoning completed with fallback in {processing_time:.2f}s")
                
                return CourseReasoningResponse(
                    success=True,
                    reasoning=fallback_reasoning,
                    course_name=course_info.get("course_name", ""),
                    model_used="fallback",
                    prompt_length=0,
                    response_length=len(fallback_reasoning)
                )
                
        except Exception as e:
            logger.warning(f"LLM service unavailable: {str(e)}")
            # Provide fallback reasoning when LLM is not available
            fallback_reasoning = _generate_fallback_reasoning(course_info, recommendation_context)
            
            processing_time = time.time() - start_time
            logger.info(f"Course reasoning completed with fallback in {processing_time:.2f}s")
            
            return CourseReasoningResponse(
                success=True,
                reasoning=fallback_reasoning,
                course_name=course_info.get("course_name", ""),
                model_used="fallback",
                prompt_length=0,
                response_length=len(fallback_reasoning)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in course reasoning after {processing_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up Neo4j connection if it was created
        if 'neo4j_service' in locals() and neo4j_service:
            try:
                neo4j_service.close()
            except Exception as e:
                logger.debug(f"Error closing Neo4j connection: {str(e)}")


def _generate_fallback_reasoning(course_info: dict, recommendation_context: dict) -> str:
    """
    Generate a fallback reasoning when LLM service is not available.
    
    Args:
        course_info: Course information
        recommendation_context: Recommendation context
        
    Returns:
        str: Fallback reasoning explanation
    """
    course_name = course_info.get("course_name", "this course")
    
    # Handle domains - could be a list of lists
    domains = course_info.get("domains", [])
    if isinstance(domains, list):
        # Flatten the list if it contains nested lists
        flat_domains = []
        for item in domains:
            if isinstance(item, list):
                flat_domains.extend(item)
            else:
                flat_domains.append(str(item))
        domains = flat_domains
    else:
        domains = [str(domains)] if domains else []
    
    # Handle skills - could be a list of lists
    skills = course_info.get("skills_associated", [])
    if isinstance(skills, list):
        # Flatten the list if it contains nested lists
        flat_skills = []
        for item in skills:
            if isinstance(item, list):
                flat_skills.extend(item)
            else:
                flat_skills.append(str(item))
        skills = flat_skills
    else:
        skills = [str(skills)] if skills else []
    
    semantic_score = recommendation_context.get("semantic_score", 0)
    keyword_matches = recommendation_context.get("keyword_matches", [])
    prerequisite_status = recommendation_context.get("prerequisite_status", "unknown")
    
    # Build fallback reasoning
    reasoning_parts = []
    
    # Add semantic score explanation
    if semantic_score > 0.7:
        reasoning_parts.append(f"This course was recommended based on strong semantic similarity to your background.")
    elif semantic_score > 0.5:
        reasoning_parts.append(f"This course shows good alignment with your profile and experience.")
    else:
        reasoning_parts.append(f"This course was identified as potentially relevant to your academic path.")
    
    # Add keyword matches
    if keyword_matches:
        reasoning_parts.append(f"Your resume contains relevant keywords that match this course's focus areas.")
    
    # Add domain information
    if domains:
        # Filter out any non-string items and limit to first 3 domains
        valid_domains = [str(d) for d in domains if d and str(d).strip() and str(d).lower() != 'nan'][:3]
        if valid_domains:
            domain_list = ", ".join(valid_domains)
            reasoning_parts.append(f"This course covers {domain_list} domains that align with your academic interests.")
    
    # Add skills information
    if skills:
        # Filter out any non-string items and limit to first 5 skills
        valid_skills = [str(s) for s in skills if s and str(s).strip() and str(s).lower() != 'nan'][:5]
        if valid_skills:
            skill_list = ", ".join(valid_skills)
            reasoning_parts.append(f"You'll learn valuable skills including {skill_list}.")
    
    # Add prerequisite status
    if prerequisite_status == "eligible":
        reasoning_parts.append("You meet all prerequisites for this course.")
    elif prerequisite_status == "ineligible":
        missing_prereqs = recommendation_context.get("missing_prerequisites", [])
        if missing_prereqs:
            prereq_list = ", ".join([str(p) for p in missing_prereqs[:3] if p])
            if prereq_list:
                reasoning_parts.append(f"To take this course, you'll need to complete: {prereq_list}.")
    
    # Combine all parts
    if reasoning_parts:
        reasoning = " ".join(reasoning_parts)
        reasoning += " This course represents a good next step in your academic journey."
    else:
        reasoning = f"{course_name} was recommended based on your academic profile and career goals. This course will help you develop relevant skills and knowledge for your field of study."
    
    return reasoning


@router.get("/{user_id}")
async def get_user_recommendation(
    user_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get the most recent recommendation for a specific user.
    
    This endpoint retrieves the latest recommendation stored in the database
    for the specified user. The response format matches exactly what the
    /recommendations/match_courses endpoint returns.
    
    Args:
        user_id (str): User ID to get recommendation for
        token (str): JWT access token for user authentication
        db (Session): Database session dependency
        
    Returns:
        dict: The stored recommendation response with the same format as match_courses:
            - eligible_matches: List of courses user can take
            - ineligible_matches: List of courses requiring prerequisites
            - total_matches: Total number of courses processed
            - prerequisite_analysis: Summary of prerequisite checking
            - user_completed_courses: List of user's completed courses
            - processing_metrics: Performance and timing information
            - system_health: Status of all external services
            
    Raises:
        HTTPException: If user not found (404), no recommendation exists (404), 
                      or access denied (403)
    """
    logger.info(f"Retrieving recommendation for user: {user_id}")
    
    try:
        # Verify the requesting user has access to this recommendation
        user_service = UserService(db)
        requesting_user = user_service.get_user_by_token(token)
        
        # Check if requesting user is admin or the same user
        if requesting_user.role != "ADMIN" and str(requesting_user.id) != user_id:
            logger.warning(f"User {requesting_user.id} attempted to access recommendation for user {user_id}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get the most recent recommendation for the user
        from app.models.recommendation import Recommendation
        from sqlalchemy import desc
        
        recommendation = db.query(Recommendation)\
            .filter(Recommendation.user_id == user_id)\
            .order_by(desc(Recommendation.created_at))\
            .first()
        
        if not recommendation:
            logger.warning(f"No recommendation found for user: {user_id}")
            raise HTTPException(status_code=404, detail="No recommendation found for user")
        
        # Construct the response in the exact same format as match_courses
        response = {
            "eligible_matches": recommendation.eligible_matches or [],
            "ineligible_matches": recommendation.ineligible_matches or [],
            "total_matches": recommendation.total_matches or 0,
            "prerequisite_analysis": recommendation.prerequisite_analysis or {},
            "user_completed_courses": recommendation.user_completed_courses or [],
            "processing_metrics": recommendation.processing_metrics or {},
            "system_health": recommendation.system_health or {}
        }
        
        logger.info(f"Successfully retrieved recommendation for user {user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving recommendation for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving recommendation: {str(e)}")


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
        
        # Store failed recommendation in database
        if user:
            try:
                from app.models.recommendation import Recommendation
                
                recommendation = Recommendation(
                    user_id=user.id,
                    recommendation_type="semantic",
                    top_k_requested=top_k,
                    eligible_matches=[],
                    ineligible_matches=[],
                    total_matches=0,
                    processing_metrics={},
                    system_health=system_health,
                    prerequisite_analysis={},
                    user_completed_courses=user.completed_courses or [],
                    resume_info=resume_info,
                    keyword_analysis=keyword_analysis,
                    error_info=error_info
                )
                
                db.add(recommendation)
                db.commit()
                logger.info(f"💾 Failed recommendation stored in database for user {user.id}")
                
            except Exception as db_error:
                logger.error(f"Failed to store failed recommendation in database: {str(db_error)}")
                db.rollback()
        
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
        
        # Store failed recommendation in database
        if user:
            try:
                from app.models.recommendation import Recommendation
                
                recommendation = Recommendation(
                    user_id=user.id,
                    recommendation_type="semantic",
                    top_k_requested=top_k,
                    eligible_matches=[],
                    ineligible_matches=[],
                    total_matches=0,
                    processing_metrics={},
                    system_health=system_health,
                    prerequisite_analysis={},
                    user_completed_courses=user.completed_courses or [],
                    resume_info=resume_info,
                    keyword_analysis=keyword_analysis,
                    error_info=error_info
                )
                
                db.add(recommendation)
                db.commit()
                logger.info(f"💾 Failed recommendation stored in database for user {user.id}")
                
            except Exception as db_error:
                logger.error(f"Failed to store failed recommendation in database: {str(db_error)}")
                db.rollback()
        
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


@router.get("/analytics")
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


@router.get("/llm/health", response_model=LLMHealthResponse)
async def check_llm_health():
    """
    Check the health and connectivity of the LLM service.
    
    Returns:
        LLMHealthResponse: Health status of the LLM service
    """
    health_status = await llm_service.test_connection()
    return LLMHealthResponse(**health_status)


@router.get("/llm/test")
async def test_llm_service():
    """
    Test the LLM service with a simple prompt.
    
    Returns:
        dict: Test results
    """
    try:
        # Test with a simple prompt
        test_payload = {
            "model": llm_service.model_name,
            "prompt": "Hello, who are you?",
            "stream": False
        }
        
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                llm_service.ollama_url,
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "response": result.get("response", ""),
                    "status_code": response.status_code
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "response_text": response.text,
                    "status_code": response.status_code
                }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "status_code": 500
        } 