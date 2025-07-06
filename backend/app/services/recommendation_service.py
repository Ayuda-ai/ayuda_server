"""
Recommendation Service

This service handles all course matching and recommendation logic.
It provides semantic, keyword, and hybrid matching capabilities with
prerequisite checking using Neo4j.

The service is responsible for:
- Semantic similarity matching using Pinecone
- Keyword matching using Redis
- Hybrid scoring combining semantic and keyword approaches
- Course data retrieval and processing
- Prerequisite checking integration
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.services.redis_service import RedisService
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Service for handling course recommendations and matching.
    
    This service provides comprehensive course matching capabilities including:
    - Semantic similarity matching using Pinecone embeddings
    - Keyword matching using Redis
    - Hybrid scoring combining multiple approaches
    - Course data retrieval and processing
    """
    
    def __init__(self, db: Session):
        """
        Initialize the recommendation service.
        
        Args:
            db (Session): Database session for course data access
        """
        self.db = db
        self.redis_service = RedisService()
        self.index = None
        self._initialize_pinecone()
        logger.debug("RecommendationService initialized with database session and Redis service")
    
    def _initialize_pinecone(self):
        """
        Initialize Pinecone connection for semantic matching.
        
        This method sets up the Pinecone index connection for semantic
        similarity search between resume embeddings and course embeddings.
        """
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index = pc.Index(settings.pinecone_index)
            logger.info(f"Pinecone initialized successfully with index: {settings.pinecone_index}")
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone: {str(e)}")
            self.index = None
    
    def get_semantic_course_matches(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get semantic course matches using Pinecone similarity search.
        
        This method performs semantic similarity search between the provided
        embedding and all course embeddings stored in Pinecone. It uses
        cosine similarity to find the most relevant courses.
        
        Args:
            embedding (List[float]): 384-dimensional embedding vector
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Any]]: List of matching courses with scores and metadata
            
        Raises:
            Exception: If Pinecone is not available or search fails
        """
        if not self.index:
            logger.warning("Pinecone not available for semantic matching")
            return []
        
        try:
            # Query Pinecone for similar courses
            query_response = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            matches = []
            for match in query_response.matches:
                course_data = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                matches.append(course_data)
            
            logger.debug(f"Semantic matching found {len(matches)} courses")
            return matches
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {str(e)}")
            return []
    
    def get_hybrid_course_matches(self, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get hybrid course matches combining semantic and keyword approaches.
        
        This method performs a comprehensive course matching approach that combines:
        1. Semantic similarity search between resume embeddings and course embeddings
        2. Keyword matching between resume text and course skills using Redis
        3. Hybrid scoring with configurable weights (70% semantic, 30% keyword by default)
        
        Args:
            user_id (str): User ID to get resume data for
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Any]]: List of enhanced course matches with hybrid scores
            
        Raises:
            Exception: If user data retrieval or matching fails
        """
        try:
            # Get user's resume embedding and text
            from app.services.user_service import UserService
            user_service = UserService(self.db)
            
            embedding = user_service.get_resume_embedding_from_postgresql(user_id)
            resume_text = user_service.get_resume_text_from_postgresql(user_id)
            
            if not embedding:
                logger.warning(f"No resume embedding found for user: {user_id}")
                return []
            
            # Get semantic matches first
            semantic_matches = self.get_semantic_course_matches(embedding, top_k * 3)  # Get more candidates for better hybrid matching
            
            if not semantic_matches:
                logger.warning(f"No semantic matches found for user: {user_id}")
                return []
            
            # Enhance semantic matches with keyword scoring
            enhanced_matches = self.redis_service.enhance_semantic_matches_with_keywords(
                semantic_matches, resume_text
            )
            
            # Sort by hybrid score and return top_k
            enhanced_matches.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            final_matches = enhanced_matches[:top_k]
            
            logger.info(f"Hybrid course matching completed for user {user_id}. Found {len(final_matches)} matches.")
            return final_matches
            
        except Exception as e:
            logger.error(f"Hybrid course matching failed: {str(e)}")
            return []
    
    def get_all_courses(self) -> List[Dict[str, Any]]:
        """
        Get all courses from the database for comprehensive prerequisite checking.
        
        This method retrieves all courses from the database to ensure that
        courses with no prerequisites are included in recommendations, even
        if they have lower hybrid scores.
        
        Returns:
            List[Dict[str, Any]]: List of all courses with their metadata
        """
        logger.info("Retrieving all courses from database for comprehensive prerequisite checking")
        try:
            from app.models.course import Course
            
            courses = self.db.query(Course).all()
            course_list = []
            
            for course in courses:
                # Parse domains and skills_associated from string to list
                domains = []
                if course.domains:
                    try:
                        # Handle the string format: "['Software Development', 'Data']"
                        domains_str = course.domains.strip()
                        if domains_str.startswith('[') and domains_str.endswith(']'):
                            # Remove outer brackets and split by comma
                            domains_content = domains_str[1:-1]
                            domains = [d.strip().strip("'\"") for d in domains_content.split(',')]
                            # Filter out 'nan' values
                            domains = [d for d in domains if d.lower() != 'nan']
                        else:
                            domains = [course.domains]
                    except Exception as e:
                        logger.warning(f"Error parsing domains for course {course.course_id}: {str(e)}")
                        domains = [course.domains] if course.domains else []
                
                skills_associated = []
                if course.skills_associated:
                    try:
                        # Handle the string format: "['python', 'java', 'databases']"
                        skills_str = course.skills_associated.strip()
                        if skills_str.startswith('[') and skills_str.endswith(']'):
                            # Remove outer brackets and split by comma
                            skills_content = skills_str[1:-1]
                            skills_associated = [s.strip().strip("'\"") for s in skills_content.split(',')]
                            # Filter out 'nan' values
                            skills_associated = [s for s in skills_associated if s.lower() != 'nan']
                        else:
                            skills_associated = [course.skills_associated]
                    except Exception as e:
                        logger.warning(f"Error parsing skills for course {course.course_id}: {str(e)}")
                        skills_associated = [course.skills_associated] if course.skills_associated else []
                
                course_data = {
                    "id": str(course.id),
                    "course_id": course.course_id,
                    "course_name": course.course_name,
                    "domains": domains,
                    "major": course.major,
                    "skills_associated": skills_associated,
                    "prerequisites": course.prerequisites,
                    "description": course.course_description
                }
                course_list.append(course_data)
            
            logger.info(f"Retrieved {len(course_list)} courses from database")
            return course_list
            
        except Exception as e:
            logger.error(f"Error retrieving courses from database: {str(e)}")
            return []
    
    def get_course_by_id(self, course_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific course by its ID.
        
        Args:
            course_id (str): Course ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Course data if found, None otherwise
        """
        try:
            from app.models.course import Course
            
            course = self.db.query(Course).filter(Course.course_id == course_id).first()
            if not course:
                return None
            
            # Parse domains and skills_associated
            domains = []
            if course.domains:
                try:
                    domains_str = course.domains.strip()
                    if domains_str.startswith('[') and domains_str.endswith(']'):
                        domains_content = domains_str[1:-1]
                        domains = [d.strip().strip("'\"") for d in domains_content.split(',')]
                        domains = [d for d in domains if d.lower() != 'nan']
                    else:
                        domains = [course.domains]
                except Exception:
                    domains = [course.domains] if course.domains else []
            
            skills_associated = []
            if course.skills_associated:
                try:
                    skills_str = course.skills_associated.strip()
                    if skills_str.startswith('[') and skills_str.endswith(']'):
                        skills_content = skills_str[1:-1]
                        skills_associated = [s.strip().strip("'\"") for s in skills_content.split(',')]
                        skills_associated = [s for s in skills_associated if s.lower() != 'nan']
                    else:
                        skills_associated = [course.skills_associated]
                except Exception:
                    skills_associated = [course.skills_associated] if course.skills_associated else []
            
            return {
                "id": str(course.id),
                "course_id": course.course_id,
                "course_name": course.course_name,
                "domains": domains,
                "major": course.major,
                "skills_associated": skills_associated,
                "prerequisites": course.prerequisites,
                "description": course.course_description
            }
            
        except Exception as e:
            logger.error(f"Error retrieving course {course_id}: {str(e)}")
            return None
    
    def get_courses_by_major(self, major: str) -> List[Dict[str, Any]]:
        """
        Get all courses for a specific major.
        
        Args:
            major (str): Major code to filter courses by
            
        Returns:
            List[Dict[str, Any]]: List of courses for the specified major
        """
        try:
            from app.models.course import Course
            
            courses = self.db.query(Course).filter(Course.major == major).all()
            course_list = []
            
            for course in courses:
                # Parse domains and skills_associated
                domains = []
                if course.domains:
                    try:
                        domains_str = course.domains.strip()
                        if domains_str.startswith('[') and domains_str.endswith(']'):
                            domains_content = domains_str[1:-1]
                            domains = [d.strip().strip("'\"") for d in domains_content.split(',')]
                            domains = [d for d in domains if d.lower() != 'nan']
                        else:
                            domains = [course.domains]
                    except Exception:
                        domains = [course.domains] if course.domains else []
                
                skills_associated = []
                if course.skills_associated:
                    try:
                        skills_str = course.skills_associated.strip()
                        if skills_str.startswith('[') and skills_str.endswith(']'):
                            skills_content = skills_str[1:-1]
                            skills_associated = [s.strip().strip("'\"") for s in skills_content.split(',')]
                            skills_associated = [s for s in skills_associated if s.lower() != 'nan']
                        else:
                            skills_associated = [course.skills_associated]
                    except Exception:
                        skills_associated = [course.skills_associated] if course.skills_associated else []
                
                course_data = {
                    "id": str(course.id),
                    "course_id": course.course_id,
                    "course_name": course.course_name,
                    "domains": domains,
                    "major": course.major,
                    "skills_associated": skills_associated,
                    "prerequisites": course.prerequisites,
                    "description": course.course_description
                }
                course_list.append(course_data)
            
            logger.info(f"Retrieved {len(course_list)} courses for major: {major}")
            return course_list
            
        except Exception as e:
            logger.error(f"Error retrieving courses for major {major}: {str(e)}")
            return []
    
    def search_courses(self, query: str, major: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search courses by name, description, or skills.
        
        This method performs a simple text-based search on course names,
        descriptions, and skills to find relevant courses.
        
        Args:
            query (str): Search query string
            major (Optional[str]): Major to filter by (optional)
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching courses
        """
        try:
            from app.models.course import Course
            from sqlalchemy import or_
            
            # Build query filter
            search_filter = or_(
                Course.course_name.ilike(f"%{query}%"),
                Course.course_description.ilike(f"%{query}%"),
                Course.skills_associated.ilike(f"%{query}%"),
                Course.domains.ilike(f"%{query}%")
            )
            
            # Add major filter if specified
            if major:
                search_filter = search_filter & (Course.major == major)
            
            courses = self.db.query(Course).filter(search_filter).limit(limit).all()
            course_list = []
            
            for course in courses:
                # Parse domains and skills_associated
                domains = []
                if course.domains:
                    try:
                        domains_str = course.domains.strip()
                        if domains_str.startswith('[') and domains_str.endswith(']'):
                            domains_content = domains_str[1:-1]
                            domains = [d.strip().strip("'\"") for d in domains_content.split(',')]
                            domains = [d for d in domains if d.lower() != 'nan']
                        else:
                            domains = [course.domains]
                    except Exception:
                        domains = [course.domains] if course.domains else []
                
                skills_associated = []
                if course.skills_associated:
                    try:
                        skills_str = course.skills_associated.strip()
                        if skills_str.startswith('[') and skills_str.endswith(']'):
                            skills_content = skills_str[1:-1]
                            skills_associated = [s.strip().strip("'\"") for s in skills_content.split(',')]
                            skills_associated = [s for s in skills_associated if s.lower() != 'nan']
                        else:
                            skills_associated = [course.skills_associated]
                    except Exception:
                        skills_associated = [course.skills_associated] if course.skills_associated else []
                
                course_data = {
                    "id": str(course.id),
                    "course_id": course.course_id,
                    "course_name": course.course_name,
                    "domains": domains,
                    "major": course.major,
                    "skills_associated": skills_associated,
                    "prerequisites": course.prerequisites,
                    "description": course.course_description
                }
                course_list.append(course_data)
            
            logger.info(f"Search found {len(course_list)} courses for query: {query}")
            return course_list
            
        except Exception as e:
            logger.error(f"Error searching courses: {str(e)}")
            return [] 