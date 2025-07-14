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
        cosine similarity to find the most relevant courses across ALL majors.
        
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
            # Query Pinecone for similar courses - NO major filtering to get courses from all majors
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
            
            logger.debug(f"Semantic matching found {len(matches)} courses across all majors")
            return matches
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {str(e)}")
            return []
    
    def get_hybrid_course_matches(self, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get hybrid course matches using a prerequisite-based recommendation algorithm.
        
        This method implements a sophisticated recommendation algorithm that:
        1. Finds courses that the user can take based on their completed prerequisites
        2. Prioritizes courses that build on the user's completed courses
        3. Uses semantic similarity to rank eligible courses
        4. Ensures logical academic progression
        5. Considers user background and preferences
        
        Args:
            user_id (str): User ID to get resume data for
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Any]]: List of recommended courses with scores and metadata
            
        Raises:
            Exception: If user data retrieval or matching fails
        """
        try:
            # Get user information
            from app.services.user_service import UserService
            from app.services.neo4j_service import Neo4jService
            user_service = UserService(self.db)
            
            # Get user information for filtering
            from app.models.user import User
            user = user_service.db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"User not found: {user_id}")
                return []
            
            # Get user's completed courses
            completed_courses = user.completed_courses or []
            logger.info(f"User {user_id} has completed {len(completed_courses)} courses: {completed_courses}")
            
            # Special logging for new users
            if not completed_courses:
                logger.info(f"🆕 NEW USER DETECTED: User {user_id} has no completed courses - applying new user algorithm")
                logger.info(f"📚 Will prioritize foundational courses with no prerequisites")
                logger.info(f"🎯 Will focus on courses matching user's background and major: {user.major}")
            
            # Get user's enhanced embedding for semantic scoring
            embedding = user_service.create_enhanced_user_embedding(user_id)
            resume_text = user_service.get_resume_text_from_postgresql(user_id)
            
            # Initialize Neo4j service for prerequisite checking
            neo4j_service = Neo4jService()
            
            # Get all courses from database
            all_courses = self.get_all_courses()
            logger.info(f"Retrieved {len(all_courses)} total courses from database")
            
            # Step 1: Find all courses the user can take based on prerequisites
            eligible_courses = []
            ineligible_courses = []
            
            for course in all_courses:
                course_id = course.get('course_id', '')
                course_name = course.get('course_name', '')
                
                # Skip if user has already completed this course
                if course_id in completed_courses:
                    logger.debug(f"Skipping completed course: {course_id}")
                    continue
                
                # Check prerequisites using Neo4j
                if neo4j_service.is_configured():
                    try:
                        prereq_status = neo4j_service.check_prerequisites_completion(
                            course_id, completed_courses
                        )
                        
                        if prereq_status["prerequisites_met"]:
                            eligible_courses.append({
                                'course_data': course,
                                'prerequisite_status': prereq_status,
                                'eligible': True
                            })
                            logger.debug(f"✅ Course {course_id} is eligible")
                        else:
                            ineligible_courses.append({
                                'course_data': course,
                                'prerequisite_status': prereq_status,
                                'eligible': False
                            })
                            logger.debug(f"❌ Course {course_id} is ineligible: {prereq_status.get('missing_prerequisites', [])}")
                    except Exception as e:
                        logger.warning(f"Could not check prerequisites for {course_id}: {str(e)}")
                        # If Neo4j fails, assume course is eligible
                        eligible_courses.append({
                            'course_data': course,
                            'prerequisite_status': {'prerequisites_met': True},
                            'eligible': True
                        })
                else:
                    # If Neo4j not available, assume course is eligible
                    eligible_courses.append({
                        'course_data': course,
                        'prerequisite_status': {'prerequisites_met': True},
                        'eligible': True
                    })
            
            logger.info(f"Found {len(eligible_courses)} eligible courses and {len(ineligible_courses)} ineligible courses")
            
            # Step 2: Score eligible courses based on multiple factors
            scored_courses = []
            
            for course_info in eligible_courses:
                course = course_info['course_data']
                course_id = course.get('course_id', '')
                course_name = course.get('course_name', '')
                
                # Initialize scores
                semantic_score = 0.0
                keyword_score = 0.0
                prerequisite_bonus = 0.0
                progression_bonus = 0.0
                background_bonus = 0.0
                
                # Calculate semantic score if embedding exists
                if embedding:
                    try:
                        semantic_matches = self.get_semantic_course_matches(embedding, 50)
                        for match in semantic_matches:
                            if match.get('id') == course_id:
                                semantic_score = match.get('score', 0.0)
                                break
                    except Exception as e:
                        logger.warning(f"Could not calculate semantic score for {course_id}: {str(e)}")
                
                # Calculate keyword score if resume text exists
                if resume_text:
                    try:
                        extracted_keywords, matching_keywords = self.redis_service.get_keyword_matches(resume_text)
                        course_skills = course.get('skills_associated', [])
                        
                        # Calculate keyword overlap
                        if course_skills and matching_keywords:
                            course_skills_lower = [skill.lower() for skill in course_skills]
                            keyword_matches = [kw for kw in matching_keywords if any(skill in kw.lower() or kw.lower() in skill for skill in course_skills_lower)]
                            keyword_score = len(keyword_matches) / len(matching_keywords) if matching_keywords else 0.0
                    except Exception as e:
                        logger.warning(f"Could not calculate keyword score for {course_id}: {str(e)}")
                
                # Calculate prerequisite bonus (courses that build on completed courses get higher scores)
                if completed_courses:
                    # Check if this course has prerequisites that the user has completed
                    prereq_status = course_info['prerequisite_status']
                    completed_prereqs = prereq_status.get('completed_prerequisites', [])
                    total_prereqs = prereq_status.get('total_prerequisites', 0)
                    
                    if total_prereqs > 0:
                        prerequisite_bonus = len(completed_prereqs) / total_prereqs * 0.3
                        logger.debug(f"Course {course_id} prerequisite bonus: {prerequisite_bonus} ({len(completed_prereqs)}/{total_prereqs})")
                else:
                    # Special handling for users with no completed courses
                    prereq_status = course_info['prerequisite_status']
                    total_prereqs = prereq_status.get('total_prerequisites', 0)
                    
                    if total_prereqs == 0:
                        # Course has no prerequisites - perfect for new users
                        prerequisite_bonus = 0.3  # Maximum bonus for foundational courses
                        logger.debug(f"Course {course_id} is foundational (no prerequisites) - perfect for new user")
                    else:
                        # Course has prerequisites but user hasn't completed any
                        # This course is not eligible for new users
                        prerequisite_bonus = 0.0
                        logger.debug(f"Course {course_id} requires prerequisites - not suitable for new user")
                
                # Calculate progression bonus (courses that are next logical step)
                progression_bonus = self._calculate_progression_bonus(course, completed_courses, all_courses)
                
                # Calculate background bonus (filtering based on user background)
                background_bonus = self._calculate_background_bonus(course, user.major, resume_text or "")
                
                # Special handling for new users - adjust scoring weights
                if not completed_courses:
                    # For new users, prioritize foundational courses and background fit
                    hybrid_score = (
                        semantic_score * 0.35 +     # 35% semantic similarity
                        keyword_score * 0.25 +      # 25% keyword matching (higher weight)
                        prerequisite_bonus +         # 30% prerequisite bonus (foundational courses)
                        background_bonus * 0.1      # 10% background fit (higher weight)
                    )
                    logger.debug(f"New user scoring for {course_id}: semantic={semantic_score:.3f}, keyword={keyword_score:.3f}, prereq_bonus={prerequisite_bonus:.3f}, background={background_bonus:.3f}")
                else:
                    # Standard scoring for users with completed courses
                    hybrid_score = (
                        semantic_score * 0.4 +      # 40% semantic similarity
                        keyword_score * 0.2 +       # 20% keyword matching
                        prerequisite_bonus +         # 30% prerequisite completion bonus
                        progression_bonus * 0.05 +  # 5% progression bonus
                        background_bonus * 0.05     # 5% background fit
                    )
                
                # Create course match object
                course_match = {
                    'id': course_id,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'hybrid_score': hybrid_score,
                    'metadata': {
                        'course_name': course_name,
                        'description': course.get('description', ''),
                        'domains': course.get('domains', []),
                        'major': course.get('major', ''),
                        'skills_associated': course.get('skills_associated', []),
                        'type': 'course',
                        'prerequisite_bonus': prerequisite_bonus,
                        'progression_bonus': progression_bonus,
                        'background_bonus': background_bonus
                    },
                    'matching_keywords': [],
                    'prerequisite_status': course_info['prerequisite_status'],
                    'eligible': True
                }
                
                scored_courses.append(course_match)
                
                logger.debug(f"Course {course_id} scores - semantic: {semantic_score:.3f}, keyword: {keyword_score:.3f}, hybrid: {hybrid_score:.3f}")
            
            # Step 3: Sort by hybrid score and return top_k
            scored_courses.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            final_matches = scored_courses[:top_k]
            
            # Also get some ineligible courses for future planning
            ineligible_matches = []
            for course_info in ineligible_courses:
                course = course_info['course_data']
                course_id = course.get('course_id', '')
                course_name = course.get('course_name', '')
                
                # Calculate basic scores for ineligible courses
                semantic_score = 0.0
                keyword_score = 0.0
                background_bonus = 0.0
                
                # Calculate semantic score if embedding exists
                if embedding:
                    try:
                        semantic_matches = self.get_semantic_course_matches(embedding, 50)
                        for match in semantic_matches:
                            if match.get('id') == course_id:
                                semantic_score = match.get('score', 0.0)
                                break
                    except Exception as e:
                        logger.warning(f"Could not calculate semantic score for ineligible course {course_id}: {str(e)}")
                
                # Calculate keyword score if resume text exists
                if resume_text:
                    try:
                        extracted_keywords, matching_keywords = self.redis_service.get_keyword_matches(resume_text)
                        course_skills = course.get('skills_associated', [])
                        
                        if course_skills and matching_keywords:
                            course_skills_lower = [skill.lower() for skill in course_skills]
                            keyword_matches = [kw for kw in matching_keywords if any(skill in kw.lower() or kw.lower() in skill for skill in course_skills_lower)]
                            keyword_score = len(keyword_matches) / len(matching_keywords) if matching_keywords else 0.0
                    except Exception as e:
                        logger.warning(f"Could not calculate keyword score for ineligible course {course_id}: {str(e)}")
                
                # Calculate background bonus
                background_bonus = self._calculate_background_bonus(course, user.major, resume_text or "")
                
                # Calculate hybrid score for ineligible courses (lower weight since they're not immediately available)
                hybrid_score = (
                    semantic_score * 0.3 +      # 30% semantic similarity
                    keyword_score * 0.15 +      # 15% keyword matching
                    background_bonus * 0.05     # 5% background fit
                )
                
                # Create ineligible course match object
                ineligible_match = {
                    'id': course_id,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'hybrid_score': hybrid_score,
                    'metadata': {
                        'course_name': course_name,
                        'description': course.get('description', ''),
                        'domains': course.get('domains', []),
                        'major': course.get('major', ''),
                        'skills_associated': course.get('skills_associated', []),
                        'type': 'course',
                        'is_ineligible': True
                    },
                    'matching_keywords': [],
                    'prerequisite_status': course_info['prerequisite_status'],
                    'eligible': False
                }
                
                ineligible_matches.append(ineligible_match)
            
            # Sort ineligible courses by hybrid score and take top ones
            ineligible_matches.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            top_ineligible_matches = ineligible_matches[:top_k]
            
            logger.info(f"Prerequisite-based recommendation completed for user {user_id}. Found {len(final_matches)} eligible matches and {len(top_ineligible_matches)} ineligible matches from {len(eligible_courses)} eligible courses and {len(ineligible_courses)} ineligible courses.")
            
            # Log the diversity of recommended courses
            majors_in_recommendations = set()
            for match in final_matches + top_ineligible_matches:
                major = match.get('metadata', {}).get('major', '')
                if major:
                    majors_in_recommendations.add(major)
            
            logger.info(f"Recommended courses span {len(majors_in_recommendations)} majors: {majors_in_recommendations}")
            
            # Return both eligible and ineligible matches
            return final_matches + top_ineligible_matches
            
        except Exception as e:
            logger.error(f"Prerequisite-based recommendation failed: {str(e)}")
            return []
    
    def _calculate_progression_bonus(self, course: Dict[str, Any], completed_courses: List[str], all_courses: List[Dict[str, Any]]) -> float:
        """
        Calculate progression bonus for courses that are logical next steps.
        
        Args:
            course: Course data
            completed_courses: User's completed courses
            all_courses: All available courses
            
        Returns:
            float: Progression bonus score (0.0 to 0.3)
        """
        course_id = course.get('course_id', '')
        course_name = course.get('course_name', '').lower()
        
        if not completed_courses:
            # Special handling for new users
            # Give bonus to foundational courses that are prerequisites for many other courses
            prereq_count = 0
            for other_course in all_courses:
                other_prereqs = other_course.get('prerequisites', [])
                if course_id in other_prereqs:
                    prereq_count += 1
            
            # Bonus based on how many courses this course is a prerequisite for
            if prereq_count >= 5:
                return 0.2  # High bonus for very foundational courses
            elif prereq_count >= 2:
                return 0.1  # Medium bonus for foundational courses
            elif prereq_count == 1:
                return 0.05  # Small bonus for basic prerequisites
            else:
                return 0.0  # No bonus for courses that aren't prerequisites
        
        # Standard logic for users with completed courses
        # Check if this course is a prerequisite for any of the user's completed courses
        # This would indicate it's a foundational course that should have been taken earlier
        foundational_penalty = 0.0
        
        # Check if this course builds on completed courses (positive progression)
        progression_bonus = 0.0
        
        # Look for courses that have this course as a prerequisite
        for other_course in all_courses:
            other_prereqs = other_course.get('prerequisites', [])
            if course_id in other_prereqs:
                # This course is a prerequisite for another course
                if other_course.get('course_id') in completed_courses:
                    # The user has completed a course that requires this course
                    # This is a foundational course that should have been taken earlier
                    foundational_penalty = 0.1
                    logger.debug(f"Course {course_id} is foundational (prerequisite for completed course)")
                else:
                    # This course is a prerequisite for a course the user hasn't taken
                    # This is good progression
                    progression_bonus += 0.05
                    logger.debug(f"Course {course_id} is prerequisite for future course")
        
        # Check for advanced course indicators
        advanced_indicators = ['advanced', 'advanced', 'senior', 'capstone', 'project', 'seminar']
        is_advanced = any(indicator in course_name for indicator in advanced_indicators)
        
        if is_advanced and len(completed_courses) < 5:
            # Penalize advanced courses for users with few completed courses
            foundational_penalty += 0.1
        
        return max(0.0, progression_bonus - foundational_penalty)
    
    def _calculate_background_bonus(self, course: Dict[str, Any], user_major: str, resume_text: str) -> float:
        """
        Calculate background bonus based on user's major and background.
        
        Args:
            course: Course data
            user_major: User's major
            resume_text: User's resume text
            
        Returns:
            float: Background bonus score (0.0 to 0.3)
        """
        bonus = 0.0
        
        # Major alignment bonus
        course_major = course.get('major', '')
        if course_major == user_major:
            bonus += 0.1
            logger.debug(f"Major alignment bonus for {course.get('course_id')}")
        
        # Research background check
        if resume_text:
            research_keywords = ['research', 'thesis', 'publication', 'paper', 'journal', 'conference']
            resume_lower = resume_text.lower()
            research_count = sum(1 for keyword in research_keywords if keyword in resume_lower)
            
            course_name = course.get('course_name', '').lower()
            research_course_indicators = ['thesis', 'research', 'capstone', 'seminar']
            is_research_course = any(indicator in course_name for indicator in research_course_indicators)
            
            if is_research_course and research_count >= 2:
                bonus += 0.1
                logger.debug(f"Research background bonus for {course.get('course_id')}")
            elif is_research_course and research_count < 2:
                bonus -= 0.1
                logger.debug(f"Research course penalty for {course.get('course_id')}")
        
        return max(0.0, min(0.3, bonus))
    
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

    def filter_courses_by_user_background(self, courses: List[Dict[str, Any]], user_major: str, resume_text: str = "") -> List[Dict[str, Any]]:
        """
        Filter courses based on user background to ensure logical recommendations.
        
        This method applies intelligent filtering to ensure recommendations are logical:
        1. Filter out research/thesis courses for users without research background
        2. Ensure prerequisites are recommended before advanced courses
        3. Consider user's major and background
        
        Args:
            courses (List[Dict[str, Any]]): List of course matches
            user_major (str): User's major
            resume_text (str): User's resume text for background analysis
            
        Returns:
            List[Dict[str, Any]]: Filtered and enhanced course list
        """
        logger.info(f"Filtering {len(courses)} courses based on user background")
        
        if not courses:
            return courses
        
        # Keywords that indicate research background
        research_keywords = {
            'research', 'thesis', 'dissertation', 'publication', 'paper', 'journal',
            'conference', 'academic', 'scholarly', 'investigation', 'study', 'analysis',
            'methodology', 'experiment', 'laboratory', 'lab', 'scientific', 'peer-reviewed'
        }
        
        # Course types that require research background
        research_course_indicators = {
            'thesis', 'research', 'dissertation', 'capstone', 'project', 'seminar',
            'advanced', 'graduate', 'masters', 'doctoral', 'phd'
        }
        
        # Check if user has research background
        has_research_background = False
        if resume_text:
            resume_lower = resume_text.lower()
            research_matches = research_keywords.intersection(set(resume_lower.split()))
            has_research_background = len(research_matches) >= 2  # At least 2 research keywords
        
        logger.info(f"User research background: {has_research_background}")
        
        filtered_courses = []
        
        for course in courses:
            course_name = course.get('metadata', {}).get('course_name', '').lower()
            course_id = course.get('id', '').lower()
            
            # Check if this is a research/thesis course
            is_research_course = any(indicator in course_name for indicator in research_course_indicators)
            
            # Filter out research courses for users without research background
            if is_research_course and not has_research_background:
                logger.debug(f"Filtering out research course for non-research user: {course.get('id')}")
                continue
            
            # Add the course to filtered list
            filtered_courses.append(course)
        
        logger.info(f"Filtered courses: {len(filtered_courses)} remaining out of {len(courses)}")
        return filtered_courses

    def ensure_prerequisites_included(self, courses: List[Dict[str, Any]], user_completed_courses: List[str]) -> List[Dict[str, Any]]:
        """
        Ensure that prerequisites are included in recommendations when advanced courses are recommended.
        
        This method analyzes the recommended courses and ensures that if an advanced course
        is recommended, its prerequisites are also included in the recommendations.
        
        Args:
            courses (List[Dict[str, Any]]): List of recommended courses
            user_completed_courses (List[str]): User's completed courses
            
        Returns:
            List[Dict[str, Any]]: Enhanced course list with prerequisites included
        """
        logger.info(f"Ensuring prerequisites are included for {len(courses)} recommended courses")
        
        if not courses:
            return courses
        
        # Get all courses from database to find prerequisites
        all_courses = self.get_all_courses()
        course_prerequisites = {}
        
        # Build prerequisite mapping
        for course in all_courses:
            course_id = course.get('course_id', '')
            prerequisites = course.get('prerequisites', [])
            if prerequisites:
                course_prerequisites[course_id] = prerequisites
        
        # Find missing prerequisites for recommended courses
        missing_prerequisites = set()
        
        for course in courses:
            course_id = course.get('id', '')
            if course_id in course_prerequisites:
                prereqs = course_prerequisites[course_id]
                for prereq in prereqs:
                    if prereq not in user_completed_courses and prereq not in missing_prerequisites:
                        missing_prerequisites.add(prereq)
                        logger.debug(f"Adding missing prerequisite: {prereq} for course: {course_id}")
        
        # Add missing prerequisites to recommendations
        enhanced_courses = courses.copy()
        
        for prereq_id in missing_prerequisites:
            # Find the prerequisite course in all courses
            prereq_course = None
            for course in all_courses:
                if course.get('course_id') == prereq_id:
                    prereq_course = {
                        'id': course.get('course_id'),
                        'score': 0.8,  # High score for prerequisite
                        'metadata': {
                            'course_name': course.get('course_name', ''),
                            'major': course.get('major', ''),
                            'domains': course.get('domains', []),
                            'skills_associated': course.get('skills_associated', []),
                            'type': 'course',
                            'is_prerequisite': True
                        },
                        'semantic_score': 0.8,
                        'keyword_score': 0.8,
                        'hybrid_score': 0.8,
                        'matching_keywords': []
                    }
                    break
            
            if prereq_course:
                enhanced_courses.append(prereq_course)
                logger.info(f"Added prerequisite course: {prereq_id}")
        
        logger.info(f"Enhanced recommendations: {len(enhanced_courses)} courses (added {len(missing_prerequisites)} prerequisites)")
        return enhanced_courses 