import logging
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, func, or_, and_, case
from app.models.course import Course
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class CourseSearchService:
    """
    Service for performing fuzzy course searches across multiple fields.
    
    This service provides comprehensive course search functionality using:
    1. PostgreSQL full-text search for semantic matching
    2. ILIKE queries for partial text matching
    3. Combined scoring for result ranking
    4. Pagination support for large result sets
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def search_courses(
        self, 
        query: str, 
        limit: int = 10, 
        offset: int = 0,
        major_filter: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Perform fuzzy search across course_id, course_name, and course_description.
        
        Args:
            query (str): Search query string
            limit (int): Maximum number of results to return (default: 10, max: 100)
            offset (int): Number of results to skip for pagination (default: 0)
            major_filter (str, optional): Filter results by major (CSYE, INFO, DAMG)
            
        Returns:
            Dict containing:
                - courses: List of matching courses with scores
                - total_count: Total number of matching courses
                - has_more: Boolean indicating if more results exist
                - search_metadata: Information about the search
        """
        try:
            # Validate and sanitize inputs
            query = query.strip()
            if not query:
                return self._empty_result()
            
            # Clamp limit to reasonable bounds
            limit = max(1, min(limit, 100))
            offset = max(0, offset)
            
            logger.info(f"Searching courses with query: '{query}', limit: {limit}, offset: {offset}")
            
            # Build search query with multiple matching strategies
            search_results = self._perform_search(query, limit + 1, offset, major_filter)
            
            # Determine if there are more results
            has_more = len(search_results) > limit
            if has_more:
                search_results = search_results[:limit]
            
            # Get total count for pagination
            total_count = self._get_total_count(query, major_filter)
            
            # Format results
            formatted_results = self._format_search_results(search_results)
            
            # Prepare response
            response = {
                "courses": formatted_results,
                "total_count": total_count,
                "has_more": has_more,
                "search_metadata": {
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                    "major_filter": major_filter,
                    "results_returned": len(formatted_results)
                }
            }
            
            logger.info(f"Search completed. Found {total_count} total matches, returning {len(formatted_results)} results")
            return response
            
        except Exception as e:
            logger.error(f"Error in course search: {str(e)}")
            raise
    
    def _perform_search(
        self, 
        query: str, 
        limit: int, 
        offset: int, 
        major_filter: Optional[str]
    ) -> List[Tuple[Course, float]]:
        """
        Perform the actual search using multiple strategies.
        
        Returns:
            List of tuples containing (Course, score)
        """
        # Build base query
        base_query = self.db.query(Course)
        
        # Apply major filter if specified
        if major_filter:
            base_query = base_query.filter(Course.major.ilike(f"%{major_filter}%"))
        
        # Create search conditions using multiple strategies
        search_conditions = self._build_search_conditions(query)
        
        # Apply search conditions
        base_query = base_query.filter(or_(*search_conditions))
        
        # Add scoring and ordering
        scored_query = self._add_search_scoring(base_query, query)
        
        # Apply pagination
        results = scored_query.offset(offset).limit(limit).all()
        
        return results
    
    def _build_search_conditions(self, query: str) -> List:
        """
        Build search conditions using multiple matching strategies.
        """
        conditions = []
        
        # Strategy 1: Exact match in course_id (highest priority)
        conditions.append(Course.course_id.ilike(f"%{query}%"))
        
        # Strategy 2: Partial match in course_name
        conditions.append(Course.course_name.ilike(f"%{query}%"))
        
        # Strategy 3: Partial match in course_description
        conditions.append(Course.course_description.ilike(f"%{query}%"))
        
        # Strategy 4: Word boundary matches for better precision
        words = query.split()
        for word in words:
            if len(word) >= 3:  # Only search for words with 3+ characters
                conditions.append(Course.course_name.ilike(f"%{word}%"))
                conditions.append(Course.course_description.ilike(f"%{word}%"))
        
        return conditions
    
    def _add_search_scoring(self, query, search_query: str):
        """
        Add scoring to rank results by relevance.
        """
        from sqlalchemy import case
        
        # New SQLAlchemy case syntax: positional arguments instead of list
        score_expr = func.coalesce(
            case(
                (Course.course_id.ilike(f"%{search_query}%"), 100.0),
                else_=0.0
            ) +
            case(
                (Course.course_name.ilike(f"%{search_query}%"), 50.0),
                else_=0.0
            ) +
            case(
                (Course.course_description.ilike(f"%{search_query}%"), 25.0),
                else_=0.0
            ),
            0.0
        )
        
        return query.add_columns(score_expr.label('search_score')).order_by(
            score_expr.desc(),
            Course.course_id.asc()
        )
    
    def _get_total_count(self, query: str, major_filter: Optional[str]) -> int:
        """
        Get total count of matching courses for pagination.
        """
        try:
            base_query = self.db.query(func.count(Course.id))
            
            # Apply major filter if specified
            if major_filter:
                base_query = base_query.filter(Course.major.ilike(f"%{major_filter}%"))
            
            # Apply search conditions
            search_conditions = self._build_search_conditions(query)
            base_query = base_query.filter(or_(*search_conditions))
            
            return base_query.scalar() or 0
            
        except Exception as e:
            logger.error(f"Error getting total count: {str(e)}")
            return 0
    
    def _format_search_results(self, results: List[Tuple[Course, float]]) -> List[Dict]:
        """
        Format search results for API response.
        """
        formatted_results = []
        
        for course, score in results:
            formatted_course = {
                "id": str(course.id),
                "course_id": course.course_id,
                "course_name": course.course_name,
                "course_description": course.course_description,
                "major": course.major,
                "domains": course.domains or [],
                "skills_associated": course.skills_associated or [],
                "prerequisites": course.prerequisites or [],
                "search_score": float(score),
                "created_at": course.created_at.isoformat() if course.created_at else None
            }
            formatted_results.append(formatted_course)
        
        return formatted_results
    
    def _empty_result(self) -> Dict[str, any]:
        """
        Return empty search result structure.
        """
        return {
            "courses": [],
            "total_count": 0,
            "has_more": False,
            "search_metadata": {
                "query": "",
                "limit": 10,
                "offset": 0,
                "major_filter": None,
                "results_returned": 0
            }
        }
    
    def get_course_by_id(self, course_id: str) -> Optional[Course]:
        """
        Get a specific course by its course_id.
        
        Args:
            course_id (str): The course ID to search for
            
        Returns:
            Course object if found, None otherwise
        """
        try:
            return self.db.query(Course).filter(Course.course_id == course_id).first()
        except Exception as e:
            logger.error(f"Error getting course by ID {course_id}: {str(e)}")
            return None
    
    def get_courses_by_major(self, major: str, limit: int = 50) -> List[Course]:
        """
        Get all courses for a specific major.
        
        Args:
            major (str): The major to filter by (CSYE, INFO, DAMG)
            limit (int): Maximum number of courses to return
            
        Returns:
            List of Course objects
        """
        try:
            return self.db.query(Course).filter(
                Course.major.ilike(f"%{major}%")
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting courses by major {major}: {str(e)}")
            return [] 