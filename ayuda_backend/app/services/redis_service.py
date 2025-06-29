import redis
import logging
import re
from typing import List, Dict, Set, Tuple
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisService:
    """
    Service class for handling Redis operations related to keyword matching.
    
    This service provides functionality for:
    - Extracting keywords from resume text
    - Checking keyword existence in Redis
    - Finding courses with matching keywords
    - Hybrid scoring for course recommendations
    """
    
    def __init__(self):
        """
        Initialize Redis connection using configuration settings.
        """
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis connected successfully to {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None

    def extract_keywords_from_text(self, text: str) -> Set[str]:
        """
        Extract potential keywords from resume text.
        
        This method processes the text to find words that might be stored
        as keywords in Redis. It normalizes the text and extracts meaningful
        terms that could match course-related keywords.
        
        Args:
            text (str): Resume text to extract keywords from
            
        Returns:
            Set[str]: Set of potential keywords found in the text
        """
        logger.debug(f"Extracting keywords from text of length: {len(text)}")
        
        if not text or not text.strip():
            logger.warning("Empty text provided for keyword extraction")
            return set()
        
        # Normalize text: lowercase, remove special characters, split into words
        normalized_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = set(normalized_text.split())
        
        # Filter out common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Filter words: remove stop words, short words, and numbers
        keywords = {
            word for word in words 
            if (len(word) >= 3 and 
                word not in stop_words and 
                not word.isdigit() and
                not word.isalpha() or len(word) > 3)  # Allow longer words
        }
        
        logger.debug(f"Extracted {len(keywords)} potential keywords from text")
        return keywords

    def check_keywords_in_redis(self, keywords: Set[str]) -> Set[str]:
        """
        Check which keywords exist in Redis and return the matching ones.
        
        This method queries Redis for each potential keyword to see if it
        exists as a stored keyword. It uses the 'keyword:' prefix pattern
        that matches the Go script's storage format.
        
        Args:
            keywords (Set[str]): Set of potential keywords to check
            
        Returns:
            Set[str]: Set of keywords that exist in Redis
        """
        if not self.redis_client:
            logger.warning("Redis client not available, skipping keyword check")
            return set()
        
        if not keywords:
            logger.debug("No keywords provided for Redis check")
            return set()
        
        logger.debug(f"Checking {len(keywords)} keywords in Redis")
        matching_keywords = set()
        
        try:
            for keyword in keywords:
                redis_key = f"keyword:{keyword}"
                if self.redis_client.exists(redis_key):
                    matching_keywords.add(keyword)
            
            logger.info(f"Found {len(matching_keywords)} matching keywords in Redis")
            return matching_keywords
            
        except Exception as e:
            logger.error(f"Error checking keywords in Redis: {str(e)}")
            return set()

    def calculate_keyword_score(self, resume_keywords: Set[str], course_skills: List[str]) -> float:
        """
        Calculate a keyword-based similarity score between resume and course.
        
        This method compares the keywords found in the resume with the skills
        associated with a course to calculate a similarity score.
        
        Args:
            resume_keywords (Set[str]): Keywords extracted from resume
            course_skills (List[str]): Skills associated with the course
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not resume_keywords or not course_skills:
            return 0.0
        
        # Normalize course skills to match keyword format
        normalized_course_skills = set()
        for skill in course_skills:
            # Clean and normalize skill names
            skill_clean = re.sub(r'[^\w\s]', ' ', str(skill).lower()).strip()
            if skill_clean:
                normalized_course_skills.add(skill_clean)
        
        # Find intersection between resume keywords and course skills
        intersection = resume_keywords.intersection(normalized_course_skills)
        
        if not intersection:
            return 0.0
        
        # Calculate Jaccard similarity: intersection / union
        union = resume_keywords.union(normalized_course_skills)
        similarity = len(intersection) / len(union)
        
        logger.debug(f"Keyword similarity: {len(intersection)} matches, score: {similarity:.3f}")
        return similarity

    def get_keyword_matches(self, resume_text: str) -> Tuple[Set[str], Set[str]]:
        """
        Extract keywords from resume text and find matches in Redis.
        
        This method combines keyword extraction and Redis lookup to find
        relevant keywords for course matching.
        
        Args:
            resume_text (str): Resume text to process
            
        Returns:
            Tuple[Set[str], Set[str]]: (all_extracted_keywords, redis_matching_keywords)
        """
        logger.info("Processing resume text for keyword matching")
        
        # Extract keywords from resume text
        extracted_keywords = self.extract_keywords_from_text(resume_text)
        
        # Check which keywords exist in Redis
        matching_keywords = self.check_keywords_in_redis(extracted_keywords)
        
        logger.info(f"Keyword processing complete: {len(extracted_keywords)} extracted, {len(matching_keywords)} matched in Redis")
        
        return extracted_keywords, matching_keywords

    def calculate_hybrid_score(self, semantic_score: float, keyword_score: float, 
                             semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> float:
        """
        Calculate a hybrid score combining semantic and keyword similarity.
        
        This method combines the semantic similarity score from embeddings
        with the keyword matching score to create a more comprehensive
        recommendation score.
        
        Args:
            semantic_score (float): Similarity score from embedding comparison (0-1)
            keyword_score (float): Similarity score from keyword matching (0-1)
            semantic_weight (float): Weight for semantic score (default: 0.7)
            keyword_weight (float): Weight for keyword score (default: 0.3)
            
        Returns:
            float: Combined hybrid score between 0 and 1
        """
        # Ensure weights sum to 1
        total_weight = semantic_weight + keyword_weight
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        
        # Calculate weighted average
        hybrid_score = (semantic_score * semantic_weight) + (keyword_score * keyword_weight)
        
        logger.debug(f"Hybrid score calculation: semantic={semantic_score:.3f}*{semantic_weight:.2f} + keyword={keyword_score:.3f}*{keyword_weight:.2f} = {hybrid_score:.3f}")
        
        return hybrid_score

    def enhance_course_matches(self, semantic_matches: List[Dict], resume_text: str) -> List[Dict]:
        """
        Enhance semantic course matches with keyword-based scoring.
        
        This method takes the results from semantic similarity search and
        enhances them with keyword matching scores to create a hybrid
        recommendation system.
        
        Args:
            semantic_matches (List[Dict]): Course matches from semantic search
            resume_text (str): Original resume text for keyword extraction
            
        Returns:
            List[Dict]: Enhanced course matches with hybrid scores
        """
        logger.info(f"Enhancing {len(semantic_matches)} semantic matches with keyword scoring")
        
        if not semantic_matches:
            return semantic_matches
        
        # Extract keywords from resume
        extracted_keywords, matching_keywords = self.get_keyword_matches(resume_text)
        
        if not matching_keywords:
            logger.info("No matching keywords found, returning semantic matches as-is")
            return semantic_matches
        
        enhanced_matches = []
        
        for match in semantic_matches:
            # Get course skills from metadata
            course_skills = match.get('metadata', {}).get('skills_associated', [])
            
            # Calculate keyword score
            keyword_score = self.calculate_keyword_score(matching_keywords, course_skills)
            
            # Get semantic score
            semantic_score = match.get('score', 0.0)
            
            # Calculate hybrid score
            hybrid_score = self.calculate_hybrid_score(semantic_score, keyword_score)
            
            # Create enhanced match
            enhanced_match = {
                'id': match['id'],
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'hybrid_score': hybrid_score,
                'metadata': match['metadata'],
                'matching_keywords': list(matching_keywords.intersection(
                    set(self.extract_keywords_from_text(' '.join(course_skills)))
                )) if course_skills else []
            }
            
            enhanced_matches.append(enhanced_match)
        
        # Sort by hybrid score (descending)
        enhanced_matches.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        logger.info(f"Enhanced {len(enhanced_matches)} course matches with keyword scoring")
        
        return enhanced_matches 