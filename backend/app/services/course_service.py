import os
import psycopg2
import logging
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.course import Course

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables for lazy loading
_pinecone_pc = None
_pinecone_index = None
_embedding_model = None

class CourseService:
    """
    Service class for course-related operations.
    
    This service handles:
    - Course retrieval from database
    - Course embedding and indexing
    - Course search functionality
    """
    
    def __init__(self):
        """Initialize the CourseService."""
        pass
    
    def get_all_courses(self, db: Session):
        """
        Get all courses from the database using SQLAlchemy.
        
        Args:
            db (Session): SQLAlchemy database session
            
        Returns:
            List[Course]: List of all courses
        """
        try:
            courses = db.query(Course).all()
            logger.info(f"Retrieved {len(courses)} courses from database")
            return courses
        except Exception as e:
            logger.error(f"Error retrieving courses: {str(e)}")
            return []
    
    def get_course_by_id(self, db: Session, course_id: str):
        """
        Get a specific course by ID.
        
        Args:
            db (Session): SQLAlchemy database session
            course_id (str): Course ID to retrieve
            
        Returns:
            Course: The course object or None if not found
        """
        try:
            course = db.query(Course).filter(Course.id == course_id).first()
            return course
        except Exception as e:
            logger.error(f"Error retrieving course {course_id}: {str(e)}")
            return None
    
    def get_courses_by_major(self, db: Session, major: str):
        """
        Get courses by major.
        
        Args:
            db (Session): SQLAlchemy database session
            major (str): Major to filter by
            
        Returns:
            List[Course]: List of courses for the specified major
        """
        try:
            courses = db.query(Course).filter(Course.major == major).all()
            logger.info(f"Retrieved {len(courses)} courses for major: {major}")
            return courses
        except Exception as e:
            logger.error(f"Error retrieving courses for major {major}: {str(e)}")
            return []

def get_pinecone_connection():
    """Lazy load Pinecone connection"""
    global _pinecone_pc, _pinecone_index
    
    if _pinecone_pc is None:
        try:
            _pinecone_pc = Pinecone(api_key=settings.pinecone_api_key)
            _pinecone_index = _pinecone_pc.Index(settings.pinecone_index)
            logger.info(f"Pinecone initialized successfully with index: {settings.pinecone_index}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            _pinecone_pc = None
            _pinecone_index = None
    
    return _pinecone_pc, _pinecone_index

def get_embedding_model():
    """Lazy load SentenceTransformer model"""
    global _embedding_model
    
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully")
    
    return _embedding_model

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432")
)

def embed_and_index_courses():
    """
    Generate embeddings for all courses in the database and store them in Pinecone.
    
    This function performs the following operations:
    1. Retrieves all courses from the PostgreSQL database
    2. Generates 384-dimensional embeddings for each course's combined_text
    3. Creates metadata for each course including type, name, major, domains, and skills
    4. Upserts all course vectors to Pinecone for similarity search
    
    The function uses the 'all-MiniLM-L6-v2' SentenceTransformer model to generate
    embeddings that are compatible with resume embeddings for course matching.
    
    Each course vector in Pinecone has:
    - ID: The course_id from the database
    - Values: 384-dimensional embedding vector
    - Metadata: Contains type="course", course_name, major, domains, and skills_associated
    
    Returns:
        None
        
    Raises:
        Exception: If Pinecone is not initialized or upsert fails
    """
    logger.info("Starting course embedding and indexing process")
    start_time = time.time()
    
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT course_id, course_name, major, domains, skills_associated, combined_text 
            FROM courses;
        """)
        rows = cursor.fetchall()
    
    logger.info(f"Retrieved {len(rows)} courses from database")
    
    upserts = []
    embedding_time = 0
    
    # Lazy load the embedding model
    model = get_embedding_model()
    
    for i, (course_id, name, major, domains, skills, text) in enumerate(rows):
        if not text:
            logger.warning(f"Skipping course {course_id} - no combined text available")
            continue
            
        logger.debug(f"Processing course {i+1}/{len(rows)}: {course_id}")
        embed_start = time.time()
        
        embedding = model.encode(text).tolist()
        embedding_time += time.time() - embed_start

        metadata = {
            "type": "course",  # Add type for filtering
            "course_name": name,
            "major": major,
            "domains": [d.strip().strip('"') for d in str(domains).strip('{}').split(',') if d and d.lower() != 'nan'],
            "skills_associated": [s.strip().strip('"') for s in str(skills).strip('{}').split(',') if s and s.lower() != 'nan']
        }

        upserts.append({
            "id": course_id,
            "values": embedding,
            "metadata": metadata
        })

    if upserts:
        logger.info(f"Upserting {len(upserts)} course vectors to Pinecone")
        pinecone_start = time.time()
        
        # Lazy load Pinecone connection
        pc, index = get_pinecone_connection()
        
        # Check if Pinecone is initialized
        if index is None:
            logger.error("Pinecone index is not initialized. Cannot upsert course vectors.")
            return
        
        # Upsert to Pinecone
        try:
            index.upsert(vectors=upserts)
            
            pinecone_time = time.time() - pinecone_start
            total_time = time.time() - start_time
            
            logger.info(f"✅ Course embedding and indexing completed successfully")
            logger.info(f"   - Courses processed: {len(upserts)}")
            logger.info(f"   - Embedding generation time: {embedding_time:.2f}s")
            logger.info(f"   - Pinecone upsert time: {pinecone_time:.2f}s")
            logger.info(f"   - Total time: {total_time:.2f}s")
        except Exception as e:
            logger.error(f"Error upserting course vectors to Pinecone: {str(e)}")
    else:
        logger.warning("No courses to embed and index")

if __name__ == "__main__":
    embed_and_index_courses()
