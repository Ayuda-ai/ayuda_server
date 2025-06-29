from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from jose import JWTError, jwt
import fitz  # PyMuPDF
import docx
import io
import os
import numpy as np
import logging
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from datetime import datetime

from app.schemas.user import UserCreate
from app.models.user import User
from app.models.access_code import AccessCode
from app.models.allowed_domain import AllowedDomain
from app.core.security import get_password_hash
from app.core.config import settings
from app.services.redis_service import RedisService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone (shared with course service)
try:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)
    logger.info(f"Pinecone initialized successfully with index: {settings.pinecone_index}")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    pc = None
    index = None

# Load embedding model (shared with course service)
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully")

class UserService:
    """
    Service class for handling user-related operations including authentication,
    resume processing, and embedding management.
    
    This service provides functionality for:
    - User registration and validation
    - Resume text extraction and embedding generation
    - Storage and retrieval of resume embeddings in PostgreSQL and Pinecone
    - Course matching based on resume embeddings and keyword matching
    - Comprehensive logging and metrics for recommendation analysis
    """
    
    def __init__(self, db: Session):
        """
        Initialize UserService with a database session.
        
        Args:
            db (Session): SQLAlchemy database session for database operations
        """
        self.db = db
        self.redis_service = RedisService()
        logger.debug(f"UserService initialized with database session and Redis service")

    def validate_email_domain(self, email: str) -> None:
        """
        Validate that the email domain is in the allowed domains list.
        
        This method checks if the domain part of the email address exists
        in the allowed_domains table. If not found, raises an HTTPException.
        
        Args:
            email (str): Email address to validate
            
        Raises:
            HTTPException: If email domain is not allowed (status_code=400)
        """
        domain = email.split("@")[-1]
        if not self.db.query(AllowedDomain).filter(AllowedDomain.domain == domain).first():
            logger.warning(f"Email domain validation failed for domain: {domain}")
            raise HTTPException(status_code=400, detail="Email domain not allowed")
        logger.debug(f"Email domain validation successful for domain: {domain}")

    def validate_access_code(self, access_code: str) -> AccessCode:
        """
        Validate that the provided access code is valid and has remaining uses.
        
        This method checks if the access code exists, is active, and has
        remaining count greater than 0.
        
        Args:
            access_code (str): Access code to validate
            
        Returns:
            AccessCode: The validated access code object
            
        Raises:
            HTTPException: If access code is invalid or expired (status_code=400)
        """
        code = self.db.query(AccessCode).filter(
            AccessCode.code == access_code,
            AccessCode.count > 0,
            AccessCode.status == True
        ).first()
        if not code:
            logger.warning(f"Invalid access code attempt: {access_code}")
            raise HTTPException(status_code=400, detail="Invalid or expired access code")
        logger.debug(f"Access code validation successful for code: {access_code}")
        return code

    def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user account with validation and access code consumption.
        
        This method performs the following steps:
        1. Validates email domain
        2. Validates access code
        3. Creates new user with hashed password
        4. Decrements access code count
        5. Commits transaction to database
        
        Args:
            user_data (UserCreate): User registration data including email, password, etc.
            
        Returns:
            User: The newly created user object
            
        Raises:
            HTTPException: If validation fails (status_code=400)
        """
        logger.info(f"Creating new user with email: {user_data.email}")
        self.validate_email_domain(user_data.email)
        code = self.validate_access_code(user_data.access_code)

        new_user = User(
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            university=user_data.university,
            email=user_data.email,
            password_hash=get_password_hash(user_data.password),
            dob=user_data.dob,
            major=user_data.major
        )
        self.db.add(new_user)
        code.count -= 1
        self.db.commit()
        self.db.refresh(new_user)
        logger.info(f"User created successfully with ID: {new_user.id}")
        return new_user

    def get_user_by_token(self, token: str) -> User:
        """
        Retrieve user information from JWT token.
        
        This method decodes the JWT token to extract the user's email,
        then queries the database to find the corresponding user.
        
        Args:
            token (str): JWT token containing user information
            
        Returns:
            User: The user object corresponding to the token
            
        Raises:
            HTTPException: If token is invalid (status_code=403) or user not found (status_code=404)
        """
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
            email = payload.get("sub")
            logger.debug(f"JWT token decoded successfully for email: {email}")
        except JWTError as e:
            logger.warning(f"JWT token validation failed: {str(e)}")
            raise HTTPException(status_code=403, detail="Invalid token")

        user = self.db.query(User).filter(User.email == email).first()
        if not user:
            logger.warning(f"User not found for email: {email}")
            raise HTTPException(status_code=404, detail="User not found")
        logger.debug(f"User retrieved successfully: {user.id}")
        return user

    def extract_text_from_document(self, file: UploadFile) -> str:
        """
        Extract text content from uploaded PDF or DOCX files.
        
        This method supports both PDF (using PyMuPDF) and DOCX (using python-docx)
        file formats. It reads the file content, processes it based on the file type,
        and returns the extracted text as a string.
        
        Args:
            file (UploadFile): Uploaded file object with content and metadata
            
        Returns:
            str: Extracted text content from the document
            
        Raises:
            HTTPException: If file type is unsupported (status_code=400) or processing fails (status_code=500)
        """
        logger.info(f"Starting text extraction from file: {file.filename} ({file.content_type})")
        try:
            content = file.file.read()
            file_size = len(content)
            logger.debug(f"File read successfully, size: {file_size} bytes")

            if file.content_type == "application/pdf":
                try:
                    # Use PyMuPDF to extract text
                    pdf = fitz.open(stream=content, filetype="pdf")
                    text = ""
                    page_count = len(pdf)
                    for page in pdf:
                        text += page.get_text()
                    extracted_text = text.strip()
                    logger.info(f"PDF text extraction completed. Pages: {page_count}, Text length: {len(extracted_text)} chars")
                    return extracted_text
                except Exception as e:
                    logger.error(f"PDF processing error: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing PDF file: {str(e)}"
                    )

            elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    doc = docx.Document(io.BytesIO(content))
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    extracted_text = text.strip()
                    logger.info(f"DOCX text extraction completed. Text length: {len(extracted_text)} chars")
                    return extracted_text
                except Exception as e:
                    logger.error(f"DOCX processing error: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing DOCX file: {str(e)}"
                    )
            else:
                logger.warning(f"Unsupported file type: {file.content_type}")
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Only PDF and DOCX files are allowed."
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during text extraction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )
        finally:
            file.file.close()

    def create_resume_embedding(self, resume_text: str) -> list:
        """
        Generate embedding vector from resume text using SentenceTransformer model.
        
        This method takes resume text and converts it into a 384-dimensional
        embedding vector using the 'all-MiniLM-L6-v2' model. The embedding
        can be used for semantic similarity matching with course descriptions.
        
        Args:
            resume_text (str): Text content extracted from the resume
            
        Returns:
            list: 384-dimensional embedding vector as a list of floats
            
        Raises:
            HTTPException: If text is empty/too short (status_code=400) or embedding generation fails (status_code=500)
        """
        logger.info("Starting resume embedding generation")
        start_time = time.time()
        
        try:
            # Validate input text
            if not resume_text or not resume_text.strip():
                logger.warning("Empty resume text provided for embedding")
                raise HTTPException(
                    status_code=400,
                    detail="Resume text cannot be empty"
                )
            
            # Clean and normalize text
            cleaned_text = resume_text.strip()
            if len(cleaned_text) < 10:  # Minimum meaningful text length
                logger.warning(f"Resume text too short for meaningful embedding: {len(cleaned_text)} chars")
                raise HTTPException(
                    status_code=400,
                    detail="Resume text is too short to generate meaningful embedding"
                )
            
            logger.debug(f"Generating embedding for text of length: {len(cleaned_text)} chars")
            
            # Generate embedding
            embedding = model.encode(cleaned_text).tolist()
            generation_time = time.time() - start_time
            
            # Validate embedding
            if len(embedding) != 384:
                logger.error(f"Generated embedding has incorrect dimensions: {len(embedding)} (expected 384)")
                raise HTTPException(
                    status_code=500,
                    detail=f"Generated embedding has incorrect dimensions: {len(embedding)} (expected 384)"
                )
            
            # Check for NaN or infinite values
            if any(np.isnan(val) or np.isinf(val) for val in embedding):
                logger.error("Generated embedding contains invalid values (NaN or infinite)")
                raise HTTPException(
                    status_code=500,
                    detail="Generated embedding contains invalid values (NaN or infinite)"
                )
            
            logger.info(f"Resume embedding generated successfully. Dimensions: {len(embedding)}, Time: {generation_time:.2f}s")
            return embedding
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating resume embedding: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating resume embedding: {str(e)}"
            )

    def store_resume_embedding_pinecone(self, user_id: str, embedding: list, user: User) -> bool:
        """
        Store resume embedding vector in Pinecone with user metadata.
        
        This method creates a new Pinecone connection, prepares user metadata,
        and upserts the embedding vector with ID 'resume:{user_id}'. If a vector
        with the same ID already exists, it will be overwritten.
        
        Args:
            user_id (str): Unique identifier for the user
            embedding (list): 384-dimensional embedding vector
            user (User): User object containing metadata (name, major, university, email)
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        logger.info(f"Storing resume embedding in Pinecone for user: {user_id}")
        start_time = time.time()
        
        try:
            # Initialize Pinecone connection directly in this method
            logger.debug("Initializing Pinecone connection...")
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            logger.debug(f"Pinecone connected to index: {settings.pinecone_index}")
            
            # Prepare metadata
            metadata = {
                "type": "resume",
                "user_id": str(user_id),
                "user_name": f"{user.first_name} {user.last_name}",
                "major": user.major,
                "university": user.university,
                "email": user.email
            }
            
            logger.debug(f"Preparing to upsert vector with ID: resume:{user_id}")
            logger.debug(f"Vector dimensions: {len(embedding)}")
            logger.debug(f"Metadata: {metadata}")
            
            # Log the exact vector being sent (first few values)
            logger.debug(f"Vector preview (first 5 values): {embedding[:5]}")
            
            # Prepare the upsert payload
            upsert_payload = [{
                "id": f"resume:{user_id}",
                "values": embedding,
                "metadata": metadata
            }]
            
            logger.debug(f"Upsert payload prepared: {len(upsert_payload)} vector(s)")
            
            # Upsert to Pinecone
            logger.debug("Calling Pinecone upsert...")
            upsert_response = index.upsert(vectors=upsert_payload)
            
            storage_time = time.time() - start_time
            logger.info(f"Resume embedding stored in Pinecone successfully. Time: {storage_time:.2f}s")
            logger.debug(f"Pinecone upsert response: {upsert_response}")
            
            # Verify the storage immediately
            logger.debug("Verifying storage...")
            try:
                fetch_response = index.fetch(ids=[f"resume:{user_id}"])
                if f"resume:{user_id}" in fetch_response.vectors:
                    logger.info(f"✅ Verification successful - resume embedding found in Pinecone")
                    stored_vector = fetch_response.vectors[f"resume:{user_id}"]
                    logger.debug(f"Stored vector dimensions: {len(stored_vector.values)}")
                    logger.debug(f"Stored metadata: {stored_vector.metadata}")
                else:
                    logger.warning(f"⚠️ Verification failed - resume embedding not found in Pinecone")
            except Exception as verify_error:
                logger.error(f"Error during verification: {str(verify_error)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing resume embedding in Pinecone: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {e}")
            return False

    def get_resume_embedding_from_pinecone(self, user_id: str) -> list:
        """
        Retrieve resume embedding vector from Pinecone.
        
        This method fetches the resume embedding vector stored in Pinecone
        using the ID 'resume:{user_id}'.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            list: 384-dimensional embedding vector if found, None otherwise
        """
        logger.debug(f"Retrieving resume embedding from Pinecone for user: {user_id}")
        try:
            # Initialize Pinecone connection directly
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            
            results = index.fetch(ids=[f"resume:{user_id}"])
            if results and f"resume:{user_id}" in results.vectors:
                logger.debug(f"Resume embedding retrieved from Pinecone successfully")
                return results.vectors[f"resume:{user_id}"].values
            logger.warning(f"Resume embedding not found in Pinecone for user: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving resume embedding from Pinecone: {str(e)}")
            return None

    def delete_resume_embedding_from_pinecone(self, user_id: str) -> bool:
        """
        Delete resume embedding vector from Pinecone.
        
        This method removes the resume embedding vector from Pinecone
        using the ID 'resume:{user_id}'.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        logger.info(f"Deleting resume embedding from Pinecone for user: {user_id}")
        try:
            # Initialize Pinecone connection directly
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            
            index.delete(ids=[f"resume:{user_id}"])
            logger.info(f"Resume embedding deleted from Pinecone successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting resume embedding from Pinecone: {str(e)}")
            return False

    def store_resume_embedding_postgresql(self, user_id: str, embedding: list, resume_text: str = None) -> bool:
        """
        Store resume embedding vector and text in PostgreSQL database.
        
        This method updates the user's resume_embedding and resume_text fields
        in the database. If an embedding already exists, it will be overwritten.
        
        Args:
            user_id (str): Unique identifier for the user
            embedding (list): 384-dimensional embedding vector
            resume_text (str): Extracted resume text for keyword matching
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        logger.info(f"Storing resume embedding and text in PostgreSQL for user: {user_id}")
        start_time = time.time()
        
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"User not found for PostgreSQL storage: {user_id}")
                return False
            
            user.resume_embedding = embedding
            if resume_text:
                user.resume_text = resume_text
            
            self.db.commit()
            storage_time = time.time() - start_time
            logger.info(f"Resume embedding and text stored in PostgreSQL successfully. Time: {storage_time:.2f}s")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error storing resume embedding in PostgreSQL: {str(e)}")
            return False

    def get_resume_embedding_from_postgresql(self, user_id: str) -> list:
        """
        Retrieve resume embedding vector from PostgreSQL database.
        
        This method fetches the resume embedding stored in the user's
        resume_embedding field in the database.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            list: 384-dimensional embedding vector if found, None otherwise
        """
        logger.debug(f"Retrieving resume embedding from PostgreSQL for user: {user_id}")
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user and user.resume_embedding:
                logger.debug(f"Resume embedding retrieved from PostgreSQL successfully")
                return user.resume_embedding
            logger.warning(f"Resume embedding not found in PostgreSQL for user: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving resume embedding from PostgreSQL: {str(e)}")
            return None

    def get_resume_text_from_postgresql(self, user_id: str) -> str:
        """
        Retrieve resume text from PostgreSQL database.
        
        This method fetches the resume text stored in the user's
        resume_text field in the database.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            str: Resume text if found, None otherwise
        """
        logger.debug(f"Retrieving resume text from PostgreSQL for user: {user_id}")
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user and user.resume_text:
                logger.debug(f"Resume text retrieved from PostgreSQL successfully")
                return user.resume_text
            logger.warning(f"Resume text not found in PostgreSQL for user: {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving resume text from PostgreSQL: {str(e)}")
            return None

    def delete_resume_embedding_postgresql(self, user_id: str) -> bool:
        """
        Delete resume embedding vector and text from PostgreSQL database.
        
        This method sets the user's resume_embedding and resume_text fields to None,
        effectively removing the stored data.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        logger.info(f"Deleting resume embedding and text from PostgreSQL for user: {user_id}")
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user:
                user.resume_embedding = None
                user.resume_text = None
                self.db.commit()
                logger.info(f"Resume embedding and text deleted from PostgreSQL successfully")
                return True
            logger.warning(f"User not found for PostgreSQL deletion: {user_id}")
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting resume embedding from PostgreSQL: {str(e)}")
            return False

    def store_resume_embedding_hybrid(self, user_id: str, embedding: list, user: User, resume_text: str = None) -> dict:
        """
        Store resume embedding vector and text in both PostgreSQL and Pinecone.
        
        This method provides a hybrid storage approach, storing the embedding
        in both PostgreSQL (for quick retrieval) and Pinecone (for similarity search),
        and storing the resume text in PostgreSQL for keyword matching.
        
        Args:
            user_id (str): Unique identifier for the user
            embedding (list): 384-dimensional embedding vector
            user (User): User object containing metadata for Pinecone
            resume_text (str): Extracted resume text for keyword matching
            
        Returns:
            dict: Dictionary with success status for each storage system
                  {"postgresql": bool, "pinecone": bool}
        """
        logger.info(f"Storing resume embedding and text in hybrid storage for user: {user_id}")
        start_time = time.time()
        
        results = {
            "postgresql": False,
            "pinecone": False
        }
        
        # Store in PostgreSQL (embedding + text)
        results["postgresql"] = self.store_resume_embedding_postgresql(user_id, embedding, resume_text)
        
        # Store in Pinecone (embedding only)
        results["pinecone"] = self.store_resume_embedding_pinecone(user_id, embedding, user)
        
        total_time = time.time() - start_time
        logger.info(f"Hybrid storage completed. PostgreSQL: {results['postgresql']}, Pinecone: {results['pinecone']}, Total time: {total_time:.2f}s")
        
        return results

    def get_hybrid_course_matches(self, user_id: str, top_k: int = 5) -> list:
        """
        Get hybrid course matches combining semantic similarity and keyword matching.
        
        This method performs semantic similarity search using embeddings and
        enhances the results with keyword matching from Redis to provide
        more accurate course recommendations.
        
        Args:
            user_id (str): Unique identifier for the user
            top_k (int): Number of top matching courses to return
            
        Returns:
            list: Enhanced course matches with hybrid scores and detailed metrics
        """
        try:
            # Get resume embedding and text
            embedding = self.get_resume_embedding_from_postgresql(user_id)
            resume_text = self.get_resume_text_from_postgresql(user_id)
            
            if not embedding:
                logger.warning(f"No resume embedding found for hybrid matching - user: {user_id}")
                return []
            
            if not resume_text:
                logger.warning(f"No resume text found for keyword matching - user: {user_id}")
                # Fall back to semantic-only matching
                return self.get_semantic_course_matches(embedding, top_k)
            
            # Get semantic matches first
            semantic_matches = self.get_semantic_course_matches(embedding, top_k)
            
            if not semantic_matches:
                logger.warning(f"No semantic matches found for user: {user_id}")
                return []
            
            # Process keyword matching
            extracted_keywords, matching_keywords = self.redis_service.get_keyword_matches(resume_text)
            
            # Enhance with keyword matching
            enhanced_matches = self.redis_service.enhance_course_matches(semantic_matches, resume_text)
            
            logger.info(f"Hybrid course matching completed for user {user_id}. Found {len(enhanced_matches)} matches.")
            
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"Error in hybrid course matching: {str(e)}")
            return []

    def get_semantic_course_matches(self, embedding: list, top_k: int = 5) -> list:
        """
        Get course matches using only semantic similarity.
        
        This method performs semantic similarity search using the resume embedding
        against course embeddings in Pinecone.
        
        Args:
            embedding (list): Resume embedding vector
            top_k (int): Number of top matching courses to return
            
        Returns:
            list: Course matches with semantic scores
        """
        try:
            # Initialize Pinecone connection
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            
            # Query Pinecone for similar courses
            results = index.query(
                vector=embedding,
                filter={"type": "course"},
                top_k=top_k,
                include_metadata=True
            )
            
            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            logger.debug(f"Semantic matching found {len(matches)} courses")
            return matches
            
        except Exception as e:
            logger.error(f"Error in semantic course matching: {str(e)}")
            return []