"""
Admin Service

This service handles all admin-only operations including user management,
course management, system administration, and data synchronization.

The service is responsible for:
- User management (CRUD operations)
- Course management (CRUD operations)
- System health monitoring
- Data synchronization with external services
- System backup and restore operations
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
import uuid
from datetime import datetime

from app.models.user import User
from app.models.course import Course
from app.schemas.user import UserOut
from app.schemas.course import CourseCreate, CourseUpdate, CourseOut
from app.services.neo4j_service import Neo4jService
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class AdminService:
    """
    Service for handling admin-only operations.
    
    This service provides comprehensive administrative capabilities including:
    - User management (CRUD operations)
    - Course management (CRUD operations)
    - System health monitoring
    - Data synchronization with external services
    - System backup and restore operations
    """
    
    def __init__(self, db: Session):
        """
        Initialize the admin service.
        
        Args:
            db (Session): Database session for data access
        """
        self.db = db
        self.neo4j_service = Neo4jService()
        logger.debug("AdminService initialized with database session")
    
    # ============================================================================
    # USER MANAGEMENT METHODS
    # ============================================================================
    
    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[UserOut]:
        """
        Get all users with pagination support.
        
        Args:
            skip (int): Number of records to skip for pagination
            limit (int): Maximum number of records to return
            
        Returns:
            List[UserOut]: List of user information (password_hash excluded)
        """
        try:
            users = self.db.query(User).offset(skip).limit(limit).all()
            user_list = []
            
            for user in users:
                user_data = {
                    "id": str(user.id),
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "university": user.university,
                    "email": user.email,
                    "dob": user.dob,
                    "major": user.major,
                    "role": user.role,
                    "profile_enhanced": user.profile_enhanced,
                    "completed_courses": user.completed_courses or [],
                    "additional_skills": user.additional_skills,
                    "created_at": user.created_at
                }
                user_list.append(user_data)
            
            logger.info(f"Retrieved {len(user_list)} users for admin")
            return user_list
            
        except Exception as e:
            logger.error(f"Error retrieving users: {str(e)}")
            raise
    
    def get_user_by_id(self, user_id: str) -> Optional[UserOut]:
        """
        Get specific user by ID.
        
        Args:
            user_id (str): UUID of the user to retrieve
            
        Returns:
            Optional[UserOut]: User information if found, None otherwise
        """
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return None
            
            user_data = {
                "id": str(user.id),
                "first_name": user.first_name,
                "last_name": user.last_name,
                "university": user.university,
                "email": user.email,
                "dob": user.dob,
                "major": user.major,
                "role": user.role,
                "profile_enhanced": user.profile_enhanced,
                "completed_courses": user.completed_courses or [],
                "additional_skills": user.additional_skills,
                "created_at": user.created_at
            }
            
            logger.info(f"Retrieved user {user_id} for admin")
            return user_data
            
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {str(e)}")
            raise
    
    def update_user(self, user_id: str, user_update: dict) -> Optional[UserOut]:
        """
        Update user information.
        
        Args:
            user_id (str): UUID of the user to update
            user_update (dict): User data to update
            
        Returns:
            Optional[UserOut]: Updated user information if successful, None otherwise
        """
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return None
            
            # Update allowed fields
            allowed_fields = [
                'first_name', 'last_name', 'university', 'email', 'dob',
                'major', 'role', 'profile_enhanced', 'completed_courses',
                'additional_skills'
            ]
            
            for field, value in user_update.items():
                if field in allowed_fields and hasattr(user, field):
                    setattr(user, field, value)
            
            self.db.commit()
            
            # Return updated user data
            return self.get_user_by_id(user_id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user {user_id}: {str(e)}")
            raise
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user from the system.
        
        Args:
            user_id (str): UUID of the user to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return False
            
            # Delete user's resume data from external services
            try:
                from app.services.user_service import UserService
                user_service = UserService(self.db)
                
                # Delete from Pinecone
                user_service.delete_resume_embedding_from_pinecone(user_id)
                
                # Delete from PostgreSQL
                user_service.delete_resume_embedding_postgresql(user_id)
                
            except Exception as e:
                logger.warning(f"Failed to delete user's external data: {str(e)}")
            
            # Delete from database
            self.db.delete(user)
            self.db.commit()
            
            logger.info(f"Deleted user {user_id} for admin")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting user {user_id}: {str(e)}")
            raise
    
    # ============================================================================
    # COURSE MANAGEMENT METHODS
    # ============================================================================
    
    def get_all_courses(self, skip: int = 0, limit: int = 100, major: Optional[str] = None) -> List[CourseOut]:
        """
        Get all courses with optional filtering and pagination.
        
        Args:
            skip (int): Number of records to skip for pagination
            limit (int): Maximum number of records to return
            major (Optional[str]): Major code to filter courses by
            
        Returns:
            List[CourseOut]: List of course information
        """
        try:
            query = self.db.query(Course)
            
            if major:
                query = query.filter(Course.major == major)
            
            courses = query.offset(skip).limit(limit).all()
            course_list = []
            
            for course in courses:
                # Parse domains and skills_associated
                domains = self._parse_list_field(course.domains)
                skills_associated = self._parse_list_field(course.skills_associated)
                
                course_data = {
                    "id": str(course.id),
                    "course_id": course.course_id,
                    "course_name": course.course_name,
                    "domains": domains,
                    "major": course.major,
                    "skills_associated": skills_associated,
                    "prerequisites": course.prerequisites,
                    "description": course.course_description,
                    "created_at": course.created_at,
                    "updated_at": course.updated_at
                }
                course_list.append(course_data)
            
            logger.info(f"Retrieved {len(course_list)} courses for admin")
            return course_list
            
        except Exception as e:
            logger.error(f"Error retrieving courses: {str(e)}")
            raise
    
    def get_course_by_id(self, course_id: str) -> Optional[CourseOut]:
        """
        Get specific course by ID.
        
        Args:
            course_id (str): Course ID to retrieve
            
        Returns:
            Optional[CourseOut]: Course information if found, None otherwise
        """
        try:
            course = self.db.query(Course).filter(Course.course_id == course_id).first()
            
            if not course:
                return None
            
            # Parse domains and skills_associated
            domains = self._parse_list_field(course.domains)
            skills_associated = self._parse_list_field(course.skills_associated)
            
            course_data = {
                "id": str(course.id),
                "course_id": course.course_id,
                "course_name": course.course_name,
                "domains": domains,
                "major": course.major,
                "skills_associated": skills_associated,
                "prerequisites": course.prerequisites,
                "description": course.course_description,
                "created_at": course.created_at,
                "updated_at": course.updated_at
            }
            
            logger.info(f"Retrieved course {course_id} for admin")
            return course_data
            
        except Exception as e:
            logger.error(f"Error retrieving course {course_id}: {str(e)}")
            raise
    
    def create_course(self, course_data: CourseCreate) -> CourseOut:
        """
        Create a new course.
        
        Args:
            course_data (CourseCreate): Course creation data
            
        Returns:
            CourseOut: Created course information
        """
        try:
            # Check if course already exists
            existing_course = self.db.query(Course).filter(Course.course_id == course_data.course_id).first()
            if existing_course:
                raise ValueError(f"Course with ID {course_data.course_id} already exists")
            
            # Create new course
            new_course = Course(
                course_id=course_data.course_id,
                course_name=course_data.course_name,
                domains=str(course_data.domains) if course_data.domains else None,
                major=course_data.major,
                skills_associated=str(course_data.skills_associated) if course_data.skills_associated else None,
                prerequisites=course_data.prerequisites,
                course_description=course_data.description
            )
            
            self.db.add(new_course)
            self.db.commit()
            self.db.refresh(new_course)
            
            # Index course in external services
            try:
                self._index_course_in_external_services(new_course)
            except Exception as e:
                logger.warning(f"Failed to index course in external services: {str(e)}")
            
            logger.info(f"Created course {course_data.course_id} for admin")
            return self.get_course_by_id(course_data.course_id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating course: {str(e)}")
            raise
    
    def update_course(self, course_id: str, course_update: CourseUpdate) -> Optional[CourseOut]:
        """
        Update course information.
        
        Args:
            course_id (str): Course ID to update
            course_update (CourseUpdate): Course update data
            
        Returns:
            Optional[CourseOut]: Updated course information if successful, None otherwise
        """
        try:
            course = self.db.query(Course).filter(Course.course_id == course_id).first()
            
            if not course:
                return None
            
            # Update allowed fields
            if course_update.course_name is not None:
                course.course_name = course_update.course_name
            if course_update.domains is not None:
                course.domains = str(course_update.domains)
            if course_update.major is not None:
                course.major = course_update.major
            if course_update.skills_associated is not None:
                course.skills_associated = str(course_update.skills_associated)
            if course_update.prerequisites is not None:
                course.prerequisites = course_update.prerequisites
            if course_update.description is not None:
                course.course_description = course_update.description
            
            course.updated_at = datetime.utcnow()
            self.db.commit()
            
            # Re-index course in external services
            try:
                self._index_course_in_external_services(course)
            except Exception as e:
                logger.warning(f"Failed to re-index course in external services: {str(e)}")
            
            logger.info(f"Updated course {course_id} for admin")
            return self.get_course_by_id(course_id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating course {course_id}: {str(e)}")
            raise
    
    def delete_course(self, course_id: str) -> bool:
        """
        Delete a course from the system.
        
        Args:
            course_id (str): Course ID to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            course = self.db.query(Course).filter(Course.course_id == course_id).first()
            
            if not course:
                return False
            
            # Remove from external services
            try:
                self._remove_course_from_external_services(course)
            except Exception as e:
                logger.warning(f"Failed to remove course from external services: {str(e)}")
            
            # Delete from database
            self.db.delete(course)
            self.db.commit()
            
            logger.info(f"Deleted course {course_id} for admin")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting course {course_id}: {str(e)}")
            raise
    
    # ============================================================================
    # SYSTEM MANAGEMENT METHODS
    # ============================================================================
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            Dict[str, Any]: System health information
        """
        try:
            # Check database connection
            try:
                self.db.execute("SELECT 1")
                database_status = True
            except Exception:
                database_status = False
            
            # Check Redis connection
            try:
                from app.services.redis_service import RedisService
                redis_service = RedisService()
                redis_status = redis_service.redis_client is not None
            except Exception:
                redis_status = False
            
            # Check Pinecone connection
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=settings.pinecone_api_key)
                index = pc.Index(settings.pinecone_index)
                index.describe_index_stats()
                pinecone_status = True
            except Exception:
                pinecone_status = False
            
            # Check Neo4j connection
            neo4j_status = self.neo4j_service.is_configured() and self.neo4j_service.test_connection()
            
            # Get counts
            user_count = self.db.query(func.count(User.id)).scalar()
            course_count = self.db.query(func.count(Course.id)).scalar()
            
            health_status = {
                "database": {
                    "status": database_status,
                    "user_count": user_count,
                    "course_count": course_count
                },
                "redis": {
                    "status": redis_status
                },
                "pinecone": {
                    "status": pinecone_status,
                    "index": settings.pinecone_index if pinecone_status else None
                },
                "neo4j": {
                    "status": neo4j_status,
                    "configured": self.neo4j_service.is_configured()
                },
                "overall_status": all([database_status, redis_status, pinecone_status, neo4j_status]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("System health check completed")
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
            raise
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics and metrics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            # User statistics
            total_users = self.db.query(func.count(User.id)).scalar()
            admin_users = self.db.query(func.count(User.id)).filter(User.role == "admin").scalar()
            regular_users = total_users - admin_users
            users_with_resume = self.db.query(func.count(User.id)).filter(User.resume_embedding.isnot(None)).scalar()
            
            # Course statistics
            total_courses = self.db.query(func.count(Course.id)).scalar()
            courses_by_major = self.db.query(Course.major, func.count(Course.id)).group_by(Course.major).all()
            
            # Recent activity
            recent_users = self.db.query(User).order_by(User.created_at.desc()).limit(5).all()
            recent_courses = self.db.query(Course).order_by(Course.created_at.desc()).limit(5).all()
            
            stats = {
                "users": {
                    "total": total_users,
                    "admin": admin_users,
                    "regular": regular_users,
                    "with_resume": users_with_resume,
                    "profile_enhanced": self.db.query(func.count(User.id)).filter(User.profile_enhanced == True).scalar()
                },
                "courses": {
                    "total": total_courses,
                    "by_major": dict(courses_by_major)
                },
                "recent_activity": {
                    "recent_users": [
                        {
                            "id": str(user.id),
                            "email": user.email,
                            "created_at": user.created_at.isoformat()
                        }
                        for user in recent_users
                    ],
                    "recent_courses": [
                        {
                            "id": str(course.id),
                            "course_id": course.course_id,
                            "course_name": course.course_name,
                            "created_at": course.created_at.isoformat()
                        }
                        for course in recent_courses
                    ]
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("System statistics retrieved")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system statistics: {str(e)}")
            raise
    
    def sync_data_to_neo4j(self, clear_first: bool = False) -> Dict[str, Any]:
        """
        Sync data to Neo4j for prerequisite checking.
        
        Args:
            clear_first (bool): Whether to clear existing data before syncing
            
        Returns:
            Dict[str, Any]: Sync operation results
        """
        try:
            if not self.neo4j_service.is_configured():
                raise ValueError("Neo4j is not configured")
            
            # Get all courses
            courses = self.db.query(Course).all()
            
            # Sync to Neo4j
            sync_results = self.neo4j_service.sync_courses_to_neo4j(courses, clear_first=clear_first)
            
            logger.info(f"Neo4j sync completed: {len(courses)} courses processed")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error syncing data to Neo4j: {str(e)}")
            raise
    
    def sync_data_to_pinecone(self, clear_first: bool = False) -> Dict[str, Any]:
        """
        Sync data to Pinecone for semantic matching.
        
        Args:
            clear_first (bool): Whether to clear existing data before syncing
            
        Returns:
            Dict[str, Any]: Sync operation results
        """
        try:
            # Get all courses
            courses = self.db.query(Course).all()
            
            # Index courses in Pinecone
            indexed_count = 0
            for course in courses:
                try:
                    self._index_course_in_pinecone(course)
                    indexed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to index course {course.course_id}: {str(e)}")
            
            sync_results = {
                "total_courses": len(courses),
                "indexed_count": indexed_count,
                "failed_count": len(courses) - indexed_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Pinecone sync completed: {indexed_count}/{len(courses)} courses indexed")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error syncing data to Pinecone: {str(e)}")
            raise
    
    def create_system_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a system backup.
        
        Args:
            backup_name (Optional[str]): Custom name for the backup
            
        Returns:
            Dict[str, Any]: Backup operation results
        """
        try:
            # Generate backup ID
            backup_id = str(uuid.uuid4())
            backup_name = backup_name or f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Get all data
            users = self.db.query(User).all()
            courses = self.db.query(Course).all()
            
            # Create backup data
            backup_data = {
                "backup_id": backup_id,
                "backup_name": backup_name,
                "timestamp": datetime.utcnow().isoformat(),
                "users": [
                    {
                        "id": str(user.id),
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "university": user.university,
                        "email": user.email,
                        "dob": user.dob,
                        "major": user.major,
                        "role": user.role,
                        "profile_enhanced": user.profile_enhanced,
                        "completed_courses": user.completed_courses,
                        "additional_skills": user.additional_skills,
                        "created_at": user.created_at.isoformat()
                    }
                    for user in users
                ],
                "courses": [
                    {
                        "id": str(course.id),
                        "course_id": course.course_id,
                        "course_name": course.course_name,
                        "domains": course.domains,
                        "major": course.major,
                        "skills_associated": course.skills_associated,
                        "prerequisites": course.prerequisites,
                        "course_description": course.course_description,
                        "created_at": course.created_at.isoformat(),
                        "updated_at": course.updated_at.isoformat() if course.updated_at else None
                    }
                    for course in courses
                ]
            }
            
            # TODO: Save backup to file system or cloud storage
            # For now, just return the backup data
            backup_results = {
                "backup_id": backup_id,
                "backup_name": backup_name,
                "user_count": len(users),
                "course_count": len(courses),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            logger.info(f"System backup created: {backup_name} ({backup_id})")
            return backup_results
            
        except Exception as e:
            logger.error(f"Error creating system backup: {str(e)}")
            raise
    
    def restore_system_backup(self, backup_id: str) -> Dict[str, Any]:
        """
        Restore system from backup.
        
        Args:
            backup_id (str): ID of the backup to restore from
            
        Returns:
            Dict[str, Any]: Restore operation results
        """
        try:
            # TODO: Load backup from file system or cloud storage
            # For now, just return a placeholder response
            restore_results = {
                "backup_id": backup_id,
                "status": "not_implemented",
                "message": "Backup restore functionality not yet implemented",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.warning(f"Backup restore requested for {backup_id} but not implemented")
            return restore_results
            
        except Exception as e:
            logger.error(f"Error restoring system backup: {str(e)}")
            raise
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _parse_list_field(self, field_value: Optional[str]) -> List[str]:
        """
        Parse a string field that contains a list representation.
        
        Args:
            field_value (Optional[str]): String field to parse
            
        Returns:
            List[str]: Parsed list of strings
        """
        if not field_value:
            return []
        
        try:
            # Handle the string format: "['Software Development', 'Data']"
            field_str = field_value.strip()
            if field_str.startswith('[') and field_str.endswith(']'):
                # Remove outer brackets and split by comma
                content = field_str[1:-1]
                items = [item.strip().strip("'\"") for item in content.split(',')]
                # Filter out 'nan' values
                items = [item for item in items if item.lower() != 'nan']
                return items
            else:
                return [field_value]
        except Exception as e:
            logger.warning(f"Error parsing list field: {str(e)}")
            return [field_value] if field_value else []
    
    def _index_course_in_external_services(self, course: Course):
        """
        Index a course in external services (Pinecone and Neo4j).
        
        Args:
            course (Course): Course to index
        """
        try:
            # Index in Pinecone
            self._index_course_in_pinecone(course)
            
            # Index in Neo4j
            if self.neo4j_service.is_configured():
                self.neo4j_service.sync_courses_to_neo4j([course], clear_first=False)
                
        except Exception as e:
            logger.warning(f"Failed to index course {course.course_id} in external services: {str(e)}")
    
    def _index_course_in_pinecone(self, course: Course):
        """
        Index a course in Pinecone for semantic matching.
        
        Args:
            course (Course): Course to index
        """
        try:
            from pinecone import Pinecone
            from sentence_transformers import SentenceTransformer
            
            # Initialize Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            
            # Initialize sentence transformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create course text for embedding
            course_text = f"{course.course_name} {course.course_description or ''}"
            if course.skills_associated:
                course_text += f" {' '.join(self._parse_list_field(course.skills_associated))}"
            
            # Generate embedding
            embedding = model.encode(course_text).tolist()
            
            # Prepare metadata
            metadata = {
                "course_name": course.course_name,
                "course_id": course.course_id,
                "major": course.major,
                "domains": self._parse_list_field(course.domains),
                "skills_associated": self._parse_list_field(course.skills_associated),
                "prerequisites": course.prerequisites,
                "type": "course"
            }
            
            # Upsert to Pinecone
            index.upsert(
                vectors=[{
                    "id": f"course:{course.course_id}",
                    "values": embedding,
                    "metadata": metadata
                }]
            )
            
            logger.debug(f"Indexed course {course.course_id} in Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to index course {course.course_id} in Pinecone: {str(e)}")
            raise
    
    def _remove_course_from_external_services(self, course: Course):
        """
        Remove a course from external services.
        
        Args:
            course (Course): Course to remove
        """
        try:
            # Remove from Pinecone
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=settings.pinecone_api_key)
                index = pc.Index(settings.pinecone_index)
                index.delete(ids=[f"course:{course.course_id}"])
                logger.debug(f"Removed course {course.course_id} from Pinecone")
            except Exception as e:
                logger.warning(f"Failed to remove course {course.course_id} from Pinecone: {str(e)}")
            
            # Remove from Neo4j
            try:
                if self.neo4j_service.is_configured():
                    self.neo4j_service.delete_course_from_neo4j(course.course_id)
                    logger.debug(f"Removed course {course.course_id} from Neo4j")
            except Exception as e:
                logger.warning(f"Failed to remove course {course.course_id} from Neo4j: {str(e)}")
                
        except Exception as e:
            logger.warning(f"Failed to remove course {course.course_id} from external services: {str(e)}") 