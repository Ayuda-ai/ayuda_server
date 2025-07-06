from neo4j import GraphDatabase
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class Neo4jService:
    """
    Service class for Neo4j graph database operations.
    
    This service handles all graph operations including:
    - Course and prerequisite management
    - Prerequisites checking and path analysis
    - Graph analytics and insights
    - Data synchronization with PostgreSQL
    """
    
    def __init__(self):
        """Initialize Neo4j connection using environment variables."""
        # Check if Neo4j is configured
        if not settings.neo4j_uri or not settings.neo4j_username or not settings.neo4j_password:
            logger.warning("⚠️ Neo4j not configured. Graph operations will be disabled.")
            self.driver = None
            return
            
        try:
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Neo4j connection established successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {str(e)}")
            self.driver = None
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
    
    def is_configured(self) -> bool:
        """Check if Neo4j is properly configured."""
        return self.driver is not None
    
    def test_connection(self) -> bool:
        """Test Neo4j connection."""
        if not self.is_configured():
            return False
            
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {str(e)}")
            return False
    
    def create_course_node(self, course_data: Dict[str, Any]) -> bool:
        """
        Create a course node in Neo4j.
        
        Args:
            course_data (Dict[str, Any]): Course data including id, name, description, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot create course node.")
            return False
            
        try:
            with self.driver.session() as session:
                # Create course node with MERGE to avoid duplicates
                # Store both UUID and course_id for easier matching
                query = """
                MERGE (c:Course {id: $course_id})
                SET c.name = $name,
                    c.description = $description,
                    c.major = $major,
                    c.domains = $domains,
                    c.skills_associated = $skills_associated,
                    c.credits = $credits,
                    c.course_id = $course_code,
                    c.updated_at = datetime()
                RETURN c
                """
                
                # Add course_code to the data
                course_data_with_code = {
                    **course_data,
                    "course_code": course_data.get("course_code", course_data.get("course_id", ""))
                }
                
                result = session.run(query, course_data_with_code)
                course = result.single()
                
                if course:
                    logger.info(f"✅ Course node created/updated: {course_data['course_id']}")
                    return True
                else:
                    logger.error(f"❌ Failed to create course node: {course_data['course_id']}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating course node: {str(e)}")
            return False
    
    def add_prerequisite_relationship(self, course_id: str, prerequisite_id: str, relationship_type: str = "OR") -> bool:
        """
        Add a prerequisite relationship between two courses.
        
        Args:
            course_id (str): ID of the course that requires the prerequisite
            prerequisite_id (str): ID of the prerequisite course
            relationship_type (str): Type of relationship ("OR" or "AND")
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot add prerequisite relationship.")
            return False
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH (course:Course {id: $course_id})
                MATCH (prereq:Course {id: $prerequisite_id})
                MERGE (course)-[r:PREREQUISITE {type: $relationship_type}]->(prereq)
                RETURN r
                """
                
                result = session.run(query, {
                    "course_id": course_id,
                    "prerequisite_id": prerequisite_id,
                    "relationship_type": relationship_type
                })
                
                relationship = result.single()
                if relationship:
                    logger.info(f"✅ Prerequisite relationship created: {course_id} -> {prerequisite_id}")
                    return True
                else:
                    logger.error(f"❌ Failed to create prerequisite relationship: {course_id} -> {prerequisite_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating prerequisite relationship: {str(e)}")
            return False
    
    def get_course_prerequisites(self, course_id: str) -> List[Dict[str, Any]]:
        """
        Get all prerequisites for a specific course.
        
        Args:
            course_id (str): ID of the course
            
        Returns:
            List[Dict[str, Any]]: List of prerequisite courses with relationship types
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot get course prerequisites.")
            return []
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH (course:Course {id: $course_id})-[r:PREREQUISITE]->(prereq:Course)
                RETURN prereq.id as id,
                       prereq.name as name,
                       prereq.major as major,
                       r.type as relationship_type
                ORDER BY prereq.id
                """
                
                result = session.run(query, {"course_id": course_id})
                prerequisites = [record.data() for record in result]
                
                logger.info(f"✅ Retrieved {len(prerequisites)} prerequisites for course: {course_id}")
                return prerequisites
                
        except Exception as e:
            logger.error(f"Error getting course prerequisites: {str(e)}")
            return []
    
    def get_prerequisite_path(self, course_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Get the complete prerequisite path for a course.
        
        Args:
            course_id (str): ID of the course
            max_depth (int): Maximum depth to search for prerequisites
            
        Returns:
            List[Dict[str, Any]]: List of courses in the prerequisite path
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot get prerequisite path.")
            return []
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = (course:Course {id: $course_id})-[:PREREQUISITE*1..$max_depth]->(prereq:Course)
                RETURN DISTINCT prereq.id as id,
                               prereq.name as name,
                               prereq.major as major,
                               length(path) as depth
                ORDER BY depth DESC, prereq.id
                """
                
                result = session.run(query, {
                    "course_id": course_id,
                    "max_depth": max_depth
                })
                
                path = [record.data() for record in result]
                logger.info(f"✅ Retrieved prerequisite path for course: {course_id} (depth: {max_depth})")
                return path
                
        except Exception as e:
            logger.error(f"Error getting prerequisite path: {str(e)}")
            return []
    
    def check_prerequisites_completion(self, course_id: str, completed_courses: List[str]) -> Dict[str, Any]:
        """
        Check if a user has completed the prerequisites for a course.
        
        Args:
            course_id (str): Course code to check (e.g., "CSYE7125") - this is the course_id field, not UUID
            completed_courses (List[str]): List of course IDs the user has completed (course_id values like "CSYE6225")
            
        Returns:
            Dict[str, Any]: Prerequisites completion status
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot check prerequisites completion.")
            return {
                "course_id": course_id,
                "has_prerequisites": False,
                "prerequisites_met": True,
                "missing_prerequisites": [],
                "completed_prerequisites": [],
                "prerequisite_groups": [],
                "neo4j_configured": False
            }
            
        try:
            with self.driver.session() as session:
                # Get all prerequisites for the course using course_id field (not UUID)
                query = """
                MATCH (course:Course {course_id: $course_id})-[r:PREREQUISITE]->(prereq:Course)
                RETURN prereq.id as id,
                       prereq.course_id as course_id,
                       prereq.name as name,
                       prereq.major as major,
                       r.type as relationship_type
                ORDER BY prereq.course_id
                """
                
                result = session.run(query, {"course_id": course_id})
                prerequisites = [record.data() for record in result]
                
                logger.debug(f"Found {len(prerequisites)} prerequisites for course {course_id}: {prerequisites}")
                
                if not prerequisites:
                    return {
                        "course_id": course_id,
                        "has_prerequisites": False,
                        "prerequisites_met": True,
                        "missing_prerequisites": [],
                        "completed_prerequisites": [],
                        "prerequisite_groups": []
                    }
                
                # Handle cases where relationship_type might be None or not set
                # Default to "OR" if not specified
                for prereq in prerequisites:
                    if prereq.get("relationship_type") is None:
                        prereq["relationship_type"] = "OR"
                
                # Group prerequisites by relationship type
                or_prerequisites = [p for p in prerequisites if p["relationship_type"] == "OR"]
                and_prerequisites = [p for p in prerequisites if p["relationship_type"] == "AND"]
                
                logger.debug(f"OR prerequisites: {len(or_prerequisites)}, AND prerequisites: {len(and_prerequisites)}")
                
                # Check completion for each group using course_id values
                or_completed = [p for p in or_prerequisites if p["course_id"] in completed_courses]
                and_completed = [p for p in and_prerequisites if p["course_id"] in completed_courses]
                
                # Determine if prerequisites are met
                or_met = len(or_completed) > 0 if or_prerequisites else True
                and_met = len(and_completed) == len(and_prerequisites) if and_prerequisites else True
                prerequisites_met = or_met and and_met
                
                # Get missing prerequisites
                missing_or = [p for p in or_prerequisites if p["course_id"] not in completed_courses] if not or_met else []
                missing_and = [p for p in and_prerequisites if p["course_id"] not in completed_courses]
                missing_prerequisites = missing_or + missing_and
                
                # Get completed prerequisites
                completed_prerequisites = or_completed + and_completed
                
                # Create prerequisite groups for display
                prerequisite_groups = []
                if or_prerequisites:
                    prerequisite_groups.append({
                        "type": "OR",
                        "courses": or_prerequisites,
                        "completed": or_completed,
                        "met": or_met
                    })
                if and_prerequisites:
                    prerequisite_groups.append({
                        "type": "AND", 
                        "courses": and_prerequisites,
                        "completed": and_completed,
                        "met": and_met
                    })
                
                logger.info(f"Prerequisites check for {course_id}: {len(prerequisites)} total, {len(completed_prerequisites)} completed, {len(missing_prerequisites)} missing, met: {prerequisites_met}")
                
                return {
                    "course_id": course_id,
                    "has_prerequisites": True,
                    "prerequisites_met": prerequisites_met,
                    "missing_prerequisites": missing_prerequisites,
                    "completed_prerequisites": completed_prerequisites,
                    "prerequisite_groups": prerequisite_groups,
                    "total_prerequisites": len(prerequisites),
                    "completed_count": len(completed_prerequisites),
                    "missing_count": len(missing_prerequisites)
                }
                
        except Exception as e:
            logger.error(f"Error checking prerequisites completion: {str(e)}")
            return {
                "course_id": course_id,
                "has_prerequisites": False,
                "prerequisites_met": False,
                "error": str(e)
            }
    
    def get_courses_by_prerequisites(self, completed_courses: List[str]) -> List[Dict[str, Any]]:
        """
        Get courses that the user can take based on their completed prerequisites.
        
        Args:
            completed_courses (List[str]): List of course IDs the user has completed
            
        Returns:
            List[Dict[str, Any]]: List of available courses with prerequisite status
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot get courses by prerequisites.")
            return []
            
        try:
            with self.driver.session() as session:
                # Get all courses with their prerequisite completion status
                query = """
                MATCH (course:Course)
                OPTIONAL MATCH (course)-[r:PREREQUISITE]->(prereq:Course)
                WITH course, collect({id: prereq.id, type: r.type}) as prerequisites
                RETURN course.id as id,
                       course.name as name,
                       course.major as major,
                       prerequisites
                ORDER BY course.id
                """
                
                result = session.run(query)
                courses = []
                
                for record in result:
                    course_data = record.data()
                    course_id = course_data["id"]
                    prerequisites = course_data["prerequisites"]
                    
                    # Check prerequisite completion for this course
                    completion_status = self.check_prerequisites_completion(course_id, completed_courses)
                    
                    courses.append({
                        "id": course_id,
                        "name": course_data["name"],
                        "major": course_data["major"],
                        "prerequisites_met": completion_status["prerequisites_met"],
                        "total_prerequisites": completion_status["total_prerequisites"],
                        "completed_prerequisites": completion_status["completed_count"],
                        "missing_prerequisites": completion_status["missing_count"],
                        "prerequisite_groups": completion_status["prerequisite_groups"]
                    })
                
                logger.info(f"✅ Retrieved {len(courses)} courses with prerequisite status")
                return courses
                
        except Exception as e:
            logger.error(f"Error getting courses by prerequisites: {str(e)}")
            return []
    
    def remove_prerequisite_relationship(self, course_id: str, prerequisite_id: str) -> bool:
        """
        Remove a prerequisite relationship between two courses.
        
        Args:
            course_id (str): ID of the course
            prerequisite_id (str): ID of the prerequisite course
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot remove prerequisite relationship.")
            return False
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH (course:Course {id: $course_id})-[r:PREREQUISITE]->(prereq:Course {id: $prerequisite_id})
                DELETE r
                RETURN count(r) as deleted_count
                """
                
                result = session.run(query, {
                    "course_id": course_id,
                    "prerequisite_id": prerequisite_id
                })
                
                deleted_count = result.single()["deleted_count"]
                
                if deleted_count > 0:
                    logger.info(f"✅ Prerequisite relationship removed: {course_id} -> {prerequisite_id}")
                    return True
                else:
                    logger.warning(f"⚠️ No prerequisite relationship found: {course_id} -> {prerequisite_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error removing prerequisite relationship: {str(e)}")
            return False
    
    def get_course_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about courses and prerequisites.
        
        Returns:
            Dict[str, Any]: Analytics data
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot get course analytics.")
            return {}
            
        try:
            with self.driver.session() as session:
                # Get total courses
                total_courses_query = "MATCH (c:Course) RETURN count(c) as total_courses"
                total_courses = session.run(total_courses_query).single()["total_courses"]
                
                # Get courses with prerequisites
                courses_with_prereqs_query = """
                MATCH (c:Course)-[:PREREQUISITE]->()
                RETURN count(DISTINCT c) as courses_with_prerequisites
                """
                courses_with_prereqs = session.run(courses_with_prereqs_query).single()["courses_with_prerequisites"]
                
                # Get total prerequisite relationships
                total_prereq_relationships_query = """
                MATCH ()-[r:PREREQUISITE]->()
                RETURN count(r) as total_relationships
                """
                total_relationships = session.run(total_prereq_relationships_query).single()["total_relationships"]
                
                # Get average prerequisites per course
                avg_prereqs_query = """
                MATCH (c:Course)-[r:PREREQUISITE]->()
                RETURN avg(size(collect(r))) as avg_prerequisites
                """
                avg_prereqs = session.run(avg_prereqs_query).single()["avg_prerequisites"] or 0
                
                # Get courses by major
                courses_by_major_query = """
                MATCH (c:Course)
                RETURN c.major as major, count(c) as count
                ORDER BY count DESC
                """
                courses_by_major = [record.data() for record in session.run(courses_by_major_query)]
                
                return {
                    "total_courses": total_courses,
                    "courses_with_prerequisites": courses_with_prereqs,
                    "courses_without_prerequisites": total_courses - courses_with_prereqs,
                    "total_prerequisite_relationships": total_relationships,
                    "average_prerequisites_per_course": round(avg_prereqs, 2),
                    "courses_by_major": courses_by_major
                }
                
        except Exception as e:
            logger.error(f"Error getting course analytics: {str(e)}")
            return {}
    
    def clear_all_data(self) -> bool:
        """
        Clear all course nodes and relationships from Neo4j.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot clear data.")
            return False
            
        try:
            with self.driver.session() as session:
                # Delete all relationships first
                delete_relationships_query = "MATCH ()-[r:PREREQUISITE]->() DELETE r"
                session.run(delete_relationships_query)
                
                # Delete all course nodes
                delete_nodes_query = "MATCH (c:Course) DELETE c"
                session.run(delete_nodes_query)
                
                logger.info("✅ Cleared all course nodes and relationships from Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing Neo4j data: {str(e)}")
            return False

    def sync_courses_from_postgresql(self, courses_data: List[Dict[str, Any]], clear_first: bool = False) -> bool:
        """
        Sync course data from PostgreSQL to Neo4j.
        
        Args:
            courses_data (List[Dict[str, Any]]): List of course data from PostgreSQL
            clear_first (bool): Whether to clear existing data before syncing
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("⚠️ Neo4j not configured. Cannot sync courses from PostgreSQL.")
            return False
            
        try:
            # Clear existing data if requested
            if clear_first:
                logger.info("Clearing existing Neo4j data...")
                if not self.clear_all_data():
                    logger.error("Failed to clear existing data")
                    return False
            
            success_count = 0
            total_count = len(courses_data)
            relationship_count = 0
            
            # Log the first course data structure for debugging
            if courses_data:
                logger.info(f"Sample course data structure: {list(courses_data[0].keys())}")
                logger.info(f"Sample course ID type: {type(courses_data[0]['id'])}")
            
            # First pass: Create all course nodes
            for i, course_data in enumerate(courses_data):
                try:
                    # Convert UUID to string and prepare course data for Neo4j
                    neo4j_course_data = {
                        "course_id": str(course_data["id"]),  # Convert UUID to string
                        "course_code": course_data.get("course_id", ""),  # The course_id field (e.g., CSYE6225)
                        "name": course_data["course_name"],
                        "description": course_data["course_description"],
                        "major": course_data["major"],
                        "domains": course_data["domains"] or [],  # Handle None values
                        "skills_associated": course_data["skills_associated"] or [],  # Handle None values
                        "credits": 3  # Default credits
                    }
                    
                    logger.debug(f"Processing course {i+1}/{total_count}: {neo4j_course_data['course_id']}")
                    
                    if self.create_course_node(neo4j_course_data):
                        success_count += 1
                    else:
                        logger.error(f"Failed to create course node for: {neo4j_course_data['course_id']}")
                        
                except Exception as e:
                    logger.error(f"Error processing course {i+1}/{total_count}: {str(e)}")
                    continue
            
            # Second pass: Create prerequisite relationships
            logger.info("Creating prerequisite relationships...")
            for i, course_data in enumerate(courses_data):
                try:
                    course_id = str(course_data["id"])
                    prerequisites = course_data.get("prerequisites", [])
                    
                    if prerequisites:
                        # Filter out "nan" values and empty strings
                        valid_prerequisites = [prereq for prereq in prerequisites 
                                            if prereq and prereq.strip() and prereq.strip().lower() != "nan"]
                        
                        if valid_prerequisites:
                            logger.info(f"Course {course_data.get('course_id', 'N/A')} (UUID: {course_id}) has valid prerequisites: {valid_prerequisites}")
                            logger.debug(f"Processing prerequisites for course {i+1}/{total_count}: {course_id}")
                            
                            for prereq in valid_prerequisites:
                                logger.debug(f"Looking for prerequisite: '{prereq}'")
                                
                                # Try to find the prerequisite course in our data
                                prereq_found = False
                                for other_course in courses_data:
                                    other_course_id = str(other_course["id"])
                                    other_course_name = other_course["course_name"]
                                    other_course_id_field = other_course.get("course_id", "")  # The course_id field (e.g., CSYE6225)
                                    
                                    # Multiple matching strategies
                                    matches = []
                                    
                                    # 1. Exact match with course_id field (most likely to work)
                                    if prereq.strip() == other_course_id_field:
                                        matches.append(f"course_id field: {other_course_id_field}")
                                    
                                    # 2. Match with spaces removed (CSYE 6200 -> CSYE6200)
                                    prereq_no_spaces = prereq.strip().replace(" ", "")
                                    if prereq_no_spaces == other_course_id_field:
                                        matches.append(f"course_id field (no spaces): {other_course_id_field}")
                                    
                                    # 3. Match course_id with spaces added (CSYE6200 -> CSYE 6200)
                                    course_id_with_spaces = other_course_id_field.replace("", " ").strip()
                                    if prereq.strip() == course_id_with_spaces:
                                        matches.append(f"course_id with spaces: {course_id_with_spaces}")
                                    
                                    # 4. Exact match with UUID
                                    if prereq.strip() == other_course_id:
                                        matches.append(f"UUID: {other_course_id}")
                                    
                                    # 5. Partial match with course name
                                    if prereq.strip().lower() in other_course_name.lower():
                                        matches.append(f"course name: {other_course_name}")
                                    
                                    # 6. Reverse match (course name contains prerequisite)
                                    if other_course_name.lower() in prereq.strip().lower():
                                        matches.append(f"reverse name match: {other_course_name}")
                                    
                                    if matches:
                                        logger.info(f"Found prerequisite '{prereq}' matches {other_course_name} ({', '.join(matches)})")
                                        
                                        # Create prerequisite relationship (OR relationship by default)
                                        if self.add_prerequisite_relationship(course_id, other_course_id, "OR"):
                                            relationship_count += 1
                                            logger.info(f"✅ Created prerequisite: {course_data.get('course_id', 'N/A')} -> {other_course_id_field}")
                                        else:
                                            logger.error(f"❌ Failed to create prerequisite: {course_data.get('course_id', 'N/A')} -> {other_course_id_field}")
                                        prereq_found = True
                                        break
                                
                                if not prereq_found:
                                    logger.warning(f"❌ Prerequisite '{prereq}' not found for course {course_data.get('course_id', 'N/A')}")
                                    logger.warning(f"Available course_ids: {[c.get('course_id', 'N/A') for c in courses_data[:10]]}...")
                        else:
                            logger.debug(f"Course {course_data.get('course_id', 'N/A')} has no valid prerequisites (all were 'nan' or empty)")
                        
                except Exception as e:
                    logger.error(f"Error processing prerequisites for course {i+1}/{total_count}: {str(e)}")
                    continue
            
            logger.info(f"✅ Synced {success_count}/{total_count} courses from PostgreSQL to Neo4j")
            logger.info(f"✅ Created {relationship_count} prerequisite relationships")
            return success_count == total_count
            
        except Exception as e:
            logger.error(f"Error syncing courses from PostgreSQL: {str(e)}")
            return False 