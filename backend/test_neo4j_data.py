#!/usr/bin/env python3
"""
Test script to verify Neo4j data structure and prerequisite relationships
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.neo4j_service import Neo4jService
from app.services.user_service import UserService
from app.db.session import get_db

def test_neo4j_data():
    """Test Neo4j data structure and prerequisite relationships"""
    
    print("🔍 Testing Neo4j data structure...")
    
    # Initialize services
    neo4j_service = Neo4jService()
    db = next(get_db())
    user_service = UserService(db)
    
    # Test 1: Check if Neo4j is configured
    print(f"✅ Neo4j configured: {neo4j_service.is_configured()}")
    print(f"✅ Neo4j connection: {neo4j_service.test_connection()}")
    
    # Test 2: Get course analytics
    analytics = neo4j_service.get_course_analytics()
    print(f"📊 Course analytics: {analytics}")
    
    # Test 3: Check specific courses that should have prerequisites
    test_courses = [
        ("CSYE7125", "Advanced Cloud Computing"),
        ("CSYE7550", "Distributed Intelligent Agents in the Metaverse"),
        ("CSYE7220", "Deployment and Operation of Software Applications")
    ]
    
    print("\n🔍 Testing prerequisite checking for specific courses...")
    
    for course_id, course_name in test_courses:
        print(f"\n--- Testing {course_id}: {course_name} ---")
        
        # Check prerequisites with empty completed courses
        prereq_status = neo4j_service.check_prerequisites_completion(course_id, [])
        print(f"Prerequisite status: {prereq_status}")
        
        # Check prerequisites with some completed courses
        prereq_status_with_completed = neo4j_service.check_prerequisites_completion(course_id, ["CSYE6225"])
        print(f"Prerequisite status (with CSYE6225 completed): {prereq_status_with_completed}")
    
    # Test 4: Check what courses are actually in Neo4j
    print("\n🔍 Checking courses in Neo4j...")
    
    # Get a sample of courses from the database
    courses_data = user_service.get_all_courses()
    print(f"Total courses in database: {len(courses_data)}")
    
    # Check first 5 courses
    for i, course in enumerate(courses_data[:5]):
        course_uuid = str(course["id"])
        course_code = course.get("course_id", "N/A")
        course_name = course.get("course_name", "N/A")
        
        print(f"Course {i+1}: {course_code} ({course_uuid}) - {course_name}")
        
        # Check if this course has prerequisites
        prereq_status = neo4j_service.check_prerequisites_completion(course_uuid, [])
        print(f"  Has prerequisites: {prereq_status['has_prerequisites']}")
        if prereq_status['has_prerequisites']:
            print(f"  Prerequisites met: {prereq_status['prerequisites_met']}")
            print(f"  Missing prerequisites: {len(prereq_status['missing_prerequisites'])}")
    
    print("\n✅ Neo4j data testing completed!")

if __name__ == "__main__":
    test_neo4j_data() 