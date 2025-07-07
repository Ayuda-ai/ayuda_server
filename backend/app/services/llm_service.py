import httpx
import logging
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service class for LLM operations using Ollama.
    
    This service handles all interactions with the custom Llama model
    for course recommendation reasoning and explanations.
    """
    
    def __init__(self):
        """Initialize LLM service with Ollama configuration."""
        self.ollama_url = settings.ollama_address
        self.model_name = settings.ollama_model
        self.timeout = 30  # 30 seconds timeout
        
    def _build_reasoning_prompt(self, user_profile: Dict[str, Any], course_info: Dict[str, Any], recommendation_context: Dict[str, Any]) -> str:
        """
        Build a comprehensive prompt for course recommendation reasoning.
        
        Args:
            user_profile: Complete user profile data
            course_info: Course information
            recommendation_context: Context about why the course was recommended
            
        Returns:
            str: Formatted prompt for the LLM
        """
        # Extract key information
        resume_text = user_profile.get("resume_text", "")
        work_experience = user_profile.get("work_experience", "")
        completed_courses = user_profile.get("completed_courses", [])
        additional_skills = user_profile.get("additional_skills", "")
        major = user_profile.get("major", "")
        university = user_profile.get("university", "")
        
        course_name = course_info.get("course_name", "")
        course_description = course_info.get("description", "")
        prerequisites = course_info.get("prerequisites", [])
        domains = course_info.get("domains", [])
        skills_associated = course_info.get("skills_associated", [])
        
        semantic_score = recommendation_context.get("semantic_score", 0)
        keyword_matches = recommendation_context.get("keyword_matches", [])
        prerequisite_status = recommendation_context.get("prerequisite_status", "unknown")
        missing_prerequisites = recommendation_context.get("missing_prerequisites", [])
        
        # Helper function to safely convert lists to strings
        def safe_join(items, default="None"):
            if not items:
                return default
            try:
                # Convert all items to strings and filter out None/empty values
                string_items = []
                for item in items:
                    if item is not None and str(item).strip():
                        string_items.append(str(item).strip())
                return ', '.join(string_items) if string_items else default
            except Exception:
                return default
        
        # Helper function to find relevant skills that match user profile
        def find_relevant_skills(user_skills_text, course_skills):
            if not course_skills or not user_skills_text:
                return []
            
            user_skills_lower = user_skills_text.lower()
            relevant_skills = []
            
            for skill in course_skills:
                if skill and str(skill).lower() in user_skills_lower:
                    relevant_skills.append(str(skill))
            
            return relevant_skills
        
        # Helper function to extract key skills from resume
        def extract_resume_skills(resume_text):
            if not resume_text:
                return []
            
            # Common programming languages and technologies
            common_skills = [
                'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'postgresql',
                'mongodb', 'docker', 'kubernetes', 'aws', 'azure', 'git', 'html', 'css',
                'typescript', 'angular', 'vue', 'php', 'c++', 'c#', '.net', 'spring',
                'django', 'flask', 'express', 'redis', 'elasticsearch', 'kafka', 'rabbitmq'
            ]
            
            resume_lower = resume_text.lower()
            found_skills = []
            
            for skill in common_skills:
                if skill in resume_lower:
                    found_skills.append(skill.title())
            
            return found_skills
        
        # Find relevant skills that match user's profile
        relevant_skills = find_relevant_skills(additional_skills, skills_associated)
        
        # Extract skills from resume
        resume_skills = extract_resume_skills(resume_text)
        
        # Build the prompt
        prompt = f"""You are an expert academic advisor providing personalized course recommendations. 

STUDENT PROFILE:
- Major: {major}
- University: {university}
- Completed Courses: {safe_join(completed_courses)}
- Additional Skills: {additional_skills if additional_skills else 'None'}
- Skills from Resume: {safe_join(resume_skills) if resume_skills else 'None'}
- Work Experience: {work_experience if work_experience else 'None'}
- Resume Summary: {resume_text[:300] + '...' if len(resume_text) > 300 else resume_text}

COURSE INFORMATION:
- Course Name: {course_name}
- Description: {course_description}
- Prerequisites: {safe_join(prerequisites)}
- Domains: {safe_join(domains)}
- Skills Covered: {safe_join(skills_associated)}
- Skills Matching Your Profile: {safe_join(relevant_skills) if relevant_skills else 'None'}

RECOMMENDATION CONTEXT:
- Semantic Similarity Score: {semantic_score:.2f}
- Keyword Matches: {safe_join(keyword_matches)}
- Prerequisite Status: {prerequisite_status}
- Missing Prerequisites: {safe_join(missing_prerequisites)}

Provide a concise explanation (1 paragraph, max 100 words) of why this course was recommended to you. Focus on: (do not explicitly mention these headers in your response)

1. **Direct alignment** with your background and experience
2. **Specific skills** from the course that match your profile
3. **Career benefits** relevant to your goals
4. **Academic progression** for your learning path

IMPORTANT GUIDELINES:
- Address the user as "you" (not by name)
- Follow consistent grammar throughout your response, addressing the user as "you/your/yours/yourself"
- Only mention skills/technologies that are actually listed in the course's "Skills Covered"
- Be specific and factual based on the provided information
- Keep it concise and to the point
- Focus on why this course fits your unique situation

Write a clear, personalized explanation that helps you understand why this course is a good match for your background and goals."""

        return prompt
    
    async def get_course_reasoning(self, user_profile: Dict[str, Any], course_info: Dict[str, Any], recommendation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get reasoning for a course recommendation using the custom Llama model.
        
        Args:
            user_profile: Complete user profile data
            course_info: Course information
            recommendation_context: Context about why the course was recommended
            
        Returns:
            Dict[str, Any]: Reasoning response with explanation and metadata
        """
        try:
            # Build the prompt
            prompt = self._build_reasoning_prompt(user_profile, course_info, recommendation_context)
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Make request to Ollama
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.ollama_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    reasoning = result.get("response", "")
                    
                    logger.info(f"Successfully generated reasoning for course: {course_info.get('course_name', 'Unknown')}")
                    
                    return {
                        "success": True,
                        "reasoning": reasoning,
                        "course_name": course_info.get("course_name", ""),
                        "model_used": self.model_name,
                        "prompt_length": len(prompt),
                        "response_length": len(reasoning)
                    }
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return {
                        "success": False,
                        "error": f"Ollama API error: {response.status_code}",
                        "reasoning": "Unable to generate reasoning at this time."
                    }
                    
        except httpx.TimeoutException:
            logger.error("Timeout while calling Ollama API")
            return {
                "success": False,
                "error": "Request timeout",
                "reasoning": "Unable to generate reasoning due to timeout."
            }
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "reasoning": "Unable to generate reasoning at this time."
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Ollama API.
        
        Returns:
            Dict[str, Any]: Connection test results
        """
        try:
            import asyncio
            
            # Simple test prompt
            test_payload = {
                "model": self.model_name,
                "prompt": "Hello, who are you?",
                "stream": False
            }
            
            async def test_call():
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(
                        self.ollama_url,
                        json=test_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    return response
            
            # Run the test
            response = asyncio.run(test_call())
            
            if response.status_code == 200:
                return {
                    "connected": True,
                    "model": self.model_name,
                    "url": self.ollama_url
                }
            else:
                return {
                    "connected": False,
                    "error": f"HTTP {response.status_code}",
                    "model": self.model_name,
                    "url": self.ollama_url
                }
                
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "model": self.model_name,
                "url": self.ollama_url
            } 