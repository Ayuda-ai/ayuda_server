import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

class RecommendationType(Enum):
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    KEYWORD_ONLY = "keyword_only"

@dataclass
class RecommendationMetrics:
    """Data class for storing recommendation performance metrics"""
    user_id: str
    recommendation_id: str
    recommendation_type: RecommendationType
    timestamp: datetime
    total_courses_queried: int
    semantic_matches_found: int
    keyword_matches_found: int
    final_recommendations: int
    semantic_processing_time: float
    keyword_processing_time: float
    total_processing_time: float
    semantic_scores: List[float]
    keyword_scores: List[float]
    hybrid_scores: List[float]
    top_semantic_score: float
    top_keyword_score: float
    top_hybrid_score: float
    semantic_weight: float
    keyword_weight: float
    resume_text_length: int
    extracted_keywords_count: int
    matching_keywords_count: int
    redis_connection_status: bool
    pinecone_connection_status: bool

@dataclass
class CourseRecommendationDetail:
    """Data class for storing detailed information about each recommended course"""
    course_id: str
    course_name: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float
    matching_keywords: List[str]
    course_skills: List[str]
    rank_position: int
    score_improvement: Optional[float] = None  # How much hybrid improved over semantic

logger = logging.getLogger(__name__)

class RecommendationLogger:
    """
    Comprehensive logging service for recommendation system.
    Automatically generates detailed JSON log files for every recommendation request.
    """
    
    def __init__(self):
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
    
    def log_recommendation_request(
        self,
        user_id: str,
        user_email: str,
        request_type: str,  # "hybrid", "semantic", "debug"
        top_k: int,
        start_time: float,
        end_time: float,
        matches: List[Dict[str, Any]],
        processing_metrics: Dict[str, Any],
        system_health: Dict[str, Any],
        resume_info: Dict[str, Any],
        keyword_analysis: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a complete recommendation request with all metrics and analytics.
        
        Args:
            user_id: User ID
            user_email: User email
            request_type: Type of recommendation request
            top_k: Number of recommendations requested
            start_time: Request start timestamp
            end_time: Request end timestamp
            matches: List of recommended courses
            processing_metrics: Performance metrics
            system_health: System health status
            resume_info: Resume information
            keyword_analysis: Keyword analysis results
            error_info: Error information if any
            
        Returns:
            str: Path to the generated log file
        """
        try:
            # Generate unique recommendation ID
            recommendation_id = str(uuid.uuid4())
            
            # Calculate processing time
            total_processing_time = end_time - start_time
            
            # Create comprehensive log data
            log_data = {
                "recommendation_id": recommendation_id,
                "timestamp": datetime.now().isoformat(),
                "request_info": {
                    "user_id": user_id,
                    "user_email": user_email,
                    "request_type": request_type,
                    "top_k_requested": top_k,
                    "start_time": start_time,
                    "end_time": end_time,
                    "total_processing_time": total_processing_time
                },
                "user_info": {
                    "user_id": user_id,
                    "email": user_email,
                    "major": resume_info.get("major", "Unknown"),
                    "university": resume_info.get("university", "Unknown")
                },
                "resume_info": resume_info,
                "system_health": system_health,
                "processing_metrics": processing_metrics,
                "recommendation_results": {
                    "total_matches": len(matches),
                    "matches": matches,
                    "score_distribution": self._analyze_score_distribution(matches, request_type),
                    "course_types": self._analyze_course_types(matches),
                    "top_recommendations": matches[:3] if matches else []
                },
                "keyword_analysis": keyword_analysis or {},
                "performance_analysis": {
                    "processing_time_breakdown": processing_metrics,
                    "total_time": total_processing_time,
                    "performance_rating": self._rate_performance(total_processing_time),
                    "recommendations_per_second": len(matches) / total_processing_time if total_processing_time > 0 else 0
                },
                "quality_metrics": {
                    "relevance_score": self._calculate_relevance_score(matches),
                    "diversity_score": self._calculate_diversity_score(matches),
                    "score_quality": self._analyze_score_quality(matches, request_type)
                },
                "algorithm_details": {
                    "semantic_model": "all-MiniLM-L6-v2",
                    "embedding_dimensions": 384,
                    "semantic_weight": 0.7,
                    "keyword_weight": 0.3,
                    "similarity_metric": "cosine_similarity"
                },
                "error_info": error_info or {},
                "success": error_info is None
            }
            
            # Generate filename with timestamp and recommendation ID
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recommendation_{request_type}_{timestamp_str}_{recommendation_id[:8]}.json"
            filepath = self.logs_dir / filename
            
            # Write JSON log file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            # Log to console
            logger.info(f"📊 Recommendation logged: {filename}")
            logger.info(f"   User: {user_email}")
            logger.info(f"   Type: {request_type}")
            logger.info(f"   Matches: {len(matches)}")
            logger.info(f"   Time: {total_processing_time:.3f}s")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error logging recommendation: {str(e)}")
            return ""
    
    def _analyze_score_distribution(self, matches: List[Dict[str, Any]], request_type: str) -> Dict[str, Any]:
        """Analyze the distribution of scores in recommendations"""
        if not matches:
            return {"error": "No matches to analyze"}
        
        if request_type == "hybrid":
            semantic_scores = [match.get("semantic_score", 0) for match in matches]
            keyword_scores = [match.get("keyword_score", 0) for match in matches]
            hybrid_scores = [match.get("hybrid_score", 0) for match in matches]
            
            return {
                "semantic_scores": {
                    "min": min(semantic_scores),
                    "max": max(semantic_scores),
                    "avg": sum(semantic_scores) / len(semantic_scores),
                    "distribution": self._categorize_scores(semantic_scores)
                },
                "keyword_scores": {
                    "min": min(keyword_scores),
                    "max": max(keyword_scores),
                    "avg": sum(keyword_scores) / len(keyword_scores),
                    "distribution": self._categorize_scores(keyword_scores)
                },
                "hybrid_scores": {
                    "min": min(hybrid_scores),
                    "max": max(hybrid_scores),
                    "avg": sum(hybrid_scores) / len(hybrid_scores),
                    "distribution": self._categorize_scores(hybrid_scores)
                }
            }
        else:
            # Semantic-only
            scores = [match.get("score", 0) for match in matches]
            return {
                "semantic_scores": {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores),
                    "distribution": self._categorize_scores(scores)
                }
            }
    
    def _categorize_scores(self, scores: List[float]) -> Dict[str, int]:
        """Categorize scores into ranges"""
        categories = {
            "excellent": len([s for s in scores if s >= 0.8]),
            "good": len([s for s in scores if 0.6 <= s < 0.8]),
            "fair": len([s for s in scores if 0.4 <= s < 0.6]),
            "poor": len([s for s in scores if s < 0.4])
        }
        return categories
    
    def _analyze_course_types(self, matches: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze the types of courses recommended"""
        course_types = {}
        
        for match in matches:
            course_name = match.get("metadata", {}).get("course_name", "").lower()
            
            if any(keyword in course_name for keyword in ["machine learning", "ai", "artificial intelligence"]):
                course_types["ML/AI"] = course_types.get("ML/AI", 0) + 1
            elif any(keyword in course_name for keyword in ["data", "analytics", "statistics"]):
                course_types["Data Science"] = course_types.get("Data Science", 0) + 1
            elif any(keyword in course_name for keyword in ["web", "frontend", "backend", "full stack"]):
                course_types["Web Development"] = course_types.get("Web Development", 0) + 1
            elif any(keyword in course_name for keyword in ["software", "programming", "development"]):
                course_types["Software Engineering"] = course_types.get("Software Engineering", 0) + 1
            else:
                course_types["Other"] = course_types.get("Other", 0) + 1
        
        return course_types
    
    def _rate_performance(self, processing_time: float) -> str:
        """Rate the performance based on processing time"""
        if processing_time < 1.0:
            return "excellent"
        elif processing_time < 3.0:
            return "good"
        elif processing_time < 5.0:
            return "fair"
        else:
            return "poor"
    
    def _calculate_relevance_score(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate overall relevance score based on top scores"""
        if not matches:
            return 0.0
        
        # Get the best score (first match has highest score)
        if "hybrid_score" in matches[0]:
            best_score = matches[0].get("hybrid_score", 0)
        else:
            best_score = matches[0].get("score", 0)
        
        return best_score
    
    def _calculate_diversity_score(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate diversity score based on course types"""
        if not matches:
            return 0.0
        
        course_types = self._analyze_course_types(matches)
        unique_types = len(course_types)
        total_matches = len(matches)
        
        # Diversity score: more unique types = higher diversity
        return min(unique_types / total_matches, 1.0) if total_matches > 0 else 0.0
    
    def _analyze_score_quality(self, matches: List[Dict[str, Any]], request_type: str) -> Dict[str, Any]:
        """Analyze the quality of scores"""
        if not matches:
            return {"error": "No matches to analyze"}
        
        if request_type == "hybrid":
            hybrid_scores = [match.get("hybrid_score", 0) for match in matches]
            semantic_scores = [match.get("semantic_score", 0) for match in matches]
            keyword_scores = [match.get("keyword_score", 0) for match in matches]
            
            # Calculate improvements
            improvements = []
            for i in range(len(matches)):
                if i < len(semantic_scores) and i < len(hybrid_scores):
                    improvement = hybrid_scores[i] - semantic_scores[i]
                    improvements.append(improvement)
            
            return {
                "avg_hybrid_score": sum(hybrid_scores) / len(hybrid_scores),
                "avg_semantic_score": sum(semantic_scores) / len(semantic_scores),
                "avg_keyword_score": sum(keyword_scores) / len(keyword_scores),
                "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
                "positive_improvements": len([i for i in improvements if i > 0]),
                "total_improvements": len(improvements)
            }
        else:
            scores = [match.get("score", 0) for match in matches]
            return {
                "avg_score": sum(scores) / len(scores),
                "score_range": max(scores) - min(scores),
                "high_quality_matches": len([s for s in scores if s >= 0.7])
            }
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent recommendation logs"""
        try:
            log_files = sorted(self.logs_dir.glob("recommendation_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            recent_logs = []
            
            for log_file in log_files[:limit]:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        recent_logs.append({
                            "filename": log_file.name,
                            "recommendation_id": log_data.get("recommendation_id"),
                            "timestamp": log_data.get("timestamp"),
                            "user_email": log_data.get("user_info", {}).get("email"),
                            "request_type": log_data.get("request_info", {}).get("request_type"),
                            "total_matches": log_data.get("recommendation_results", {}).get("total_matches"),
                            "processing_time": log_data.get("request_info", {}).get("total_processing_time"),
                            "success": log_data.get("success", False)
                        })
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {str(e)}")
            
            return recent_logs
            
        except Exception as e:
            logger.error(f"Error getting recent logs: {str(e)}")
            return []
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics from all recommendation logs"""
        try:
            log_files = list(self.logs_dir.glob("recommendation_*.json"))
            
            if not log_files:
                return {"error": "No recommendation logs found"}
            
            total_requests = len(log_files)
            successful_requests = 0
            total_processing_time = 0
            request_types = {}
            total_matches = 0
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        
                        if log_data.get("success", False):
                            successful_requests += 1
                        
                        processing_time = log_data.get("request_info", {}).get("total_processing_time", 0)
                        total_processing_time += processing_time
                        
                        request_type = log_data.get("request_info", {}).get("request_type", "unknown")
                        request_types[request_type] = request_types.get(request_type, 0) + 1
                        
                        total_matches += log_data.get("recommendation_results", {}).get("total_matches", 0)
                        
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {str(e)}")
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "avg_processing_time": total_processing_time / total_requests if total_requests > 0 else 0,
                "total_processing_time": total_processing_time,
                "request_types": request_types,
                "total_matches_generated": total_matches,
                "avg_matches_per_request": total_matches / total_requests if total_requests > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting log statistics: {str(e)}")
            return {"error": str(e)}

    def log_recommendation_start(self, user_id: str, recommendation_type: RecommendationType, 
                               top_k: int, resume_text_length: int) -> str:
        """
        Log the start of a recommendation process.
        
        Args:
            user_id (str): User identifier
            recommendation_type (RecommendationType): Type of recommendation being performed
            top_k (int): Number of recommendations requested
            resume_text_length (int): Length of resume text being processed
            
        Returns:
            str: Unique recommendation ID for tracking
        """
        recommendation_id = str(uuid.uuid4())
        
        self.logger.info(f"🎯 RECOMMENDATION START - ID: {recommendation_id}")
        self.logger.info(f"   User: {user_id}")
        self.logger.info(f"   Type: {recommendation_type.value}")
        self.logger.info(f"   Top-K: {top_k}")
        self.logger.info(f"   Resume Length: {resume_text_length} chars")
        
        return recommendation_id

    def log_semantic_processing(self, recommendation_id: str, embedding_dim: int, 
                              processing_time: float, matches_found: int, 
                              top_scores: List[float], pinecone_status: bool):
        """
        Log semantic similarity processing details.
        
        Args:
            recommendation_id (str): Unique recommendation ID
            embedding_dim (int): Dimension of the embedding vector
            processing_time (float): Time taken for semantic processing
            matches_found (int): Number of semantic matches found
            top_scores (List[float]): Top semantic similarity scores
            pinecone_status (bool): Whether Pinecone connection was successful
        """
        self.logger.info(f"🧠 SEMANTIC PROCESSING - ID: {recommendation_id}")
        self.logger.info(f"   Embedding Dimensions: {embedding_dim}")
        self.logger.info(f"   Processing Time: {processing_time:.3f}s")
        self.logger.info(f"   Matches Found: {matches_found}")
        self.logger.info(f"   Top Scores: {[f'{score:.3f}' for score in top_scores[:5]]}")
        self.logger.info(f"   Pinecone Status: {'✅' if pinecone_status else '❌'}")
        
        if processing_time > 2.0:
            self.logger.warning(f"⚠️ Slow semantic processing: {processing_time:.3f}s")

    def log_keyword_processing(self, recommendation_id: str, extracted_keywords: int,
                             matching_keywords: int, processing_time: float,
                             redis_status: bool, keyword_matches: List[str]):
        """
        Log keyword matching processing details.
        
        Args:
            recommendation_id (str): Unique recommendation ID
            extracted_keywords (int): Number of keywords extracted from resume
            matching_keywords (int): Number of keywords found in Redis
            processing_time (float): Time taken for keyword processing
            redis_status (bool): Whether Redis connection was successful
            keyword_matches (List[str]): List of matching keywords
        """
        self.logger.info(f"🔍 KEYWORD PROCESSING - ID: {recommendation_id}")
        self.logger.info(f"   Extracted Keywords: {extracted_keywords}")
        self.logger.info(f"   Matching Keywords: {matching_keywords}")
        self.logger.info(f"   Processing Time: {processing_time:.3f}s")
        self.logger.info(f"   Redis Status: {'✅' if redis_status else '❌'}")
        self.logger.info(f"   Matching Keywords: {keyword_matches[:10]}...")
        
        if processing_time > 1.0:
            self.logger.warning(f"⚠️ Slow keyword processing: {processing_time:.3f}s")

    def log_hybrid_scoring(self, recommendation_id: str, semantic_weight: float,
                          keyword_weight: float, score_improvements: List[float]):
        """
        Log hybrid scoring details and improvements.
        
        Args:
            recommendation_id (str): Unique recommendation ID
            semantic_weight (float): Weight given to semantic scores
            keyword_weight (float): Weight given to keyword scores
            score_improvements (List[float]): How much hybrid improved over semantic
        """
        self.logger.info(f"⚖️ HYBRID SCORING - ID: {recommendation_id}")
        self.logger.info(f"   Semantic Weight: {semantic_weight:.2f}")
        self.logger.info(f"   Keyword Weight: {keyword_weight:.2f}")
        
        if score_improvements:
            avg_improvement = sum(score_improvements) / len(score_improvements)
            max_improvement = max(score_improvements)
            self.logger.info(f"   Average Score Improvement: {avg_improvement:.3f}")
            self.logger.info(f"   Max Score Improvement: {max_improvement:.3f}")
            
            if avg_improvement > 0.1:
                self.logger.info(f"🎉 Significant improvement from hybrid approach!")

    def log_course_recommendations(self, recommendation_id: str, 
                                 recommendations: List[CourseRecommendationDetail]):
        """
        Log detailed information about each recommended course.
        
        Args:
            recommendation_id (str): Unique recommendation ID
            recommendations (List[CourseRecommendationDetail]): List of course recommendations
        """
        self.logger.info(f"📚 COURSE RECOMMENDATIONS - ID: {recommendation_id}")
        
        for i, rec in enumerate(recommendations[:5], 1):  # Log top 5
            self.logger.info(f"   #{i} - {rec.course_name} ({rec.course_id})")
            self.logger.info(f"      Semantic: {rec.semantic_score:.3f}")
            self.logger.info(f"      Keyword: {rec.keyword_score:.3f}")
            self.logger.info(f"      Hybrid: {rec.hybrid_score:.3f}")
            if rec.score_improvement:
                self.logger.info(f"      Improvement: {rec.score_improvement:+.3f}")
            self.logger.info(f"      Matching Keywords: {rec.matching_keywords[:5]}...")
            self.logger.info(f"      Course Skills: {rec.course_skills[:5]}...")

    def log_recommendation_completion(self, metrics: RecommendationMetrics):
        """
        Log comprehensive recommendation completion metrics.
        
        Args:
            metrics (RecommendationMetrics): Complete metrics for the recommendation
        """
        self.logger.info(f"✅ RECOMMENDATION COMPLETED - ID: {metrics.recommendation_id}")
        self.logger.info(f"   Total Processing Time: {metrics.total_processing_time:.3f}s")
        self.logger.info(f"   Final Recommendations: {metrics.final_recommendations}")
        self.logger.info(f"   Top Hybrid Score: {metrics.top_hybrid_score:.3f}")
        self.logger.info(f"   Top Semantic Score: {metrics.top_semantic_score:.3f}")
        self.logger.info(f"   Top Keyword Score: {metrics.top_keyword_score:.3f}")
        
        # Performance analysis
        if metrics.total_processing_time > 5.0:
            self.logger.warning(f"⚠️ Slow recommendation processing: {metrics.total_processing_time:.3f}s")
        
        if metrics.final_recommendations == 0:
            self.logger.warning(f"⚠️ No recommendations generated")
        
        # Save detailed metrics to JSON file for analysis
        self._save_metrics_to_file(metrics)

    def log_error(self, recommendation_id: str, error: Exception, context: str = ""):
        """
        Log recommendation errors with context.
        
        Args:
            recommendation_id (str): Unique recommendation ID
            error (Exception): The error that occurred
            context (str): Additional context about the error
        """
        self.logger.error(f"❌ RECOMMENDATION ERROR - ID: {recommendation_id}")
        self.logger.error(f"   Context: {context}")
        self.logger.error(f"   Error Type: {type(error).__name__}")
        self.logger.error(f"   Error Message: {str(error)}")

    def _save_metrics_to_file(self, metrics: RecommendationMetrics):
        """
        Save detailed metrics to a JSON file for later analysis.
        
        Args:
            metrics (RecommendationMetrics): Metrics to save
        """
        try:
            # Convert datetime to string for JSON serialization
            metrics_dict = asdict(metrics)
            metrics_dict['timestamp'] = metrics.timestamp.isoformat()
            metrics_dict['recommendation_type'] = metrics.recommendation_type.value
            
            # Create logs directory if it doesn't exist
            import os
            os.makedirs('logs', exist_ok=True)
            
            # Save to file with timestamp
            filename = f"logs/recommendation_metrics_{metrics.recommendation_id}.json"
            with open(filename, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
                
            self.logger.debug(f"Metrics saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics to file: {str(e)}")

    def get_performance_summary(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Generate a performance summary for a user over a specified time period.
        
        Args:
            user_id (str): User identifier
            days (int): Number of days to analyze
            
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        # This would typically query a database of stored metrics
        # For now, return a placeholder structure
        return {
            "user_id": user_id,
            "period_days": days,
            "total_recommendations": 0,
            "average_processing_time": 0.0,
            "average_hybrid_score": 0.0,
            "average_semantic_score": 0.0,
            "average_keyword_score": 0.0,
            "success_rate": 0.0,
            "most_common_keywords": [],
            "recommendation_types_used": []
        }

    def log_system_health(self, redis_status: bool, pinecone_status: bool, 
                         database_status: bool):
        """
        Log system health status for monitoring.
        
        Args:
            redis_status (bool): Redis connection status
            pinecone_status (bool): Pinecone connection status
            database_status (bool): Database connection status
        """
        self.logger.info(f"🏥 SYSTEM HEALTH CHECK")
        self.logger.info(f"   Redis: {'✅' if redis_status else '❌'}")
        self.logger.info(f"   Pinecone: {'✅' if pinecone_status else '❌'}")
        self.logger.info(f"   Database: {'✅' if database_status else '❌'}")
        
        if not all([redis_status, pinecone_status, database_status]):
            self.logger.warning(f"⚠️ System health issues detected") 