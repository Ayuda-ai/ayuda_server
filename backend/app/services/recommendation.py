'''
    This is a Recommendation Service
    It is responsible for showing recommendations when the /courses/recommendations API is called
    This acts as a driver from the Client API call to the handling of various utility methods to generate recommendations

    Flow of Recommendation

    User -------> calls /courses/recommendations
    API endpoint -------> calls recommend_driver() function below
    recommend_driver() --------> calls courses_data.get_courses_data() to get courses data
    recommend_driver() --------> calls dataframe.make_dataframe() to get the courses df
    recommend_driver() --------> gets filters from user profile (filters for Career path & Previous Courses domain ??? Think about implementation)
    recommend_driver() --------> gets list of skills associated from the user model
    recommend_driver() --------> passes user_skills and courses_skills to vectorizer
    recommend_driver() --------> passes the vectorized lists to the similiarity function
    Returns the list of recommended course objects/course_ids

'''
import json
from flask import jsonify
import numpy as np
from .courses_data import get_courses_data
from ..utils.vectorizer import get_vectorized_user_matrix, get_vectorized_course_matrix
from ..utils.similarity import get_cosine_similarity
from .user_data import get_current_user_data
from ..utils.dataframe import make_threefield_dataframe, sort_dataframe, make_dataframe
import pandas as pd

def recommend_driver():
    # get organized course data from the DB
    courses_data = get_courses_data()
    course_skills = make_dataframe(courses_data)
    vectorized_course_skills, course_vectorizer = get_vectorized_course_matrix(course_skills)

    ## User skills
    user_data = get_current_user_data()
    user_skills = json.loads(json.dumps(user_data, indent=4))['skills']
    user_skills_combined = ", ".join(user_skills)
    user_skills = [user_skills_combined]
    vectorized_user_skills = get_vectorized_user_matrix(user_skills_combined, vectorizer=course_vectorizer)

    ## Compute Similarity
    similarity_scores = get_cosine_similarity(vectorized_user_skills, vectorized_course_skills)

    # Displaying similarity scores with course IDs
    recommendation_df = pd.DataFrame({
        'course_id': course_skills['course_id'],
        'course_name': course_skills['course_name'],
        'similarity_score': similarity_scores
    })

    recommendations = sort_dataframe(recommendation_df, sortBy='similarity_score', ascending=False)
    filtered_recommendations = recommendations[recommendations['similarity_score'] > 0.2]

    ## Convert to dictionary, then convert to dataframe
    recommendation_dict = {
        'course_id': filtered_recommendations['course_id'],
        'course_name': filtered_recommendations['course_name'],
        'similarity_score': filtered_recommendations['similarity_score']
    }

    dict_df = pd.DataFrame(recommendation_dict)
    # convert the above dataframe to list of dicts
    list_of_recomms = dict_df.to_dict(orient='records')

    recomms = json.dumps(list_of_recomms, indent=4)
    # return filtered_recommendations.to_json()
    print(type(recomms))
    return recomms
