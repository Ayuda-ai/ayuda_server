import requests
from flask import jsonify


def get_courses_data():
    response = requests.get('http://127.0.0.1:5000/api/v1/courses/get-all')
    if response.status_code == 200:
        courses_data = response.json()  # Parses the JSON response data into a dictionary
        data = {
            '_id': [str(course['_id']) for course in courses_data],
            'course_id': [course['course_id'] for course in courses_data],
            'course_name': [course['course_name'] for course in courses_data],
            'course_description': [course['course_description'] for course in courses_data],
            'prerequisite_1': [course['prerequisite_1'] for course in courses_data],
            'prerequisite_2': [course['prerequisite_2'] for course in courses_data],
            'prerequisite_3': [course['prerequisite_3'] for course in courses_data],
            'major': [course['major'] for course in courses_data],
            'domain_1': [course['domain_1'] for course in courses_data],
            'domain_2': [course['domain_2'] for course in courses_data],
            'skills_associated': [course['skills_associated'] for course in courses_data]
        }
        return data
    else:
        return jsonify({"error": "Failed to convert courses to dataframe"}), response.status_code

    # Prepare data for DataFrame
# data = {
#     '_id': [str(course['_id'])] for course in course_data
#     'course_id': [course['course_id'] for course in courses_data],
#     'course_name': [course['course_name'] for course in courses_data],
#     'course_description': [course['course_description'] for course in courses_data],
#     'prerequisite_1': [course['prerequisite_1'] for course in courses_data],
#     'prerequisite_2': [course['prerequisite_2'] for course in courses_data],
#     'prerequisite_3': [course['prerequisite_3'] for course in courses_data],
#     'major': [course['major'] for course in courses_data],
#     'domain_1': [course['domain_1'] for course in courses_data],
#     'domain_2': [course['domain_2'] for course in courses_data],
#     'skills_associated': [course['skills_associated'] for course in courses_data]
# }

# # Create DataFrame
# courses_df = pd.DataFrame(data)
