from flask import request, jsonify, Blueprint, current_app
from ..utils.api_base_route import get_api_base_route
from ..services.recommendation import recommend_driver
import json

blueprint = Blueprint('courses_api', __name__)

baseRoute = get_api_base_route()

'''
    This API is used to fetch all Course objects from the Courses Collection in the DB
'''
@blueprint.route(baseRoute + '/courses/get-all', methods=['GET'])
def get_all_courses():
    mongo = current_app.mongo
    # get data
    courses_data = list(mongo.db.Courses.find({}))
    # Convert the '_id' field from ObjectId to string to make it JSON serializable
    for item in courses_data:
        item['_id'] = str(item['_id'])
    return jsonify(courses_data)


@blueprint.route(baseRoute + '/courses/get-course', methods=['GET'])
def get_course_by_courseid():
    mongo = current_app.mongo
    course_id = request.args.get('course_id')
    course = mongo.db.Courses.find_one({"course_id": course_id})
    if course:
        course['_id'] = str(course['_id'])  # Convert ObjectId to string
        return jsonify(course), 200
    else:
        return jsonify({"error": "Course not found"}), 404


@blueprint.route(baseRoute + '/courses/recommendations', methods=['GET'])
def get_course_recommendations():
    mongo = current_app.mongo
    courses_data = recommend_driver()

    return courses_data