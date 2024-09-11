from flask import request, jsonify, Blueprint, current_app
from ..utils.date_time import get_datetime
from ..utils.api_base_route import get_api_base_route

blueprint = Blueprint('user_api', __name__)

baseRoute = get_api_base_route()


@blueprint.route(baseRoute + '/users/get-user', methods=['GET'])
def get_user_by_email():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email parameter is required"}), 400

    mongo = current_app.mongo
    # Perform an update in the database to set the last_login time
    update_result = mongo.db.Users.update_one(
        {"email": email},
        {"$set": {"last_login": get_datetime()}}
    )

    # Check if the update was successful (i.e., if any document was actually found and updated)
    if update_result.matched_count == 0:
        return jsonify({"error": "User not found"}), 404

    # Retrieve the updated user to return in response
    user = mongo.db.Users.find_one({"email": email})
    if user:
        user['_id'] = str(user['_id'])  # Convert ObjectId to string
        return jsonify(user), 200
    else:
        return jsonify({"error": "User not found after update"}), 404


@blueprint.route('/submit-profile', methods=['POST'])
def submit_profile():
    if request.method == 'POST':
        # get mongo instance
        mongo = current_app.mongo
        # Assuming the user data is sent in a JSON format
        user_data = request.json
        user_data['last_login'] = get_datetime()
        print(user_data)
        # Insert the data into the 'Users' collection
        user_id = mongo.db.Users.insert_one(user_data).inserted_id

        # Return a success response
        return jsonify({"message": "Profile submitted successfully", "user_id": str(user_id)}), 201
    else:
        return jsonify({"error": "Method not allowed"}), 405


@blueprint.route(baseRoute + '/users/update-user', methods=['PATCH'])
def update_user():
    try:
        data = request.json
        email = data.get('email')
        skills = data.get('skills')
        career_path = data.get('career_path')

        if not email or not skills or not career_path:
            return jsonify({'error': 'Email, skills, and career path are required'}), 400

        # Find the user by email
        # get mongo instance
        mongo = current_app.mongo
        user = mongo.db.Users.find_one({'email': email})

        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Update the user's skills and career path
        # If the skills in the array are not in user document, add them
        # If the career_path in the array are not in user document, add them
        update_result = mongo.db.Users.update_one(
            {'email': email},
            {'$set': {
                'skills': skills,
                'career_path': career_path
            }}
        )

        if update_result.modified_count == 0:
            return jsonify({'error': 'No changes made to the user'}), 200

        return jsonify({'message': 'User updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
