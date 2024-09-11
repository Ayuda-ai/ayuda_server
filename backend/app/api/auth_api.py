from flask import Flask, request, jsonify, Blueprint, current_app
import requests
from ..database.db import get_config
from ..utils.date_time import get_datetime
from ..utils.api_base_route import get_api_base_route

baseRoute = get_api_base_route()

blueprint = Blueprint('auth_api', __name__)

config = get_config()

# Hardcoded user data for demonstration purposes.
# In a real application, you should use a database and secure password handling.
users = {
    "vijayakumarhebbar.k@northeastern.edu": "PassApr11"
}

# access_codes = ["PassApr11", "Code0Ayuda"]

# Accepted mail domain
accepted_mail_domain = "@northeastern.edu"

# registered emails
registered_users = ["vijayakumarhebbar.k@northeastern.edu", "sahay.r@northeastern.edu"]

'''
Get all access codes from DB
'''
@blueprint.route(baseRoute + '/auth/codes', methods=['GET'])
def get_access_codes():
    mongo = current_app.mongo
    codes = mongo.db.AccessCodes.find()
    codes_list = list(codes)
    # MongoDB's ObjectIds are not JSON serializable by default, so we convert them to strings
    for code in codes_list:
        code['_id'] = str(code['_id'])
    return jsonify(codes_list)

'''
The below method logs users in based on the access code entered
'''
@blueprint.route(baseRoute + '/auth/access', methods=['POST'])
def code_login():
    data = request.json
    access_code = data.get('access_code')
    mongo = current_app.mongo

    # check if the entered access_code is present in the DB
    code_found = mongo.db.AccessCodes.find_one({"code": access_code})
    print(code_found)

    if code_found:
        # Check if the validity of the code is greater than 0
        if code_found['validity'] > 0:
            # decrement the count of validity of the access code
            mongo.db.AccessCodes.update_one(
                {"_id": code_found["_id"]},
                {"$inc": {"validity": -1}}
            )
            return jsonify({"message": "Login successful", "status": "success"}), 200
        else:
            return jsonify({"message": "Invalid Access Code", "status": "error"}), 401
    else:
        return jsonify({"message": "Invalid Access Code", "status": "error"}), 401

'''
This API endpoint logs users based on their email ID
'''
@blueprint.route(baseRoute + '/auth/login', methods=['POST'])
def email_login():
    data = request.json
    email = data.get('email')
    mongo = current_app.mongo

    # check if the entered mail id is not from NEU
    if not email.endswith(accepted_mail_domain):
        return jsonify({"message": "Use Northeastern Email ID only", "status": "error"}), 401

    user = mongo.db.Users.find_one({"email": email})

    # check this on the DB
    if user:
        # update last login and redirect to homepage
        mongo.db.Users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": get_datetime()}}
        )
        user['_id'] = str(user['_id'])  # Convert ObjectId to string
        return jsonify(user), 200
    else:
        return jsonify({"message": "Email not found in DB, redirecting to profile builder page", "status": "error"}), 400

'''
    This API registers user to the app by creating a User model on the DB
'''
@blueprint.route(baseRoute + '/auth/register', methods=['POST'])
def register():
    if request.method == 'POST':
        # get mongo instance
        mongo = current_app.mongo
        # Assuming the user data is sent in a JSON format
        user_data = request.json

        # check if the entered email is an accepted email domain
        if not user_data['email'].endswith(accepted_mail_domain):
            return jsonify({"message": "Use Northeastern Email ID only", "status": "error"}), 401
            
        user_data['last_login'] = get_datetime()
        print(user_data)
        # Insert the data into the 'Users' collection
        user_id = mongo.db.Users.insert_one(user_data).inserted_id

        # Return a success response
        return jsonify({"message": "Profile submitted successfully", "user_id": str(user_id)}), 201
    else:
        return jsonify({"error": "Method not allowed"}), 405