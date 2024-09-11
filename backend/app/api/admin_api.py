from flask import request, jsonify, Blueprint, current_app
from ..database.db import get_config
from ..utils.api_base_route import get_api_base_route
import pandas as pd

baseRoute = get_api_base_route()

blueprint = Blueprint('admin_api', __name__)

config = get_config()

'''
This Admin API allows admins to add new access codes for users
access_code = {
    'admin_key'
    'code'
    'validity'
}
'''
@blueprint.route(baseRoute + '/admin/add-code', methods=['POST'])
def add_access_codes():
    if request.method == 'POST':
        # get mongo instance
        mongo = current_app.mongo
        access_code = request.json

        print(config['ADMIN']['ADMIN_KEY'])

        if not access_code or 'admin_key' not in access_code:
            return jsonify({"error": "Missing 'admin_key' field"}), 400
        elif access_code['admin_key'] != config['ADMIN']['ADMIN_KEY']:
            return jsonify({"error": "Wrong 'admin_key' entered!"}), 400

        if 'validity' not in access_code:
            return jsonify({"error": "Missing 'validity' field"}), 400

        # Validate that the "code" field exists
        if 'code' not in access_code:
            return jsonify({"error": "Missing 'code' field"}), 400

        # result = mongo.db.AccessCodes.insert_one({"code": access_code['code']})
        result = mongo.db.AccessCodes.insert_one(access_code)
        # Respond with success
        return jsonify({"message": "Access Code added successfully", "id": str(result.inserted_id)}), 201
    else:
        return jsonify({"error": "Method not allowed"}), 405

'''
    This API is used to populate Courses collection with data present in the mgen_courses.csv
'''
@blueprint.route(baseRoute + '/admin/upload-courses', methods=['POST'])
def upload_courses():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        mongo = current_app.mongo

        # Convert DataFrame to dictionary
        data_dict = data.to_dict(orient='records')

        # Insert documents into MongoDB
        result = mongo.db.Courses.insert_many(data_dict)
        return jsonify({"message": f"Inserted {len(result.inserted_ids)} documents."}), 201
    
    return jsonify({"error": "Unsupported file type"}), 400