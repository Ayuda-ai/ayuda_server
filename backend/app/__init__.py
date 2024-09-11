from flask import Flask
from .database.db import get_mongodb
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    # configure app here with app.config settings

    # Enable CORS for all routes and set specific origins
    cors = CORS(app, origins=[
        "http://localhost:3000/",
    ])

    # Initialize MongoDB
    mongo = get_mongodb(app)

    # Store the mongo instance in app for global access, if necessary
    app.mongo = mongo

    # Importing API modules
    from app.api import user_api, auth_api, admin_api, courses_api

    # Register Blueprints or add routes from your API
    app.register_blueprint(user_api.blueprint)
    app.register_blueprint(auth_api.blueprint)
    app.register_blueprint(admin_api.blueprint)
    app.register_blueprint(courses_api.blueprint)

    return app
