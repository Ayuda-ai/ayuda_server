from flask import Flask
from .database.db import get_mongodb
from flask_cors import CORS

def init_db(mongo):
    """Initialize the MongoDB database and create required collections."""
    # Access the database using PyMongo
    db = mongo.db
    
    # List of collections to create
    collections_to_create = ["AccessCodes", "Courses", "Users"]

    print("Creating the database instance...")
    # Check if collections exist, create them if they don't
    existing_collections = db.list_collection_names()
    for collection in collections_to_create:
        if collection not in existing_collections:
            db.create_collection(collection)
            print(f"Collection '{collection}' created in database '{db.name}'.")

def create_app():
    app = Flask(__name__)
    # configure app here with app.config settings

    # Enable CORS for all routes and set specific origins
    cors = CORS(app, origins=[
        "http://localhost:3000",
        "*",   
    ], supports_credentials=True)

    # Initialize MongoDB
    mongo = get_mongodb(app)

    # Store the mongo instance in app for global access, if necessary
    app.mongo = mongo

    # Initialize the database and collections
    with app.app_context():
        init_db(mongo)

    print("Registering all the API endpoints...")
    # Importing API modules
    from app.api import user_api, auth_api, admin_api, courses_api

    # Register Blueprints or add routes from your API
    print("Registering USER API")
    app.register_blueprint(user_api.blueprint)
    print("Registering AUTH API")
    app.register_blueprint(auth_api.blueprint)
    print("Registering ADMIN API")
    app.register_blueprint(admin_api.blueprint)
    print("Registering COURSES API")
    app.register_blueprint(courses_api.blueprint)

    return app
