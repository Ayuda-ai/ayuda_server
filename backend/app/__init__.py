from flask import Flask, send_from_directory
from .database.db import get_mongodb
from flask_cors import CORS
import os


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
            print(
                f"Collection '{collection}' created in database '{db.name}'.")


def create_app():
    app = Flask(__name__,
                static_folder=os.path.join(os.path.dirname(os.path.abspath(
                    __file__)), "../static"),  # Path to the React build directory
                static_url_path="/"  # Serve static files from root
                )
    # configure app here with app.config settings

    # TESTS
    print(f"Static folder path: {os.path.abspath(app.static_folder)}")

    # Enable CORS for all routes and set specific origins
    cors = CORS(app, origins=[
        "http://localhost:3000",    # Dev server
        # Allow all origins (update for production security)
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

    # Route for serving the React app
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_react(path):
        print(f"Requested path: {path}")
        resolved_path = os.path.join(app.static_folder, path)
        print(f"Resolved path: {resolved_path}")

        # Serve the requested file if it exists
        if path != '' and os.path.exists(resolved_path):
            print(f"Serving file: {resolved_path}")
            return send_from_directory(app.static_folder, path)

        # Otherwise, serve React's index.html for the root or unmatched paths
        index_path = os.path.join(app.static_folder, 'index.html')
        print(f"Serving index.html from: {index_path}")
        return send_from_directory(app.static_folder, 'index.html')

    return app
