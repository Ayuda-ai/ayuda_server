from fastapi import FastAPI
from app.api import user, auth, admin
from app.db.session import engine
from app.models import user as user_model
from app.models import access_code, allowed_domain
from alembic import command
from alembic.config import Config
import os

# Initialize FastAPI application
app = FastAPI(
    title="Ayuda Course Recommender",
    description="AI-powered course recommendation system using resume embeddings and semantic search",
    version="1.0.0"
)

def run_migrations():
    """
    Run Alembic database migrations programmatically.
    
    This function executes database migrations during application startup
    to ensure the database schema is up to date. It uses Alembic to
    apply any pending migrations to the database.
    
    The function:
    1. Determines the base directory of the project
    2. Loads the Alembic configuration from alembic.ini
    3. Runs all pending migrations up to the latest version
    
    This ensures that the database schema matches the current application
    model definitions before the API starts accepting requests.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_cfg = Config(os.path.join(base_dir, "alembic.ini"))
    command.upgrade(alembic_cfg, "head")

@app.on_event("startup")
def startup_event():
    """
    Application startup event handler.
    
    This function is called when the FastAPI application starts up.
    It performs necessary initialization tasks such as:
    - Running database migrations to ensure schema is up to date
    - Setting up any required application state
    
    This ensures the application is properly configured before
    accepting any requests.
    """
    print("🔄 Running migrations at startup...")
    run_migrations()
    print("✅ Migrations applied successfully.")

# Include API routers with their respective prefixes and tags
# Users router - handles user registration, profile management, and resume processing
app.include_router(user.router, prefix="/users", tags=["Users"])

# Auth router - handles user authentication and JWT token generation
app.include_router(auth.router, prefix="/auth", tags=["Auth"])

# Admin router - handles administrative functions like course import
app.include_router(admin.router, prefix="/admin", tags=["Admin Utilities"])
