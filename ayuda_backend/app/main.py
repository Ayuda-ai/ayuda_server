from fastapi import FastAPI
from app.api import user, auth
from app.db.session import engine
from app.models import user as user_model
from app.models import access_code, allowed_domain

# Create tables (only for initial development, use Alembic for production)
user_model.Base.metadata.create_all(bind=engine)
access_code.Base.metadata.create_all(bind=engine)
allowed_domain.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Create tables if they don't exist
@app.on_event("startup")
def create_required_tables():
    print("🔄 Checking for required tables...")
    user_model.Base.metadata.create_all(bind=engine)
    access_code.Base.metadata.create_all(bind=engine)
    allowed_domain.Base.metadata.create_all(bind=engine)
    print("✅ Tables checked/created")

app.include_router(user.router, prefix="/users", tags=["Users"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
