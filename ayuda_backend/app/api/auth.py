from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
import logging

from app.db.session import get_db
from app.models.user import User
from app.schemas.token import Token
from app.core.security import verify_password, create_access_token

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Authenticate user and generate JWT access token.
    
    This endpoint handles user authentication using email and password.
    It verifies the provided credentials against the database and generates
    a JWT access token upon successful authentication.
    
    The endpoint uses OAuth2PasswordRequestForm which expects:
    - username: User's email address
    - password: User's password
    
    The password is verified against the hashed password stored in the database.
    If authentication succeeds, a JWT token is generated with the user's email
    as the subject claim.
    
    Args:
        form_data (OAuth2PasswordRequestForm): Form data containing email (username) and password
        db (Session): Database session dependency
        
    Returns:
        Token: Authentication response containing:
            - access_token: JWT token for API access
            - token_type: Token type (always "bearer")
            
    Raises:
        HTTPException: If credentials are invalid (status_code=401)
    """
    logger.info(f"Login attempt for email: {form_data.username}")
    
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        logger.warning(f"Failed login attempt for email: {form_data.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.email})
    logger.info(f"Successful login for user: {user.email}")
    return {"access_token": token, "token_type": "bearer"}
