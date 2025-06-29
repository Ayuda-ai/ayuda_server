from passlib.context import CryptContext
from datetime import datetime, timedelta
from app.core.config import settings
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from app.models.user import User
from app.db.session import get_db

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Validate JWT token and retrieve the current authenticated user.
    
    This function is used as a dependency in protected endpoints to ensure
    that the request is made by an authenticated user. It decodes the JWT token,
    extracts the user email from the subject claim, and retrieves the user
    from the database.
    
    Args:
        token (str): JWT access token from Authorization header
        db (Session): Database session dependency
        
    Returns:
        User: The authenticated user object
        
    Raises:
        HTTPException: If token is invalid, expired, or user not found (status_code=401)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_email: str = payload.get("sub")
        if user_email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == user_email).first()
    if user is None:
        raise credentials_exception

    return user

def get_password_hash(password: str) -> str:
    """
    Hash a plain text password using bcrypt.
    
    This function takes a plain text password and returns a secure hash
    that can be safely stored in the database. The hash includes a salt
    and is generated using the bcrypt algorithm.
    
    Args:
        password (str): Plain text password to hash
        
    Returns:
        str: Hashed password string
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against a hashed password.
    
    This function compares a plain text password with a previously hashed
    password to determine if they match. It handles the salt extraction
    and comparison automatically.
    
    Args:
        plain_password (str): Plain text password to verify
        hashed_password (str): Previously hashed password to compare against
        
    Returns:
        bool: True if passwords match, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    """
    Create a JWT access token with expiration.
    
    This function creates a JWT token containing user information and an
    expiration time. The token is signed using the application's secret key
    and can be used for API authentication.
    
    The token payload must include a 'sub' field (subject) which typically
    contains the user's email address. The expiration time is set based on
    the ACCESS_TOKEN_EXPIRE_MINUTES configuration setting.
    
    Args:
        data (dict): Token payload data, must include 'sub' field
        
    Returns:
        str: JWT token string
        
    Raises:
        ValueError: If the payload doesn't include a 'sub' field
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    # Make sure 'sub' field exists
    if "sub" not in to_encode:
        raise ValueError("Token payload must include a 'sub' field (user id)")

    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
