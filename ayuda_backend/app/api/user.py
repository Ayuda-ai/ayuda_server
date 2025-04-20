from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.schemas.user import UserCreate, UserOut
from app.db.session import get_db
from app.models.user import User
from app.models.access_code import AccessCode
from app.models.allowed_domain import AllowedDomain
from app.core.security import get_password_hash
from app.core.config import settings

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

@router.post("/signup", response_model=UserOut)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    # Validate email domain
    domain = user_data.email.split("@")[-1]
    if not db.query(AllowedDomain).filter(AllowedDomain.domain == domain).first():
        raise HTTPException(status_code=400, detail="Email domain not allowed")

    # Validate access code
    code = db.query(AccessCode).filter(
        AccessCode.code == user_data.access_code,
        AccessCode.count > 0,
        AccessCode.status == True
    ).first()
    if not code:
        raise HTTPException(status_code=400, detail="Invalid or expired access code")

    # Create user
    new_user = User(
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        university=user_data.university,
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        dob=user_data.dob,
        major=user_data.major
    )
    db.add(new_user)
    code.count -= 1
    db.commit()
    db.refresh(new_user)

    return new_user

@router.get("/me", response_model=UserOut)
def get_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        email = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user
