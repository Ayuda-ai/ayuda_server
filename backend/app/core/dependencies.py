from fastapi import Depends, HTTPException, status
from app.models.user import User
from app.core.security import get_current_user  # Assuming you already have a way to get current user from JWT/session

def admin_required(current_user: User = Depends(get_current_user)):
    """
    Dependency function to ensure the current user has admin privileges.
    
    This function is used as a dependency in admin-only endpoints to verify
    that the authenticated user has the "ADMIN" role. It first gets the
    current user from the JWT token, then checks their role.
    
    Args:
        current_user (User): The authenticated user object (from get_current_user dependency)
        
    Returns:
        User: The authenticated admin user object
        
    Raises:
        HTTPException: If user is not an admin (status_code=403)
    """
    if current_user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required."
        )
    return current_user
