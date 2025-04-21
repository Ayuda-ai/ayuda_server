from sqlalchemy import Column, String, Integer, Boolean
from app.db.session import Base

class AccessCode(Base):
    __tablename__ = "access_codes"

    code = Column(String(64), primary_key=True)
    count = Column(Integer, nullable=False, default=0)
    status = Column(Boolean, nullable=False, default=True)
