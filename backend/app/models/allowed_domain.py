from sqlalchemy import Column, String
from app.db.session import Base

class AllowedDomain(Base):
    __tablename__ = "allowed_domains"

    domain = Column(String(100), primary_key=True)
