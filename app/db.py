from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from app.config import settings
from sqlalchemy import String

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class ModelResult(Base):
    __tablename__ = "model_result"
    model_name = Column(String, nullable=True)

    id = Column(Integer, primary_key=True, index=True)
    feature_1 = Column(Float, nullable=False)
    feature_2 = Column(Float, nullable=False)
    feature_3 = Column(Float, nullable=False)
    feature_4 = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
