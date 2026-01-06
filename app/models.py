from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from .database import Base

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(Base):
    __tablename__ = "holdings"
    id = Column(Integer, primary_key=True, index=True)

    ticker = Column(String, nullable=False)
    weight = Column(Float, nullable=False)

    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    portfolio = relationship("Portfolio", back_populates="holdings")
