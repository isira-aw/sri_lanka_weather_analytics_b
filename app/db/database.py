from sqlalchemy import create_engine, Column, Integer, Float, String, Date, ForeignKey, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

# SQLite database - NO SETUP NEEDED! Just a file.
DATABASE_URL = "sqlite:///./weather_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Location(Base):
    __tablename__ = "location"
    
    location_id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)
    utc_offset_seconds = Column(Integer)
    timezone = Column(String(100))
    timezone_abbreviation = Column(String(10))
    city_name = Column(String(100), nullable=False)
    
    weather_records = relationship("Weather", back_populates="location")

class Weather(Base):
    __tablename__ = "weather"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey("location.location_id"), nullable=False)
    date = Column(Date, nullable=False)
    weather_code = Column(Integer)
    temperature_2m_max = Column(Float)
    temperature_2m_min = Column(Float)
    temperature_2m_mean = Column(Float)
    apparent_temperature_max = Column(Float)
    apparent_temperature_min = Column(Float)
    apparent_temperature_mean = Column(Float)
    daylight_duration = Column(Float)
    sunshine_duration = Column(Float)
    precipitation_sum = Column(Float)
    rain_sum = Column(Float)
    precipitation_hours = Column(Float)
    wind_speed_10m_max = Column(Float)
    wind_gusts_10m_max = Column(Float)
    wind_direction_10m_dominant = Column(Float)
    shortwave_radiation_sum = Column(Float)
    et0_fao_evapotranspiration = Column(Float)
    sunrise = Column(Time)
    sunset = Column(Time)
    
    location = relationship("Location", back_populates="weather_records")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database - creates tables automatically"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")
