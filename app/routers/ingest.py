from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db, Location, Weather
import csv
import io
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class WeatherInsert(BaseModel):
    location_id: int
    date: str
    weather_code: Optional[int] = None
    temperature_2m_max: Optional[float] = None
    temperature_2m_min: Optional[float] = None
    temperature_2m_mean: Optional[float] = None
    apparent_temperature_max: Optional[float] = None
    apparent_temperature_min: Optional[float] = None
    apparent_temperature_mean: Optional[float] = None
    daylight_duration: Optional[float] = None
    sunshine_duration: Optional[float] = None
    precipitation_sum: Optional[float] = None
    rain_sum: Optional[float] = None
    precipitation_hours: Optional[float] = None
    wind_speed_10m_max: Optional[float] = None
    wind_gusts_10m_max: Optional[float] = None
    wind_direction_10m_dominant: Optional[float] = None
    shortwave_radiation_sum: Optional[float] = None
    et0_fao_evapotranspiration: Optional[float] = None
    sunrise: Optional[str] = None
    sunset: Optional[str] = None

@router.post("/upload-location")
async def upload_location_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload location CSV and insert all rows into database"""
    try:
        content = await file.read()
        decoded = content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(decoded))
        
        count = 0
        for row in reader:
            location = Location(
                location_id=int(row['location_id']),
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                elevation=float(row['elevation']) if row['elevation'] else None,
                utc_offset_seconds=int(row['utc_offset_seconds']) if row['utc_offset_seconds'] else None,
                timezone=row['timezone'],
                timezone_abbreviation=row['timezone_abbreviation'],
                city_name=row['city_name']
            )
            db.merge(location)
            count += 1
        
        db.commit()
        return {"message": f"Successfully uploaded {count} location records", "count": count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error uploading location data: {str(e)}")

@router.post("/upload-weather")
async def upload_weather_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload weather CSV and insert all rows into database"""
    try:
        content = await file.read()
        decoded = content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(decoded))
        
        count = 0
        batch_size = 1000
        batch = []
        
        for row in reader:
            # Parse date
            date_str = row['date']
            try:
                date_obj = datetime.strptime(date_str, '%m/%d/%Y').date()
            except:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                except:
                    continue
            
            # Parse time fields
            sunrise_time = None
            sunset_time = None
            if row.get('sunrise'):
                try:
                    sunrise_time = datetime.strptime(row['sunrise'], '%H:%M').time()
                except:
                    pass
            if row.get('sunset'):
                try:
                    sunset_time = datetime.strptime(row['sunset'], '%H:%M').time()
                except:
                    pass
            
            weather = Weather(
                location_id=int(row['location_id']),
                date=date_obj,
                weather_code=int(row['weather_code (wmo code)']) if row.get('weather_code (wmo code)') else None,
                temperature_2m_max=float(row['temperature_2m_max (°C)']) if row.get('temperature_2m_max (°C)') else None,
                temperature_2m_min=float(row['temperature_2m_min (°C)']) if row.get('temperature_2m_min (°C)') else None,
                temperature_2m_mean=float(row['temperature_2m_mean (°C)']) if row.get('temperature_2m_mean (°C)') else None,
                apparent_temperature_max=float(row['apparent_temperature_max (°C)']) if row.get('apparent_temperature_max (°C)') else None,
                apparent_temperature_min=float(row['apparent_temperature_min (°C)']) if row.get('apparent_temperature_min (°C)') else None,
                apparent_temperature_mean=float(row['apparent_temperature_mean (°C)']) if row.get('apparent_temperature_mean (°C)') else None,
                daylight_duration=float(row['daylight_duration (s)']) if row.get('daylight_duration (s)') else None,
                sunshine_duration=float(row['sunshine_duration (s)']) if row.get('sunshine_duration (s)') else None,
                precipitation_sum=float(row['precipitation_sum (mm)']) if row.get('precipitation_sum (mm)') else None,
                rain_sum=float(row['rain_sum (mm)']) if row.get('rain_sum (mm)') else None,
                precipitation_hours=float(row['precipitation_hours (h)']) if row.get('precipitation_hours (h)') else None,
                wind_speed_10m_max=float(row['wind_speed_10m_max (km/h)']) if row.get('wind_speed_10m_max (km/h)') else None,
                wind_gusts_10m_max=float(row['wind_gusts_10m_max (km/h)']) if row.get('wind_gusts_10m_max (km/h)') else None,
                wind_direction_10m_dominant=float(row['wind_direction_10m_dominant (°)']) if row.get('wind_direction_10m_dominant (°)') else None,
                shortwave_radiation_sum=float(row['shortwave_radiation_sum (MJ/m²)']) if row.get('shortwave_radiation_sum (MJ/m²)') else None,
                et0_fao_evapotranspiration=float(row['et0_fao_evapotranspiration (mm)']) if row.get('et0_fao_evapotranspiration (mm)') else None,
                sunrise=sunrise_time,
                sunset=sunset_time
            )
            batch.append(weather)
            count += 1
            
            if len(batch) >= batch_size:
                db.bulk_save_objects(batch)
                db.commit()
                batch = []
        
        if batch:
            db.bulk_save_objects(batch)
            db.commit()
        
        return {"message": f"Successfully uploaded {count} weather records", "count": count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error uploading weather data: {str(e)}")

@router.post("/insert-weather")
async def insert_weather_record(weather_data: WeatherInsert, db: Session = Depends(get_db)):
    """Insert a single weather record (real-time)"""
    try:
        # Parse date
        date_obj = datetime.strptime(weather_data.date, '%Y-%m-%d').date()
        
        # Parse time fields
        sunrise_time = None
        sunset_time = None
        if weather_data.sunrise:
            try:
                sunrise_time = datetime.strptime(weather_data.sunrise, '%H:%M').time()
            except:
                pass
        if weather_data.sunset:
            try:
                sunset_time = datetime.strptime(weather_data.sunset, '%H:%M').time()
            except:
                pass
        
        weather = Weather(
            location_id=weather_data.location_id,
            date=date_obj,
            weather_code=weather_data.weather_code,
            temperature_2m_max=weather_data.temperature_2m_max,
            temperature_2m_min=weather_data.temperature_2m_min,
            temperature_2m_mean=weather_data.temperature_2m_mean,
            apparent_temperature_max=weather_data.apparent_temperature_max,
            apparent_temperature_min=weather_data.apparent_temperature_min,
            apparent_temperature_mean=weather_data.apparent_temperature_mean,
            daylight_duration=weather_data.daylight_duration,
            sunshine_duration=weather_data.sunshine_duration,
            precipitation_sum=weather_data.precipitation_sum,
            rain_sum=weather_data.rain_sum,
            precipitation_hours=weather_data.precipitation_hours,
            wind_speed_10m_max=weather_data.wind_speed_10m_max,
            wind_gusts_10m_max=weather_data.wind_gusts_10m_max,
            wind_direction_10m_dominant=weather_data.wind_direction_10m_dominant,
            shortwave_radiation_sum=weather_data.shortwave_radiation_sum,
            et0_fao_evapotranspiration=weather_data.et0_fao_evapotranspiration,
            sunrise=sunrise_time,
            sunset=sunset_time
        )
        
        db.add(weather)
        db.commit()
        db.refresh(weather)
        
        return {"message": "Weather record inserted successfully", "id": weather.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error inserting weather record: {str(e)}")

@router.get("/locations")
async def get_locations(db: Session = Depends(get_db)):
    """Get all locations"""
    locations = db.query(Location).all()
    return [{"location_id": loc.location_id, "city_name": loc.city_name} for loc in locations]
