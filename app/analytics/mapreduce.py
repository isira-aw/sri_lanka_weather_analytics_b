from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from app.db.database import get_db, Weather, Location
from collections import defaultdict
from datetime import datetime

router = APIRouter()

def mapper_monthly_precipitation_temp(weather_records):
    """Map function: emit (location_id, year, month) -> (precipitation, temperature, count)"""
    mapped_data = []
    for record in weather_records:
        year = record.date.year
        month = record.date.month
        key = (record.location_id, year, month)
        value = (
            record.precipitation_sum or 0,
            record.temperature_2m_mean or 0,
            1
        )
        mapped_data.append((key, value))
    return mapped_data

def reducer_monthly_precipitation_temp(mapped_data):
    """Reduce function: aggregate precipitation and temperature by location, year, month"""
    reduced = defaultdict(lambda: {'precipitation': 0, 'temperature': 0, 'count': 0})
    
    for key, value in mapped_data:
        location_id, year, month = key
        precipitation, temperature, count = value
        
        reduced[key]['precipitation'] += precipitation
        reduced[key]['temperature'] += temperature
        reduced[key]['count'] += count
    
    # Calculate means
    result = []
    for key, values in reduced.items():
        location_id, year, month = key
        mean_temp = values['temperature'] / values['count'] if values['count'] > 0 else 0
        result.append({
            'location_id': location_id,
            'year': year,
            'month': month,
            'total_precipitation': round(values['precipitation'], 2),
            'mean_temperature': round(mean_temp, 2)
        })
    
    return result

@router.get("/monthly-precipitation-temp")
async def monthly_precipitation_temperature(db: Session = Depends(get_db)):
    """
    MapReduce Analysis 1: Monthly total precipitation and mean temperature per district (last 10 years)
    """
    # Get last 10 years of data
    current_year = datetime.now().year
    start_year = current_year - 10
    
    # Fetch weather records for last 10 years
    weather_records = db.query(Weather).filter(
        extract('year', Weather.date) >= start_year
    ).all()
    
    if not weather_records:
        return {"error": "No weather data found"}
    
    # Map phase
    mapped_data = mapper_monthly_precipitation_temp(weather_records)
    
    # Reduce phase
    reduced_data = reducer_monthly_precipitation_temp(mapped_data)
    
    # Join with location data to get city names
    location_map = {loc.location_id: loc.city_name for loc in db.query(Location).all()}
    
    for record in reduced_data:
        record['city_name'] = location_map.get(record['location_id'], 'Unknown')
    
    # Sort by year, month, location
    reduced_data.sort(key=lambda x: (x['year'], x['month'], x['location_id']))
    
    return {
        "message": "Monthly precipitation and temperature aggregated by district (last 10 years)",
        "total_records": len(reduced_data),
        "data": reduced_data[:100]  # Return first 100 for API response
    }

@router.get("/highest-precipitation-month")
async def highest_precipitation_month(db: Session = Depends(get_db)):
    """
    MapReduce Analysis 2: Month & year with highest total precipitation
    """
    # Fetch all weather records
    weather_records = db.query(Weather).all()
    
    if not weather_records:
        return {"error": "No weather data found"}
    
    # Map: emit (year, month) -> precipitation
    mapped_data = []
    for record in weather_records:
        year = record.date.year
        month = record.date.month
        key = (year, month)
        value = record.precipitation_sum or 0
        mapped_data.append((key, value))
    
    # Reduce: sum precipitation by year-month
    reduced = defaultdict(float)
    for key, value in mapped_data:
        reduced[key] += value
    
    # Find maximum
    if not reduced:
        return {"error": "No precipitation data found"}
    
    max_key = max(reduced, key=reduced.get)
    max_precipitation = reduced[max_key]
    
    year, month = max_key
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    return {
        "message": "Month and year with highest total precipitation",
        "year": year,
        "month": month,
        "month_name": month_names[month],
        "total_precipitation_mm": round(max_precipitation, 2)
    }

@router.get("/district-precipitation-summary")
async def district_precipitation_summary(db: Session = Depends(get_db)):
    """
    Additional MapReduce: Total precipitation by district for visualization
    """
    weather_records = db.query(Weather).all()
    
    # Map: emit location_id -> precipitation
    mapped_data = [(record.location_id, record.precipitation_sum or 0) for record in weather_records]
    
    # Reduce: sum by location
    reduced = defaultdict(float)
    for location_id, precipitation in mapped_data:
        reduced[location_id] += precipitation
    
    # Join with location data
    location_map = {loc.location_id: loc.city_name for loc in db.query(Location).all()}
    
    result = [
        {
            'location_id': location_id,
            'city_name': location_map.get(location_id, 'Unknown'),
            'total_precipitation_mm': round(precipitation, 2)
        }
        for location_id, precipitation in reduced.items()
    ]
    
    # Sort by precipitation descending
    result.sort(key=lambda x: x['total_precipitation_mm'], reverse=True)
    
    return {
        "message": "Total precipitation by district",
        "data": result
    }
