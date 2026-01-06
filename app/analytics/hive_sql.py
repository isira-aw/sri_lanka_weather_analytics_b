from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, extract, case, and_, or_
from app.db.database import get_db, Weather, Location

router = APIRouter()

@router.get("/top-temperate-cities")
async def top_temperate_cities(db: Session = Depends(get_db)):
    """
    Hive-style SQL Analysis 1: Top 10 most temperate cities (lowest max temperature)
    """
    # SQL equivalent:
    # SELECT l.city_name, AVG(w.temperature_2m_max) as avg_max_temp
    # FROM weather w JOIN location l ON w.location_id = l.location_id
    # GROUP BY l.city_name
    # ORDER BY avg_max_temp ASC
    # LIMIT 10
    
    result = db.query(
        Location.city_name,
        func.avg(Weather.temperature_2m_max).label('avg_max_temp'),
        func.min(Weather.temperature_2m_max).label('min_max_temp'),
        func.max(Weather.temperature_2m_max).label('max_max_temp')
    ).join(Weather, Location.location_id == Weather.location_id)\
     .group_by(Location.city_name)\
     .order_by(func.avg(Weather.temperature_2m_max).asc())\
     .limit(10)\
     .all()
    
    data = [
        {
            'city_name': row[0],
            'avg_max_temperature': round(row[1], 2),
            'min_max_temperature': round(row[2], 2),
            'max_max_temperature': round(row[3], 2)
        }
        for row in result
    ]
    
    return {
        "message": "Top 10 most temperate cities (lowest average maximum temperature)",
        "data": data
    }

@router.get("/evapotranspiration-by-season")
async def evapotranspiration_by_season(db: Session = Depends(get_db)):
    """
    Hive-style SQL Analysis 2: Average evapotranspiration per agricultural season
    Season 1: September - March
    Season 2: April - August
    """
    # SQL equivalent using CASE WHEN for season classification
    # SELECT 
    #   l.city_name,
    #   CASE 
    #     WHEN EXTRACT(MONTH FROM date) IN (9,10,11,12,1,2,3) THEN 'Sep-Mar'
    #     ELSE 'Apr-Aug'
    #   END as season,
    #   AVG(et0_fao_evapotranspiration) as avg_et0
    # FROM weather w JOIN location l ON w.location_id = l.location_id
    # GROUP BY l.city_name, season
    
    season_case = case(
        (
            or_(
                extract('month', Weather.date).in_([9, 10, 11, 12]),
                extract('month', Weather.date).in_([1, 2, 3])
            ),
            'Sep-Mar'
        ),
        else_='Apr-Aug'
    )
    
    result = db.query(
        Location.city_name,
        season_case.label('season'),
        func.avg(Weather.et0_fao_evapotranspiration).label('avg_et0'),
        func.count(Weather.id).label('record_count')
    ).join(Weather, Location.location_id == Weather.location_id)\
     .filter(Weather.et0_fao_evapotranspiration.isnot(None))\
     .group_by(Location.city_name, season_case)\
     .order_by(Location.city_name, season_case)\
     .all()
    
    data = [
        {
            'city_name': row[0],
            'season': row[1],
            'avg_evapotranspiration_mm': round(row[2], 2),
            'record_count': row[3]
        }
        for row in result
    ]
    
    return {
        "message": "Average evapotranspiration by agricultural season",
        "seasons": {
            "Sep-Mar": "September to March",
            "Apr-Aug": "April to August"
        },
        "data": data
    }

@router.get("/extreme-weather-days")
async def extreme_weather_days(db: Session = Depends(get_db)):
    """
    Additional Hive-style Analysis: Count extreme weather days
    Extreme = high rain (>50mm) AND high wind gusts (>40 km/h)
    """
    # SQL with WHERE clause filtering
    result = db.query(
        Location.city_name,
        func.count(Weather.id).label('extreme_days')
    ).join(Weather, Location.location_id == Weather.location_id)\
     .filter(
        and_(
            Weather.precipitation_sum > 50,
            Weather.wind_gusts_10m_max > 40
        )
     )\
     .group_by(Location.city_name)\
     .order_by(func.count(Weather.id).desc())\
     .all()
    
    data = [
        {
            'city_name': row[0],
            'extreme_weather_days': row[1]
        }
        for row in result
    ]
    
    return {
        "message": "Extreme weather days by district (precipitation > 50mm AND wind gusts > 40 km/h)",
        "data": data
    }

@router.get("/high-temperature-months")
async def high_temperature_months(db: Session = Depends(get_db)):
    """
    Additional Analysis: Percentage of months with mean temperature > 30°C per year
    """
    # Count total months and high-temp months per year per location
    total_months = db.query(
        Location.city_name,
        extract('year', Weather.date).label('year'),
        func.count(func.distinct(extract('month', Weather.date))).label('total_months')
    ).join(Weather, Location.location_id == Weather.location_id)\
     .group_by(Location.city_name, extract('year', Weather.date))\
     .subquery()
    
    high_temp_months = db.query(
        Location.city_name,
        extract('year', Weather.date).label('year'),
        func.count(func.distinct(extract('month', Weather.date))).label('high_temp_months')
    ).join(Weather, Location.location_id == Weather.location_id)\
     .filter(Weather.temperature_2m_mean > 30)\
     .group_by(Location.city_name, extract('year', Weather.date))\
     .subquery()
    
    # This is simplified - in production, would use a proper SQL join
    # For now, fetch data and calculate in Python
    all_records = db.query(
        Location.city_name,
        extract('year', Weather.date).label('year'),
        extract('month', Weather.date).label('month'),
        func.avg(Weather.temperature_2m_mean).label('avg_temp')
    ).join(Weather, Location.location_id == Weather.location_id)\
     .group_by(Location.city_name, extract('year', Weather.date), extract('month', Weather.date))\
     .all()
    
    # Process in Python
    from collections import defaultdict
    stats = defaultdict(lambda: {'total': 0, 'high_temp': 0})
    
    for row in all_records:
        city, year, month, avg_temp = row
        key = (city, year)
        stats[key]['total'] += 1
        if avg_temp and avg_temp > 30:
            stats[key]['high_temp'] += 1
    
    data = [
        {
            'city_name': key[0],
            'year': int(key[1]),
            'total_months': values['total'],
            'high_temp_months': values['high_temp'],
            'percentage': round((values['high_temp'] / values['total'] * 100) if values['total'] > 0 else 0, 2)
        }
        for key, values in stats.items()
    ]
    
    data.sort(key=lambda x: (x['city_name'], x['year']))
    
    return {
        "message": "Percentage of months with mean temperature > 30°C per year",
        "data": data[:50]  # Return first 50 records
    }
