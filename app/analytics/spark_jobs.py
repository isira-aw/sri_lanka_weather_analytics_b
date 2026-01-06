from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from app.db.database import get_db, Weather, Location
import pandas as pd
from datetime import datetime, timedelta

router = APIRouter()

def create_spark_dataframe(db: Session):
    """Create a pandas DataFrame from database (simulating Spark DataFrame)"""
    # In production, this would use PySpark to read from database
    # For this project, we simulate with pandas
    
    query = db.query(
        Weather.location_id,
        Weather.date,
        Weather.temperature_2m_max,
        Weather.shortwave_radiation_sum,
        Location.city_name
    ).join(Location, Weather.location_id == Location.location_id)
    
    df = pd.read_sql(query.statement, db.bind)
    return df

@router.get("/shortwave-radiation-percentage")
async def shortwave_radiation_percentage(db: Session = Depends(get_db)):
    """
    Spark Analysis 1: Percentage of monthly shortwave radiation > 15 MJ/m²
    """
    try:
        # Create DataFrame (simulating Spark)
        df = create_spark_dataframe(db)
        
        if df.empty:
            return {"error": "No data found"}
        
        # Add year and month columns
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        
        # Group by city, year, month
        grouped = df.groupby(['city_name', 'year', 'month']).agg({
            'shortwave_radiation_sum': ['count', lambda x: (x > 15).sum()]
        }).reset_index()
        
        grouped.columns = ['city_name', 'year', 'month', 'total_days', 'high_radiation_days']
        
        # Calculate percentage
        grouped['percentage'] = (grouped['high_radiation_days'] / grouped['total_days'] * 100).round(2)
        
        # Convert to dict
        result = grouped.to_dict('records')
        
        return {
            "message": "Percentage of days with shortwave radiation > 15 MJ/m² by month",
            "total_records": len(result),
            "data": result[:100]  # Return first 100
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/weekly-max-temperatures")
async def weekly_max_temperatures(db: Session = Depends(get_db)):
    """
    Spark Analysis 2: Weekly maximum temperatures for hottest months
    First identify hottest months, then calculate weekly max temps
    """
    try:
        # Create DataFrame
        df = create_spark_dataframe(db)
        
        if df.empty:
            return {"error": "No data found"}
        
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Find hottest months (top 3 months by average max temperature)
        monthly_avg = df.groupby(['year', 'month'])['temperature_2m_max'].mean().reset_index()
        monthly_avg = monthly_avg.sort_values('temperature_2m_max', ascending=False).head(3)
        
        hottest_months = list(zip(monthly_avg['year'], monthly_avg['month']))
        
        # Filter data for hottest months
        df_hot = df[df.apply(lambda row: (row['year'], row['month']) in hottest_months, axis=1)]
        
        # Add week number
        df_hot['week'] = df_hot['date'].dt.isocalendar().week
        
        # Group by city, year, month, week
        weekly_max = df_hot.groupby(['city_name', 'year', 'month', 'week']).agg({
            'temperature_2m_max': 'max',
            'date': ['min', 'max']
        }).reset_index()
        
        weekly_max.columns = ['city_name', 'year', 'month', 'week', 'max_temperature', 'week_start', 'week_end']
        
        result = weekly_max.to_dict('records')
        
        # Format dates
        for r in result:
            r['week_start'] = r['week_start'].strftime('%Y-%m-%d')
            r['week_end'] = r['week_end'].strftime('%Y-%m-%d')
            r['max_temperature'] = round(r['max_temperature'], 2)
        
        return {
            "message": "Weekly maximum temperatures for hottest months",
            "hottest_months": [{"year": int(y), "month": int(m)} for y, m in hottest_months],
            "data": result
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/precipitation-patterns")
async def precipitation_patterns(db: Session = Depends(get_db)):
    """
    Additional Spark Analysis: Precipitation patterns by season and district
    """
    try:
        query = db.query(
            Weather.location_id,
            Weather.date,
            Weather.precipitation_sum,
            Location.city_name
        ).join(Location, Weather.location_id == Location.location_id)
        
        df = pd.read_sql(query.statement, db.bind)
        
        if df.empty:
            return {"error": "No data found"}
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        
        # Define seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        df['season'] = df['month'].apply(get_season)
        
        # Aggregate by city and season
        seasonal_precip = df.groupby(['city_name', 'season']).agg({
            'precipitation_sum': ['mean', 'sum', 'max']
        }).reset_index()
        
        seasonal_precip.columns = ['city_name', 'season', 'avg_precipitation', 'total_precipitation', 'max_precipitation']
        
        result = seasonal_precip.to_dict('records')
        
        for r in result:
            r['avg_precipitation'] = round(r['avg_precipitation'], 2)
            r['total_precipitation'] = round(r['total_precipitation'], 2)
            r['max_precipitation'] = round(r['max_precipitation'], 2)
        
        return {
            "message": "Precipitation patterns by season and district",
            "data": result
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/temperature-distribution")
async def temperature_distribution(db: Session = Depends(get_db)):
    """
    Additional Spark Analysis: Temperature distribution analysis
    """
    try:
        query = db.query(
            Location.city_name,
            Weather.temperature_2m_max,
            Weather.temperature_2m_min,
            Weather.temperature_2m_mean
        ).join(Location, Weather.location_id == Location.location_id)
        
        df = pd.read_sql(query.statement, db.bind)
        
        if df.empty:
            return {"error": "No data found"}
        
        # Calculate statistics by city
        stats = df.groupby('city_name').agg({
            'temperature_2m_max': ['mean', 'min', 'max', 'std'],
            'temperature_2m_min': ['mean', 'min', 'max', 'std'],
            'temperature_2m_mean': ['mean', 'min', 'max', 'std']
        }).reset_index()
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
        
        result = stats.to_dict('records')
        
        # Round values
        for r in result:
            for key in r:
                if isinstance(r[key], float):
                    r[key] = round(r[key], 2)
        
        return {
            "message": "Temperature distribution statistics by district",
            "data": result
        }
    except Exception as e:
        return {"error": str(e)}
