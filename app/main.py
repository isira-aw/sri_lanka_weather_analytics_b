from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ingest
from app.analytics import mapreduce, hive_sql, spark_jobs
from app.ml import train_model
from app.db.database import init_db

# Initialize database on startup
init_db()

app = FastAPI(title="Sri Lanka Weather Analytics API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://srilankaweatherfrontend-production.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/api/ingest", tags=["Data Ingestion"])
app.include_router(mapreduce.router, prefix="/api/analytics/mapreduce", tags=["MapReduce Analytics"])
app.include_router(hive_sql.router, prefix="/api/analytics/hive", tags=["Hive-SQL Analytics"])
app.include_router(spark_jobs.router, prefix="/api/analytics/spark", tags=["Spark Analytics"])
app.include_router(train_model.router, prefix="/api/ml", tags=["Machine Learning"])

@app.get("/")
def read_root():
    return {
        "message": "Sri Lanka Weather Analytics API", 
        "status": "running",
        "database": "SQLite (No setup needed!)"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "SQLite"}
