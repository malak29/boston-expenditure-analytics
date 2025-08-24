from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    app_name: str = "Boston Expenditure Analytics API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    data_source_url: str = Field(
        default="https://customerExpenses.boston.gov/customerExpensesstore/dump/0b7c9c5f-d1c2-46e7-b738-6ab37a110eef?bom=True",
        env="DATA_SOURCE_URL"
    )
    
    ml_models_path: str = Field(default="./models", env="ML_MODELS_PATH")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    pagination_default_size: int = 50
    pagination_max_size: int = 1000
    
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()