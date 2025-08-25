from celery import Celery
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncio
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

from app.core.config import settings
from app.services.ml_service import ml_service
from app.services.cache_service import cache_service
from loguru import logger

celery_app = Celery(
    "boston_analytics",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "app.tasks.ml_tasks.*": {"queue": "ml_queue"},
        "app.tasks.data_tasks.*": {"queue": "data_queue"}
    }
)

async def get_async_session():
    engine = create_async_engine(settings.database_url)
    async_session = AsyncSession(engine)
    try:
        yield async_session
        await async_session.commit()
    except Exception:
        await async_session.rollback()
        raise
    finally:
        await async_session.close()
        await engine.dispose()

@celery_app.task(bind=True, name="train_clustering_model")
def train_clustering_model(
    self,
    task_id: str,
    n_clusters: int = 5,
    fiscal_year: Optional[int] = None,
    features: Optional[List[str]] = None
):
    try:
        logger.info(f"Starting clustering model training - Task ID: {task_id}")
        
        self.update_state(
            state="PROGRESS",
            meta={"stage": "data_preparation", "progress": 10}
        )
        
        async def train_model():
            async for session in get_async_session():
                df = await ml_service.prepare_clustering_data(session, fiscal_year)
                
                self.update_state(
                    state="PROGRESS", 
                    meta={"stage": "model_training", "progress": 50}
                )
                
                cluster_labels, silhouette_score, model_info = ml_service.perform_kmeans_clustering(
                    df, n_clusters, features
                )
                
                self.update_state(
                    state="PROGRESS",
                    meta={"stage": "saving_metadata", "progress": 90}
                )
                
                metadata = await ml_service.save_model_metadata(session, model_info, "KMeans")
                
                return {
                    "task_id": task_id,
                    "model_id": str(metadata.id),
                    "silhouette_score": silhouette_score,
                    "n_clusters": n_clusters,
                    "training_samples": len(df),
                    "model_path": model_info["model_path"],
                    "features_used": features or ["vendor_name", "account_descr", "dept_name"],
                    "completed_at": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(train_model())
        
        logger.info(f"Clustering model training completed - Task ID: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Clustering model training failed - Task ID: {task_id}, Error: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "task_id": task_id}
        )
        raise

@celery_app.task(bind=True, name="train_classification_model")
def train_classification_model(
    self,
    task_id: str,
    threshold: float = 1000.0,
    fiscal_year: Optional[int] = None,
    features: Optional[List[str]] = None
):
    try:
        logger.info(f"Starting classification model training - Task ID: {task_id}")
        
        self.update_state(
            state="PROGRESS",
            meta={"stage": "data_preparation", "progress": 10}
        )
        
        async def train_model():
            async for session in get_async_session():
                df = await ml_service.prepare_clustering_data(session, fiscal_year)
                
                self.update_state(
                    state="PROGRESS",
                    meta={"stage": "model_training", "progress": 50}
                )
                
                model_info = ml_service.train_expenditure_classifier(df, threshold, features)
                
                self.update_state(
                    state="PROGRESS",
                    meta={"stage": "saving_metadata", "progress": 90}
                )
                
                model_info["silhouette_score"] = None
                metadata = await ml_service.save_model_metadata(session, model_info, "RandomForestClassifier")
                
                return {
                    "task_id": task_id,
                    "model_id": str(metadata.id),
                    "accuracy": model_info["accuracy"],
                    "threshold": threshold,
                    "training_samples": model_info["training_samples"],
                    "test_samples": model_info["test_samples"],
                    "model_path": model_info["model_path"],
                    "features_used": features or ["account_descr", "dept_name", "vendor_name"],
                    "feature_importance": model_info["feature_importance"],
                    "completed_at": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(train_model())
        
        logger.info(f"Classification model training completed - Task ID: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Classification model training failed - Task ID: {task_id}, Error: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "task_id": task_id}
        )
        raise

@celery_app.task(bind=True, name="batch_prediction_task")
def batch_prediction_task(
    self,
    task_id: str,
    input_data_list: List[Dict[str, Any]],
    model_type: str = "classification"
):
    try:
        logger.info(f"Starting batch prediction - Task ID: {task_id}")
        
        self.update_state(
            state="PROGRESS",
            meta={"stage": "initialization", "progress": 5}
        )
        
        async def process_predictions():
            predictions = []
            total_items = len(input_data_list)
            
            async for session in get_async_session():
                from sqlalchemy import select, desc
                from app.models.database import MLModelMetadata
                
                model_query = select(MLModelMetadata).where(
                    MLModelMetadata.model_type == "RandomForestClassifier" if model_type == "classification" else "KMeans",
                    MLModelMetadata.is_active == True
                ).order_by(desc(MLModelMetadata.trained_at))
                
                model_result = await session.execute(model_query)
                model_metadata = model_result.scalar_one_or_none()
                
                if not model_metadata:
                    raise Exception(f"No active {model_type} model found")
                
                encoders_path = model_metadata.file_path.replace(
                    "rf_classifier" if model_type == "classification" else "kmeans_model",
                    "classifier_encoders" if model_type == "classification" else "clustering_encoders"
                )
                
                for i, input_data in enumerate(input_data_list):
                    try: