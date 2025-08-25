from fastapi import APIRouter, Depends, Query, Body, BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

from app.core.database import get_db_session
from app.services.ml_service import ml_service
from app.services.cache_service import cache_service
from app.services.analytics_service import analytics_service
from app.tasks.ml_tasks import train_clustering_model, train_classification_model
from app.core.exceptions import MLModelException, ValidationException
from loguru import logger

router = APIRouter()

@router.post("/clustering/train")
async def train_clustering_model_endpoint(
    background_tasks: BackgroundTasks,
    n_clusters: int = Query(5, ge=2, le=20, description="Number of clusters"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    features: Optional[str] = Query(None, description="Comma-separated feature names"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        features_list = None
        if features:
            features_list = [f.strip() for f in features.split(",")]
            valid_features = ["vendor_name", "account_descr", "dept_name"]
            invalid_features = [f for f in features_list if f not in valid_features]
            
            if invalid_features:
                raise ValidationException(
                    f"Invalid features: {invalid_features}. Valid options: {valid_features}"
                )
        
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            train_clustering_model,
            task_id=task_id,
            n_clusters=n_clusters,
            fiscal_year=fiscal_year,
            features=features_list
        )
        
        return {
            "message": "Clustering model training started",
            "task_id": task_id,
            "parameters": {
                "n_clusters": n_clusters,
                "fiscal_year": fiscal_year,
                "features": features_list
            }
        }
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error starting clustering training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start clustering model training")

@router.post("/classification/train")
async def train_classification_model_endpoint(
    background_tasks: BackgroundTasks,
    threshold: float = Query(1000.0, ge=0, description="Threshold for high/low value classification"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    features: Optional[str] = Query(None, description="Comma-separated feature names"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        features_list = None
        if features:
            features_list = [f.strip() for f in features.split(",")]
            valid_features = ["account_descr", "dept_name", "vendor_name"]
            invalid_features = [f for f in features_list if f not in valid_features]
            
            if invalid_features:
                raise ValidationException(
                    f"Invalid features: {invalid_features}. Valid options: {valid_features}"
                )
        
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            train_classification_model,
            task_id=task_id,
            threshold=threshold,
            fiscal_year=fiscal_year,
            features=features_list
        )
        
        return {
            "message": "Classification model training started",
            "task_id": task_id,
            "parameters": {
                "threshold": threshold,
                "fiscal_year": fiscal_year,
                "features": features_list
            }
        }
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error starting classification training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start classification model training")

@router.post("/predict/expenditure-type")
async def predict_expenditure_type(
    input_data: Dict[str, Any] = Body(..., description="Input data for prediction"),
    model_version: Optional[str] = Query(None, description="Specific model version"),
    use_cache: bool = Query(True, description="Use cached predictions if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        required_fields = ["account_descr", "dept_name", "vendor_name"]
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            raise ValidationException(f"Missing required fields: {missing_fields}")
        
        if use_cache:
            cached_prediction = await cache_service.get_cached_prediction("expenditure_classifier", input_data)
            if cached_prediction:
                logger.info("Returning cached prediction")
                return cached_prediction
        
        from sqlalchemy import select, desc
        from app.models.database import MLModelMetadata
        
        model_query = select(MLModelMetadata).where(
            MLModelMetadata.model_type == "RandomForestClassifier",
            MLModelMetadata.is_active == True
        ).order_by(desc(MLModelMetadata.trained_at))
        
        if model_version:
            model_query = model_query.where(MLModelMetadata.model_version == model_version)
        
        model_result = await session.execute(model_query)
        model_metadata = model_result.scalar_one_or_none()
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail="No trained classification model found")
        
        encoders_path = model_metadata.file_path.replace("rf_classifier", "classifier_encoders")
        
        prediction = await ml_service.predict_expenditure_type(
            model_metadata.file_path,
            encoders_path,
            input_data
        )
        
        result = {
            "prediction": prediction,
            "model_info": {
                "model_name": model_metadata.model_name,
                "model_version": model_metadata.model_version,
                "trained_at": model_metadata.trained_at.isoformat(),
                "accuracy": model_metadata.accuracy_score
            },
            "input_data": input_data
        }
        
        if use_cache:
            await cache_service.cache_ml_prediction("expenditure_classifier", input_data, result)
        
        return result
        
    except ValidationException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in expenditure type prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict expenditure type")

@router.get("/models/status")
async def get_models_status(session: AsyncSession = Depends(get_db_session)):
    try:
        from sqlalchemy import select
        from app.models.database import MLModelMetadata
        
        query = select(MLModelMetadata).order_by(MLModelMetadata.trained_at.desc())
        result = await session.execute(query)
        models = result.scalars().all()
        
        models_info = []
        for model in models:
            models_info.append({
                "id": str(model.id),
                "model_name": model.model_name,
                "model_type": model.model_type,
                "model_version": model.model_version,
                "is_active": model.is_active,
                "accuracy_score": model.accuracy_score,
                "silhouette_score": model.silhouette_score,
                "training_data_size": model.training_data_size,
                "trained_at": model.trained_at.isoformat(),
                "features_used": model.features_used
            })
        
        active_models = [m for m in models_info if m["is_active"]]
        
        return {
            "total_models": len(models_info),
            "active_models": len(active_models),
            "models": models_info,
            "active_models_summary": active_models
        }
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models status")

@router.delete("/models/{model_id}")
async def deactivate_model(
    model_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, update
        from app.models.database import MLModelMetadata
        
        model_query = select(MLModelMetadata).where(MLModelMetadata.id == model_id)
        result = await session.execute(model_query)
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        update_query = update(MLModelMetadata).where(
            MLModelMetadata.id == model_id
        ).values(is_active=False)
        
        await session.execute(update_query)
        
        return {
            "message": f"Model {model.model_name} deactivated successfully",
            "model_id": str(model_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating model: {e}")
        raise HTTPException(status_code=500, detail="Failed to deactivate model")

@router.get("/performance/metrics")
async def get_model_performance_metrics(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func
        from app.models.database import MLModelMetadata
        
        query = select(
            MLModelMetadata.model_type,
            func.avg(MLModelMetadata.accuracy_score).label("avg_accuracy"),
            func.avg(MLModelMetadata.silhouette_score).label("avg_silhouette"),
            func.count(MLModelMetadata.id).label("model_count"),
            func.max(MLModelMetadata.trained_at).label("latest_training")
        ).group_by(MLModelMetadata.model_type)
        
        if model_type:
            query = query.where(MLModelMetadata.model_type == model_type)
        
        result = await session.execute(query)
        metrics = result.all()
        
        return {
            "model_performance": [
                {
                    "model_type": metric.model_type,
                    "average_accuracy": float(metric.avg_accuracy or 0),
                    "average_silhouette": float(metric.avg_silhouette or 0),
                    "model_count": metric.model_count,
                    "latest_training": metric.latest_training.isoformat() if metric.latest_training else None
                }
                for metric in metrics
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model performance metrics")