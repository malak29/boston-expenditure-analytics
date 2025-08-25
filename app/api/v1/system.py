from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import os
import psutil

from app.core.database import get_db_session, db_manager
from app.services.cache_service import cache_service
from app.utils.monitoring import metrics, performance_monitor
from app.core.config import settings
from loguru import logger

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version
    }

@router.get("/health/detailed")
async def detailed_health_check(session: AsyncSession = Depends(get_db_session)):
    try:
        health_data = await performance_monitor.health_check_comprehensive(session)
        
        celery_health = await check_celery_health()
        health_data["celery"] = celery_health
        
        disk_usage = check_disk_usage()
        health_data["disk"] = disk_usage
        
        return health_data
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/metrics")
async def get_prometheus_metrics():
    try:
        return Response(
            content=metrics.get_metrics(),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")

@router.get("/metrics/summary")
async def get_metrics_summary(session: AsyncSession = Depends(get_db_session)):
    try:
        from sqlalchemy import select, func, desc
        from app.models.database import ExpenditureRecord, VendorPerformance, MLModelMetadata
        
        db_stats_query = select(
            func.count(ExpenditureRecord.id).label("total_records"),
            func.sum(ExpenditureRecord.monetary_amount).label("total_amount"),
            func.count(func.distinct(ExpenditureRecord.vendor_name)).label("unique_vendors"),
            func.count(func.distinct(ExpenditureRecord.dept_name)).label("unique_departments")
        )
        
        db_result = await session.execute(db_stats_query)
        db_stats = db_result.first()
        
        ml_models_query = select(
            func.count(MLModelMetadata.id).label("total_models"),
            func.count(case((MLModelMetadata.is_active == True, 1))).label("active_models")
        )
        
        ml_result = await session.execute(ml_models_query)
        ml_stats = ml_result.first()
        
        cache_stats = await cache_service.get_cache_stats()
        
        system_stats = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "database": {
                "total_records": db_stats.total_records or 0,
                "total_amount": float(db_stats.total_amount or 0),
                "unique_vendors": db_stats.unique_vendors or 0,
                "unique_departments": db_stats.unique_departments or 0
            },
            "ml_models": {
                "total_models": ml_stats.total_models or 0,
                "active_models": ml_stats.active_models or 0
            },
            "cache": cache_stats,
            "system": system_stats
        }
        
    except Exception as e:
        logger.error(f"Metrics summary failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics summary")

@router.get("/performance/database")
async def get_database_performance(session: AsyncSession = Depends(get_db_session)):
    try:
        performance_data = await performance_monitor.check_database_performance(session)
        return performance_data
        
    except Exception as e:
        logger.error(f"Database performance check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to check database performance")

@router.get("/status/services")
async def get_services_status():
    try:
        services_status = {}
        
        try:
            db_healthy = await db_manager.health_check()
            services_status["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "type": "PostgreSQL"
            }
        except Exception:
            services_status["database"] = {"status": "unhealthy", "type": "PostgreSQL"}
        
        try:
            cache_healthy = await cache_service.health_check()
            services_status["cache"] = {
                "status": "healthy" if cache_healthy else "unhealthy",
                "type": "Redis"
            }
        except Exception:
            services_status["cache"] = {"status": "unhealthy", "type": "Redis"}
        
        celery_status = await check_celery_health()
        services_status["task_queue"] = celery_status
        
        overall_healthy = all(
            service.get("status") == "healthy" 
            for service in services_status.values()
        )
        
        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "services": services_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Services status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to check services status")

@router.post("/maintenance/cleanup")
async def trigger_maintenance_cleanup(
    cleanup_type: str = Query("all", regex="^(logs|cache|models|all)$", description="Type of cleanup"),
    days_old: int = Query(7, ge=1, le=90, description="Clean items older than X days")
):
    try:
        cleanup_results = {}
        
        if cleanup_type in ["logs", "all"]:
            log_cleanup = await cleanup_old_logs(days_old)
            cleanup_results["logs"] = log_cleanup
        
        if cleanup_type in ["cache", "all"]:
            cache_cleanup = await cleanup_cache()
            cleanup_results["cache"] = cache_cleanup
        
        if cleanup_type in ["models", "all"]:
            from app.tasks.ml_tasks import cleanup_old_models
            model_cleanup_task = cleanup_old_models.delay(days_old)
            cleanup_results["models"] = {
                "task_id": model_cleanup_task.id,
                "status": "started"
            }
        
        return {
            "cleanup_type": cleanup_type,
            "days_old": days_old,
            "results": cleanup_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Maintenance cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger maintenance cleanup")

@router.get("/logs/recent")
async def get_recent_logs(
    level: str = Query("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    limit: int = Query(100, ge=1, le=1000, description="Number of log entries"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back")
):
    try:
        log_file_path = "/app/logs/app.log"
        
        if not os.path.exists(log_file_path):
            return {"logs": [], "message": "No log file found"}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        filtered_logs = []
        for line in reversed(lines[-limit*2:]):
            if level.upper() in line:
                try:
                    timestamp_str = line.split(' ')[0] + ' ' + line.split(' ')[1]
                    log_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    if log_time >= cutoff_time:
                        filtered_logs.append({
                            "timestamp": timestamp_str,
                            "level": level,
                            "message": line.strip()
                        })
                        
                        if len(filtered_logs) >= limit:
                            break
                except Exception:
                    continue
        
        return {
            "logs": filtered_logs,
            "level_filter": level,
            "hours_back": hours,
            "total_entries": len(filtered_logs)
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent logs")

async def check_celery_health() -> Dict[str, Any]:
    try:
        from app.tasks.ml_tasks import celery_app
        
        inspect = celery_app.control.inspect()
        
        stats = inspect.stats()
        active_tasks = inspect.active()
        
        worker_count = len(stats) if stats else 0
        total_active_tasks = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        
        return {
            "status": "healthy" if worker_count > 0 else "unhealthy",
            "worker_count": worker_count,
            "active_tasks": total_active_tasks,
            "type": "Celery"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "type": "Celery"
        }

def check_disk_usage() -> Dict[str, Any]:
    try:
        disk_usage = psutil.disk_usage("/")
        
        return {
            "total_bytes": disk_usage.total,
            "used_bytes": disk_usage.used,
            "free_bytes": disk_usage.free,
            "usage_percentage": round((disk_usage.used / disk_usage.total) * 100, 2),
            "status": "healthy" if disk_usage.free > (1024**3) else "warning"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "unhealthy"}

async def cleanup_old_logs(days_old: int = 7) -> Dict[str, Any]:
    try:
        log_dir = "/app/logs"
        if not os.path.exists(log_dir):
            return {"message": "No logs directory found", "files_removed": 0}
        
        cutoff_time = datetime.utcnow() - timedelta(days=days_old)
        files_removed = 0
        total_size_freed = 0
        
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_time and filename.endswith('.log'):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    files_removed += 1
                    total_size_freed += file_size
        
        return {
            "files_removed": files_removed,
            "size_freed_bytes": total_size_freed,
            "cutoff_date": cutoff_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Log cleanup failed: {e}")
        return {"error": str(e)}

async def cleanup_cache() -> Dict[str, Any]:
    try:
        deleted_analytics = await cache_service.invalidate_pattern("*", "analytics")
        deleted_vendors = await cache_service.invalidate_pattern("*", "vendor_analytics") 
        deleted_departments = await cache_service.invalidate_pattern("*", "department_analytics")
        deleted_predictions = await cache_service.invalidate_pattern("*", "ml_predictions")
        
        total_deleted = deleted_analytics + deleted_vendors + deleted_departments + deleted_predictions
        
        return {
            "total_keys_deleted": total_deleted,
            "categories_cleared": {
                "analytics": deleted_analytics,
                "vendor_analytics": deleted_vendors,
                "department_analytics": deleted_departments, 
                "ml_predictions": deleted_predictions
            }
        }
        
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return {"error": str(e)}