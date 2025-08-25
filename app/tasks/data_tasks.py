from celery import Celery
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from app.core.config import settings
from app.services.data_ingestion_service import data_ingestion_service
from app.services.cache_service import cache_service
from loguru import logger

celery_app = Celery(
    "boston_analytics_data",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=7200,
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "daily-incremental-update": {
            "task": "app.tasks.data_tasks.scheduled_incremental_update",
            "schedule": 86400.0,
        },
        "weekly-data-quality-check": {
            "task": "app.tasks.data_tasks.scheduled_data_quality_check",
            "schedule": 604800.0,
        },
        "monthly-analytics-refresh": {
            "task": "app.tasks.data_tasks.scheduled_analytics_refresh",
            "schedule": 2592000.0,
        }
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

@celery_app.task(bind=True, name="full_refresh_task")
def full_refresh_task(self, task_id: str, source_url: Optional[str] = None, clear_cache: bool = True):
    try:
        logger.info(f"Starting full data refresh - Task ID: {task_id}")
        
        self.update_state(
            state="PROGRESS",
            meta={"stage": "fetching_data", "progress": 10}
        )
        
        async def refresh_data():
            async for session in get_async_session():
                self.update_state(
                    state="PROGRESS",
                    meta={"stage": "processing_data", "progress": 30}
                )
                
                result = await data_ingestion_service.full_data_refresh(session, source_url)
                
                self.update_state(
                    state="PROGRESS",
                    meta={"stage": "updating_analytics", "progress": 70}
                )
                
                from app.services.database_service import vendor_service
                from sqlalchemy import select, func
                from app.models.database import ExpenditureRecord
                
                vendor_stats_query = select(
                    ExpenditureRecord.vendor_name,
                    func.sum(ExpenditureRecord.monetary_amount).label("total_expenditure"),
                    func.count(ExpenditureRecord.id).label("transaction_count"),
                    func.min(ExpenditureRecord.entered).label("first_transaction"),
                    func.max(ExpenditureRecord.entered).label("last_transaction")
                ).group_by(ExpenditureRecord.vendor_name)
                
                vendor_result = await session.execute(vendor_stats_query)
                vendor_stats = vendor_result.all()
                
                for stats in vendor_stats:
                    await vendor_service.update_vendor_performance(
                        session,
                        stats.vendor_name,
                        {
                            "amount": stats.total_expenditure,
                            "date": stats.last_transaction
                        }
                    )
                
                self.update_state(
                    state="PROGRESS",
                    meta={"stage": "clearing_cache", "progress": 90}
                )
                
                if clear_cache:
                    await cache_service.invalidate_pattern("*", "analytics")
                    await cache_service.invalidate_pattern("*", "vendor_analytics")
                    await cache_service.invalidate_pattern("*", "department_analytics")
                
                return result
        
        result = asyncio.run(refresh_data())
        
        final_result = {
            "task_id": task_id,
            "completed_at": datetime.utcnow().isoformat(),
            "cache_cleared": clear_cache,
            **result
        }
        
        logger.info(f"Full data refresh completed - Task ID: {task_id}")
        return final_result
        
    except Exception as e:
        logger.error(f"Full data refresh failed - Task ID: {task_id}, Error: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "task_id": task_id}
        )
        raise

@celery_app.task(bind=True, name="incremental_update_task")
def incremental_update_task(self, task_id: str, since_date: Optional[datetime] = None):
    try:
        logger.info(f"Starting incremental update - Task ID: {task_id}")
        
        self.update_state(
            state="PROGRESS",
            meta={"stage": "fetching_new_data", "progress": 20}
        )
        
        async def update_data():
            async for session in get_async_session():
                result = await data_ingestion_service.incremental_update(session, since_date)
                
                self.update_state(
                    state="PROGRESS", 
                    meta={"stage": "updating_cache", "progress": 80}
                )
                
                await cache_service.invalidate_pattern("analytics_*", "analytics")
                
                return result
        
        result = asyncio.run(update_data())
        
        final_result = {
            "task_id": task_id,
            "completed_at": datetime.utcnow().isoformat(),
            **result
        }
        
        logger.info(f"Incremental update completed - Task ID: {task_id}")
        return final_result
        
    except Exception as e:
        logger.error(f"Incremental update failed - Task ID: {task_id}, Error: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "task_id": task_id}
        )
        raise

@celery_app.task(name="scheduled_incremental_update")
def scheduled_incremental_update():
    try:
        logger.info("Running scheduled incremental update")
        
        yesterday = datetime.utcnow() - timedelta(days=1)
        task_id = f"scheduled_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        async def update():
            async for session in get_async_session():
                return await data_ingestion_service.incremental_update(session, yesterday)
        
        result = asyncio.run(update())
        
        logger.info(f"Scheduled incremental update completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Scheduled incremental update failed: {e}")
        raise

@celery_app.task(name="scheduled_data_quality_check")
def scheduled_data_quality_check():
    try:
        logger.info("Running scheduled data quality check")
        
        async def quality_check():
            async for session in get_async_session():
                from sqlalchemy import select, func, case
                from app.models.database import ExpenditureRecord
                
                current_year = datetime.utcnow().year
                
                quality_query = select(
                    func.count(ExpenditureRecord.id).label("total_records"),
                    func.sum(
                        case((ExpenditureRecord.vendor_name.is_(None), 1), else_=0)
                    ).label("missing_vendor"),
                    func.sum(
                        case((ExpenditureRecord.monetary_amount <= 0, 1), else_=0)
                    ).label("invalid_amounts"),
                    func.sum(
                        case((ExpenditureRecord.dept_name.is_(None), 1), else_=0)
                    ).label("missing_department")
                ).where(ExpenditureRecord.fiscal_year == current_year)
                
                result = await session.execute(quality_query)
                stats = result.first()
                
                total_records = stats.total_records or 0
                quality_issues = (stats.missing_vendor or 0) + (stats.invalid_amounts or 0) + (stats.missing_department or 0)
                quality_score = ((total_records - quality_issues) / total_records * 100) if total_records > 0 else 100
                
                return {
                    "fiscal_year": current_year,
                    "total_records": total_records,
                    "quality_score": round(quality_score, 2),
                    "issues": {
                        "missing_vendor": stats.missing_vendor or 0,
                        "invalid_amounts": stats.invalid_amounts or 0,
                        "missing_department": stats.missing_department or 0
                    }
                }
        
        result = asyncio.run(quality_check())
        
        if result["quality_score"] < 95:
            logger.warning(f"Data quality below threshold: {result['quality_score']}%")
        
        logger.info(f"Data quality check completed: {result['quality_score']}% quality score")
        return result
        
    except Exception as e:
        logger.error(f"Scheduled data quality check failed: {e}")
        raise

@celery_app.task(name="scheduled_analytics_refresh") 
def scheduled_analytics_refresh():
    try:
        logger.info("Running scheduled analytics refresh")
        
        async def refresh_analytics():
            await cache_service.invalidate_pattern("*", "analytics")
            await cache_service.invalidate_pattern("*", "vendor_analytics")
            await cache_service.invalidate_pattern("*", "department_analytics")
            
            async for session in get_async_session():
                from app.services.analytics_service import analytics_service
                
                current_year = datetime.utcnow().year
                
                trends = await analytics_service.get_spending_trends(session, "yearly", [current_year])
                await cache_service.cache_analytics_result("spending_trends", {"period": "yearly", "fiscal_years": [current_year]}, trends)
                
                insights = await analytics_service.get_departmental_insights(session, current_year, 10)
                await cache_service.cache_analytics_result("departmental_insights", {"fiscal_year": current_year, "top_n": 10}, insights)
                
                concentration = await analytics_service.get_vendor_concentration_analysis(session, current_year)
                await cache_service.cache_analytics_result("vendor_concentration", {"fiscal_year": current_year}, concentration)
                
                return {
                    "cache_refreshed": True,
                    "fiscal_year": current_year,
                    "analytics_precomputed": ["spending_trends", "departmental_insights", "vendor_concentration"]
                }
        
        result = asyncio.run(refresh_analytics())
        
        logger.info(f"Analytics refresh completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Scheduled analytics refresh failed: {e}")
        raise

@celery_app.task(name="data_export_task")
def data_export_task(
    export_format: str,
    filters: Dict[str, Any],
    task_id: str
):
    try:
        logger.info(f"Starting data export - Task ID: {task_id}, Format: {export_format}")
        
        async def export_data():
            async for session in get_async_session():
                from sqlalchemy import select, and_
                from app.models.database import ExpenditureRecord
                
                query = select(ExpenditureRecord)
                conditions = []
                
                if filters.get("fiscal_year"):
                    conditions.append(ExpenditureRecord.fiscal_year == filters["fiscal_year"])
                
                if filters.get("dept_name"):
                    conditions.append(ExpenditureRecord.dept_name.ilike(f"%{filters['dept_name']}%"))
                
                if filters.get("vendor_name"):
                    conditions.append(ExpenditureRecord.vendor_name.ilike(f"%{filters['vendor_name']}%"))
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                result = await session.execute(query)
                records = result.scalars().all()
                
                data = []
                for record in records:
                    data.append({
                        "voucher": record.voucher,
                        "entered": record.entered.isoformat(),
                        "vendor_name": record.vendor_name,
                        "dept_name": record.dept_name,
                        "account_descr": record.account_descr,
                        "monetary_amount": record.monetary_amount,
                        "fiscal_year": record.fiscal_year
                    })
                
                df = pd.DataFrame(data)
                
                if export_format == "csv":
                    export_path = f"/app/data/export_{task_id}.csv"
                    df.to_csv(export_path, index=False)
                elif export_format == "excel":
                    export_path = f"/app/data/export_{task_id}.xlsx"
                    df.to_excel(export_path, index=False)
                elif export_format == "json":
                    export_path = f"/app/data/export_{task_id}.json"
                    df.to_json(export_path, orient="records", date_format="iso")
                
                return {
                    "export_path": export_path,
                    "record_count": len(data),
                    "file_format": export_format,
                    "filters_applied": filters
                }
        
        result = asyncio.run(export_data())
        
        logger.info(f"Data export completed - Task ID: {task_id}")
        return {
            "task_id": task_id,
            "completed_at": datetime.utcnow().isoformat(),
            **result
        }
        
    except Exception as e:
        logger.error(f"Data export failed - Task ID: {task_id}, Error: {e}")
        raise

@celery_app.task(name="data_aggregation_task")
def data_aggregation_task(aggregation_type: str, parameters: Dict[str, Any]):
    try:
        logger.info(f"Starting data aggregation - Type: {aggregation_type}")
        
        async def aggregate():
            async for session in get_async_session():
                from sqlalchemy import select, func, desc
                from app.models.database import ExpenditureRecord
                
                if aggregation_type == "vendor_monthly":
                    query = select(
                        ExpenditureRecord.vendor_name,
                        ExpenditureRecord.fiscal_year,
                        ExpenditureRecord.fiscal_month,
                        func.sum(ExpenditureRecord.monetary_amount).label("monthly_spending"),
                        func.count(ExpenditureRecord.id).label("transaction_count")
                    ).group_by(
                        ExpenditureRecord.vendor_name,
                        ExpenditureRecord.fiscal_year,
                        ExpenditureRecord.fiscal_month
                    ).order_by(
                        ExpenditureRecord.vendor_name,
                        ExpenditureRecord.fiscal_year,
                        ExpenditureRecord.fiscal_month
                    )
                
                elif aggregation_type == "department_category":
                    query = select(
                        ExpenditureRecord.dept_name,
                        ExpenditureRecord.account_descr,
                        func.sum(ExpenditureRecord.monetary_amount).label("category_spending"),
                        func.count(ExpenditureRecord.id).label("transaction_count")
                    ).group_by(
                        ExpenditureRecord.dept_name,
                        ExpenditureRecord.account_descr
                    ).order_by(desc("category_spending"))
                
                fiscal_year = parameters.get("fiscal_year")
                if fiscal_year:
                    query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
                
                result = await session.execute(query)
                aggregated_data = result.all()
                
                output_data = []
                for row in aggregated_data:
                    if aggregation_type == "vendor_monthly":
                        output_data.append({
                            "vendor_name": row.vendor_name,
                            "fiscal_year": row.fiscal_year,
                            "fiscal_month": row.fiscal_month,
                            "monthly_spending": float(row.monthly_spending),
                            "transaction_count": row.transaction_count
                        })
                    elif aggregation_type == "department_category":
                        output_data.append({
                            "dept_name": row.dept_name,
                            "account_descr": row.account_descr,
                            "category_spending": float(row.category_spending),
                            "transaction_count": row.transaction_count
                        })
                
                return {
                    "aggregation_type": aggregation_type,
                    "parameters": parameters,
                    "record_count": len(output_data),
                    "data": output_data
                }
        
        result = asyncio.run(aggregate())
        
        logger.info(f"Data aggregation completed - Type: {aggregation_type}")
        return result
        
    except Exception as e:
        logger.error(f"Data aggregation failed - Type: {aggregation_type}, Error: {e}")
        raise

@celery_app.task(name="performance_optimization_task")
def performance_optimization_task():
    try:
        logger.info("Starting performance optimization task")
        
        async def optimize():
            async for session in get_async_session():
                from sqlalchemy import text
                
                optimization_queries = [
                    "ANALYZE expenditure_records;",
                    "ANALYZE vendor_performance;", 
                    "ANALYZE department_analytics;",
                    "REINDEX INDEX idx_vendor_name;",
                    "REINDEX INDEX idx_dept_name;",
                    "REINDEX INDEX idx_fiscal_year;"
                ]
                
                for query in optimization_queries:
                    await session.execute(text(query))
                
                vacuum_query = "VACUUM (ANALYZE) expenditure_records;"
                await session.execute(text(vacuum_query))
                
                return {
                    "optimizations_performed": len(optimization_queries) + 1,
                    "queries_executed": optimization_queries + [vacuum_query]
                }
        
        result = asyncio.run(optimize())
        
        logger.info(f"Performance optimization completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")
        raise

@celery_app.task(name="data_backup_task")
def data_backup_task(backup_type: str = "incremental"):
    try:
        logger.info(f"Starting data backup - Type: {backup_type}")
        
        async def backup():
            async for session in get_async_session():
                from sqlalchemy import text, select, func
                from app.models.database import ExpenditureRecord
                import subprocess
                import os
                from datetime import datetime
                
                backup_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_dir = "/app/data/backups"
                os.makedirs(backup_dir, exist_ok=True)
                
                if backup_type == "full":
                    backup_file = f"{backup_dir}/full_backup_{backup_timestamp}.sql"
                    
                    dump_command = [
                        "pg_dump",
                        settings.database_url.replace("postgresql+asyncpg://", "postgresql://"),
                        "-f", backup_file,
                        "--no-owner",
                        "--no-privileges"
                    ]
                    
                    subprocess.run(dump_command, check=True)
                    
                    file_size = os.path.getsize(backup_file)
                    
                elif backup_type == "incremental":
                    record_count_query = select(func.count(ExpenditureRecord.id))
                    result = await session.execute(record_count_query)
                    record_count = result.scalar()
                    
                    backup_info_file = f"{backup_dir}/incremental_info_{backup_timestamp}.json"
                    
                    import json
                    backup_info = {
                        "backup_type": "incremental",
                        "timestamp": backup_timestamp,
                        "record_count": record_count,
                        "last_backup": datetime.utcnow().isoformat()
                    }
                    
                    with open(backup_info_file, 'w') as f:
                        json.dump(backup_info, f, indent=2)
                    
                    file_size = os.path.getsize(backup_info_file)
                
                return {
                    "backup_type": backup_type,
                    "backup_file": backup_file if backup_type == "full" else backup_info_file,
                    "file_size_bytes": file_size,
                    "timestamp": backup_timestamp
                }
        
        result = asyncio.run(backup())
        
        logger.info(f"Data backup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Data backup failed: {e}")
        raise