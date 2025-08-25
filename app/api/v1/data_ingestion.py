from fastapi import APIRouter, Depends, Query, Body, BackgroundTasks, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import io

from app.core.database import get_db_session
from app.services.data_ingestion_service import data_ingestion_service
from app.services.cache_service import cache_service
from app.tasks.data_tasks import full_refresh_task, incremental_update_task
from app.core.exceptions import ValidationException, DataProcessingException
from loguru import logger

router = APIRouter()

@router.post("/refresh/full")
async def trigger_full_data_refresh(
    background_tasks: BackgroundTasks,
    source_url: Optional[str] = Query(None, description="Override default data source URL"),
    clear_cache: bool = Query(True, description="Clear all caches after refresh"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            full_refresh_task,
            task_id=task_id,
            source_url=source_url,
            clear_cache=clear_cache
        )
        
        return {
            "message": "Full data refresh started",
            "task_id": task_id,
            "source_url": source_url or "default",
            "clear_cache": clear_cache
        }
        
    except Exception as e:
        logger.error(f"Error starting full data refresh: {e}")
        raise HTTPException(status_code=500, detail="Failed to start full data refresh")

@router.post("/refresh/incremental")
async def trigger_incremental_update(
    background_tasks: BackgroundTasks,
    since_date: Optional[datetime] = Query(None, description="Update since this date (ISO format)"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            incremental_update_task,
            task_id=task_id,
            since_date=since_date
        )
        
        return {
            "message": "Incremental data update started",
            "task_id": task_id,
            "since_date": since_date.isoformat() if since_date else "auto-detect"
        }
        
    except Exception as e:
        logger.error(f"Error starting incremental update: {e}")
        raise HTTPException(status_code=500, detail="Failed to start incremental update")

@router.post("/upload/csv")
async def upload_csv_data(
    file: UploadFile = File(..., description="CSV file to upload"),
    replace_existing: bool = Query(False, description="Replace all existing data"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        if not file.filename.endswith('.csv'):
            raise ValidationException("File must be a CSV file")
        
        contents = await file.read()
        csv_content = contents.decode('utf-8')
        
        import pandas as pd
        df = pd.read_csv(io.StringIO(csv_content))
        
        df_clean = data_ingestion_service.clean_and_validate_data(df)
        records = data_ingestion_service.transform_to_schema(df_clean)
        
        if replace_existing:
            from sqlalchemy import delete
            from app.models.database import ExpenditureRecord
            await session.execute(delete(ExpenditureRecord))
            logger.info("Cleared existing data for replacement")
        
        result = await data_ingestion_service.ingest_data_batch(session, records)
        
        await cache_service.invalidate_pattern("*", "analytics")
        await cache_service.invalidate_pattern("*", "vendor_analytics")
        
        return {
            "message": "CSV data uploaded successfully",
            "filename": file.filename,
            "raw_records": len(df),
            "clean_records": len(df_clean),
            "replace_existing": replace_existing,
            **result
        }
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error uploading CSV data: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload CSV data")

@router.get("/status/ingestion")
async def get_data_ingestion_status(session: AsyncSession = Depends(get_db_session)):
    try:
        from sqlalchemy import select, func
        from app.models.database import ExpenditureRecord
        
        stats_query = select(
            func.count(ExpenditureRecord.id).label("total_records"),
            func.min(ExpenditureRecord.entered).label("earliest_record"),
            func.max(ExpenditureRecord.entered).label("latest_record"),
            func.sum(ExpenditureRecord.monetary_amount).label("total_amount")
        )
        
        stats_result = await session.execute(stats_query)
        stats = stats_result.first()
        
        fiscal_year_query = select(
            ExpenditureRecord.fiscal_year,
            func.count(ExpenditureRecord.id).label("record_count"),
            func.sum(ExpenditureRecord.monetary_amount).label("year_amount")
        ).group_by(ExpenditureRecord.fiscal_year).order_by(ExpenditureRecord.fiscal_year)
        
        fy_result = await session.execute(fiscal_year_query)
        fiscal_years = fy_result.all()
        
        return {
            "database_status": {
                "total_records": stats.total_records or 0,
                "total_amount": float(stats.total_amount or 0),
                "earliest_record": stats.earliest_record.isoformat() if stats.earliest_record else None,
                "latest_record": stats.latest_record.isoformat() if stats.latest_record else None
            },
            "fiscal_years": [
                {
                    "fiscal_year": fy.fiscal_year,
                    "record_count": fy.record_count,
                    "total_amount": float(fy.year_amount)
                }
                for fy in fiscal_years
            ] if fiscal_years else []
        }
        
    except Exception as e:
        logger.error(f"Error getting data ingestion status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get data ingestion status")

@router.post("/validate/data-quality")
async def validate_data_quality(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Validate specific fiscal year"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, case, and_
        from app.models.database import ExpenditureRecord
        
        base_query = select(ExpenditureRecord)
        if fiscal_year:
            base_query = base_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
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
            ).label("missing_department"),
            func.avg(ExpenditureRecord.monetary_amount).label("avg_amount"),
            func.stddev(ExpenditureRecord.monetary_amount).label("stddev_amount")
        )
        
        if fiscal_year:
            quality_query = quality_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        quality_result = await session.execute(quality_query)
        quality_stats = quality_result.first()
        
        duplicates_query = select(
            func.count().label("duplicate_count")
        ).select_from(
            select(
                ExpenditureRecord.voucher,
                ExpenditureRecord.vendor_name,
                ExpenditureRecord.monetary_amount,
                func.count().label("cnt")
            ).group_by(
                ExpenditureRecord.voucher,
                ExpenditureRecord.vendor_name,
                ExpenditureRecord.monetary_amount
            ).having(func.count() > 1).subquery()
        )
        
        if fiscal_year:
            duplicates_query = duplicates_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        duplicates_result = await session.execute(duplicates_query)
        duplicate_count = duplicates_result.scalar() or 0
        
        total_records = quality_stats.total_records or 0
        quality_score = 100.0
        
        if total_records > 0:
            missing_vendor_pct = (quality_stats.missing_vendor or 0) / total_records * 100
            invalid_amounts_pct = (quality_stats.invalid_amounts or 0) / total_records * 100
            missing_dept_pct = (quality_stats.missing_department or 0) / total_records * 100
            duplicate_pct = duplicate_count / total_records * 100
            
            quality_score = max(0, 100 - missing_vendor_pct - invalid_amounts_pct - missing_dept_pct - duplicate_pct)
        
        return {
            "fiscal_year": fiscal_year,
            "data_quality": {
                "quality_score": round(quality_score, 2),
                "total_records": total_records,
                "issues": {
                    "missing_vendor": quality_stats.missing_vendor or 0,
                    "invalid_amounts": quality_stats.invalid_amounts or 0,
                    "missing_department": quality_stats.missing_department or 0,
                    "duplicate_records": duplicate_count
                },
                "statistics": {
                    "average_amount": float(quality_stats.avg_amount or 0),
                    "stddev_amount": float(quality_stats.stddev_amount or 0)
                }
            },
            "recommendations": [
                "Review vendor name standardization" if (quality_stats.missing_vendor or 0) > 0 else None,
                "Investigate negative or zero amounts" if (quality_stats.invalid_amounts or 0) > 0 else None,
                "Complete department information" if (quality_stats.missing_department or 0) > 0 else None,
                "Review duplicate transactions" if duplicate_count > 0 else None
            ]
        }
        
    except Exception as e:
        logger.error(f"Error validating data quality: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate data quality")

@router.get("/schema/mapping")
async def get_data_schema_mapping():
    return {
        "required_columns": [
            "Voucher",
            "Entered", 
            "Vendor Name",
            "Monetary Amount"
        ],
        "optional_columns": [
            "Voucher Line",
            "Distribution Line",
            "Month",
            "Fiscal Month", 
            "Fiscal Year",
            "Year",
            "Account",
            "Account Descr",
            "Dept",
            "Dept Name",
            "6 Digit Org Name"
        ],
        "column_types": {
            "Voucher": "string",
            "Voucher Line": "integer",
            "Distribution Line": "integer", 
            "Entered": "datetime",
            "Month": "integer",
            "Fiscal Month": "integer",
            "Fiscal Year": "integer",
            "Year": "integer",
            "Vendor Name": "string",
            "Account": "string",
            "Account Descr": "string",
            "Dept": "string",
            "Dept Name": "string",
            "6 Digit Org Name": "string",
            "Monetary Amount": "float"
        },
        "validation_rules": {
            "Monetary Amount": "Must be >= 0",
            "Entered": "Must be valid datetime",
            "Fiscal Year": "Must be between 2000-2100",
            "Vendor Name": "Cannot be empty"
        }
    }