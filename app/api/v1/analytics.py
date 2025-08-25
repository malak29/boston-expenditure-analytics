from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from app.core.database import get_db_session
from app.services.analytics_service import analytics_service
from app.services.cache_service import cache_service
from app.core.exceptions import ValidationException
from loguru import logger

router = APIRouter()

@router.get("/trends/spending")
async def get_spending_trends(
    period: str = Query("yearly", regex="^(yearly|monthly)$", description="Analysis period"),
    fiscal_years: Optional[str] = Query(None, description="Comma-separated fiscal years (e.g., 2022,2023,2024)"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        fiscal_years_list = None
        if fiscal_years:
            try:
                fiscal_years_list = [int(year.strip()) for year in fiscal_years.split(",")]
            except ValueError:
                raise ValidationException("Invalid fiscal years format. Use comma-separated integers.")
        
        cache_params = {
            "period": period,
            "fiscal_years": fiscal_years_list
        }
        
        if use_cache:
            cached_result = await cache_service.get_cached_analytics("spending_trends", cache_params)
            if cached_result:
                logger.info("Returning cached spending trends")
                return cached_result
        
        result = await analytics_service.get_spending_trends(session, period, fiscal_years_list)
        
        if use_cache:
            await cache_service.cache_analytics_result("spending_trends", cache_params, result)
        
        return result
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error in spending trends analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze spending trends")

@router.get("/departments/insights")
async def get_departmental_insights(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    top_n: int = Query(10, ge=1, le=50, description="Number of top departments to analyze"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        cache_params = {
            "fiscal_year": fiscal_year,
            "top_n": top_n
        }
        
        if use_cache:
            cached_result = await cache_service.get_cached_analytics("departmental_insights", cache_params)
            if cached_result:
                logger.info("Returning cached departmental insights")
                return cached_result
        
        result = await analytics_service.get_departmental_insights(session, fiscal_year, top_n)
        
        if use_cache:
            await cache_service.cache_analytics_result("departmental_insights", cache_params, result, ttl=1800)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in departmental insights analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze departmental insights")

@router.get("/vendors/concentration")
async def get_vendor_concentration(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        cache_params = {"fiscal_year": fiscal_year}
        
        if use_cache:
            cached_result = await cache_service.get_cached_analytics("vendor_concentration", cache_params)
            if cached_result:
                logger.info("Returning cached vendor concentration analysis")
                return cached_result
        
        result = await analytics_service.get_vendor_concentration_analysis(session, fiscal_year)
        
        if use_cache:
            await cache_service.cache_analytics_result("vendor_concentration", cache_params, result, ttl=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in vendor concentration analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze vendor concentration")

@router.get("/outliers/spending")
async def get_spending_outliers(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    threshold_multiplier: float = Query(3.0, ge=1.0, le=5.0, description="IQR multiplier for outlier detection"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        cache_params = {
            "fiscal_year": fiscal_year,
            "threshold_multiplier": threshold_multiplier
        }
        
        if use_cache:
            cached_result = await cache_service.get_cached_analytics("spending_outliers", cache_params)
            if cached_result:
                logger.info("Returning cached spending outliers")
                return cached_result
        
        result = await analytics_service.get_spending_outliers(session, fiscal_year, threshold_multiplier)
        
        if use_cache:
            await cache_service.cache_analytics_result("spending_outliers", cache_params, result, ttl=1800)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in outlier detection: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect spending outliers")

@router.get("/distribution/categories")
async def get_category_distribution(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    groupby: str = Query("account_descr", regex="^(account_descr|dept_name|org_name_6_digit)$", description="Grouping field"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        cache_params = {
            "fiscal_year": fiscal_year,
            "groupby": groupby
        }
        
        if use_cache:
            cached_result = await cache_service.get_cached_analytics("category_distribution", cache_params)
            if cached_result:
                logger.info("Returning cached category distribution")
                return cached_result
        
        result = await analytics_service.get_category_distribution(session, fiscal_year, groupby)
        
        if use_cache:
            await cache_service.cache_analytics_result("category_distribution", cache_params, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in category distribution analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze category distribution")

@router.post("/cache/invalidate")
async def invalidate_analytics_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to invalidate"),
):
    try:
        if pattern:
            deleted_count = await cache_service.invalidate_pattern(f"*{pattern}*", "analytics")
        else:
            deleted_count = await cache_service.invalidate_pattern("*", "analytics")
        
        return {
            "message": f"Invalidated {deleted_count} cache entries",
            "pattern": pattern or "all analytics cache"
        }
        
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")

@router.get("/cache/stats")
async def get_cache_statistics():
    try:
        stats = await cache_service.get_cache_stats()
        health = await cache_service.health_check()
        
        return {
            "cache_healthy": health,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")