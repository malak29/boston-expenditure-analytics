from fastapi import APIRouter, Depends, Query, Path, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
import uuid

from app.core.database import get_db_session
from app.services.database_service import vendor_service
from app.services.cache_service import cache_service
from app.schemas.expenditure import VendorPerformance, PaginationParams
from app.core.exceptions import NotFoundException
from loguru import logger

router = APIRouter()

@router.get("/", response_model=List[VendorPerformance])
async def get_vendors(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    search: Optional[str] = Query(None, description="Search vendor names"),
    min_expenditure: Optional[float] = Query(None, ge=0, description="Minimum total expenditure"),
    risk_category: Optional[str] = Query(None, regex="^(low|medium|high)$", description="Risk category filter"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, desc, and_
        from app.models.database import VendorPerformance
        
        query = select(VendorPerformance)
        conditions = []
        
        if search:
            conditions.append(VendorPerformance.vendor_name.ilike(f"%{search}%"))
        
        if min_expenditure is not None:
            conditions.append(VendorPerformance.total_expenditure >= min_expenditure)
        
        if risk_category:
            conditions.append(VendorPerformance.risk_category == risk_category)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(VendorPerformance.total_expenditure))
        
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        result = await session.execute(query)
        vendors = result.scalars().all()
        
        return vendors
        
    except Exception as e:
        logger.error(f"Error fetching vendors: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vendors")

@router.get("/{vendor_id}", response_model=VendorPerformance)
async def get_vendor(
    vendor_id: uuid.UUID = Path(..., description="Vendor ID"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select
        from app.models.database import VendorPerformance
        
        query = select(VendorPerformance).where(VendorPerformance.id == vendor_id)
        result = await session.execute(query)
        vendor = result.scalar_one_or_none()
        
        if not vendor:
            raise NotFoundException("Vendor")
        
        return vendor
        
    except NotFoundException:
        raise
    except Exception as e:
        logger.error(f"Error fetching vendor {vendor_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vendor")

@router.get("/name/{vendor_name}/analytics")
async def get_vendor_analytics(
    vendor_name: str = Path(..., description="Vendor name"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        cache_key = f"vendor_analytics_{vendor_name}_{fiscal_year}"
        
        if use_cache:
            cached_result = await cache_service.get(cache_key, namespace="vendor_analytics")
            if cached_result:
                logger.info(f"Returning cached vendor analytics for {vendor_name}")
                return cached_result
        
        from sqlalchemy import select, func, desc, extract
        from app.models.database import ExpenditureRecord
        
        base_query = select(ExpenditureRecord).where(
            ExpenditureRecord.vendor_name.ilike(f"%{vendor_name}%")
        )
        
        if fiscal_year:
            base_query = base_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        total_query = select(
            func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
            func.count(ExpenditureRecord.id).label("transaction_count"),
            func.avg(ExpenditureRecord.monetary_amount).label("avg_transaction"),
            func.min(ExpenditureRecord.entered).label("first_transaction"),
            func.max(ExpenditureRecord.entered).label("last_transaction")
        ).where(ExpenditureRecord.vendor_name.ilike(f"%{vendor_name}%"))
        
        if fiscal_year:
            total_query = total_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        total_result = await session.execute(total_query)
        totals = total_result.first()
        
        monthly_query = select(
            ExpenditureRecord.fiscal_month,
            func.sum(ExpenditureRecord.monetary_amount).label("monthly_spending"),
            func.count(ExpenditureRecord.id).label("monthly_transactions")
        ).where(
            ExpenditureRecord.vendor_name.ilike(f"%{vendor_name}%")
        ).group_by(ExpenditureRecord.fiscal_month)
        
        if fiscal_year:
            monthly_query = monthly_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        monthly_query = monthly_query.order_by(ExpenditureRecord.fiscal_month)
        monthly_result = await session.execute(monthly_query)
        monthly_data = monthly_result.all()
        
        dept_query = select(
            ExpenditureRecord.dept_name,
            func.sum(ExpenditureRecord.monetary_amount).label("dept_spending")
        ).where(
            ExpenditureRecord.vendor_name.ilike(f"%{vendor_name}%")
        ).group_by(ExpenditureRecord.dept_name)
        
        if fiscal_year:
            dept_query = dept_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        dept_query = dept_query.order_by(desc("dept_spending"))
        dept_result = await session.execute(dept_query)
        dept_data = dept_result.all()
        
        result = {
            "vendor_name": vendor_name,
            "fiscal_year": fiscal_year,
            "summary": {
                "total_spending": float(totals.total_spending or 0),
                "transaction_count": totals.transaction_count or 0,
                "average_transaction": float(totals.avg_transaction or 0),
                "first_transaction": totals.first_transaction.isoformat() if totals.first_transaction else None,
                "last_transaction": totals.last_transaction.isoformat() if totals.last_transaction else None
            },
            "monthly_breakdown": [
                {
                    "fiscal_month": month.fiscal_month,
                    "spending": float(month.monthly_spending),
                    "transactions": month.monthly_transactions
                }
                for month in monthly_data
            ],
            "department_breakdown": [
                {
                    "dept_name": dept.dept_name,
                    "spending": float(dept.dept_spending)
                }
                for dept in dept_data
            ]
        }
        
        if use_cache:
            await cache_service.set(cache_key, result, ttl=1800, namespace="vendor_analytics")
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching vendor analytics for {vendor_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vendor analytics")

@router.get("/performance/ranking")
async def get_vendor_performance_ranking(
    metric: str = Query("total_expenditure", regex="^(total_expenditure|transaction_count|avg_monthly_spending)$"),
    limit: int = Query(20, ge=1, le=100, description="Number of vendors to return"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, desc, case
        from app.models.database import ExpenditureRecord
        
        if fiscal_year:
            subquery = select(
                ExpenditureRecord.vendor_name,
                func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
                func.count(ExpenditureRecord.id).label("transaction_count"),
                func.avg(ExpenditureRecord.monetary_amount).label("avg_spending")
            ).where(
                ExpenditureRecord.fiscal_year == fiscal_year
            ).group_by(ExpenditureRecord.vendor_name).subquery()
            
            if metric == "total_expenditure":
                order_column = desc(subquery.c.total_spending)
            elif metric == "transaction_count":
                order_column = desc(subquery.c.transaction_count)
            else:
                order_column = desc(subquery.c.avg_spending)
            
            query = select(subquery).order_by(order_column).limit(limit)
            
        else:
            from app.models.database import VendorPerformance
            
            if metric == "total_expenditure":
                order_column = desc(VendorPerformance.total_expenditure)
            elif metric == "transaction_count":
                order_column = desc(VendorPerformance.transaction_count)
            else:
                order_column = desc(VendorPerformance.avg_monthly_spending)
            
            query = select(VendorPerformance).order_by(order_column).limit(limit)
        
        result = await session.execute(query)
        vendors = result.all()
        
        ranking_data = []
        for i, vendor in enumerate(vendors, 1):
            if hasattr(vendor, 'vendor_name'):
                vendor_data = {
                    "rank": i,
                    "vendor_name": vendor.vendor_name,
                    "total_expenditure": float(getattr(vendor, 'total_spending', vendor.total_expenditure or 0)),
                    "transaction_count": getattr(vendor, 'transaction_count', vendor.transaction_count or 0),
                    "avg_spending": float(getattr(vendor, 'avg_spending', vendor.avg_monthly_spending or 0))
                }
            else:
                vendor_data = {
                    "rank": i,
                    "vendor_name": vendor[0],
                    "total_expenditure": float(vendor[1] or 0),
                    "transaction_count": vendor[2] or 0,
                    "avg_spending": float(vendor[3] or 0)
                }
            
            ranking_data.append(vendor_data)
        
        return {
            "ranking_metric": metric,
            "fiscal_year": fiscal_year,
            "vendors": ranking_data
        }
        
    except Exception as e:
        logger.error(f"Error getting vendor performance ranking: {e}")
        raise HTTPException(status_code=500, detail="Failed to get vendor performance ranking")

@router.post("/performance/recalculate")
async def recalculate_vendor_performance(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Recalculate for specific fiscal year"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, delete
        from app.models.database import VendorPerformance, ExpenditureRecord
        
        if fiscal_year:
            logger.info(f"Recalculating vendor performance for fiscal year {fiscal_year}")
        else:
            logger.info("Recalculating vendor performance for all years")
            await session.execute(delete(VendorPerformance))
        
        vendor_stats_query = select(
            ExpenditureRecord.vendor_name,
            func.sum(ExpenditureRecord.monetary_amount).label("total_expenditure"),
            func.count(ExpenditureRecord.id).label("transaction_count"),
            func.min(ExpenditureRecord.entered).label("first_transaction"),
            func.max(ExpenditureRecord.entered).label("last_transaction")
        ).group_by(ExpenditureRecord.vendor_name)
        
        if fiscal_year:
            vendor_stats_query = vendor_stats_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        vendor_stats_result = await session.execute(vendor_stats_query)
        vendor_stats = vendor_stats_result.all()
        
        updated_vendors = 0
        for stats in vendor_stats:
            vendor = await vendor_service.get_or_create_vendor_performance(session, stats.vendor_name)
            
            vendor.total_expenditure = float(stats.total_expenditure)
            vendor.transaction_count = stats.transaction_count
            vendor.first_transaction = stats.first_transaction
            vendor.last_transaction = stats.last_transaction
            
            if stats.transaction_count > 0:
                months_active = max(1, (stats.last_transaction - stats.first_transaction).days / 30)
                vendor.avg_monthly_spending = float(stats.total_expenditure / months_active)
            
            if stats.total_expenditure > 100000:
                vendor.risk_category = "high"
            elif stats.total_expenditure > 10000:
                vendor.risk_category = "medium"
            else:
                vendor.risk_category = "low"
            
            vendor.performance_score = min(1.0, stats.transaction_count / 100.0)
            
            updated_vendors += 1
        
        await cache_service.invalidate_pattern("*", "vendor_analytics")
        
        return {
            "message": f"Recalculated performance for {updated_vendors} vendors",
            "fiscal_year": fiscal_year,
            "vendors_updated": updated_vendors
        }
        
    except Exception as e:
        logger.error(f"Error recalculating vendor performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to recalculate vendor performance")