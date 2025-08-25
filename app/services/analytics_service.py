from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, asc, and_, case
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

from app.models.database import ExpenditureRecord, DepartmentAnalytics
from app.core.exceptions import DataProcessingException, NotFoundException
from app.core.config import settings

class AnalyticsService:
    
    async def get_spending_trends(
        self,
        session: AsyncSession,
        period: str = "yearly",
        fiscal_years: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        try:
            if period == "yearly":
                query = select(
                    ExpenditureRecord.fiscal_year,
                    func.sum(ExpenditureRecord.monetary_amount).label("total_amount"),
                    func.count(ExpenditureRecord.id).label("transaction_count"),
                    func.avg(ExpenditureRecord.monetary_amount).label("avg_amount")
                ).group_by(ExpenditureRecord.fiscal_year)
                
            elif period == "monthly":
                query = select(
                    ExpenditureRecord.fiscal_year,
                    ExpenditureRecord.fiscal_month,
                    func.sum(ExpenditureRecord.monetary_amount).label("total_amount"),
                    func.count(ExpenditureRecord.id).label("transaction_count")
                ).group_by(ExpenditureRecord.fiscal_year, ExpenditureRecord.fiscal_month)
            
            if fiscal_years:
                query = query.where(ExpenditureRecord.fiscal_year.in_(fiscal_years))
            
            query = query.order_by(asc(ExpenditureRecord.fiscal_year))
            
            result = await session.execute(query)
            trends = result.all()
            
            return {
                "period": period,
                "data": [
                    {
                        "fiscal_year": trend.fiscal_year,
                        "fiscal_month": getattr(trend, 'fiscal_month', None),
                        "total_amount": float(trend.total_amount),
                        "transaction_count": trend.transaction_count,
                        "avg_amount": float(getattr(trend, 'avg_amount', 0))
                    }
                    for trend in trends
                ]
            }
        except Exception as e:
            raise DataProcessingException("spending trends calculation", {"error": str(e)})
    
    async def get_departmental_insights(
        self,
        session: AsyncSession,
        fiscal_year: Optional[int] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        try:
            query = select(
                ExpenditureRecord.dept_name,
                func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
                func.count(ExpenditureRecord.id).label("transaction_count"),
                func.avg(ExpenditureRecord.monetary_amount).label("avg_transaction"),
                func.min(ExpenditureRecord.monetary_amount).label("min_transaction"),
                func.max(ExpenditureRecord.monetary_amount).label("max_transaction")
            ).group_by(ExpenditureRecord.dept_name)
            
            if fiscal_year:
                query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
            
            query = query.order_by(desc("total_spending")).limit(top_n)
            
            result = await session.execute(query)
            departments = result.all()
            
            return {
                "fiscal_year": fiscal_year,
                "departments": [
                    {
                        "dept_name": dept.dept_name,
                        "total_spending": float(dept.total_spending),
                        "transaction_count": dept.transaction_count,
                        "avg_transaction": float(dept.avg_transaction),
                        "min_transaction": float(dept.min_transaction),
                        "max_transaction": float(dept.max_transaction)
                    }
                    for dept in departments
                ]
            }
        except Exception as e:
            raise DataProcessingException("departmental insights calculation", {"error": str(e)})
    
    async def get_vendor_concentration_analysis(
        self,
        session: AsyncSession,
        fiscal_year: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            total_query = select(func.sum(ExpenditureRecord.monetary_amount))
            if fiscal_year:
                total_query = total_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
            
            total_result = await session.execute(total_query)
            total_spending = float(total_result.scalar() or 0)
            
            vendor_query = select(
                ExpenditureRecord.vendor_name,
                func.sum(ExpenditureRecord.monetary_amount).label("vendor_spending")
            ).group_by(ExpenditureRecord.vendor_name)
            
            if fiscal_year:
                vendor_query = vendor_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
            
            vendor_query = vendor_query.order_by(desc("vendor_spending"))
            
            vendor_result = await session.execute(vendor_query)
            vendors = vendor_result.all()
            
            concentration_data = []
            cumulative_percentage = 0
            
            for vendor in vendors:
                vendor_percentage = (vendor.vendor_spending / total_spending) * 100 if total_spending > 0 else 0
                cumulative_percentage += vendor_percentage
                
                concentration_data.append({
                    "vendor_name": vendor.vendor_name,
                    "spending": float(vendor.vendor_spending),
                    "percentage": round(vendor_percentage, 2),
                    "cumulative_percentage": round(cumulative_percentage, 2)
                })
            
            top_10_percentage = sum(v["percentage"] for v in concentration_data[:10])
            
            return {
                "fiscal_year": fiscal_year,
                "total_spending": total_spending,
                "vendor_count": len(concentration_data),
                "top_10_concentration": round(top_10_percentage, 2),
                "concentration_data": concentration_data
            }
        except Exception as e:
            raise DataProcessingException("vendor concentration analysis", {"error": str(e)})
    
    async def get_spending_outliers(
        self,
        session: AsyncSession,
        fiscal_year: Optional[int] = None,
        threshold_multiplier: float = 3.0
    ) -> Dict[str, Any]:
        try:
            query = select(ExpenditureRecord.monetary_amount)
            if fiscal_year:
                query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
            
            result = await session.execute(query)
            amounts = [float(row[0]) for row in result.all()]
            
            if not amounts:
                return {"outliers": [], "statistics": {}}
            
            q1 = np.percentile(amounts, 25)
            q3 = np.percentile(amounts, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold_multiplier * iqr
            upper_bound = q3 + threshold_multiplier * iqr
            
            outlier_query = select(ExpenditureRecord).where(
                and_(
                    or_(
                        ExpenditureRecord.monetary_amount < lower_bound,
                        ExpenditureRecord.monetary_amount > upper_bound
                    ),
                    ExpenditureRecord.fiscal_year == fiscal_year if fiscal_year else True
                )
            ).order_by(desc(ExpenditureRecord.monetary_amount))
            
            outlier_result = await session.execute(outlier_query)
            outliers = outlier_result.scalars().all()
            
            return {
                "fiscal_year": fiscal_year,
                "statistics": {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "total_records": len(amounts),
                    "outlier_count": len(outliers)
                },
                "outliers": [
                    {
                        "id": str(outlier.id),
                        "vendor_name": outlier.vendor_name,
                        "dept_name": outlier.dept_name,
                        "monetary_amount": outlier.monetary_amount,
                        "entered": outlier.entered.isoformat(),
                        "voucher": outlier.voucher
                    }
                    for outlier in outliers[:100]
                ]
            }
        except Exception as e:
            raise DataProcessingException("outlier detection", {"error": str(e)})
    
    async def get_category_distribution(
        self,
        session: AsyncSession,
        fiscal_year: Optional[int] = None,
        groupby: str = "account_descr"
    ) -> Dict[str, Any]:
        try:
            groupby_column = getattr(ExpenditureRecord, groupby)
            
            query = select(
                groupby_column.label("category"),
                func.sum(ExpenditureRecord.monetary_amount).label("total_amount"),
                func.count(ExpenditureRecord.id).label("transaction_count")
            ).group_by(groupby_column)
            
            if fiscal_year:
                query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
            
            query = query.order_by(desc("total_amount"))
            
            result = await session.execute(query)
            categories = result.all()
            
            total_spending = sum(cat.total_amount for cat in categories)
            
            distribution_data = []
            other_threshold = 0.01 * total_spending
            other_amount = 0
            other_count = 0
            
            for cat in categories:
                if cat.total_amount >= other_threshold:
                    distribution_data.append({
                        "category": cat.category or "Unknown",
                        "amount": float(cat.total_amount),
                        "percentage": round((cat.total_amount / total_spending) * 100, 2) if total_spending > 0 else 0,
                        "transaction_count": cat.transaction_count
                    })
                else:
                    other_amount += cat.total_amount
                    other_count += cat.transaction_count
            
            if other_amount > 0:
                distribution_data.append({
                    "category": "Other",
                    "amount": float(other_amount),
                    "percentage": round((other_amount / total_spending) * 100, 2) if total_spending > 0 else 0,
                    "transaction_count": other_count
                })
            
            return {
                "fiscal_year": fiscal_year,
                "groupby": groupby,
                "total_spending": float(total_spending),
                "distribution": distribution_data
            }
        except Exception as e:
            raise DataProcessingException("category distribution calculation", {"error": str(e)})

analytics_service = AnalyticsService()