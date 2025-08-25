from fastapi import APIRouter, Depends, Query, Path, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from app.core.database import get_db_session
from app.services.cache_service import cache_service
from app.schemas.expenditure import DepartmentAnalytics, PaginationParams
from app.core.exceptions import NotFoundException, ValidationException
from loguru import logger

router = APIRouter()

@router.get("/", response_model=List[DepartmentAnalytics])
async def get_departments(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    search: Optional[str] = Query(None, description="Search department names"),
    min_expenditure: Optional[float] = Query(None, ge=0, description="Minimum total expenditure"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, desc, and_
        from app.models.database import DepartmentAnalytics
        
        query = select(DepartmentAnalytics)
        conditions = []
        
        if fiscal_year:
            conditions.append(DepartmentAnalytics.fiscal_year == fiscal_year)
        
        if search:
            conditions.append(DepartmentAnalytics.dept_name.ilike(f"%{search}%"))
        
        if min_expenditure is not None:
            conditions.append(DepartmentAnalytics.total_expenditure >= min_expenditure)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(DepartmentAnalytics.total_expenditure))
        
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        result = await session.execute(query)
        departments = result.scalars().all()
        
        return departments
        
    except Exception as e:
        logger.error(f"Error fetching departments: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch departments")

@router.get("/{dept_name}/spending-analysis")
async def get_department_spending_analysis(
    dept_name: str = Path(..., description="Department name"),
    fiscal_years: Optional[str] = Query(None, description="Comma-separated fiscal years"),
    comparison_mode: str = Query("yearly", regex="^(yearly|monthly)$", description="Comparison granularity"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        fiscal_years_list = None
        if fiscal_years:
            try:
                fiscal_years_list = [int(year.strip()) for year in fiscal_years.split(",")]
            except ValueError:
                raise ValidationException("Invalid fiscal years format")
        
        cache_key = f"dept_analysis_{dept_name}_{fiscal_years}_{comparison_mode}"
        
        if use_cache:
            cached_result = await cache_service.get(cache_key, namespace="department_analytics")
            if cached_result:
                logger.info(f"Returning cached department analysis for {dept_name}")
                return cached_result
        
        from sqlalchemy import select, func, desc
        from app.models.database import ExpenditureRecord
        
        base_conditions = [ExpenditureRecord.dept_name.ilike(f"%{dept_name}%")]
        if fiscal_years_list:
            base_conditions.append(ExpenditureRecord.fiscal_year.in_(fiscal_years_list))
        
        if comparison_mode == "yearly":
            trend_query = select(
                ExpenditureRecord.fiscal_year,
                func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
                func.count(ExpenditureRecord.id).label("transaction_count"),
                func.avg(ExpenditureRecord.monetary_amount).label("avg_transaction")
            ).where(and_(*base_conditions)).group_by(
                ExpenditureRecord.fiscal_year
            ).order_by(ExpenditureRecord.fiscal_year)
        else:
            trend_query = select(
                ExpenditureRecord.fiscal_year,
                ExpenditureRecord.fiscal_month,
                func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
                func.count(ExpenditureRecord.id).label("transaction_count")
            ).where(and_(*base_conditions)).group_by(
                ExpenditureRecord.fiscal_year, ExpenditureRecord.fiscal_month
            ).order_by(ExpenditureRecord.fiscal_year, ExpenditureRecord.fiscal_month)
        
        trend_result = await session.execute(trend_query)
        trends = trend_result.all()
        
        vendor_breakdown_query = select(
            ExpenditureRecord.vendor_name,
            func.sum(ExpenditureRecord.monetary_amount).label("vendor_spending"),
            func.count(ExpenditureRecord.id).label("vendor_transactions")
        ).where(and_(*base_conditions)).group_by(
            ExpenditureRecord.vendor_name
        ).order_by(desc("vendor_spending")).limit(20)
        
        vendor_result = await session.execute(vendor_breakdown_query)
        vendor_breakdown = vendor_result.all()
        
        category_query = select(
            ExpenditureRecord.account_descr,
            func.sum(ExpenditureRecord.monetary_amount).label("category_spending")
        ).where(and_(*base_conditions)).group_by(
            ExpenditureRecord.account_descr
        ).order_by(desc("category_spending"))
        
        category_result = await session.execute(category_query)
        categories = category_result.all()
        
        total_spending = sum(trend.total_spending for trend in trends)
        total_transactions = sum(trend.transaction_count for trend in trends)
        
        result = {
            "department_name": dept_name,
            "fiscal_years": fiscal_years_list,
            "comparison_mode": comparison_mode,
            "summary": {
                "total_spending": float(total_spending),
                "total_transactions": total_transactions,
                "average_transaction": float(total_spending / total_transactions) if total_transactions > 0 else 0
            },
            "spending_trends": [
                {
                    "fiscal_year": trend.fiscal_year,
                    "fiscal_month": getattr(trend, 'fiscal_month', None),
                    "total_spending": float(trend.total_spending),
                    "transaction_count": trend.transaction_count,
                    "avg_transaction": float(getattr(trend, 'avg_transaction', 0))
                }
                for trend in trends
            ],
            "top_vendors": [
                {
                    "vendor_name": vendor.vendor_name,
                    "spending": float(vendor.vendor_spending),
                    "transactions": vendor.vendor_transactions,
                    "percentage_of_dept_spending": round(
                        (vendor.vendor_spending / total_spending) * 100, 2
                    ) if total_spending > 0 else 0
                }
                for vendor in vendor_breakdown
            ],
            "spending_categories": [
                {
                    "category": cat.account_descr or "Unknown",
                    "spending": float(cat.category_spending),
                    "percentage": round(
                        (cat.category_spending / total_spending) * 100, 2
                    ) if total_spending > 0 else 0
                }
                for cat in categories
            ]
        }
        
        if use_cache:
            await cache_service.set(cache_key, result, ttl=1800, namespace="department_analytics")
        
        return result
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing department {dept_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze department spending")

@router.get("/comparison/cross-department")
async def compare_departments(
    departments: str = Query(..., description="Comma-separated department names"),
    fiscal_year: int = Query(..., ge=2000, le=2100, description="Fiscal year for comparison"),
    metric: str = Query("total_spending", regex="^(total_spending|avg_transaction|transaction_count)$"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        dept_list = [dept.strip() for dept in departments.split(",")]
        if len(dept_list) < 2:
            raise ValidationException("At least 2 departments required for comparison")
        
        cache_key = f"dept_comparison_{hash(departments)}_{fiscal_year}_{metric}"
        
        if use_cache:
            cached_result = await cache_service.get(cache_key, namespace="department_analytics")
            if cached_result:
                return cached_result
        
        from sqlalchemy import select, func, case
        from app.models.database import ExpenditureRecord
        
        comparison_data = []
        
        for dept in dept_list:
            dept_query = select(
                func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
                func.count(ExpenditureRecord.id).label("transaction_count"),
                func.avg(ExpenditureRecord.monetary_amount).label("avg_transaction")
            ).where(
                ExpenditureRecord.dept_name.ilike(f"%{dept}%"),
                ExpenditureRecord.fiscal_year == fiscal_year
            )
            
            result = await session.execute(dept_query)
            stats = result.first()
            
            comparison_data.append({
                "department": dept,
                "total_spending": float(stats.total_spending or 0),
                "transaction_count": stats.transaction_count or 0,
                "avg_transaction": float(stats.avg_transaction or 0)
            })
        
        comparison_data.sort(key=lambda x: x[metric], reverse=True)
        
        total_across_depts = sum(dept[metric] for dept in comparison_data)
        
        for dept in comparison_data:
            dept["percentage_of_total"] = round(
                (dept[metric] / total_across_depts) * 100, 2
            ) if total_across_depts > 0 else 0
            dept["rank"] = comparison_data.index(dept) + 1
        
        result = {
            "comparison_metric": metric,
            "fiscal_year": fiscal_year,
            "departments_compared": len(dept_list),
            "total_across_departments": total_across_depts,
            "department_comparison": comparison_data
        }
        
        if use_cache:
            await cache_service.set(cache_key, result, ttl=3600, namespace="department_analytics")
        
        return result
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error comparing departments: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare departments")

@router.post("/{dept_name}/cluster-analysis")
async def analyze_department_clusters(
    dept_name: str = Path(..., description="Department name"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    n_clusters: int = Query(3, ge=2, le=10, description="Number of clusters"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select
        from app.models.database import ExpenditureRecord
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        query = select(
            ExpenditureRecord.vendor_name,
            ExpenditureRecord.account_descr,
            ExpenditureRecord.monetary_amount,
            ExpenditureRecord.fiscal_month
        ).where(ExpenditureRecord.dept_name.ilike(f"%{dept_name}%"))
        
        if fiscal_year:
            query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        result = await session.execute(query)
        data = result.all()
        
        if len(data) < n_clusters * 2:
            raise ValidationException(f"Insufficient data for clustering. Need at least {n_clusters * 2} records.")
        
        df = pd.DataFrame([
            {
                "vendor_name": row.vendor_name,
                "account_descr": row.account_descr or "Unknown",
                "monetary_amount": row.monetary_amount,
                "fiscal_month": row.fiscal_month or 1
            }
            for row in data
        ])
        
        from sklearn.preprocessing import LabelEncoder
        
        le_vendor = LabelEncoder()
        le_account = LabelEncoder()
        
        df["vendor_encoded"] = le_vendor.fit_transform(df["vendor_name"])
        df["account_encoded"] = le_account.fit_transform(df["account_descr"])
        
        features = ["vendor_encoded", "account_encoded", "monetary_amount", "fiscal_month"]
        X = df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        df["cluster"] = cluster_labels
        
        cluster_analysis = []
        for cluster_id in range(n_clusters):
            cluster_data = df[df["cluster"] == cluster_id]
            
            cluster_summary = {
                "cluster_id": cluster_id,
                "size": len(cluster_data),
                "total_spending": float(cluster_data["monetary_amount"].sum()),
                "avg_spending": float(cluster_data["monetary_amount"].mean()),
                "dominant_vendors": cluster_data["vendor_name"].value_counts().head(5).to_dict(),
                "dominant_accounts": cluster_data["account_descr"].value_counts().head(5).to_dict(),
                "spending_range": {
                    "min": float(cluster_data["monetary_amount"].min()),
                    "max": float(cluster_data["monetary_amount"].max())
                }
            }
            cluster_analysis.append(cluster_summary)
        
        return {
            "department_name": dept_name,
            "fiscal_year": fiscal_year,
            "clustering_results": {
                "n_clusters": n_clusters,
                "silhouette_score": silhouette_avg,
                "total_records": len(df),
                "clusters": cluster_analysis
            }
        }
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing department clusters for {dept_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze department clusters")

@router.get("/efficiency/metrics")
async def get_department_efficiency_metrics(
    fiscal_year: int = Query(..., ge=2000, le=2100, description="Fiscal year for analysis"),
    metric_type: str = Query("cost_per_transaction", regex="^(cost_per_transaction|spending_variance|budget_utilization)$"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, desc
        from app.models.database import ExpenditureRecord
        import numpy as np
        
        if metric_type == "cost_per_transaction":
            query = select(
                ExpenditureRecord.dept_name,
                func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
                func.count(ExpenditureRecord.id).label("transaction_count"),
                func.avg(ExpenditureRecord.monetary_amount).label("cost_per_transaction")
            ).where(
                ExpenditureRecord.fiscal_year == fiscal_year
            ).group_by(ExpenditureRecord.dept_name).order_by(desc("cost_per_transaction"))
            
        elif metric_type == "spending_variance":
            query = select(
                ExpenditureRecord.dept_name,
                func.stddev(ExpenditureRecord.monetary_amount).label("spending_variance"),
                func.avg(ExpenditureRecord.monetary_amount).label("avg_spending"),
                func.count(ExpenditureRecord.id).label("transaction_count")
            ).where(
                ExpenditureRecord.fiscal_year == fiscal_year
            ).group_by(ExpenditureRecord.dept_name).order_by(desc("spending_variance"))
        
        result = await session.execute(query)
        departments = result.all()
        
        efficiency_data = []
        for dept in departments:
            if metric_type == "cost_per_transaction":
                efficiency_score = 1 / (1 + dept.cost_per_transaction / 1000) if dept.cost_per_transaction > 0 else 0
                efficiency_data.append({
                    "dept_name": dept.dept_name,
                    "total_spending": float(dept.total_spending),
                    "transaction_count": dept.transaction_count,
                    "cost_per_transaction": float(dept.cost_per_transaction),
                    "efficiency_score": round(efficiency_score, 3)
                })
            elif metric_type == "spending_variance":
                cv = (dept.spending_variance / dept.avg_spending) if dept.avg_spending > 0 else 0
                consistency_score = 1 / (1 + cv) if cv > 0 else 1
                efficiency_data.append({
                    "dept_name": dept.dept_name,
                    "spending_variance": float(dept.spending_variance or 0),
                    "avg_spending": float(dept.avg_spending),
                    "coefficient_of_variation": round(cv, 3),
                    "consistency_score": round(consistency_score, 3),
                    "transaction_count": dept.transaction_count
                })
        
        return {
            "fiscal_year": fiscal_year,
            "metric_type": metric_type,
            "departments": efficiency_data
        }
        
    except Exception as e:
        logger.error(f"Error calculating department efficiency metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate efficiency metrics")

@router.get("/{dept_name}/forecasting")
async def forecast_department_spending(
    dept_name: str = Path(..., description="Department name"),
    forecast_periods: int = Query(12, ge=1, le=24, description="Number of months to forecast"),
    confidence_level: float = Query(0.95, ge=0.8, le=0.99, description="Confidence level for prediction intervals"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, extract
        from app.models.database import ExpenditureRecord
        import pandas as pd
        from datetime import datetime, timedelta
        
        historical_query = select(
            extract('year', ExpenditureRecord.entered).label('year'),
            extract('month', ExpenditureRecord.entered).label('month'),
            func.sum(ExpenditureRecord.monetary_amount).label('monthly_spending')
        ).where(
            ExpenditureRecord.dept_name.ilike(f"%{dept_name}%")
        ).group_by(
            extract('year', ExpenditureRecord.entered),
            extract('month', ExpenditureRecord.entered)
        ).order_by('year', 'month')
        
        result = await session.execute(historical_query)
        historical_data = result.all()
        
        if len(historical_data) < 6:
            raise ValidationException("Insufficient historical data for forecasting (minimum 6 months required)")
        
        df = pd.DataFrame([
            {
                "year": int(row.year),
                "month": int(row.month),
                "spending": float(row.monthly_spending)
            }
            for row in historical_data
        ])
        
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values('date').reset_index(drop=True)
        
        spending_values = df['spending'].values
        
        trend = np.polyfit(range(len(spending_values)), spending_values, 1)
        seasonal_component = np.mean([
            spending_values[i::12] for i in range(min(12, len(spending_values)))
            if len(spending_values[i::12]) > 0
        ], axis=1)
        
        if len(seasonal_component) < 12:
            seasonal_component = np.tile(seasonal_component, (12 // len(seasonal_component)) + 1)[:12]
        
        forecasts = []
        last_date = df['date'].max()
        
        for i in range(1, forecast_periods + 1):
            future_date = last_date + pd.DateOffset(months=i)
            
            trend_value = trend[0] * (len(spending_values) + i) + trend[1]
            seasonal_value = seasonal_component[(future_date.month - 1) % 12]
            
            base_forecast = trend_value + (seasonal_value - np.mean(seasonal_component))
            base_forecast = max(0, base_forecast)
            
            std_dev = np.std(spending_values[-12:]) if len(spending_values) >= 12 else np.std(spending_values)
            margin_of_error = 1.96 * std_dev if confidence_level == 0.95 else 2.58 * std_dev
            
            forecasts.append({
                "period": i,
                "date": future_date.strftime("%Y-%m"),
                "forecast": round(base_forecast, 2),
                "lower_bound": round(max(0, base_forecast - margin_of_error), 2),
                "upper_bound": round(base_forecast + margin_of_error, 2)
            })
        
        historical_summary = {
            "avg_monthly_spending": float(np.mean(spending_values)),
            "trend_slope": float(trend[0]),
            "seasonal_variation": float(np.std(seasonal_component)),
            "data_points": len(spending_values)
        }
        
        return {
            "department_name": dept_name,
            "forecast_periods": forecast_periods,
            "confidence_level": confidence_level,
            "historical_summary": historical_summary,
            "forecasts": forecasts
        }
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error forecasting for department {dept_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to forecast department spending")

@router.post("/analytics/recalculate")
async def recalculate_department_analytics(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Recalculate for specific fiscal year"),
    session: AsyncSession = Depends(get_db_session)
):
    try:
        from sqlalchemy import select, func, delete
        from app.models.database import DepartmentAnalytics, ExpenditureRecord
        
        if fiscal_year:
            delete_query = delete(DepartmentAnalytics).where(
                DepartmentAnalytics.fiscal_year == fiscal_year
            )
        else:
            delete_query = delete(DepartmentAnalytics)
        
        await session.execute(delete_query)
        
        dept_stats_query = select(
            ExpenditureRecord.dept_name,
            ExpenditureRecord.fiscal_year,
            func.sum(ExpenditureRecord.monetary_amount).label("total_expenditure"),
            func.count(ExpenditureRecord.id).label("transaction_count"),
            func.avg(ExpenditureRecord.monetary_amount).label("avg_transaction_amount")
        ).group_by(
            ExpenditureRecord.dept_name, ExpenditureRecord.fiscal_year
        )
        
        if fiscal_year:
            dept_stats_query = dept_stats_query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        dept_result = await session.execute(dept_stats_query)
        dept_stats = dept_result.all()
        
        new_analytics = []
        for stats in dept_stats:
            if stats.total_expenditure > 100000:
                spending_pattern = "high_volume"
            elif stats.total_expenditure > 10000:
                spending_pattern = "medium_volume"
            else:
                spending_pattern = "low_volume"
            
            analytics = DepartmentAnalytics(
                dept_name=stats.dept_name,
                fiscal_year=stats.fiscal_year,
                total_expenditure=float(stats.total_expenditure),
                transaction_count=stats.transaction_count,
                avg_transaction_amount=float(stats.avg_transaction_amount),
                spending_pattern=spending_pattern
            )
            new_analytics.append(analytics)
        
        session.add_all(new_analytics)
        
        await cache_service.invalidate_pattern("*", "department_analytics")
        
        return {
            "message": f"Recalculated analytics for {len(new_analytics)} department-year combinations",
            "fiscal_year": fiscal_year,
            "departments_updated": len(set(analytics.dept_name for analytics in new_analytics))
        }
        
    except Exception as e:
        logger.error(f"Error recalculating department analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to recalculate department analytics")