from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid

from app.models.database import (
    ExpenditureRecord,
    VendorPerformance, 
    DepartmentAnalytics,
    MLModelMetadata
)
from app.schemas.expenditure import (
    ExpenditureRecordCreate,
    ExpenditureRecordUpdate,
    ExpenditureFilter,
    PaginationParams
)
from app.core.exceptions import NotFoundException, DatabaseException

class ExpenditureService:
    
    async def create_expenditure(
        self, 
        session: AsyncSession, 
        expenditure: ExpenditureRecordCreate
    ) -> ExpenditureRecord:
        try:
            db_expenditure = ExpenditureRecord(**expenditure.model_dump())
            session.add(db_expenditure)
            await session.flush()
            await session.refresh(db_expenditure)
            return db_expenditure
        except Exception as e:
            raise DatabaseException("Failed to create expenditure record", {"error": str(e)})
    
    async def bulk_create_expenditures(
        self,
        session: AsyncSession,
        expenditures: List[ExpenditureRecordCreate]
    ) -> int:
        try:
            db_expenditures = [
                ExpenditureRecord(**exp.model_dump()) 
                for exp in expenditures
            ]
            session.add_all(db_expenditures)
            await session.flush()
            return len(db_expenditures)
        except Exception as e:
            raise DatabaseException("Failed to bulk create expenditure records", {"error": str(e)})
    
    async def get_expenditure(
        self, 
        session: AsyncSession, 
        expenditure_id: uuid.UUID
    ) -> ExpenditureRecord:
        result = await session.execute(
            select(ExpenditureRecord).where(ExpenditureRecord.id == expenditure_id)
        )
        expenditure = result.scalar_one_or_none()
        if not expenditure:
            raise NotFoundException("Expenditure record")
        return expenditure
    
    async def get_expenditures(
        self,
        session: AsyncSession,
        filters: Optional[ExpenditureFilter] = None,
        pagination: Optional[PaginationParams] = None
    ) -> Tuple[List[ExpenditureRecord], int]:
        
        query = select(ExpenditureRecord)
        count_query = select(func.count()).select_from(ExpenditureRecord)
        
        if filters:
            conditions = []
            
            if filters.vendor_name:
                conditions.append(
                    ExpenditureRecord.vendor_name.ilike(f"%{filters.vendor_name}%")
                )
            
            if filters.dept_name:
                conditions.append(
                    ExpenditureRecord.dept_name.ilike(f"%{filters.dept_name}%")
                )
            
            if filters.fiscal_year:
                conditions.append(ExpenditureRecord.fiscal_year == filters.fiscal_year)
            
            if filters.min_amount is not None:
                conditions.append(ExpenditureRecord.monetary_amount >= filters.min_amount)
            
            if filters.max_amount is not None:
                conditions.append(ExpenditureRecord.monetary_amount <= filters.max_amount)
            
            if filters.start_date:
                conditions.append(ExpenditureRecord.entered >= filters.start_date)
            
            if filters.end_date:
                conditions.append(ExpenditureRecord.entered <= filters.end_date)
            
            if conditions:
                where_clause = and_(*conditions)
                query = query.where(where_clause)
                count_query = count_query.where(where_clause)
        
        total_count = await session.execute(count_query)
        total = total_count.scalar()
        
        query = query.order_by(desc(ExpenditureRecord.entered))
        
        if pagination:
            offset = (pagination.page - 1) * pagination.size
            query = query.offset(offset).limit(pagination.size)
        
        result = await session.execute(query)
        expenditures = result.scalars().all()
        
        return expenditures, total
    
    async def update_expenditure(
        self,
        session: AsyncSession,
        expenditure_id: uuid.UUID,
        update_data: ExpenditureRecordUpdate
    ) -> ExpenditureRecord:
        expenditure = await self.get_expenditure(session, expenditure_id)
        
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(expenditure, field, value)
        
        expenditure.updated_at = datetime.utcnow()
        await session.flush()
        await session.refresh(expenditure)
        return expenditure
    
    async def delete_expenditure(
        self,
        session: AsyncSession,
        expenditure_id: uuid.UUID
    ) -> bool:
        expenditure = await self.get_expenditure(session, expenditure_id)
        await session.delete(expenditure)
        return True
    
    async def get_expenditure_summary(
        self,
        session: AsyncSession,
        fiscal_year: Optional[int] = None
    ) -> Dict[str, Any]:
        query = select(
            func.count(ExpenditureRecord.id).label("total_records"),
            func.sum(ExpenditureRecord.monetary_amount).label("total_amount"),
            func.avg(ExpenditureRecord.monetary_amount).label("avg_amount"),
            func.min(ExpenditureRecord.monetary_amount).label("min_amount"),
            func.max(ExpenditureRecord.monetary_amount).label("max_amount")
        )
        
        if fiscal_year:
            query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
        
        result = await session.execute(query)
        summary = result.first()
        
        return {
            "total_records": summary.total_records or 0,
            "total_amount": float(summary.total_amount or 0),
            "average_amount": float(summary.avg_amount or 0),
            "minimum_amount": float(summary.min_amount or 0),
            "maximum_amount": float(summary.max_amount or 0)
        }

class VendorService:
    
    async def get_or_create_vendor_performance(
        self,
        session: AsyncSession,
        vendor_name: str
    ) -> VendorPerformance:
        result = await session.execute(
            select(VendorPerformance).where(VendorPerformance.vendor_name == vendor_name)
        )
        vendor = result.scalar_one_or_none()
        
        if not vendor:
            vendor = VendorPerformance(vendor_name=vendor_name)
            session.add(vendor)
            await session.flush()
            await session.refresh(vendor)
        
        return vendor
    
    async def update_vendor_performance(
        self,
        session: AsyncSession,
        vendor_name: str,
        expenditure_data: Dict[str, Any]
    ) -> VendorPerformance:
        vendor = await self.get_or_create_vendor_performance(session, vendor_name)
        
        vendor.total_expenditure += expenditure_data.get("amount", 0)
        vendor.transaction_count += 1
        vendor.last_transaction = expenditure_data.get("date", datetime.utcnow())
        
        if not vendor.first_transaction:
            vendor.first_transaction = vendor.last_transaction
        
        vendor.updated_at = datetime.utcnow()
        
        return vendor

expenditure_service = ExpenditureService()
vendor_service = VendorService()