from fastapi import APIRouter, Depends, Query, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from app.core.database import get_db_session
from app.services.database_service import expenditure_service
from app.schemas.expenditure import (
    ExpenditureRecord,
    ExpenditureRecordCreate,
    ExpenditureRecordUpdate,
    ExpenditureFilter,
    PaginationParams,
    PaginatedResponse
)
from app.core.exceptions import ValidationException

router = APIRouter()

@router.post("/", response_model=ExpenditureRecord, status_code=201)
async def create_expenditure(
    expenditure: ExpenditureRecordCreate,
    session: AsyncSession = Depends(get_db_session)
):
    return await expenditure_service.create_expenditure(session, expenditure)

@router.post("/bulk", status_code=201)
async def bulk_create_expenditures(
    expenditures: List[ExpenditureRecordCreate] = Body(..., max_items=1000),
    session: AsyncSession = Depends(get_db_session)
):
    if not expenditures:
        raise ValidationException("Expenditure list cannot be empty")
    
    count = await expenditure_service.bulk_create_expenditures(session, expenditures)
    return {
        "message": f"Successfully created {count} expenditure records",
        "count": count
    }

@router.get("/{expenditure_id}", response_model=ExpenditureRecord)
async def get_expenditure(
    expenditure_id: uuid.UUID = Path(..., description="Expenditure record ID"),
    session: AsyncSession = Depends(get_db_session)
):
    return await expenditure_service.get_expenditure(session, expenditure_id)

@router.get("/", response_model=PaginatedResponse)
async def get_expenditures(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=1000, description="Page size"),
    vendor_name: Optional[str] = Query(None, description="Filter by vendor name"),
    dept_name: Optional[str] = Query(None, description="Filter by department name"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    min_amount: Optional[float] = Query(None, ge=0, description="Minimum amount filter"),
    max_amount: Optional[float] = Query(None, ge=0, description="Maximum amount filter"),
    session: AsyncSession = Depends(get_db_session)
):
    pagination = PaginationParams(page=page, size=size)
    filters = ExpenditureFilter(
        vendor_name=vendor_name,
        dept_name=dept_name,
        fiscal_year=fiscal_year,
        min_amount=min_amount,
        max_amount=max_amount
    )
    
    expenditures, total = await expenditure_service.get_expenditures(
        session, filters, pagination
    )
    
    return PaginatedResponse(
        items=[ExpenditureRecord.model_validate(exp) for exp in expenditures],
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size
    )

@router.put("/{expenditure_id}", response_model=ExpenditureRecord)
async def update_expenditure(
    expenditure_id: uuid.UUID = Path(..., description="Expenditure record ID"),
    update_data: ExpenditureRecordUpdate = Body(...),
    session: AsyncSession = Depends(get_db_session)
):
    return await expenditure_service.update_expenditure(
        session, expenditure_id, update_data
    )

@router.delete("/{expenditure_id}", status_code=204)
async def delete_expenditure(
    expenditure_id: uuid.UUID = Path(..., description="Expenditure record ID"),
    session: AsyncSession = Depends(get_db_session)
):
    await expenditure_service.delete_expenditure(session, expenditure_id)
    return {"message": "Expenditure record deleted successfully"}

@router.get("/summary/overview")
async def get_expenditure_summary(
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    session: AsyncSession = Depends(get_db_session)
):
    summary = await expenditure_service.get_expenditure_summary(session, fiscal_year)
    return {
        "fiscal_year": fiscal_year,
        "summary": summary
    }

@router.get("/vendors/top")
async def get_top_vendors(
    limit: int = Query(10, ge=1, le=50, description="Number of top vendors to return"),
    fiscal_year: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by fiscal year"),
    session: AsyncSession = Depends(get_db_session)
):
    from sqlalchemy import select, func, desc
    from app.models.database import ExpenditureRecord
    
    query = select(
        ExpenditureRecord.vendor_name,
        func.sum(ExpenditureRecord.monetary_amount).label("total_spending"),
        func.count(ExpenditureRecord.id).label("transaction_count")
    ).group_by(ExpenditureRecord.vendor_name)
    
    if fiscal_year:
        query = query.where(ExpenditureRecord.fiscal_year == fiscal_year)
    
    query = query.order_by(desc("total_spending")).limit(limit)
    
    result = await session.execute(query)
    vendors = result.all()
    
    return {
        "fiscal_year": fiscal_year,
        "top_vendors": [
            {
                "vendor_name": vendor.vendor_name,
                "total_spending": float(vendor.total_spending),
                "transaction_count": vendor.transaction_count
            }
            for vendor in vendors
        ]
    }