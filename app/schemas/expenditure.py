from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from enum import Enum

class ExpenditureTypeEnum(str, Enum):
    HIGH_VALUE = "High Value"
    LOW_VALUE = "Low Value"

class RiskCategoryEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ExpenditureRecordBase(BaseModel):
    voucher: str = Field(..., min_length=1, max_length=50)
    voucher_line: Optional[int] = None
    distribution_line: Optional[int] = None
    entered: datetime
    month: Optional[int] = Field(None, ge=1, le=12)
    fiscal_month: Optional[int] = Field(None, ge=1, le=12)
    fiscal_year: Optional[int] = Field(None, ge=2000, le=2100)
    year: Optional[int] = Field(None, ge=2000, le=2100)
    vendor_name: str = Field(..., min_length=1, max_length=255)
    account: Optional[str] = Field(None, max_length=50)
    account_descr: Optional[str] = Field(None, max_length=255)
    dept: Optional[str] = Field(None, max_length=50)
    dept_name: Optional[str] = Field(None, max_length=255)
    org_name_6_digit: Optional[str] = Field(None, max_length=255)
    monetary_amount: float = Field(..., ge=0)

class ExpenditureRecordCreate(ExpenditureRecordBase):
    pass

class ExpenditureRecordUpdate(BaseModel):
    vendor_name: Optional[str] = Field(None, min_length=1, max_length=255)
    account_descr: Optional[str] = Field(None, max_length=255)
    dept_name: Optional[str] = Field(None, max_length=255)
    monetary_amount: Optional[float] = Field(None, ge=0)

class ExpenditureRecord(ExpenditureRecordBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class VendorPerformanceBase(BaseModel):
    vendor_name: str = Field(..., min_length=1, max_length=255)
    total_expenditure: float = Field(default=0.0, ge=0)
    transaction_count: int = Field(default=0, ge=0)
    avg_monthly_spending: float = Field(default=0.0, ge=0)
    performance_score: Optional[float] = Field(None, ge=0, le=1)
    risk_category: Optional[RiskCategoryEnum] = None

class VendorPerformance(VendorPerformanceBase):
    id: UUID
    first_transaction: Optional[datetime] = None
    last_transaction: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class DepartmentAnalyticsBase(BaseModel):
    dept_name: str = Field(..., min_length=1, max_length=255)
    fiscal_year: int = Field(..., ge=2000, le=2100)
    total_expenditure: float = Field(default=0.0, ge=0)
    transaction_count: int = Field(default=0, ge=0)
    avg_transaction_amount: float = Field(default=0.0, ge=0)
    cluster_id: Optional[int] = None
    spending_pattern: Optional[str] = Field(None, max_length=100)

class DepartmentAnalytics(DepartmentAnalyticsBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    size: int = Field(default=50, ge=1, le=1000)

class PaginatedResponse(BaseModel):
    items: List[dict]
    total: int
    page: int
    size: int
    pages: int

class ExpenditureFilter(BaseModel):
    vendor_name: Optional[str] = None
    dept_name: Optional[str] = None
    fiscal_year: Optional[int] = None
    min_amount: Optional[float] = Field(None, ge=0)
    max_amount: Optional[float] = Field(None, ge=0)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @validator('max_amount')
    def validate_amount_range(cls, v, values):
        if v is not None and values.get('min_amount') is not None:
            if v < values.get('min_amount'):
                raise ValueError('max_amount must be greater than or equal to min_amount')
        return v