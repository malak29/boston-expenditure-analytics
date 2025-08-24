from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class ExpenditureRecord(Base):
    __tablename__ = "expenditure_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    voucher = Column(String(50), nullable=False)
    voucher_line = Column(Integer)
    distribution_line = Column(Integer)
    entered = Column(DateTime, nullable=False)
    month = Column(Integer)
    fiscal_month = Column(Integer)
    fiscal_year = Column(Integer)
    year = Column(Integer)
    vendor_name = Column(String(255), nullable=False)
    account = Column(String(50))
    account_descr = Column(String(255))
    dept = Column(String(50))
    dept_name = Column(String(255))
    org_name_6_digit = Column(String(255))
    monetary_amount = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_vendor_name', 'vendor_name'),
        Index('idx_dept_name', 'dept_name'),
        Index('idx_fiscal_year', 'fiscal_year'),
        Index('idx_entered_date', 'entered'),
        Index('idx_monetary_amount', 'monetary_amount'),
    )

class VendorPerformance(Base):
    __tablename__ = "vendor_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vendor_name = Column(String(255), nullable=False, unique=True)
    total_expenditure = Column(Float, default=0.0)
    transaction_count = Column(Integer, default=0)
    avg_monthly_spending = Column(Float, default=0.0)
    performance_score = Column(Float)
    risk_category = Column(String(50))
    
    first_transaction = Column(DateTime)
    last_transaction = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DepartmentAnalytics(Base):
    __tablename__ = "department_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dept_name = Column(String(255), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    total_expenditure = Column(Float, default=0.0)
    transaction_count = Column(Integer, default=0)
    avg_transaction_amount = Column(Float, default=0.0)
    cluster_id = Column(Integer)
    spending_pattern = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_dept_fiscal_year', 'dept_name', 'fiscal_year'),
    )

class MLModelMetadata(Base):
    __tablename__ = "ml_model_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    model_version = Column(String(20), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    accuracy_score = Column(Float)
    silhouette_score = Column(Float)
    training_data_size = Column(Integer)
    features_used = Column(Text)
    
    is_active = Column(Boolean, default=False)
    trained_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_name_version', 'model_name', 'model_version'),
    )