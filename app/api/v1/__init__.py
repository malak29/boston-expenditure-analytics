from fastapi import APIRouter
from app.api.v1.expenditure import router as expenditure_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.ml_models import router as ml_router
from app.api.v1.vendors import router as vendors_router
from app.api.v1.departments import router as departments_router
from app.api.v1.data_ingestion import router as ingestion_router

api_router = APIRouter()

api_router.include_router(
    expenditure_router,
    prefix="/expenditures",
    tags=["expenditures"]
)

api_router.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["analytics"]
)

api_router.include_router(
    ml_router,
    prefix="/ml",
    tags=["machine-learning"]
)

api_router.include_router(
    vendors_router,
    prefix="/vendors",
    tags=["vendors"]
)

api_router.include_router(
    departments_router,
    prefix="/departments",
    tags=["departments"]
)

api_router.include_router(
    ingestion_router,
    prefix="/data",
    tags=["data-ingestion"]
)

@api_router.get("/", summary="API Information")
async def api_info():
    return {
        "message": "Boston Expenditure Analytics API v1",
        "endpoints": {
            "expenditures": "/api/v1/expenditures",
            "analytics": "/api/v1/analytics", 
            "machine_learning": "/api/v1/ml",
            "vendors": "/api/v1/vendors",
            "departments": "/api/v1/departments",
            "data_ingestion": "/api/v1/data"
        }
    }