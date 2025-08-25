from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from functools import wraps
import time
from typing import Callable, Any
from fastapi import Request, Response
import asyncio

from loguru import logger

class MetricsCollector:
    def __init__(self):
        self.registry = CollectorRegistry()
        
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_database_connections',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total', 
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.ml_predictions = Counter(
            'ml_predictions_total',
            'Total ML predictions made',
            ['model_type', 'status'],
            registry=self.registry
        )
        
        self.data_ingestion_records = Counter(
            'data_ingestion_records_total',
            'Total records ingested',
            ['source_type'],
            registry=self.registry
        )
        
        self.background_tasks = Gauge(
            'background_tasks_active',
            'Number of active background tasks',
            ['task_type'],
            registry=self.registry
        )
    
    def track_request(self, method: str, endpoint: str, status_code: int, duration: float):
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def track_cache_hit(self, cache_type: str = "general"):
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def track_cache_miss(self, cache_type: str = "general"):
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def track_ml_prediction(self, model_type: str, status: str = "success"):
        self.ml_predictions.labels(model_type=model_type, status=status).inc()
    
    def track_data_ingestion(self, source_type: str, record_count: int = 1):
        self.data_ingestion_records.labels(source_type=source_type).inc(record_count)
    
    def set_active_connections(self, count: int):
        self.active_connections.set(count)
    
    def set_background_tasks(self, task_type: str, count: int):
        self.background_tasks.labels(task_type=task_type).set(count)
    
    def get_metrics(self) -> str:
        return generate_latest(self.registry)

metrics = MetricsCollector()

def track_endpoint_metrics():
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            request: Request = kwargs.get('request') or args[0] if args else None
            
            try:
                result = await func(*args, **kwargs)
                status_code = getattr(result, 'status_code', 200)
                
                if request:
                    duration = time.time() - start_time
                    metrics.track_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=status_code,
                        duration=duration
                    )
                
                return result
                
            except Exception as e:
                if request:
                    duration = time.time() - start_time
                    metrics.track_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=500,
                        duration=duration
                    )
                raise
        
        return wrapper
    return decorator

class PerformanceMonitor:
    def __init__(self):
        self.slow_query_threshold = 5.0
        self.memory_threshold = 1024 * 1024 * 1024
    
    async def check_database_performance(self, session):
        try:
            from sqlalchemy import text
            
            slow_queries_query = text("""
                SELECT query, mean_exec_time, calls, total_exec_time
                FROM pg_stat_statements 
                WHERE mean_exec_time > :threshold
                ORDER BY mean_exec_time DESC
                LIMIT 10
            """)
            
            result = await session.execute(slow_queries_query, {"threshold": self.slow_query_threshold * 1000})
            slow_queries = result.all()
            
            connection_query = text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            conn_result = await session.execute(connection_query)
            active_connections = conn_result.scalar()
            
            metrics.set_active_connections(active_connections)
            
            return {
                "slow_queries": len(slow_queries),
                "active_connections": active_connections,
                "performance_issues": [
                    {
                        "query": query.query[:100] + "..." if len(query.query) > 100 else query.query,
                        "mean_exec_time_ms": float(query.mean_exec_time),
                        "calls": query.calls,
                        "total_exec_time_ms": float(query.total_exec_time)
                    }
                    for query in slow_queries
                ]
            }
            
        except Exception as e:
            logger.error(f"Database performance check failed: {e}")
            return {"error": str(e)}
    
    def check_memory_usage(self) -> Dict[str, Any]:
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_bytes": memory_info.rss,
                "vms_bytes": memory_info.vms,
                "memory_percentage": process.memory_percent(),
                "cpu_percentage": process.cpu_percent(),
                "memory_threshold_exceeded": memory_info.rss > self.memory_threshold
            }
            
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    async def health_check_comprehensive(self, session) -> Dict[str, Any]:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy"
        }
        
        try:
            db_perf = await self.check_database_performance(session)
            health_status["database"] = db_perf
            
            memory_status = self.check_memory_usage()
            health_status["memory"] = memory_status
            
            from app.services.cache_service import cache_service
            cache_healthy = await cache_service.health_check()
            health_status["cache"] = {"status": "healthy" if cache_healthy else "unhealthy"}
            
            if (db_perf.get("slow_queries", 0) > 5 or 
                memory_status.get("memory_threshold_exceeded", False) or 
                not cache_healthy):
                health_status["overall_status"] = "degraded"
            
        except Exception as e:
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status

performance_monitor = PerformanceMonitor()

async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    metrics.track_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        duration=duration
    )
    
    return response