from typing import Optional, Any, Dict

class AppException(Exception):
    def __init__(
        self, 
        message: str, 
        status_code: int = 500, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationException(AppException):
    def __init__(self, message: str = "Validation error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, details)

class NotFoundException(AppException):
    def __init__(self, resource: str = "Resource", details: Optional[Dict[str, Any]] = None):
        message = f"{resource} not found"
        super().__init__(message, 404, details)

class ConflictException(AppException):
    def __init__(self, message: str = "Resource already exists", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 409, details)

class DatabaseException(AppException):
    def __init__(self, message: str = "Database operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 500, details)

class ExternalServiceException(AppException):
    def __init__(self, service: str, message: str = "External service error", details: Optional[Dict[str, Any]] = None):
        full_message = f"{service}: {message}"
        super().__init__(full_message, 503, details)

class AuthenticationException(AppException):
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 401, details)

class AuthorizationException(AppException):
    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 403, details)

class RateLimitException(AppException):
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 429, details)

class MLModelException(AppException):
    def __init__(self, model_name: str, operation: str, details: Optional[Dict[str, Any]] = None):
        message = f"ML Model {model_name} failed during {operation}"
        super().__init__(message, 500, details)

class DataProcessingException(AppException):
    def __init__(self, operation: str, details: Optional[Dict[str, Any]] = None):
        message = f"Data processing failed during {operation}"
        super().__init__(message, 500, details)