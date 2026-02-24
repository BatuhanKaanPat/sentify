import os
import json
import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("sentify.api")

# Paths exempt from API key authentication
EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

API_KEY = os.environ.get("SENTIFY_API_KEY", "")


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Validates x-api-key header against SENTIFY_API_KEY env var.
    Skips health/docs endpoints. If SENTIFY_API_KEY is empty, auth is disabled.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        if not API_KEY:
            return await call_next(request)

        provided_key = request.headers.get("x-api-key", "")
        if provided_key != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Generates a request_id, measures total latency, and emits a structured
    JSON log line for every request including model_type and inference_time_ms.
    """

    def __init__(self, app, model_loader=None):
        super().__init__(app)
        self.model_loader = model_loader

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        # Initialize inference time in state (endpoints may set this)
        request.state.inference_time_ms = None

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "model_type": self.model_loader.model_type if self.model_loader else None,
            "inference_time_ms": request.state.inference_time_ms,
        }

        logger.info(json.dumps(log_data))

        # Attach request_id to response headers for traceability
        response.headers["X-Request-ID"] = request_id
        return response
