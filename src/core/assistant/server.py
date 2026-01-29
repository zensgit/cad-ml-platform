"""
FastAPI Server for CAD Assistant API.

Production-ready server with health checks, metrics, and middleware.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .api_service import CADAssistantAPI
from .caching import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
api_instance: Optional[CADAssistantAPI] = None
cache_manager: Optional[CacheManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global api_instance, cache_manager

    # Startup
    logger.info("Starting CAD Assistant API...")
    api_instance = CADAssistantAPI(
        rate_limit_enabled=True,
        requests_per_minute=60,
    )
    cache_manager = CacheManager()
    logger.info("CAD Assistant API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down CAD Assistant API...")
    if cache_manager:
        cache_manager.clear_all()
    logger.info("CAD Assistant API shut down complete")


# Create FastAPI application
app = FastAPI(
    title="CAD Assistant API",
    description="REST API for CAD/manufacturing knowledge assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AskRequestModel(BaseModel):
    """Ask request model."""

    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    options: dict = Field(default_factory=dict)


class ConversationRequestModel(BaseModel):
    """Conversation request model."""

    action: str = Field(..., pattern="^(create|get|delete|list|clear)$")
    conversation_id: Optional[str] = None


class EvaluateRequestModel(BaseModel):
    """Evaluation request model."""

    query: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Health and info endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.get("/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "CAD Assistant API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/ask", "method": "POST", "description": "Query the assistant"},
            {"path": "/conversation", "method": "POST", "description": "Manage conversations"},
            {"path": "/evaluate", "method": "POST", "description": "Evaluate response quality"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/info", "method": "GET", "description": "API information"},
            {"path": "/metrics", "method": "GET", "description": "Cache metrics"},
        ],
    }


@app.get("/metrics")
async def get_metrics():
    """Get cache and performance metrics."""
    if cache_manager:
        return {
            "cache": cache_manager.get_all_stats(),
            "timestamp": time.time(),
        }
    return {"error": "Metrics not available"}


# API endpoints
@app.post("/ask")
async def ask(
    request: AskRequestModel,
    x_client_id: str = Header(default="default"),
):
    """Query the CAD assistant."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    response = api_instance.ask(
        {
            "query": request.query,
            "conversation_id": request.conversation_id,
            "options": request.options,
        },
        client_id=x_client_id,
    )

    if not response.success:
        status_code = 400
        if response.error:
            if response.error.get("code") == "RATE_LIMIT_EXCEEDED":
                status_code = 429
            elif response.error.get("code") == "VALIDATION_ERROR":
                status_code = 400
        raise HTTPException(status_code=status_code, detail=response.error)

    return response.to_dict()


@app.post("/conversation")
async def conversation(
    request: ConversationRequestModel,
    x_client_id: str = Header(default="default"),
):
    """Manage conversations."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    response = api_instance.conversation(
        {
            "action": request.action,
            "conversation_id": request.conversation_id,
        },
        client_id=x_client_id,
    )

    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)

    return response.to_dict()


@app.post("/evaluate")
async def evaluate(
    request: EvaluateRequestModel,
    x_client_id: str = Header(default="default"),
):
    """Evaluate response quality."""
    if not api_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    response = api_instance.evaluate(
        {
            "query": request.query,
            "response": request.response,
        },
        client_id=x_client_id,
    )

    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)

    return response.to_dict()


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
            },
        },
    )


# Entry point for running directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.core.assistant.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
