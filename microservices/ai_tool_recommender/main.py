"""Main module for the AI tool recommender."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from routes import (
    add_tools,
    discovery,
    excel_handler,
    explain,
    health_routes,
    internet_search_routes,
    search_tool,
)

app = FastAPI(
    title="AI Tools API with LanceDB",
    docs_url="/ai_tool_recommender/docs",  # Set docs path
    redoc_url="/ai_tool_recommender/redoc",  # Optional: ReDoc documentation
    openapi_url="/ai_tool_recommender/openapi.json",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Secret key to encode and decode JWT tokens
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Include routes with token verification
app.include_router(add_tools, prefix="/ai_tool_recommender", tags=["Tools"])
app.include_router(search_tool, prefix="/ai_tool_recommender", tags=["Tools"])
app.include_router(explain, prefix="/ai_tool_recommender", tags=["Tools"])
app.include_router(
    internet_search_routes,
    prefix="/ai_tool_recommender/internet",
    tags=["Internet Search"],
)
app.include_router(
    health_routes,
    prefix="/ai_tool_recommender/health",
    tags=["Health & Testing"],
)
app.include_router(
    discovery,
    prefix="/ai_tool_recommender",
    tags=["Tool Discovery"],
)
app.include_router(
    excel_handler,
    prefix="/ai_tool_recommender",
    tags=["Excel Handler Style"],
)

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")  # Default to localhost
    # Configure for better concurrency handling
    uvicorn.run(
        app,
        host=host,
        port=12000,
        reload=True,
        workers=1,  # Single worker for dev, increase for production
        limit_concurrency=50,  # Limit total concurrent connections
        limit_max_requests=1000,  # Restart worker after 1000 requests
        timeout_keep_alive=30,  # Keep-alive timeout
    )
