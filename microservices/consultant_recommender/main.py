"""Main module for the consultant recommender."""

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import add_counsaltant, search

from microservices.shared.authentication.main import verify_token

app = FastAPI(
    title="Consultant Recommender API",
    docs_url="/consultant_recommender/docs",  # Set docs path
    redoc_url="/consultant_recommender/redoc",  # Optional: ReDoc documentation
    openapi_url="/consultant_recommender/openapi.json",
)
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routes

app.include_router(
    add_counsaltant.router,
    prefix="/consultant_recommender",
    tags=["Counsaltant"],
    dependencies=[Depends(verify_token)],
)
app.include_router(
    search.router,
    prefix="/consultant_recommender",
    tags=["Consultant Recommender"],
    dependencies=[Depends(verify_token)],
)

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")  # Default to localhost
    uvicorn.run(app, host=host, port=13000, reload=True)
