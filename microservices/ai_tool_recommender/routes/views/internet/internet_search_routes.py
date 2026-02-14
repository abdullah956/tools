"""Internet search recommendation endpoint."""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from microservices.ai_tool_recommender.ai_agents.tools import AIToolRecommender

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize service
ai_tool_recommender = AIToolRecommender()


class SearchRequest(BaseModel):
    """Request model for AI tool search."""

    query: str
    user_id: str
    max_results: int = 10
    include_internet_search: bool = True


class SearchResponse(BaseModel):
    """Response model for AI tool search."""

    status: str
    tools: List[Dict[str, Any]]
    query: str
    refined_query: str
    message: str = None
    error: str = None
    count: int = None


@router.post("/recommend", response_model=SearchResponse)
async def recommend_tools(request: SearchRequest) -> SearchResponse:
    """Get AI tool recommendations using internet search with link verification.

    Args:
        request: Search request with query and parameters

    Returns:
        Search response with recommended tools and workflow
    """
    try:
        logger.info(f"Received search request: {request.query}")

        # Search for AI tools using the new modular service
        result = await ai_tool_recommender.search_tools(
            query=request.query,
            max_results=request.max_results,
            include_pinecone=False,  # Only internet search for this route
            include_internet=True,
        )

        if result["status"] != "success":
            return SearchResponse(
                status="error",
                tools=[],
                query=request.query,
                refined_query=request.query,
                message=result.get("message", "No relevant AI tools found"),
                error=result.get("message", "Search failed"),
                count=0,
            )

        logger.info(f"Found tools with status: {result['status']}")

        return SearchResponse(
            status="success",
            tools=result["tools"],
            query=request.query,
            refined_query=request.query,
            message=result.get("message", "Found verified AI tools"),
            count=result.get("count", len(result["tools"])),
        )

    except Exception as e:
        logger.error(f"Error in recommend_tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))
