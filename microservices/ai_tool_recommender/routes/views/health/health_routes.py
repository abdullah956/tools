"""Health check and testing routes for Pinecone and Internet Search services."""

import logging
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from microservices.ai_tool_recommender.ai_agents.tools.internet_search import (
    InternetSearchService,
    PricingExtractor,
)
from microservices.ai_tool_recommender.ai_agents.tools.pinecone import PineconeService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
pinecone_service = PineconeService()
internet_service = InternetSearchService()
pricing_extractor = PricingExtractor()


class PricingTestRequest(BaseModel):
    """Request model for testing pricing extraction."""

    url: str
    tool_name: str = "Test Tool"


@router.post("/pricing/extract")
async def test_pricing_extraction(request: PricingTestRequest) -> Dict[str, Any]:
    """Test pricing extraction from a specific URL.

    Args:
        request: Pricing test request with URL and tool name

    Returns:
        Pricing extraction results
    """
    try:
        logger.info(f"Testing pricing extraction from: {request.url}")

        # Extract pricing from URL
        pricing_info = await pricing_extractor.extract_pricing_from_url(request.url)

        return {
            "status": (
                "success" if pricing_info.get("pricing_found", False) else "warning"
            ),
            "service": "pricing_extractor",
            "message": f"Pricing extraction {'successful' if pricing_info.get('pricing_found', False) else 'no pricing found'}",
            "test_url": request.url,
            "tool_name": request.tool_name,
            "pricing_found": pricing_info.get("pricing_found", False),
            "price_from": pricing_info.get("price_from", ""),
            "price_to": pricing_info.get("price_to", ""),
            "pricing_details": pricing_info.get("pricing_details", []),
            "pricing_buttons": pricing_info.get("pricing_buttons", []),
            "source_url": pricing_info.get("source_url", request.url),
            "error": pricing_info.get("error", ""),
            "extraction_details": {
                "url_accessible": pricing_info.get("pricing_found", False)
                or not pricing_info.get("error"),
                "pricing_sections_found": len(pricing_info.get("pricing_details", [])),
                "pricing_buttons_found": len(pricing_info.get("pricing_buttons", [])),
                "extraction_method": "website_scraping",
            },
        }

    except Exception as e:
        logger.error(f"Pricing extraction test error: {e}")
        return {
            "status": "error",
            "service": "pricing_extractor",
            "message": f"Pricing extraction test failed: {e}",
            "test_url": request.url,
            "tool_name": request.tool_name,
            "pricing_found": False,
            "price_from": "",
            "price_to": "",
            "pricing_details": [],
            "pricing_buttons": [],
            "source_url": request.url,
            "error": str(e),
            "extraction_details": {
                "url_accessible": False,
                "pricing_sections_found": 0,
                "pricing_buttons_found": 0,
                "extraction_method": "website_scraping",
            },
        }


class TestRequest(BaseModel):
    """Request model for testing services."""

    query: str = "AI video editing tools"
    max_results: int = 5


@router.get("/pinecone/health")
async def pinecone_health() -> Dict[str, Any]:
    """Health check for Pinecone service.

    Returns:
        Health status and basic info about Pinecone connection
    """
    try:
        # Check if Pinecone is initialized
        if not pinecone_service.index:
            return {
                "status": "unhealthy",
                "service": "pinecone",
                "message": "Pinecone index not initialized",
                "details": {
                    "client_initialized": pinecone_service.pinecone_client is not None,
                    "index_connected": False,
                    "embeddings_initialized": pinecone_service.embeddings is not None,
                },
            }

        # Get index stats
        try:
            stats = pinecone_service.index.describe_index_stats()
            return {
                "status": "healthy",
                "service": "pinecone",
                "message": "Pinecone service is operational",
                "details": {
                    "client_initialized": pinecone_service.pinecone_client is not None,
                    "index_connected": True,
                    "embeddings_initialized": pinecone_service.embeddings is not None,
                    "index_stats": {
                        "total_vector_count": stats.total_vector_count,
                        "dimension": stats.dimension,
                        "index_fullness": stats.index_fullness,
                    },
                },
            }
        except Exception as stats_error:
            return {
                "status": "degraded",
                "service": "pinecone",
                "message": f"Pinecone connected but stats unavailable: {stats_error}",
                "details": {
                    "client_initialized": pinecone_service.pinecone_client is not None,
                    "index_connected": True,
                    "embeddings_initialized": pinecone_service.embeddings is not None,
                    "stats_error": str(stats_error),
                },
            }

    except Exception as e:
        logger.error(f"Pinecone health check error: {e}")
        return {
            "status": "unhealthy",
            "service": "pinecone",
            "message": f"Pinecone service error: {e}",
            "details": {
                "client_initialized": pinecone_service.pinecone_client is not None,
                "index_connected": False,
                "embeddings_initialized": pinecone_service.embeddings is not None,
                "error": str(e),
            },
        }


@router.get("/tavily/health")
async def tavily_health() -> Dict[str, Any]:
    """Health check for Tavily Internet Search service.

    Returns:
        Health status and basic info about Tavily connection
    """
    try:
        # Check if Tavily is initialized
        if not internet_service.tavily_client:
            return {
                "status": "unhealthy",
                "service": "tavily",
                "message": "Tavily client not initialized",
                "details": {
                    "client_initialized": False,
                    "config_available": internet_service.config.is_configured(),
                },
            }

        return {
            "status": "healthy",
            "service": "tavily",
            "message": "Tavily service is operational",
            "details": {
                "client_initialized": True,
                "config_available": internet_service.config.is_configured(),
                "api_key_configured": bool(internet_service.config.tavily_api_key),
            },
        }

    except Exception as e:
        logger.error(f"Tavily health check error: {e}")
        return {
            "status": "unhealthy",
            "service": "tavily",
            "message": f"Tavily service error: {e}",
            "details": {
                "client_initialized": internet_service.tavily_client is not None,
                "config_available": internet_service.config.is_configured(),
                "error": str(e),
            },
        }


@router.post("/pinecone/test")
async def test_pinecone(request: TestRequest) -> Dict[str, Any]:
    """Test Pinecone service with a sample query.

    Args:
        request: Test request with query and max_results

    Returns:
        Test results from Pinecone search
    """
    try:
        logger.info(f"Testing Pinecone with query: {request.query}")

        # Test Pinecone search
        results = await pinecone_service.search_tools(
            query=request.query, max_results=request.max_results
        )
        logger.info(f"Pinecone search returned {len(results)} results")

        if not results:
            return {
                "status": "warning",
                "service": "pinecone",
                "message": "Pinecone search returned no results",
                "test_query": request.query,
                "results_count": 0,
                "results": [],
                "details": {
                    "search_executed": True,
                    "results_found": False,
                    "possible_causes": [
                        "Index might be empty",
                        "Query might not match any vectors",
                        "Namespace might be incorrect",
                    ],
                },
            }

        # Format results for response with complete data
        formatted_results = []
        for i, tool in enumerate(results):
            formatted_results.append(
                {
                    "rank": i + 1,
                    "title": tool.get("Title", "Unknown"),
                    "category": tool.get("Category", "Unknown"),
                    "description": tool.get("Description", ""),
                    "features": tool.get("Features", ""),
                    "tags": tool.get("Tags (Keywords)", ""),
                    "website": tool.get("Website", ""),
                    "twitter": tool.get("Twitter", ""),
                    "facebook": tool.get("Facebook", ""),
                    "linkedin": tool.get("Linkedin", ""),
                    "instagram": tool.get("Instagram", ""),
                    "price_from": tool.get("Price From", ""),
                    "price_to": tool.get("Price To", ""),
                    "source": tool.get("Source", "Unknown"),
                    "similarity_score": tool.get("Similarity Score", 0),
                    "relevance_score": tool.get("Relevance Score", 0),
                    "namespace": tool.get("Namespace", ""),
                    "raw_data": tool,  # Include complete raw data
                }
            )

        return {
            "status": "success",
            "service": "pinecone",
            "message": f"Pinecone search successful - found {len(results)} tools",
            "test_query": request.query,
            "results_count": len(results),
            "results": formatted_results,
            "details": {
                "search_executed": True,
                "results_found": True,
                "max_results_requested": request.max_results,
                "actual_results": len(results),
            },
        }

    except Exception as e:
        logger.error(f"Pinecone test error: {e}")
        return {
            "status": "error",
            "service": "pinecone",
            "message": f"Pinecone test failed: {e}",
            "test_query": request.query,
            "results_count": 0,
            "results": [],
            "details": {
                "search_executed": False,
                "results_found": False,
                "error": str(e),
            },
        }


@router.post("/tavily/test")
async def test_tavily(request: TestRequest) -> Dict[str, Any]:
    """Test Tavily Internet Search service with a sample query.

    Args:
        request: Test request with query and max_results

    Returns:
        Test results from Tavily search
    """
    try:
        logger.info(f"Testing Tavily with query: {request.query}")

        # Test Tavily search
        results = await internet_service.search_ai_tools(
            query=request.query, max_results=request.max_results
        )

        if not results:
            return {
                "status": "warning",
                "service": "tavily",
                "message": "Tavily search returned no results",
                "test_query": request.query,
                "results_count": 0,
                "results": [],
                "details": {
                    "search_executed": True,
                    "results_found": False,
                    "possible_causes": [
                        "No relevant tools found for query",
                        "All results filtered out as blogs/lists",
                        "Link verification failed for all results",
                    ],
                },
            }

        # Format results for response with complete data
        formatted_results = []
        for i, tool in enumerate(results):
            formatted_results.append(
                {
                    "rank": i + 1,
                    "title": tool.get("Title", "Unknown"),
                    "category": tool.get("Category", "Unknown"),
                    "description": tool.get("Description", ""),
                    "features": tool.get("Features", ""),
                    "tags": tool.get("Tags (Keywords)", ""),
                    "website": tool.get("Website", ""),
                    "twitter": tool.get("Twitter", ""),
                    "facebook": tool.get("Facebook", ""),
                    "linkedin": tool.get("Linkedin", ""),
                    "instagram": tool.get("Instagram", ""),
                    "price_from": tool.get("Price From", ""),
                    "price_to": tool.get("Price To", ""),
                    "source": tool.get("Source", "Unknown"),
                    "similarity_score": tool.get("Similarity Score", 0),
                    "relevance_score": tool.get("Relevance Score", 0),
                    "raw_data": tool,  # Include complete raw data
                }
            )

        return {
            "status": "success",
            "service": "tavily",
            "message": f"Tavily search successful - found {len(results)} verified tools",
            "test_query": request.query,
            "results_count": len(results),
            "results": formatted_results,
            "details": {
                "search_executed": True,
                "results_found": True,
                "max_results_requested": request.max_results,
                "actual_results": len(results),
                "link_verification": "enabled",
            },
        }

    except Exception as e:
        logger.error(f"Tavily test error: {e}")
        return {
            "status": "error",
            "service": "tavily",
            "message": f"Tavily test failed: {e}",
            "test_query": request.query,
            "results_count": 0,
            "results": [],
            "details": {
                "search_executed": False,
                "results_found": False,
                "error": str(e),
            },
        }
