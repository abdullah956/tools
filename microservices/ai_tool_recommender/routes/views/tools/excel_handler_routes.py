"""Excel-style tool addition API endpoints."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from microservices.ai_tool_recommender.ai_agents.tools.pinecone.service import (
    PineconeService,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ToolData(BaseModel):
    """Model for tool data."""

    Title: str
    Description: str
    Category: str = ""
    Features: str = ""
    Tags: str = ""
    Website: str = ""
    Price_From: str = ""
    Price_To: str = ""
    Twitter: str = ""
    Facebook: str = ""
    LinkedIn: str = ""
    Instagram: str = ""
    Source: str = "Manual Addition"


class BatchToolRequest(BaseModel):
    """Request model for batch tool addition."""

    tools: List[ToolData]


class BatchToolResponse(BaseModel):
    """Response model for batch tool addition."""

    status: str
    message: str
    added_count: int
    failed_count: int
    duplicate_count: int = 0
    total_count: int


class ToolAdditionDetail(BaseModel):
    """Detailed information about tool addition."""

    title: str
    website: str
    status: str  # "added", "duplicate", "failed", "skipped"
    reason: str
    similarity_score: float = 0.0


class DetailedBatchToolResponse(BaseModel):
    """Detailed response model for batch tool addition."""

    status: str
    message: str
    added_count: int
    failed_count: int
    duplicate_count: int
    skipped_count: int = 0
    total_count: int
    details: List[ToolAdditionDetail] = []


@router.post(
    "/tools/batch-add", response_model=BatchToolResponse, tags=["Excel Handler Style"]
)
async def add_tools_batch(request: BatchToolRequest):
    """Add multiple tools to Pinecone using Excel handler approach."""
    try:
        logger.info(
            f"Batch adding {len(request.tools)} tools using Excel handler approach"
        )

        # Convert Pydantic models to dictionaries
        tools_data = [tool.dict() for tool in request.tools]

        # Use Pinecone service with Excel handler approach
        pinecone_service = PineconeService()
        result = await pinecone_service.add_tools_batch(tools_data)

        return BatchToolResponse(
            status="success",
            message=f"Successfully added {result['success']} tools using Excel handler approach (skipped {result.get('duplicates', 0)} duplicates)",
            added_count=result["success"],
            failed_count=result["failed"],
            duplicate_count=result.get("duplicates", 0),
            total_count=result["total"],
        )

    except Exception as e:
        logger.error(f"Error in batch tool addition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tools/batch-add-detailed",
    response_model=DetailedBatchToolResponse,
    tags=["Excel Handler Style"],
)
async def add_tools_batch_detailed(request: BatchToolRequest):
    """Add multiple tools to Pinecone with detailed status for each tool."""
    try:
        logger.info(
            f"Detailed batch adding {len(request.tools)} tools using Excel handler approach"
        )

        # Convert Pydantic models to dictionaries
        tools_data = [tool.dict() for tool in request.tools]

        # Use Pinecone service with detailed tracking
        pinecone_service = PineconeService()
        result = await pinecone_service.add_tools_batch_detailed(tools_data)

        return DetailedBatchToolResponse(
            status="success",
            message=f"Processed {result['total']} tools: {result['success']} added, {result.get('duplicates', 0)} duplicates, {result['failed']} failed",
            added_count=result["success"],
            failed_count=result["failed"],
            duplicate_count=result.get("duplicates", 0),
            skipped_count=result.get("skipped", 0),
            total_count=result["total"],
            details=result.get("details", []),
        )

    except Exception as e:
        logger.error(f"Error in detailed batch tool addition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tools/single-add", response_model=BatchToolResponse, tags=["Excel Handler Style"]
)
async def add_single_tool(tool: ToolData):
    """Add a single tool to Pinecone using Excel handler approach."""
    try:
        logger.info(f"Adding single tool: {tool.Title}")

        # Convert to dictionary
        tool_data = tool.dict()

        # Use Pinecone service with Excel handler approach
        pinecone_service = PineconeService()
        result = await pinecone_service.add_tool(tool_data)

        if result:
            return BatchToolResponse(
                status="success",
                message=f"Successfully added tool '{tool.Title}' using Excel handler approach",
                added_count=1,
                failed_count=0,
                duplicate_count=0,
                total_count=1,
            )
        else:
            return BatchToolResponse(
                status="error",
                message=f"Failed to add tool '{tool.Title}' (may be duplicate or invalid)",
                added_count=0,
                failed_count=1,
                duplicate_count=0,
                total_count=1,
            )

    except Exception as e:
        logger.error(f"Error in single tool addition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/excel-approach/status", tags=["Excel Handler Style"])
async def get_excel_approach_status():
    """Get status of Excel handler approach for tool addition."""
    try:
        # Check Pinecone connection and get stats
        pinecone_service = PineconeService()
        stats = await pinecone_service.get_index_stats()

        return {
            "status": "success",
            "excel_handler_approach": {
                "enabled": True,
                "description": "Tools are added to Pinecone using the same approach as Excel handler",
                "method": "Batch upsert with embeddings",
                "namespace": "ai_tools",
                "embedding_model": "OpenAI Embeddings",
                "pinecone_connected": pinecone_service.index is not None,
                "index_stats": stats,
                "features": [
                    "Same embedding generation as Excel handler",
                    "Same vector format as Excel handler",
                    "Same upsert method as Excel handler",
                    "Batch processing for performance",
                    "UUID-based record IDs",
                    "Async processing with thread pool",
                    "Duplicate detection and prevention",
                    "Detailed status tracking",
                ],
            },
        }

    except Exception as e:
        logger.error(f"Error getting Excel approach status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/check-status", tags=["Excel Handler Style"])
async def check_tools_status(request: BatchToolRequest):
    """Check the status of tools in Pinecone without adding them."""
    try:
        logger.info(f"Checking status of {len(request.tools)} tools in Pinecone")

        # Convert Pydantic models to dictionaries
        tools_data = [tool.dict() for tool in request.tools]

        # Use Pinecone service to check status
        pinecone_service = PineconeService()
        results = []

        for tool_data in tools_data:
            title = tool_data.get("Title", "Unknown")
            website = tool_data.get("Website", "")

            # Check if tool exists
            duplicate_check = await pinecone_service._check_duplicate_detailed(
                tool_data
            )

            results.append(
                {
                    "title": title,
                    "website": website,
                    "exists_in_pinecone": duplicate_check["is_duplicate"],
                    "reason": duplicate_check["reason"],
                    "similarity_score": duplicate_check.get("similarity_score", 0.0),
                }
            )

        return {
            "status": "success",
            "message": f"Checked status of {len(tools_data)} tools",
            "total_checked": len(tools_data),
            "existing_tools": len([r for r in results if r["exists_in_pinecone"]]),
            "new_tools": len([r for r in results if not r["exists_in_pinecone"]]),
            "details": results,
        }

    except Exception as e:
        logger.error(f"Error checking tools status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
