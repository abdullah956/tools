"""Explain tools endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from microservices.ai_tool_recommender.ai_agents.explain_tool_service import (
    generate_tool_explanation,
)

router = APIRouter()


# Define request model
class ExplainToolRequest(BaseModel):
    """Request model for explaining AI tools."""

    json_object: dict
    query: str


@router.post("/explain_tool/")
async def explain_tool(request: ExplainToolRequest):
    """Generate an explanation for a given AI tool configuration."""
    print(request.json_object)
    print(request.query)

    return {
        "explanation": await generate_tool_explanation(
            request.json_object, request.query
        ),
    }
