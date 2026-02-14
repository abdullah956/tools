"""Serializers for the AI tool recommender."""

from typing import Optional

from pydantic import BaseModel


class AITool(BaseModel):
    """Model representing an AI tool."""

    title: str
    description: str
    category: str
    features: str
    tags: str
    website: Optional[str] = None
    twitter: Optional[str] = None
    facebook: Optional[str] = None
    linkedin: Optional[str] = None


class SearchQuery(BaseModel):
    """Model representing a search query for AI tools."""

    query: str
