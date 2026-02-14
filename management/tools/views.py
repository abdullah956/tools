"""This module contains the views for the tools."""

from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Tool
from .serializers import ToolSerializer


class ToolViewSet(viewsets.ModelViewSet):
    """This class contains the viewset for the Tool model."""

    queryset = Tool.objects.all()
    serializer_class = ToolSerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="query",
                description="Search query string",
                required=True,
                type=str,
            ),
        ],
        responses={200: ToolSerializer(many=True)},
        description=(
            "Search tools based on title, description, category, features, and tags. "
            "Returns ranked results based on relevance."
        ),
    )
    @action(detail=False, methods=["get"], url_path="search")
    def search(self, request):
        """Search tools based on the provided query."""
        try:
            query = request.query_params.get("query", "").strip()
            if not query:
                return Response(
                    {"error": "Search query is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Perform the search
            results = Tool.search(query)

            # Apply pagination
            page = self.paginate_queryset(results)
            if page is not None:
                serializer = self.get_serializer(page, many=True)
                return self.get_paginated_response(serializer.data)

            # If pagination is disabled
            serializer = self.get_serializer(results, many=True)
            return Response(serializer.data)

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
