"""Custom pagination classes for the API."""

from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response


class CustomPagination(PageNumberPagination):
    """Custom pagination class with a default page size."""

    page_size = 5  # You can set a default page size

    def get_paginated_response(self, data):
        """Return a paginated response with links and metadata."""
        return Response(
            {
                "links": {
                    "current": self.request.build_absolute_uri(),
                    "next": self.get_next_link(),
                    "previous": self.get_previous_link(),
                },
                "total_pages": self.page.paginator.num_pages,
                "count": self.page.paginator.count,
                "results": data,
            }
        )
