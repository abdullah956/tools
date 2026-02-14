"""Views for the submit resource form app."""

from rest_framework import permissions, status, viewsets
from rest_framework.response import Response

from .models import SubmitResourceForm
from .serializers import SubmitResourceFormSerializer


class SubmitResourceFormViewSet(viewsets.ModelViewSet):
    """ViewSet for managing resource submission form."""

    queryset = SubmitResourceForm.objects.all()
    serializer_class = SubmitResourceFormSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        """Handle the creation of a new resource submission."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            self.perform_create(serializer)
            return Response(
                {
                    "message": "Thank you for submitting your resource!",
                    "data": serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(
            {"message": "Invalid data provided.", "errors": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def list(self, request, *args, **kwargs):
        """Handle listing all resource submissions."""
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(
            {
                "message": "Successfully retrieved all resource submissions.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def retrieve(self, request, *args, **kwargs):
        """Handle retrieving a specific resource submission."""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(
            {
                "message": "Successfully retrieved resource submission.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def update(self, request, *args, **kwargs):
        """Handle updating a resource submission."""
        partial = kwargs.pop("partial", False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        if serializer.is_valid():
            self.perform_update(serializer)
            return Response(
                {
                    "message": "Resource submission updated successfully.",
                    "data": serializer.data,
                },
                status=status.HTTP_200_OK,
            )
        return Response(
            {
                "message": "Invalid data provided for update.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    def destroy(self, request, *args, **kwargs):
        """Handle deleting a resource submission."""
        try:
            instance = self.get_object()
            instance_id = instance.id
            self.perform_destroy(instance)
            return Response(
                {
                    "message": "Resource submission deleted successfully.",
                    "data": {"id": instance_id},
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"message": "Error deleting resource submission.", "error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
