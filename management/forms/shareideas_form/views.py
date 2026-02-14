"""Views for the share ideas form app."""

from rest_framework import permissions, status, viewsets
from rest_framework.response import Response

from .models import ShareIdeasForm
from .serializers import ShareIdeasFormSerializer


class ShareIdeasFormViewSet(viewsets.ModelViewSet):
    """ViewSet for managing ideas sharing form submissions."""

    queryset = ShareIdeasForm.objects.all()
    serializer_class = ShareIdeasFormSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        """Handle the creation of a new ideas submission."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            self.perform_create(serializer)
            return Response(
                {
                    "message": "Thank you for sharing your ideas with us!",
                    "data": serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(
            {"message": "Invalid data provided.", "errors": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def list(self, request, *args, **kwargs):
        """Handle listing all ideas submissions."""
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(
            {
                "message": "Successfully retrieved all ideas submissions.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def retrieve(self, request, *args, **kwargs):
        """Handle retrieving a specific ideas submission."""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(
            {
                "message": "Successfully retrieved ideas submission.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def update(self, request, *args, **kwargs):
        """Handle updating an ideas submission."""
        partial = kwargs.pop("partial", False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        if serializer.is_valid():
            self.perform_update(serializer)
            return Response(
                {
                    "message": "Ideas submission updated successfully.",
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
        """Handle deleting an ideas submission."""
        try:
            instance = self.get_object()
            instance_id = instance.id
            self.perform_destroy(instance)
            return Response(
                {
                    "message": "Ideas submission deleted successfully.",
                    "data": {"id": instance_id},
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"message": "Error deleting ideas submission.", "error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
