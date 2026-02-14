"""Views for the invite friends form app."""

from rest_framework import permissions, status, viewsets
from rest_framework.response import Response

from configurations.utils import send_email

from .models import InviteFriendsForm
from .serializers import InviteFriendsFormSerializer


class InviteFriendsFormViewSet(viewsets.ModelViewSet):
    """ViewSet for managing friend invitations."""

    queryset = InviteFriendsForm.objects.all()
    serializer_class = InviteFriendsFormSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        """Handle the creation of new friend invitations."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            # Don't save yet - first check if emails can be sent
            email_results = []

            subject = "Join Our Community!"
            body = """Hello!
            You have been invited to join our community.
            We're excited to have you join us!

            Click here to get started: https://example.com
            """

            # Try sending emails to all addresses
            all_emails_sent = True
            for email in serializer.validated_data["emails"]:
                success, error = send_email(email, subject, body)
                email_results.append(
                    {
                        "email": email,
                        "sent": success,
                        "error": error if not success else None,
                    }
                )
                if not success:
                    all_emails_sent = False

            # Only save if all emails were sent successfully
            if all_emails_sent:
                self.perform_create(serializer)
                return Response(
                    {
                        "message": "Thank you for inviting your friends!",
                        "data": serializer.data,
                        "email_results": email_results,
                    },
                    status=status.HTTP_201_CREATED,
                )
            else:
                return Response(
                    {
                        "message": "Failed to send some invitation emails.",
                        "email_results": email_results,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        return Response(
            {"message": "Invalid data provided.", "errors": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def list(self, request, *args, **kwargs):
        """Handle listing all friend invitations."""
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(
            {
                "message": "Successfully retrieved all friend invitations.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def retrieve(self, request, *args, **kwargs):
        """Handle retrieving a specific friend invitation."""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(
            {
                "message": "Successfully retrieved friend invitation.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def update(self, request, *args, **kwargs):
        """Handle updating friend invitations."""
        partial = kwargs.pop("partial", False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        if serializer.is_valid():
            self.perform_update(serializer)
            return Response(
                {
                    "message": "Friend invitations updated successfully.",
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
        """Handle deleting friend invitations."""
        try:
            instance = self.get_object()
            instance_id = instance.id
            self.perform_destroy(instance)
            return Response(
                {
                    "message": "Friend invitations deleted successfully.",
                    "data": {"id": instance_id},
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"message": "Error deleting friend invitations.", "error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
