"""Custom permission classes for the API."""

from rest_framework.permissions import BasePermission


class IsAuthenticatedUserOrSuperUser(BasePermission):
    """Allows access only to authenticated users and superusers."""

    def has_permission(self, request, view):
        """Check if the user is authenticated or a superuser."""
        return request.user and (
            request.user.is_authenticated or request.user.is_superuser
        )
