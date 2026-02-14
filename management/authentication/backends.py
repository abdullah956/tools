"""Custom authentication backends for the authentication app."""

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend

UserModel = get_user_model()


class EmailOrUsernameModelBackend(ModelBackend):
    """Authenticate using either email or username."""

    def authenticate(self, request, username=None, password=None, **kwargs):
        """Authenticate a user by username or email."""
        if username is None:
            username = kwargs.get(UserModel.USERNAME_FIELD)
        try:
            # Try to fetch the user by username or email
            user = UserModel.objects.get(username=username)
        except UserModel.DoesNotExist:
            try:
                user = UserModel.objects.get(email=username)
            except UserModel.DoesNotExist:
                return None

        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
