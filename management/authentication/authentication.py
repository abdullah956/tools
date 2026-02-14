"""CUSTOM AUTHENTICATION CLASS."""
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import AuthenticationFailed, InvalidToken
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken


class CustomJWTAuthentication(JWTAuthentication):
    """Custom JWT AUTH."""

    def authenticate(self, request):
        """AUTHENTICATION FUNCTION."""
        try:
            header = self.get_header(request)
            if header is None:
                return None

            raw_token = self.get_raw_token(header)
            if raw_token is None:
                return None

            validated_token = self.get_validated_token(raw_token)

            # Check if token is blacklisted
            jti = validated_token.get("jti")
            if BlacklistedToken.objects.filter(token__jti=jti).exists():
                raise InvalidToken("Token is blacklisted")

            user = self.get_user(validated_token)
            return user, validated_token

        except InvalidToken:
            raise AuthenticationFailed("Invalid token or token is blacklisted")
        except Exception as e:
            raise AuthenticationFailed(str(e))

    def authenticate_header(self, request):
        """AUTHENTICATION HEADER."""
        return "Bearer"
