"""Views for the authentication app."""

import os
import socket
import urllib.parse
from datetime import datetime
from io import BytesIO
from urllib.parse import urlencode

import boto3
import requests
from botocore.exceptions import ClientError
from django.conf import settings
from django.contrib.auth.hashers import check_password
from django.contrib.auth.tokens import default_token_generator
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.timezone import make_aware
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from envs.env_loader import EnvLoader

# from .password_reset.serializers import ForgotPasswordSerializer
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.token_blacklist.models import (
    BlacklistedToken,
    OutstandingToken,
)
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken

from authentication.email_verification.models import EmailVerificationToken
from authentication.email_verification.serializers import EmailVerificationSerializer
from authentication.models import CustomUser
from authentication.set_password.serializers import SetPasswordSerializer
from configurations.pagination import CustomPagination
from configurations.utils import send_html_email
from email_teamplates.email_verification_success import (
    email_verification_success_template,
)
from email_teamplates.email_verification_template import email_verification_template
from email_teamplates.password_change_successfully import (
    email_template as password_change_successfully_email_template,
)
from email_teamplates.set_password import email_template as set_password_email_template
from forms.earlyaccess_form.models import EarlyAccessForm
from management.authentication.change_password.serializers import (
    PasswordChangeSerializer,
)
from management.authentication.password_reset.serializers import (
    PasswordResetConfirmSerializer,
    PasswordResetRequestSerializer,
)
from management.email_teamplates.password_reset import (
    email_template as password_reset_email_template,
)
from management.email_teamplates.password_reset_successfull import (
    email_template as password_reset_successfull_email_template,
)
from onboarding_questions.models import OnboardingQuestion
from onboarding_questions.serializers import OnboardingQuestionSerializer
from prompt_optimization.models import UserQuery
from prompt_optimization.serializers import UserQuerySerializer
from subscription.models import UserSubscription
from subscription.serializers import UserSubscriptionSerializer

# Get user's workflows
from workflow.models import Workflow
from workflow.serializers import WorkflowSerializer

from .pdf_processing_agent import ContractorDataExtractor
from .serializers import (
    CodeSerializer,
    CustomUserSerializer,
    LoginSerializer,
    PDFUploadSerializer,
    ProfilePictureSerializer,
)

env_loader = EnvLoader()


@extend_schema_view(
    register=extend_schema(
        request=CustomUserSerializer,
        responses={201: CustomUserSerializer, 400: "Bad Request"},
    ),
    login=extend_schema(
        request=LoginSerializer,
        responses={200: "Token and User Details", 400: "Bad Request"},
    ),
)
class UserViewSet(viewsets.ModelViewSet):
    """ViewSet for managing user registration and authentication."""

    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer
    pagination_class = CustomPagination
    lookup_field = "unique_id"

    @action(detail=False, methods=["post"], permission_classes=[permissions.AllowAny])
    def register(self, request):
        """Register a new user."""
        try:
            # Get server IP address
            try:
                server_ip_address = socket.gethostbyname(socket.gethostname())
            except socket.gaierror:
                server_ip_address = "127.0.0.1"  # Fallback to localhost

            # Prepare request data
            data = request.data.copy()
            data["server_ip_address"] = server_ip_address

            # Validate role and contractor fields
            role = data.get("role", "user")
            if role == "contractor":
                required_fields = [
                    "expertise",
                    "experience",
                    "website",
                    "language",
                    "country",
                    "company_name",
                    "type_of_services",
                    "about",
                ]
                missing_fields = [
                    field for field in required_fields if not data.get(field)
                ]
                if missing_fields:
                    error_msg = (
                        "Following fields are required for contractors: "
                        f"{', '.join(missing_fields)}"
                    )
                    return Response(
                        {"error": error_msg},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                # Convert date and time strings to proper format if provided
                if "availability_date" in data:
                    try:
                        datetime.strptime(data["availability_date"], "%Y-%m-%d")
                    except ValueError:
                        return Response(
                            {"error": "Invalid date format. Use YYYY-MM-DD"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )

                if "availability_time" in data:
                    try:
                        datetime.strptime(data["availability_time"], "%H:%M")
                    except ValueError:
                        return Response(
                            {"error": "Invalid time format. Use HH:MM"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )

            # Check for existing email
            if CustomUser.objects.filter(email=data.get("email")).exists():
                return Response(
                    {"error": "This email is already registered."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Validate and create user
            serializer = CustomUserSerializer(data=data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            # Check early access
            is_early_access = EarlyAccessForm.objects.filter(
                email=data.get("email"), is_email_verified=True, payment_status="paid"
            ).exists()

            # Create user
            user = serializer.save(is_early_access_user=is_early_access, role=role)

            # Generate tokens
            refresh = RefreshToken.for_user(user)
            jwt_tokens = {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }

            # Create verification token
            verification_token = get_random_string(32)
            EmailVerificationToken.objects.create(user=user, token=verification_token)

            # Create verification URL
            verification_url = (
                f"{env_loader.change_password_url}{reverse('user-verify-email')}"
                f"?token={verification_token}&email={user.email}"
            )

            # Send verification email
            try:
                email_content = email_verification_template.replace(
                    "verification_url", verification_url
                )
                send_html_email(
                    recipient_email=user.email,
                    subject="Welcome to Early Access! - Please Verify Your Email",
                    html_body=email_content,
                )
            except Exception as e:
                # Log email error but don't fail registration
                print(f"Email sending failed: {str(e)}")

            # Return success response
            return Response(
                {**serializer.data, "jwt_tokens": jwt_tokens, "role": user.role},
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            # Catch any unexpected errors and return proper response
            print(f"Registration error: {str(e)}")
            return Response(
                {"error": "Registration failed. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"], permission_classes=[permissions.AllowAny])
    def login(self, request):
        """Authenticate a user and return JWT tokens."""
        serializer = LoginSerializer(data=request.data)

        try:
            if serializer.is_valid(raise_exception=True):
                validated_data = serializer.validated_data
                user = validated_data["user"]

                # Update user tokens
                user.refresh_token = validated_data["refresh"]
                user.access_token = validated_data["access"]
                user.save()

                # Prepare response data
                response_data = {
                    "refresh": validated_data["refresh"],
                    "access": validated_data["access"],
                    "user": CustomUserSerializer(user).data,
                }

                # Create response with tokens
                response = Response(response_data, status=status.HTTP_200_OK)

                # Set cookies for access and refresh tokens
                response.set_cookie(
                    key="access_token",
                    value=validated_data["access"],
                    httponly=True,
                    secure=True,
                    samesite="Lax",
                    max_age=60 * 60,  # 1 hour
                )
                response.set_cookie(
                    key="refresh_token",
                    value=validated_data["refresh"],
                    httponly=True,
                    secure=True,
                    samesite="Lax",
                    max_age=7 * 24 * 60 * 60,  # 7 days
                )

                return response

        except ValidationError as e:
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(
        detail=False,
        methods=["get"],
        url_path="google/auth",
        permission_classes=[permissions.AllowAny],
    )
    def google_auth(self, request):
        """Initiate Google OAuth2 authentication."""
        base_url = "https://accounts.google.com/o/oauth2/auth"
        params = {
            "response_type": "code",
            "client_id": env_loader.google_client_id,
            "redirect_uri": env_loader.google_redirect_uri,
            "scope": "email profile",
            "access_type": "offline",
        }
        print(env_loader.google_client_id)
        print(env_loader.google_redirect_uri)
        url = f"{base_url}?{urlencode(params)}"
        return Response({"url": url}, status=status.HTTP_200_OK)

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="code",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Authorization code from Google",
                required=True,
            ),
        ],
        responses={
            200: "Successful response with user and token data",
            400: "Bad Request",
        },
    )
    @action(
        detail=False,
        methods=["post"],
        url_path="google/callback/2",
        permission_classes=[permissions.AllowAny],
    )
    def google_callback(self, request):
        """Handle Google OAuth2 callback."""
        serializer = CodeSerializer(data=request.data)
        if serializer.is_valid():
            # Decode the code if necessary
            code = urllib.parse.unquote(serializer.validated_data["code"])
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                "code": code,
                "client_id": env_loader.google_client_id,
                "client_secret": env_loader.google_client_secret,
                "redirect_uri": env_loader.google_redirect_uri,
                "grant_type": "authorization_code",
            }
            response = requests.post(token_url, data=data, timeout=10)

            try:
                tokens = response.json()
            except ValueError:
                return JsonResponse(
                    {"error": "Invalid response from token endpoint"}, status=400
                )

            if "error" in tokens:
                return JsonResponse({"error": tokens["error"]}, status=400)

            user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            user_info_response = requests.get(
                user_info_url, headers=headers, timeout=10
            )

            try:
                user_info = user_info_response.json()
            except ValueError:
                return JsonResponse(
                    {"error": "Invalid response from user info endpoint"}, status=400
                )

            if "error" in user_info:
                return JsonResponse({"error": user_info["error"]}, status=400)

            # Get the profile picture URL from Google
            profile_picture_url = user_info.get("picture")
            s3_profile_picture_url = None

            users = CustomUser.objects.filter(email=user_info["email"])
            if users.exists():
                user = users.first()

                # Only update the profile picture if:
                # 1. User has no profile picture, or
                # 2. User's profile picture source is 'google' or 'default'
                should_update_picture = (
                    not user.profile_picture
                    or user.profile_picture_source in ["google", "default"]
                )

                if profile_picture_url and should_update_picture:
                    try:
                        # Download the profile picture
                        picture_response = requests.get(profile_picture_url, timeout=10)
                        if picture_response.status_code == 200:
                            # Create a file-like object from the image content
                            image_content = BytesIO(picture_response.content)

                            # Generate a consistent filename for the user
                            file_extension = (
                                ".jpg"  # Google usually provides JPEG images
                            )
                            filename = f"{user_info['email']}/profile-picture/google_profile{file_extension}"

                            # Initialize S3 client
                            s3_client = boto3.client(
                                "s3",
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                region_name=settings.AWS_S3_REGION_NAME,
                            )

                            # Upload to S3
                            s3_client.upload_fileobj(
                                image_content,
                                settings.AWS_STORAGE_BUCKET_NAME,
                                filename,
                                ExtraArgs={"ContentType": "image/jpeg"},
                            )

                            # Generate the S3 URL
                            s3_profile_picture_url = (
                                f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{filename}"
                            )

                        user.profile_picture = s3_profile_picture_url
                        user.profile_picture_source = "google"
                        user.save()

                    except Exception as e:
                        print(f"Failed to upload profile picture: {str(e)}")
            else:
                # Check if user is in early access list
                is_early_access = EarlyAccessForm.objects.filter(
                    email=user_info["email"],
                    is_email_verified=True,
                    payment_status="paid",
                ).exists()

                # Create new user with profile picture
                user = CustomUser.objects.create(
                    username=user_info["email"],
                    email=user_info["email"],
                    first_name=user_info.get("given_name", ""),
                    last_name=user_info.get("family_name", ""),
                    auth_source="google",
                    is_verified=True,
                    is_early_access_user=is_early_access,
                    profile_picture=s3_profile_picture_url,
                    profile_picture_source="google",  # Set initial source as Google
                )

            # If this is an existing Google user, update their profile picture
            if user.auth_source == "google" and s3_profile_picture_url:
                user.profile_picture = s3_profile_picture_url
                user.save()

            # Generate JWT token for the user with unique_id
            refresh = RefreshToken.for_user(user)
            refresh["unique_id"] = str(user.unique_id)
            jwt_tokens = {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }

            # Update user's tokens
            user.refresh_token = jwt_tokens["refresh"]
            user.access_token = jwt_tokens["access"]
            user.save()

            response = JsonResponse(
                {
                    "user": CustomUserSerializer(user).data,
                    "tokens": tokens,
                    "jwt_tokens": jwt_tokens,
                    "code": code,
                },
                status=status.HTTP_200_OK,
            )

            # Set cookies for access and refresh tokens
            response.set_cookie(
                key="access_token",
                value=jwt_tokens["access"],
                httponly=True,
                secure=True,
                samesite="Lax",
                max_age=60 * 60,  # 1 hour
            )
            response.set_cookie(
                key="refresh_token",
                value=jwt_tokens["refresh"],
                httponly=True,
                secure=True,
                samesite="Lax",
                max_age=7 * 24 * 60 * 60,  # 7 days
            )

            return response

        return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(
        detail=True,  # This is a detail action, so it operates on a specific user
        methods=["post"],
        permission_classes=[
            permissions.IsAdminUser
        ],  # Only admins can make other users admin
    )
    def make_admin(self, request, unique_id=None):
        """Make a user an admin."""
        try:
            user = self.get_object()

            # Check if user is already an admin
            if user.is_staff:
                return Response(
                    {"message": f"User {user.username} is already an admin."},
                    status=status.HTTP_200_OK,
                )

            user.is_staff = True  # is_staff gives admin access in Django
            user.save()
            return Response(
                {"message": f"User {user.username} is now an admin."},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(
        detail=False,  # This is a list action, so it returns multiple users
        methods=["get"],
        permission_classes=[permissions.IsAdminUser],  # Only admins can view admin list
    )
    def admin_users(self, request):
        """Get all admin users."""
        try:
            admin_users = CustomUser.objects.filter(is_staff=True)
            serializer = self.get_serializer(admin_users, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(
        detail=False,
        methods=["get"],
        permission_classes=[permissions.IsAuthenticated],
    )
    def current_user_data(self, request):
        """Get all data related to the current user."""
        try:
            user = request.user

            workflows = Workflow.objects.filter(owner=user)
            workflow_data = WorkflowSerializer(workflows, many=True).data

            # Get user's onboarding questions
            onboarding_questions = OnboardingQuestion.objects.filter(user=user)
            onboarding_data = OnboardingQuestionSerializer(
                onboarding_questions, many=True
            ).data

            user_queries = UserQuery.objects.filter(user=user)
            query_data = UserQuerySerializer(user_queries, many=True).data

            # Get user's active subscription
            active_subscription = UserSubscription.objects.filter(
                user=user, status="active", end_date__gt=timezone.now()
            ).first()
            subscription_data = (
                UserSubscriptionSerializer(active_subscription).data
                if active_subscription
                else None
            )

            # Return all user-related data
            return Response(
                {
                    "user": CustomUserSerializer(user).data,
                    "workflows": workflow_data,
                    "onboarding_questions": onboarding_data,
                    "user_queries": query_data,
                    "active_subscription": subscription_data,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(
        detail=True,  # This is a detail action, so it operates on a specific user
        methods=["post"],
        permission_classes=[
            permissions.IsAdminUser
        ],  # Only admins can remove admin privileges
    )
    def remove_admin(self, request, unique_id=None):
        """Remove admin privileges from a user."""
        try:
            user = self.get_object()

            # Check if user is not an admin
            if not user.is_staff:
                return Response(
                    {"message": f"User {user.username} is not an admin."},
                    status=status.HTTP_200_OK,
                )

            user.is_staff = False  # Remove admin privileges
            user.save()
            return Response(
                {"message": f"Admin privileges removed from user {user.username}."},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @extend_schema(
        request=PasswordChangeSerializer,
        responses={
            200: "Password set successfully.",
            400: "Bad Request",
        },
        description="Allow users logged in with Google to set a password.",
    )
    @action(
        detail=False,
        methods=["post"],
        permission_classes=[permissions.IsAuthenticated],
        url_path="change-password",
    )
    def change_password(self, request):
        """Change the password for the authenticated user."""
        serializer = PasswordChangeSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user
            old_password = serializer.validated_data["old_password"]
            new_password = serializer.validated_data["new_password"]

            # Check if the old password is correct
            if not check_password(old_password, user.password):
                return Response(
                    {"error": "Old password is incorrect."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Set the new password
            user.set_password(new_password)
            user.save()

            # Send email to user
            send_html_email(
                user.email,
                "Password Changed Successfully",
                password_change_successfully_email_template,
            )
            return Response(
                {"message": "Password changed successfully."},
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        request=SetPasswordSerializer,
        responses={
            200: "Password set successfully.",
            400: "Bad Request",
        },
        description="Allow users logged in with Google to set a password.",
    )
    @action(
        detail=False,
        methods=["post"],
        permission_classes=[permissions.IsAuthenticated],
        url_path="set-password",
    )
    def set_password(self, request):
        """Allow users logged in with Google to set a password."""
        serializer = SetPasswordSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user

            # Ensure the user is logged in with Google
            if user.auth_source != "google":
                return Response(
                    {"error": "This is only available for Google-authenticated users."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            new_password = serializer.validated_data["new_password"]

            # Set the new password
            user.set_password(new_password)
            user.save()

            # Send email to user
            send_html_email(
                user.email,
                "Password Set Successfully",
                set_password_email_template,
            )

            return Response(
                {"message": "Password set successfully."},
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        request=EmailVerificationSerializer,
        responses={
            200: "Email verified successfully.",
            400: "Bad Request",
        },
        description="Verify a user's email using a token and email.",
    )
    @action(detail=False, methods=["get"], permission_classes=[AllowAny])
    def verify_email(self, request):
        """Verify a user's email using a token and email from query parameters."""
        token = request.query_params.get("token")
        email = request.query_params.get("email")
        if not token or not email:
            return redirect(env_loader.frontend_url)

        try:
            # Retrieve the verification token
            verification_token = get_object_or_404(
                EmailVerificationToken, token=token, user__email=email
            )

            # Check if the token is expired
            if verification_token.is_expired():
                return redirect(env_loader.frontend_url)

            # Verify the user's email
            user = verification_token.user
            user.is_verified = True
            user.save()

            # Optionally, delete the token after successful verification
            verification_token.delete()

            # Send email verification success email
            send_html_email(
                recipient_email=user.email,
                subject="Email Verified Successfully!",
                html_body=email_verification_success_template,
            )

            return redirect(env_loader.frontend_url)
        except Exception as e:
            print(f"Error verifying email: {str(e)}")
            return redirect(f"{env_loader.frontend_url}/verification-failed")

    @extend_schema(
        request=PasswordResetRequestSerializer,
        responses={200: "Email sent if user exists."},
    )
    @action(detail=False, methods=["post"], permission_classes=[AllowAny])
    def password_reset_request(self, request):
        """Send a password reset email to the user."""
        serializer = PasswordResetRequestSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data["email"]
            user = CustomUser.objects.get(email=email)

            def send_password_reset_email(user, request):
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                # Change the domain and path according to your frontend setup
                password_reset_url = f"{env_loader.password_reset_url}/{uid}/{token}/"

                send_html_email(
                    recipient_email=user.email,
                    subject="Password Reset Request",
                    html_body=password_reset_email_template.replace(
                        "{{ password_reset_url }}", password_reset_url
                    ),
                )

            send_password_reset_email(user, request)

            return Response(
                {"message": "Password reset email sent if the user exists."},
                status=status.HTTP_200_OK,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        request=PasswordResetConfirmSerializer,
        responses={
            200: "Password reset successfully.",
            400: "Invalid token or user ID.",
        },
    )
    @action(
        detail=False,
        methods=["post"],
        permission_classes=[AllowAny],
        url_path="reset-password-confirm",
    )
    def reset_password_confirm(self, request):
        """Reset the user's password if the token is valid."""
        uidb64 = request.data.get("uid")
        token = request.data.get("token")
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = CustomUser.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, CustomUser.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            serializer = PasswordResetConfirmSerializer(data=request.data)
            if serializer.is_valid():
                new_password = serializer.validated_data["new_password1"]
                user.set_password(new_password)
                user.save()
                send_html_email(
                    recipient_email=user.email,
                    subject="Password Reset Successfully",
                    html_body=password_reset_successfull_email_template,
                )
                return Response(
                    {"message": "Password reset successfully."},
                    status=status.HTTP_200_OK,
                )
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(
                {"error": "Invalid token or user ID."},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @extend_schema(
        request=None,
        responses={
            200: {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "example": "Successfully logged out."}
                },
            },
            400: {
                "type": "object",
                "properties": {"error": {"type": "string", "example": "Error message"}},
            },
        },
        description="Logout the current user and invalidate their tokens.",
    )
    @action(
        detail=False, methods=["post"], permission_classes=[permissions.IsAuthenticated]
    )
    def logout(self, request):
        """Logout the current user."""
        try:
            # Get access token from Authorization header
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                access_token_str = auth_header.split(" ")[1]
                try:
                    # Parse the access token
                    access_token = AccessToken(access_token_str)

                    # Create OutstandingToken instance
                    outstanding_token = OutstandingToken.objects.create(
                        user=request.user,
                        jti=access_token["jti"],
                        token=access_token_str,
                        created_at=make_aware(
                            datetime.fromtimestamp(access_token["iat"])
                        ),
                        expires_at=make_aware(
                            datetime.fromtimestamp(access_token["exp"])
                        ),
                    )

                    # Blacklist the token
                    BlacklistedToken.objects.create(token=outstanding_token)

                except Exception as e:
                    print(f"Error blacklisting token: {str(e)}")

            # Clear the user's tokens in the database
            user = request.user
            user.refresh_token = None
            user.access_token = None
            user.save()

            response = Response(
                {"message": "Successfully logged out."}, status=status.HTTP_200_OK
            )

            # Delete the cookies if they exist
            response.delete_cookie("access_token")
            response.delete_cookie("refresh_token")

            return response

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        responses={
            200: CustomUserSerializer(many=True),
            403: "Permission denied",
            400: "Bad Request",
        },
        description=(
            "Get all contractors. Only accessible by regular users and super admins."
        ),
    )
    @action(
        detail=False,
        methods=["get"],
        permission_classes=[permissions.IsAuthenticated],
        url_path="contractors",
    )
    def get_contractors(self, request):
        """Get all users with contractor role."""
        try:
            # Check if the requesting user is a contractor
            if request.user.role == "contractor":
                return Response(
                    {"error": "Contractors cannot access the list of contractors."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Get all contractors
            contractors = CustomUser.objects.filter(role="contractor")

            # Apply pagination
            page = self.paginate_queryset(contractors)
            if page is not None:
                serializer = self.get_serializer(page, many=True)
                return self.get_paginated_response(serializer.data)

            # If pagination is disabled, return all results
            serializer = self.get_serializer(contractors, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        responses={
            200: CustomUserSerializer,
            403: "Permission denied",
            404: "Contractor not found",
            400: "Bad Request",
        },
        description="Get a specific contractor by ID.",
    )
    @action(
        detail=False,
        methods=["get"],
        permission_classes=[permissions.IsAuthenticated],
        url_path="contractor/(?P<contractor_id>[^/.]+)",
    )
    def get_contractor_by_id(self, request, contractor_id=None):
        """Get a specific contractor by ID."""
        try:
            # Check if the requesting user is a contractor
            if request.user.role == "contractor":
                return Response(
                    {"error": "Contractors cannot access contractor details."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Get the contractor
            try:
                contractor = CustomUser.objects.get(
                    unique_id=contractor_id, role="contractor"
                )
            except CustomUser.DoesNotExist:
                return Response(
                    {"error": "Contractor not found."}, status=status.HTTP_404_NOT_FOUND
                )

            serializer = self.get_serializer(contractor)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="query",
                description="Search query string for contractors",
                required=True,
                type=str,
            ),
        ],
        responses={
            200: CustomUserSerializer(many=True),
            403: "Permission denied",
            400: "Bad Request",
        },
        description="Search contractors based on name, company, and occupation.",
    )
    @action(
        detail=False,
        methods=["get"],
        permission_classes=[permissions.IsAuthenticated],
        url_path="contractors/search",
    )
    def search_contractors(self, request):
        """Search contractors based on name, company, and occupation."""
        try:
            # Check if the requesting user is a contractor
            if request.user.role == "contractor":
                return Response(
                    {"error": "Contractors cannot search contractor details."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Get the search query
            query = request.query_params.get("query", "").strip()
            if not query:
                return Response(
                    {"error": "Search query is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Perform the search
            results = CustomUser.search_contractors(query)

            # Apply pagination
            page = self.paginate_queryset(results)
            if page is not None:
                serializer = self.get_serializer(page, many=True)
                return self.get_paginated_response(serializer.data)

            # If pagination is disabled
            serializer = self.get_serializer(results, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        request=PDFUploadSerializer,
        responses={200: "JSON with extracted contractor information"},
        description="Extract contractor information from uploaded PDF resume/CV",
    )
    @action(
        detail=False,
        methods=["post"],
        permission_classes=[permissions.AllowAny],
        parser_classes=[MultiPartParser, FormParser],
        url_path="extract-contractor-data",
    )
    def extract_contractor_data(self, request):
        """Extract contractor information from uploaded PDF."""
        try:
            # Validate the uploaded file
            serializer = PDFUploadSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            # Process the PDF file
            pdf_file = serializer.validated_data["pdf_file"]
            extractor = ContractorDataExtractor()
            extracted_data = extractor.process_pdf(pdf_file)

            return Response(extracted_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        request=ProfilePictureSerializer,
        responses={
            200: {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "profile_picture_url": {"type": "string"},
                },
            },
            400: {"type": "object", "properties": {"error": {"type": "string"}}},
            500: {"type": "object", "properties": {"error": {"type": "string"}}},
        },
        description="Upload a profile picture. Accepts JPEG, JPG, PNG formats up to 5MB.",
    )
    @action(
        detail=False,
        methods=["post"],
        permission_classes=[permissions.IsAuthenticated],
        parser_classes=[MultiPartParser, FormParser],
    )
    def upload_profile_picture(self, request):
        """Upload a profile picture to S3 and update the user's profile."""
        try:
            serializer = ProfilePictureSerializer(data=request.FILES)
            if not serializer.is_valid():
                return Response(
                    {"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST
                )

            file = serializer.validated_data["profile_picture"]

            # Validate file type
            allowed_types = ["image/jpeg", "image/png", "image/jpg"]
            if file.content_type not in allowed_types:
                return Response(
                    {"error": "Invalid file type. Only JPEG, JPG and PNG are allowed."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Validate file size (e.g., max 5MB)
            if file.size > 5 * 1024 * 1024:
                return Response(
                    {"error": "File too large. Maximum size is 5MB."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Generate filename with the consistent path structure
            file_extension = os.path.splitext(file.name)[1]
            filename = f"{request.user.unique_id}/profile-picture/assets/profile{file_extension}"

            # Initialize S3 client
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME,
            )

            # Upload to S3
            try:
                s3_client.upload_fileobj(
                    file,
                    settings.AWS_STORAGE_BUCKET_NAME,
                    filename,
                    ExtraArgs={"ContentType": file.content_type},
                )

                # Generate the URL for the uploaded file
                s3_url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{filename}"

                # Update user's profile picture URL and source
                user = request.user
                user.profile_picture = s3_url
                user.profile_picture_source = "manual"  # Mark as manually uploaded
                user.save()

                return Response(
                    {
                        "message": "Profile picture uploaded successfully",
                        "profile_picture_url": s3_url,
                    },
                    status=status.HTTP_200_OK,
                )

            except ClientError as e:
                return Response(
                    {"error": f"Failed to upload to S3: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        except Exception as e:
            return Response(
                {"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema_view(
    linkedin_auth=extend_schema(
        responses={200: "LinkedIn OAuth URL", 400: "Bad Request"},
        description="Initiate LinkedIn OAuth2 authentication.",
    ),
    linkedin_callback=extend_schema(
        request=CodeSerializer,
        responses={
            200: "Successful response with user and token data",
            400: "Bad Request",
        },
        description="Handle LinkedIn OAuth2 callback.",
    ),
)
class LinkedInAuthViewSet(viewsets.GenericViewSet):
    """ViewSet for LinkedIn OAuth2 authentication."""

    permission_classes = [permissions.AllowAny]
    serializer_class = CodeSerializer

    @action(
        detail=False,
        methods=["get"],
        url_path="auth",
        permission_classes=[permissions.AllowAny],
    )
    def linkedin_auth(self, request):
        """Initiate LinkedIn OAuth2 authentication."""
        try:
            base_url = "https://www.linkedin.com/oauth/v2/authorization"
            params = {
                "response_type": "code",
                "client_id": env_loader.linkedin_client_id,
                "redirect_uri": env_loader.linkedin_redirect_uri,
                "scope": "openid profile email",
            }
            url = f"{base_url}?{urlencode(params)}"
            return Response({"url": url}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Failed to generate LinkedIn auth URL: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(
        detail=False,
        methods=["post"],
        url_path="callback",
        permission_classes=[permissions.AllowAny],
    )
    def linkedin_callback(self, request):
        """Handle LinkedIn OAuth2 callback."""
        serializer = CodeSerializer(data=request.data)
        if not serializer.is_valid():
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Decode the code if necessary
            code = urllib.parse.unquote(serializer.validated_data["code"])

            # Exchange code for access token
            token_url = "https://www.linkedin.com/oauth/v2/accessToken"
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": env_loader.linkedin_client_id,
                "client_secret": env_loader.linkedin_client_secret,
                "redirect_uri": env_loader.linkedin_redirect_uri,
            }

            token_response = requests.post(
                token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )

            try:
                tokens = token_response.json()
            except ValueError:
                return JsonResponse(
                    {"error": "Invalid response from LinkedIn token endpoint"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if "error" in tokens or "access_token" not in tokens:
                error_msg = tokens.get(
                    "error_description", tokens.get("error", "Unknown error")
                )
                return JsonResponse(
                    {"error": f"LinkedIn token error: {error_msg}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            access_token = tokens["access_token"]

            # Fetch user profile information
            user_info_url = "https://api.linkedin.com/v2/userinfo"
            headers = {"Authorization": f"Bearer {access_token}"}
            user_info_response = requests.get(
                user_info_url, headers=headers, timeout=10
            )

            try:
                user_info = user_info_response.json()
            except ValueError:
                return JsonResponse(
                    {"error": "Invalid response from LinkedIn user info endpoint"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if "error" in user_info or "email" not in user_info:
                error_msg = user_info.get("message", "Failed to fetch user information")
                return JsonResponse(
                    {"error": f"LinkedIn user info error: {error_msg}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Log the user info for debugging
            print(f"LinkedIn user_info response: {user_info}")

            # Extract user information - LinkedIn may return different field names
            email = user_info.get("email")

            # Try multiple field names for first name
            first_name = (
                user_info.get("given_name")
                or user_info.get("firstName")
                or user_info.get("givenName")
                or ""
            )

            # Try multiple field names for last name
            last_name = (
                user_info.get("family_name")
                or user_info.get("lastName")
                or user_info.get("familyName")
                or ""
            )

            # If name field exists but given_name/family_name don't, try to split name
            if not first_name and not last_name:
                full_name = user_info.get("name", "")
                if full_name:
                    name_parts = full_name.split(" ", 1)
                    first_name = name_parts[0] if len(name_parts) > 0 else ""
                    last_name = name_parts[1] if len(name_parts) > 1 else ""

            profile_picture_url = user_info.get("picture")
            s3_profile_picture_url = None

            # Check if user already exists
            users = CustomUser.objects.filter(email=email)
            if users.exists():
                user = users.first()

                # Update user name if it's missing or generic
                should_update_name = (
                    not user.first_name
                    or not user.last_name
                    or user.first_name in ["string", ""]
                    or user.last_name in ["string", ""]
                )

                if should_update_name and (first_name or last_name):
                    if first_name:
                        user.first_name = first_name
                    if last_name:
                        user.last_name = last_name
                    user.save()

                # Update profile picture if needed
                should_update_picture = (
                    not user.profile_picture
                    or user.profile_picture_source in ["linkedin", "default"]
                )

                if profile_picture_url and should_update_picture:
                    try:
                        # Download the profile picture
                        picture_response = requests.get(profile_picture_url, timeout=10)
                        if picture_response.status_code == 200:
                            # Create a file-like object from the image content
                            image_content = BytesIO(picture_response.content)

                            # Generate filename
                            file_extension = ".jpg"
                            filename = f"{email}/profile-picture/linkedin_profile{file_extension}"

                            # Initialize S3 client
                            s3_client = boto3.client(
                                "s3",
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                region_name=settings.AWS_S3_REGION_NAME,
                            )

                            # Upload to S3
                            s3_client.upload_fileobj(
                                image_content,
                                settings.AWS_STORAGE_BUCKET_NAME,
                                filename,
                                ExtraArgs={"ContentType": "image/jpeg"},
                            )

                            # Generate the S3 URL
                            s3_profile_picture_url = (
                                f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{filename}"
                            )

                            user.profile_picture = s3_profile_picture_url
                            user.profile_picture_source = "linkedin"
                            user.save()

                    except Exception as e:
                        print(f"Failed to upload LinkedIn profile picture: {str(e)}")
            else:
                # Check if user is in early access list
                is_early_access = EarlyAccessForm.objects.filter(
                    email=email,
                    is_email_verified=True,
                    payment_status="paid",
                ).exists()

                # Download and upload profile picture to S3 if available
                if profile_picture_url:
                    try:
                        picture_response = requests.get(profile_picture_url, timeout=10)
                        if picture_response.status_code == 200:
                            image_content = BytesIO(picture_response.content)
                            file_extension = ".jpg"
                            filename = f"{email}/profile-picture/linkedin_profile{file_extension}"

                            s3_client = boto3.client(
                                "s3",
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                region_name=settings.AWS_S3_REGION_NAME,
                            )

                            s3_client.upload_fileobj(
                                image_content,
                                settings.AWS_STORAGE_BUCKET_NAME,
                                filename,
                                ExtraArgs={"ContentType": "image/jpeg"},
                            )

                            s3_profile_picture_url = (
                                f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{filename}"
                            )
                    except Exception as e:
                        print(f"Failed to upload LinkedIn profile picture: {str(e)}")

                # Generate username from email (part before @)
                username_base = email.split("@")[0]
                username = username_base

                # Ensure username is unique
                counter = 1
                while CustomUser.objects.filter(username=username).exists():
                    username = f"{username_base}{counter}"
                    counter += 1

                # Create new user
                user = CustomUser.objects.create(
                    username=username,
                    email=email,
                    first_name=first_name,
                    last_name=last_name,
                    auth_source="linkedin",
                    is_verified=True,
                    is_early_access_user=is_early_access,
                    profile_picture=s3_profile_picture_url,
                    profile_picture_source="linkedin",
                )

            # Generate JWT tokens for the user
            refresh = RefreshToken.for_user(user)
            refresh["unique_id"] = str(user.unique_id)
            jwt_tokens = {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            }

            # Update user's tokens
            user.refresh_token = jwt_tokens["refresh"]
            user.access_token = jwt_tokens["access"]
            user.save()

            # Create response
            response = JsonResponse(
                {
                    "user": CustomUserSerializer(user).data,
                    "tokens": tokens,
                    "jwt_tokens": jwt_tokens,
                    "code": code,
                },
                status=status.HTTP_200_OK,
            )

            # Set cookies for access and refresh tokens
            response.set_cookie(
                key="access_token",
                value=jwt_tokens["access"],
                httponly=True,
                secure=True,
                samesite="Lax",
                max_age=60 * 60,  # 1 hour
            )
            response.set_cookie(
                key="refresh_token",
                value=jwt_tokens["refresh"],
                httponly=True,
                secure=True,
                samesite="Lax",
                max_age=7 * 24 * 60 * 60,  # 7 days
            )

            return response

        except requests.RequestException as e:
            return JsonResponse(
                {"error": f"Network error during LinkedIn authentication: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        except Exception as e:
            return JsonResponse(
                {"error": f"LinkedIn authentication failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
