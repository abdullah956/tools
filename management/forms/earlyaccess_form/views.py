"""Views for managing early access form submissions."""

import json
import logging
from datetime import datetime, timedelta

import stripe
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from drf_spectacular.utils import (
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
    extend_schema_view,
    inline_serializer,
)
from envs.env_loader import env_loader
from rest_framework import permissions, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from email_teamplates.earlyaccess_paymentlink import email_template
from email_teamplates.success_emaill_verification_earlyaccess import (
    early_access_success_email_template,
)
from management.configurations.utils import send_html_email
from subscription.models import SubscriptionProduct, UserSubscription
from subscription.utils import (
    create_checkout_session,
    get_rewardful_affiliate_info,
    process_early_access_payment,
)

from .models import EarlyAccessForm, generate_alphanumeric_token
from .serializers import EarlyAccessFormSerializer

stripe.api_key = env_loader.stripe_secret_key
logger = logging.getLogger(__name__)


@extend_schema_view(
    create=extend_schema(
        description="Create a new early access form submission",
        responses={201: EarlyAccessFormSerializer},
    ),
    list=extend_schema(
        description="List all early access form submissions",
        responses={200: EarlyAccessFormSerializer(many=True)},
    ),
)
class EarlyAccessFormViewSet(viewsets.ModelViewSet):
    """ViewSet for managing early access form submissions."""

    queryset = EarlyAccessForm.objects.all()
    serializer_class = EarlyAccessFormSerializer
    permission_classes = [permissions.AllowAny]

    @extend_schema(
        request=inline_serializer(
            name="EarlyAccessCheckoutRequest",
            fields={
                "email": serializers.EmailField(
                    help_text="Email address for early access"
                ),
                "product_id": serializers.CharField(
                    help_text="ID of the subscription product"
                ),
                "ref": serializers.CharField(
                    help_text="Rewardful referral ID", required=False
                ),
            },
        ),
        responses={
            200: OpenApiResponse(
                description="Checkout session created successfully",
                response=inline_serializer(
                    name="EarlyAccessCheckoutResponse",
                    fields={
                        "checkout_url": serializers.URLField(),
                        "session_id": serializers.CharField(),
                    },
                ),
            ),
            400: OpenApiResponse(
                description="Bad Request",
                response=inline_serializer(
                    name="EarlyAccessCheckoutError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
        },
        description="Create a Stripe checkout session for early access payment",
    )
    @action(detail=False, methods=["post"])
    def checkout(self, request):
        """Create a Stripe checkout session for early access payment."""
        email = request.data.get("email")
        product_id = request.data.get("product_id")
        referral_id = request.data.get("ref")  # Get referral ID if available

        if not email:
            return Response(
                {"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        if not product_id:
            return Response(
                {"error": "Product ID is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get affiliate information if referral ID is provided
            affiliate_info = None
            if referral_id:
                affiliate_info = get_rewardful_affiliate_info(referral_id)

            # Check if user already has early access with verified email
            existing_access = EarlyAccessForm.objects.filter(
                email=email, is_email_verified=True, payment_status="paid"
            ).first()

            if existing_access:
                logger.info(f"User {email} already has early access")
                return Response(
                    {
                        "message": "You already have early access to our platform.",
                        "email": email,
                        "status": "already_subscribed",
                    },
                    status=status.HTTP_200_OK,
                )

            # Generate verification token
            verification_token = generate_alphanumeric_token()

            # Update or create early access form entry
            early_access, _ = EarlyAccessForm.objects.update_or_create(
                email=email,
                defaults={
                    "payment_status": "pending",
                    "product_id": product_id,
                    "email_verification_token": verification_token,
                    "token_created_at": datetime.now(),
                    "is_email_verified": False,
                    "referral_id": referral_id,
                },
            )

            # Get the subscription product
            subscription_product = SubscriptionProduct.objects.get(
                stripe_product_id=product_id
            )

            # Construct success URL with the session ID parameter
            # Use absolute URI to ensure the URL is properly formed
            success_url = request.build_absolute_uri(
                reverse("earlyaccess_form:early-access-success")
            )
            # Make sure the success URL doesn't already have query parameters
            if "?" in success_url:
                success_url = success_url.split("?")[0]

            cancel_url = "https://app.example.com"

            # Prepare metadata for the checkout session
            metadata = {
                "email": email,
                "form_id": str(early_access.id),
                "product_id": subscription_product.stripe_product_id,
            }

            checkout_session, _ = create_checkout_session(
                product=subscription_product,
                success_url=success_url,
                cancel_url=cancel_url,
                metadata=metadata,
                customer_email=email,
                referral_id=referral_id,
            )

            # Update the form with checkout session ID
            early_access.stripe_checkout_session_id = checkout_session.id
            early_access.save()

            # Modify the email template to include affiliate information
            email_subject = "Complete Your Early Access Payment"

            # Create a clean email template with proper string formatting
            email_body = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Complete Your Early Access Payment</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                background-color: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                max-width: 600px;
                margin: 0 auto;
            }}
            h1 {{
                color: #2c3e50;
                margin-bottom: 20px;
            }}
            p {{
                color: #333;
                font-size: 16px;
                line-height: 1.6;
            }}
            .button {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                padding: 12px 24px;
                border-radius: 4px;
                margin-top: 20px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Complete Your Early Access Payment</h1>
            <p>Thank you for your interest in our early access program!</p>
            <p>To complete your registration, please click the button below to proceed to the payment page:</p>
            <p><a href="{0}" class="button">Complete Payment</a></p>
            {1}
            <p>If the button doesn't work, you can also copy and paste this link into your browser:</p>
            <p>{0}</p>
            <p>This link will expire in 24 hours.</p>
            <p>If you have any questions, please don't hesitate to contact our support team.</p>
            <p>Best regards,<br>The Team</p>
        </div>
    </body>
    </html>
    """

            # Add affiliate information if available
            affiliate_text = ""
            if affiliate_info and affiliate_info.get("name"):
                affiliate_text = f'<p>You\'ve been invited by <strong>{affiliate_info["name"]}</strong> to join example.com!</p>'

            # Format the email body with the checkout URL and affiliate information
            formatted_email_body = email_body.format(
                checkout_session.url, affiliate_text
            )

            success, error = send_html_email(
                early_access.email, email_subject, formatted_email_body
            )

            if not success:
                print(f"Warning: Failed to send checkout email: {error}")

            # Include affiliate name in the response if available
            response_data = {
                "checkout_url": checkout_session.url,
                "session_id": checkout_session.id,
                "message": "Checkout URL sent to email",
            }
            if affiliate_info and affiliate_info.get("name"):
                response_data["affiliate_name"] = affiliate_info["name"]

            return Response(response_data)

        except SubscriptionProduct.DoesNotExist:
            return Response(
                {"error": "Invalid subscription product"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except stripe.error.StripeError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except StopIteration as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        request=inline_serializer(
            name="EarlyAccessWithoutCheckoutRequest",
            fields={
                "email": serializers.EmailField(
                    help_text="Email address for early access"
                ),
                "ref": serializers.CharField(
                    help_text="Rewardful referral ID", required=False
                ),
            },
        ),
        responses={
            201: OpenApiResponse(
                description="User registered successfully",
                response=inline_serializer(
                    name="EarlyAccessWithoutCheckoutResponse",
                    fields={
                        "message": serializers.CharField(),
                        "email": serializers.EmailField(),
                    },
                ),
            ),
            400: OpenApiResponse(
                description="Bad Request",
                response=inline_serializer(
                    name="EarlyAccessWithoutCheckoutError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
        },
        description="Register for early access without checkout",
    )
    @action(detail=False, methods=["post"])
    def without_checkout(self, request):
        """Register for early access without going through checkout."""
        email = request.data.get("email")
        referral_id = request.data.get("ref")  # Get referral ID if available

        if not email:
            return Response(
                {"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        logger.info(
            f"Processing early access registration without checkout for: {email}"
        )

        try:
            # Check if user already has early access with verified email
            existing_access = EarlyAccessForm.objects.filter(email=email).first()

            if existing_access:
                return Response(
                    {
                        "message": "You already have early access to our platform.",
                        "email": email,
                        "status": "already_subscribed",
                    },
                    status=status.HTTP_200_OK,
                )

            # Generate verification token
            verification_token = generate_alphanumeric_token()

            # Update or create early access form entry
            early_access, created = EarlyAccessForm.objects.update_or_create(
                email=email,
                defaults={
                    "payment_status": "pending",  # Mark as pending until email verification
                    "has_paid": False,  # Set has_paid to false until verification
                    "email_verification_token": verification_token,
                    "token_created_at": datetime.now(),
                    "is_email_verified": False,
                    "referral_id": referral_id,
                    "payment_date": None,  # Will be set after verification
                },
            )

            # Send welcome email with verification link
            email_subject = "Welcome to Early Access - Please Verify Your Email"
            verification_url = f"{settings.BASE_URL}/early-access/forms/verify_email/?email={email}&token={verification_token}"
            print(f"verification url: {verification_url}")
            email_body = f"""
                <html>
                    <body style="font-family: Arial, sans-serif; color: #333;">
                        <h2>Welcome to Early Access!</h2>

                        <p>Thank you for registering for early access to our platform.</p>

                        <p>Please verify your email address by clicking the link below:</p>
                        <p><a href="{verification_url}">Verify Email Address</a></p>

                        <p>If you have any questions or need assistance,
                        please don't hesitate to contact our support team.</p>

                        <p>Best regards,<br>
                        The Team</p>
                    </body>
                </html>
            """

            success, error = send_html_email(
                early_access.email, email_subject, email_body
            )
            if not success:
                print(f"Warning: Failed to send welcome email: {error}")

            return Response(
                {
                    "message": "Successfully registered for early access. Please check your email to verify your address.",
                    "email": email,
                    "is_new_registration": created,
                    "status": "success",
                },
                status=status.HTTP_201_CREATED,
            )

        except (ValueError, TypeError) as e:
            return Response(
                {"error": f"Data validation error: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except EarlyAccessForm.DoesNotExist as e:
            return Response(
                {"error": f"Database error: {str(e)}"}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            # Log the unexpected error
            print(f"Unexpected error in without_checkout: {e}")
            return Response(
                {"error": "Internal server error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="email",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Email address to check payment history",
                required=True,
            ),
            OpenApiParameter(
                name="token",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Verification token",
                required=True,
            ),
        ],
        responses={
            200: inline_serializer(
                name="EarlyAccessPaymentHistoryResponse",
                fields={
                    "email": serializers.EmailField(),
                    "has_paid": serializers.BooleanField(),
                    "payment_status": serializers.CharField(),
                    "payment_date": serializers.DateTimeField(allow_null=True),
                },
            ),
            400: OpenApiResponse(
                description="Bad Request",
                response=inline_serializer(
                    name="PaymentHistoryError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
            404: OpenApiResponse(
                description="No payment history found",
                response=inline_serializer(
                    name="PaymentHistoryError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
        },
        description="Get payment history using both email and verification token",
    )
    @action(detail=False, methods=["get"])
    def payment_history(self, request):
        """Get payment history using both email and verification token."""
        email = request.query_params.get("email")
        token = request.query_params.get("token")

        if not email or not token:
            return Response(
                {"error": "Both email and token are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            early_access = EarlyAccessForm.objects.get(
                email=email, email_verification_token=token
            )

            return Response(
                {
                    "email": early_access.email,
                    "has_paid": early_access.has_paid,
                    "payment_status": early_access.payment_status,
                    "payment_date": early_access.payment_date,
                }
            )
        except EarlyAccessForm.DoesNotExist:
            return Response(
                {"error": "No payment history found or email/token is invalid"},
                status=status.HTTP_404_NOT_FOUND,
            )

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="email",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Email address to cancel subscription",
                required=True,
            ),
            OpenApiParameter(
                name="token",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Verification token",
                required=True,
            ),
        ],
        responses={
            200: inline_serializer(
                name="CancelSubscriptionResponse",
                fields={
                    "message": serializers.CharField(),
                },
            ),
            400: OpenApiResponse(
                description="Bad Request",
                response=inline_serializer(
                    name="CancelSubscriptionError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
            404: OpenApiResponse(
                description="No subscription found",
                response=inline_serializer(
                    name="CancelSubscriptionError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
        },
        description="Cancel subscription using email and verification token",
    )
    @action(detail=False, methods=["post"])
    def cancel_subscription(self, request):
        """Cancel subscription using email and verification token."""
        email = request.query_params.get("email")
        token = request.query_params.get("token")

        if not email or not token:
            return Response(
                {"error": "Both email and token are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            early_access = EarlyAccessForm.objects.get(
                email=email, email_verification_token=token
            )

            if not early_access.has_paid:
                return Response(
                    {"error": "No active subscription found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Process cancellation
            early_access.has_paid = False
            early_access.payment_status = "cancelled"
            early_access.email_verification_token = None  # Clear the token
            early_access.token_created_at = None
            early_access.save()

            # Send cancellation confirmation email
            email_subject = "Subscription Cancellation Confirmed"
            email_body = """
                <html>
                    <body style="font-family: Arial, sans-serif; color: #333;">
                        <h2>Subscription Cancellation Confirmed</h2>

                        <p>Your subscription has been successfully cancelled.</p>

                        <p>If you change your mind, you can always subscribe again.</p>

                        <p>Thank you for trying our service!</p>

                        <p>Best regards,<br>
                        The Team</p>
                    </body>
                </html>
            """

            success, error = send_html_email(
                early_access.email, email_subject, email_body
            )
            if not success:
                print(
                    "Warning: Failed to send cancellation confirmation "
                    f"email: {error}"
                )

            return Response({"message": "Subscription successfully cancelled"})

        except EarlyAccessForm.DoesNotExist:
            return Response(
                {"error": "No subscription found or email/token is invalid"},
                status=status.HTTP_404_NOT_FOUND,
            )

    @action(detail=False, methods=["post"])
    def verify_cancellation(self, request):
        """Verify cancellation token and process the cancellation."""
        token = request.query_params.get("token")

        if not token:
            return Response(
                {"error": "Verification token is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Check if token is valid and not expired (24 hours)
            one_day_ago = datetime.now() - timedelta(hours=24)
            early_access = EarlyAccessForm.objects.get(
                email_verification_token=token,
                token_created_at__gte=one_day_ago,
                has_paid=True,
            )

            # Process cancellation
            early_access.has_paid = False
            early_access.payment_status = "cancelled"
            early_access.email_verification_token = None  # Clear the token
            early_access.token_created_at = None
            early_access.save()

            # Send cancellation confirmation email
            email_subject = "Subscription Cancellation Confirmed"
            email_body = """
                <html>
                    <body style="font-family: Arial, sans-serif; color: #333;">
                        <h2>Subscription Cancellation Confirmed</h2>

                        <p>Your subscription has been successfully cancelled.</p>

                        <p>If you change your mind, you can always subscribe again.</p>

                        <p>Thank you for trying our service!</p>

                        <p>Best regards,<br>
                        The Team</p>
                    </body>
                </html>
            """

            success, error = send_html_email(
                early_access.email, email_subject, email_body
            )
            if not success:
                print(
                    "Warning: Failed to send cancellation confirmation "
                    f"email: {error}"
                )

            return Response({"message": "Subscription successfully cancelled"})

        except EarlyAccessForm.DoesNotExist:
            return Response(
                {"error": "Invalid or expired verification token"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="email",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Email address to verify",
                required=True,
            ),
            OpenApiParameter(
                name="token",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Verification token",
                required=True,
            ),
        ],
        responses={
            200: inline_serializer(
                name="EmailVerificationResponse",
                fields={
                    "message": serializers.CharField(),
                    "email": serializers.EmailField(),
                    "is_verified": serializers.BooleanField(),
                },
            ),
            400: OpenApiResponse(
                description="Bad Request",
                response=inline_serializer(
                    name="EmailVerificationError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
            404: OpenApiResponse(
                description="No record found",
                response=inline_serializer(
                    name="EmailVerificationError",
                    fields={
                        "error": serializers.CharField(),
                    },
                ),
            ),
        },
        description="Verify email address using verification token",
    )
    @action(detail=False, methods=["get"])
    def verify_email(self, request):
        """Verify email address using verification token."""
        email = request.query_params.get("email")
        token = request.query_params.get("token")

        if not email or not token:
            logger.error(f"Email verification failed: Missing email or token")
            return Response(
                {"error": "Both email and token are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        logger.info(f"Processing email verification for: {email}")

        try:
            # Check if token is valid
            early_access = EarlyAccessForm.objects.get(
                email=email, email_verification_token=token
            )

            # If email is already verified, just redirect to frontend
            if early_access.is_email_verified:
                logger.info(
                    f"Email {email} is already verified, redirecting to frontend"
                )
                return redirect(f"{settings.FRONTEND_URL}/{email}")

            # Mark email as verified
            early_access.is_email_verified = True
            early_access.save()

            logger.info(f"Email verification successful for: {email}")

            # Send confirmation email
            email_subject = "Email Verification Successful"
            email_body = early_access_success_email_template

            success, error = send_html_email(
                early_access.email, email_subject, email_body
            )
            if not success:
                print(
                    f"Warning: Failed to send verification confirmation email: {error}"
                )
            return redirect(f"{settings.FRONTEND_URL}/signup/{email}")

        except EarlyAccessForm.DoesNotExist:
            return redirect(f"{settings.FRONTEND_URL}/verification-failed")
        except (ValueError, TypeError) as e:
            return redirect(f"{settings.FRONTEND_URL}/verification-failed")
        except Exception as e:
            # Log the unexpected error
            print(f"Unexpected error in verify_email: {e}")
            return redirect(f"{settings.FRONTEND_URL}/verification-failed")


def checkout_success(request):
    """Handle successful checkout and redirect to appropriate page."""
    session_id = request.GET.get("session_id")
    print(f"Checkout success called with session_id: {session_id}")
    print(f"Request URL: {request.build_absolute_uri()}")

    if not session_id:
        print("No session_id provided in early access checkout success")
        return redirect(settings.FRONTEND_URL)

    try:
        # Retrieve the Stripe checkout session
        session = stripe.checkout.Session.retrieve(session_id)
        print(f"Retrieved Stripe session: {session.id}")

        # Find the corresponding early access form
        early_access = EarlyAccessForm.objects.get(
            stripe_checkout_session_id=session.id,
            payment_status="pending",
        )
        print(f"Found early access form: {early_access.id}")

        # Use the shared utility function to process the payment
        process_early_access_payment(early_access, session)
        print(f"Processed payment for early access form: {early_access.id}")

        # Send payment confirmation email
        email_subject = "Payment Successful - Welcome to Early Access!"
        # flake8: noqa: E501
        email_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333;">
                    <h2>Thank you for your payment!</h2>
                    <p>Your early access has been activated.</p>

                    <h3>Payment Details:</h3>
                    <ul>
                        <li>Date: {early_access.payment_date.strftime('%Y-%m-%d %H:%M:%S')}</li>
                        <li>Status: Confirmed</li>
                    </ul>

                    <p>You can now access our platform at:
                    <a href="https://app.example.com">https://app.example.com</a></p>

                    <p>If you have any questions or need assistance,
                    please don't hesitate to contact our
                    support team.</p>

                    <p>Best regards,<br>
                    The Team</p>
                </body>
            </html>
        """

        success, error = send_html_email(early_access.email, email_subject, email_body)
        if not success:
            print(f"Warning: Failed to send payment confirmation email: {error}")

        # Redirect to the frontend with a success parameter
        frontend_success_url = "https://app.example.com"
        return redirect(f"{frontend_success_url}")

    except stripe.error.StripeError as e:
        print(f"Stripe error in early access checkout_success: {e}")
        frontend_error_url = f"{settings.FRONTEND_URL}/early-access/error"
        return redirect(frontend_error_url)
    except EarlyAccessForm.DoesNotExist:
        print(f"No early access form found for session ID: {session_id}")
        frontend_error_url = f"{settings.FRONTEND_URL}/early-access/error"
        return redirect(frontend_error_url)
    except StopIteration as e:
        print(f"Unexpected error in early access checkout_success: {e}")
        frontend_error_url = f"{settings.FRONTEND_URL}/early-access/error"
        return redirect(frontend_error_url)


def checkout_cancel(request):
    """Handle cancelled checkout and redirect to appropriate page."""
    session_id = request.GET.get("session_id")
    if session_id:
        EarlyAccessForm.objects.filter(
            stripe_checkout_session_id=session_id, payment_status="pending"
        ).delete()

    return JsonResponse(
        {
            "status": "cancelled",
            "message": "Checkout was cancelled",
            "redirect_url": f"{settings.FRONTEND_URL}/early-access/cancel",
        }
    )


@method_decorator(csrf_exempt, name="dispatch")
class WebhookView(APIView):
    """Handle Stripe webhooks according to best practices."""

    permission_classes = [permissions.AllowAny]

    @extend_schema(
        description="Handle Stripe webhook events",
        responses={
            200: inline_serializer(
                name="WebhookSuccess",
                fields={
                    "status": serializers.CharField(),
                },
            ),
            400: inline_serializer(
                name="WebhookError",
                fields={
                    "error": serializers.CharField(),
                },
            ),
        },
    )
    def post(self, request):
        """Handle Stripe webhook events.

        Follows Stripe webhook best practices:
        1. Returns 200 response quickly to avoid timeouts
        2. Verifies Stripe signature
        3. Handles specific event types
        4. Proper error logging
        """
        payload = request.body
        sig_header = request.META.get("HTTP_STRIPE_SIGNATURE")
        webhook_secret = settings.STRIPE_WEBHOOK_SECRET

        # Log the webhook request
        logger.info(f"Received webhook with signature: {sig_header[:10]}...")

        if not sig_header:
            logger.error("Webhook error: No signature header provided")
            return Response(
                {"error": "No Stripe signature header provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not webhook_secret:
            logger.error("Webhook error: No webhook secret configured")
            return Response(
                {"error": "Webhook secret not configured"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            # Verify the event came from Stripe
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)

            # Quickly return a 200 response to acknowledge receipt
            # Process the event asynchronously

            # Get event data
            event_type = event["type"]
            event_id = event.get("id", "")
            logger.info(f"Webhook received: {event_type} (ID: {event_id})")

            # Handle different event types
            if event_type == "checkout.session.completed":
                # Process this event immediately but return response quickly
                self._handle_checkout_session_completed(event)
            elif event_type == "payment_intent.succeeded":
                # For any other payment events you want to handle
                logger.info(f"Payment intent success for event ID: {event_id}")
            elif event_type == "charge.succeeded":
                logger.info(f"Charge succeeded for event ID: {event_id}")
            else:
                logger.info(f"Unhandled event type: {event_type}")

            # Always return a success response to Stripe
            return Response({"status": "success"})

        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {str(e)}")
            return Response(
                {"error": "Invalid signature"}, status=status.HTTP_400_BAD_REQUEST
            )
        except json.JSONDecodeError as e:
            logger.error(f"Webhook invalid payload: {str(e)}")
            return Response(
                {"error": "Invalid payload"}, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Webhook error: {str(e)}", exc_info=True)
            return Response(
                {"error": "Internal server error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _handle_checkout_session_completed(self, event):
        """Process checkout.session.completed event.

        This is done in a separate method to allow for future asynchronous processing.
        """
        try:
            event_id = event.get("id")
            session = event["data"]["object"]
            logger.info(
                f"Processing checkout session: {session.id} (Event ID: {event_id})"
            )

            # Store processed event IDs to prevent duplicate processing
            # In a production system, this should be stored in a database or cache
            # For simplicity, we're just checking if the payment is already completed

            # Find the corresponding early access form
            early_access = EarlyAccessForm.objects.filter(
                stripe_checkout_session_id=session.id
            ).first()

            if early_access:
                # Check if already processed to prevent duplicates
                if early_access.payment_status == "paid" and early_access.has_paid:
                    logger.info(f"Session {session.id} already processed. Skipping.")
                    return

                # Use the shared utility function to process the payment
                process_early_access_payment(early_access, session)
                logger.info(
                    f"Successfully processed payment for early access: {early_access.email}"
                )

                # Send payment confirmation email
                email_subject = "Payment Successful - Welcome to Early Access!"
                email_body = f"""
                    <html>
                        <body style="font-family: Arial, sans-serif; color: #333;">
                            <h2>Thank you for your payment!</h2>
                            <p>Your early access has been activated.</p>

                            <h3>Payment Details:</h3>
                            <ul>
                                <li>Date: {early_access.payment_date.strftime('%Y-%m-%d %H:%M:%S')}</li>
                                <li>Status: Confirmed</li>
                            </ul>

                            <p>You can now access our platform at:
                            <a href="https://app.example.com">app.example.com</a></p>

                            <p>If you have any questions or need assistance,
                            please don't hesitate to contact our
                            support team.</p>

                            <p>Best regards,<br>
                            The Team</p>
                        </body>
                    </html>
                """

                success, error = send_html_email(
                    early_access.email, email_subject, email_body
                )
                if not success:
                    logger.error(f"Failed to send confirmation email: {error}")
            else:
                # Check if this might be a regular subscription webhook
                subscription = UserSubscription.objects.filter(
                    stripe_checkout_session_id=session.id
                ).first()

                if subscription:
                    logger.info(
                        "This appears to be a regular subscription webhook, "
                        "not handling here"
                    )
                else:
                    logger.warning(f"Unknown checkout session: {session.id}")
        except Exception as e:
            logger.error(f"Error processing checkout session: {str(e)}", exc_info=True)
            # We don't raise the exception here to avoid sending an error response
            # to Stripe, as we've already sent a success response
