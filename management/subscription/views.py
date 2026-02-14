"""Views for managing subscriptions, checkout process, and webhook handling."""

import traceback

import stripe
from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from drf_spectacular.utils import OpenApiResponse, extend_schema, inline_serializer
from envs.env_loader import env_loader
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.mixins import ListModelMixin, RetrieveModelMixin
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from configurations.pagination import CustomPagination

from .models import SubscriptionProduct, UserSubscription
from .serializers import StripeProductSerializer, UserSubscriptionSerializer

stripe.api_key = env_loader.stripe_secret_key


@extend_schema(tags=["Subscription Products"])
class SubscriptionProductViewSet(viewsets.ModelViewSet):
    """ViewSet for managing subscription products."""

    serializer_class = StripeProductSerializer
    permission_classes = [AllowAny]  # Allow public access to subscription products
    pagination_class = CustomPagination  # Apply pagination

    def get_queryset(self):
        """Return products with their prices directly from Stripe."""
        try:
            products = stripe.Product.list(active=True)
            # Filter out products with no active prices
            products_with_prices = []
            for product in products.data:
                prices = stripe.Price.list(product=product.id, active=True)
                if prices.data:
                    products_with_prices.append(product)
            return products_with_prices
        except stripe.error.StripeError as e:
            print(f"Stripe error: {e}")
            return []

    def list(self, request, *args, **kwargs):
        """List all active products from Stripe."""
        try:
            products = self.get_queryset()

            # Apply pagination to the Stripe products
            page = self.paginate_queryset(products)
            if page is not None:
                serializer = self.get_serializer(page, many=True)
                return self.get_paginated_response(serializer.data)

            # If pagination is disabled
            serializer = self.get_serializer(products, many=True)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def create(self, request, *args, **kwargs):
        """Create a product in both Stripe and database."""
        try:
            # Validate price
            try:
                price = float(request.data.get("price", 0))
                if price <= 0:
                    return Response(
                        {"error": "Price must be greater than 0"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            except (ValueError, TypeError):
                return Response(
                    {"error": "Invalid price format"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Create in Stripe first
            stripe_product = stripe.Product.create(
                name=request.data["name"],
                description=request.data["description"],
                metadata={
                    "duration_months": str(request.data.get("duration_months", 1)),
                    "regular_price": str(request.data.get("regular_price", 0)),
                    "searches_per_month": str(
                        request.data.get("searches_per_month", 0)
                    ),
                    "active_projects": str(request.data.get("active_projects", 0)),
                    "early_adopter_benefits": str(
                        request.data.get("early_adopter_benefits", False)
                    ),
                    "community_access": str(
                        request.data.get("community_access", False)
                    ),
                    "priority_support": str(
                        request.data.get("priority_support", False)
                    ),
                },
            )

            # Create price in Stripe
            stripe_price = stripe.Price.create(
                product=stripe_product.id,
                unit_amount=int(price * 100),  # Convert to cents
                currency="usd",
            )

            # Create in database
            SubscriptionProduct.objects.create(
                name=request.data["name"],
                description=request.data["description"],
                price=price,
                duration_months=int(request.data.get("duration_months", 1)),
                regular_price=float(request.data.get("regular_price", 0)),
                searches_per_month=int(request.data.get("searches_per_month", 0)),
                active_projects=int(request.data.get("active_projects", 0)),
                early_adopter_benefits=bool(
                    request.data.get("early_adopter_benefits", False)
                ),
                community_access=bool(request.data.get("community_access", False)),
                priority_support=bool(request.data.get("priority_support", False)),
                stripe_product_id=stripe_product.id,
                stripe_price_id=stripe_price.id,
            )

            serializer = self.get_serializer(stripe_product)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except stripe.error.StripeError as e:
            return Response(
                {"error": f"Stripe error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a product directly from Stripe."""
        try:
            product = stripe.Product.retrieve(kwargs["pk"])
            if not product.active:
                return Response(
                    {"error": "This product is no longer active."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            serializer = self.get_serializer(product)
            return Response(serializer.data)
        except stripe.error.StripeError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)

    @extend_schema(
        request=inline_serializer(
            name="CheckoutRequest",
            fields={
                "price_id": serializers.CharField(
                    help_text="Stripe price ID to purchase"
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
                    name="CheckoutResponse",
                    fields={
                        "checkout_url": serializers.CharField(),
                        "session_id": serializers.CharField(),
                    },
                ),
            ),
            400: OpenApiResponse(description="Bad Request"),
            404: OpenApiResponse(description="Product not found"),
        },
        description="Create a Stripe Checkout Session for the subscription product",
    )
    @action(detail=False, methods=["post"])
    def checkout(self, request):
        """Create a Stripe Checkout Session for the subscription product."""
        if request.user.is_staff:
            return Response(
                {"error": "Admin users cannot create checkouts."},
                status=status.HTTP_403_FORBIDDEN,
            )

        price_id = request.data.get("price_id")
        if not price_id:
            return Response(
                {"error": "Price ID is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Verify the price exists and get its product
            price = stripe.Price.retrieve(price_id)
            stripe_product = stripe.Product.retrieve(price.product)

            if not stripe_product.active:
                return Response(
                    {"error": "This product is no longer active."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Calculate duration in months based on the price type
            duration_months = 1  # Default for one-time payments
            if price.recurring:
                if price.recurring.interval == "year":
                    duration_months = 12
                elif price.recurring.interval == "month":
                    duration_months = 1

            # Get or create the SubscriptionProduct instance
            try:
                subscription_product = SubscriptionProduct.get_or_sync_product(
                    price_id
                )  # Pass price_id instead of product_id
            except ValueError as e:
                return Response(
                    {"error": f"Invalid product: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Check if user already has an active subscription
            active_subscription = UserSubscription.objects.filter(
                user=request.user, status="active"
            ).first()

            if active_subscription:
                return Response(
                    {
                        "error": "You already have an active subscription. "
                        "Please cancel it before purchasing a new one."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Delete any existing pending subscriptions for this user
            UserSubscription.objects.filter(
                user=request.user, status="pending"
            ).delete()

            success_url = request.data.get(
                "success_url",
                request.build_absolute_uri(reverse("subscription:checkout-success")),
            )
            cancel_url = request.data.get(
                "cancel_url",
                request.build_absolute_uri(reverse("subscription:checkout-cancel")),
            )

            # Create a new subscription record with the correct duration
            subscription = UserSubscription.objects.create(
                user=request.user,
                product=subscription_product,
                status="pending",
            )

            # Set the checkout session mode based on the price type
            mode = "subscription" if price.recurring else "payment"

            # Create Stripe checkout session
            checkout_session = stripe.checkout.Session.create(
                customer_email=request.user.email,
                payment_method_types=["card"],
                line_items=[
                    {
                        "price": price_id,
                        "quantity": 1,
                    }
                ],
                mode=mode,
                success_url=f"{success_url}?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{cancel_url}?session_id={{CHECKOUT_SESSION_ID}}",
                metadata={
                    "subscription_id": str(subscription.id),
                    "user_id": str(request.user.id),
                    "product_id": stripe_product.id,
                    "price_id": price_id,
                    "is_recurring": str(bool(price.recurring)),
                    "interval": price.recurring.interval
                    if price.recurring
                    else "one_time",
                    "duration_months": str(duration_months),
                },
            )

            # Update subscription with checkout session ID
            subscription.stripe_checkout_session_id = checkout_session.id
            subscription.save()

            return Response(
                {
                    "checkout_url": checkout_session.url,
                    "session_id": checkout_session.id,
                }
            )

        except stripe.error.StripeError as e:
            print(f"Stripe error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            return Response(
                {"error": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def get_prices(self, stripe_product):
        """Get all prices for the product."""
        try:
            # Get all active prices for this product
            prices = stripe.Price.list(
                product=stripe_product.id, active=True, expand=["data.product"]
            )

            return [
                {
                    "id": price.id,
                    "unit_amount": price.unit_amount / 100,  # Convert cents to dollars
                    "currency": price.currency,
                    "recurring": {
                        "interval": price.recurring.interval
                        if price.recurring
                        else None,
                        "interval_count": price.recurring.interval_count
                        if price.recurring
                        else None,
                    }
                    if price.recurring
                    else None,
                }
                for price in prices.data
            ]
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return []


@extend_schema(tags=["User Subscriptions"])
class UserSubscriptionViewSet(ListModelMixin, RetrieveModelMixin, GenericViewSet):
    """ViewSet for managing user subscriptions.

    Users can view their own subscriptions.
    Admin users can view all subscriptions.
    """

    serializer_class = UserSubscriptionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Return subscriptions for the current user."""
        if self.request.user.is_staff:
            return UserSubscription.objects.all()
        return UserSubscription.objects.filter(user=self.request.user)

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a subscription using product_id."""
        try:
            # First verify the product exists in Stripe
            try:
                stripe_product = stripe.Product.retrieve(kwargs["pk"])
                if not stripe_product.active:
                    return Response(
                        {"error": "This product is no longer active."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            except stripe.error.StripeError as e:
                return Response(
                    {"error": f"Invalid product ID: {str(e)}"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Find the subscription for this user and product
            subscription = UserSubscription.objects.filter(
                user=request.user, product__stripe_product_id=kwargs["pk"]
            ).first()

            if not subscription:
                return Response(
                    {"error": "No subscription found for this product."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            serializer = self.get_serializer(subscription)
            return Response(serializer.data)

        except Exception as e:
            print(f"Error retrieving subscription: {e}")
            traceback.print_exc()
            return Response(
                {
                    "error": "An unexpected error occurred while retrieving the subscription."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        methods=["GET"],
        responses={
            200: OpenApiResponse(
                description="Subscription cancelled successfully",
                response=inline_serializer(
                    name="CancelResponse",
                    fields={
                        "message": serializers.CharField(),
                    },
                ),
            ),
            400: OpenApiResponse(description="Bad Request"),
            403: OpenApiResponse(description="Forbidden"),
        },
        description="Cancel a user's subscription",
    )
    @action(detail=True, methods=["get"])
    def cancel(self, request, pk=None):
        """Cancel a user's subscription."""
        try:
            # First verify the product exists in Stripe
            try:
                stripe_product = stripe.Product.retrieve(pk)
                if not stripe_product.active:
                    return Response(
                        {"error": "This product is no longer active."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            except stripe.error.StripeError as e:
                return Response(
                    {"error": f"Invalid product ID: {str(e)}"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Find the subscription for this user and product
            subscription = UserSubscription.objects.filter(
                user=request.user, product__stripe_product_id=pk, status="active"
            ).first()

            if not subscription:
                return Response(
                    {"error": "No active subscription found for this product."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            if subscription.user != request.user or request.user.is_staff:
                return Response(
                    {
                        "error": "You do not have permission to cancel this subscription."
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            subscription.status = "cancelled"
            subscription.save()

            return Response(
                {
                    "message": "Subscription cancelled successfully.",
                    "product_name": subscription.product.name,
                    "cancelled_at": timezone.now().isoformat(),
                }
            )

        except Exception as e:
            print(f"Error cancelling subscription: {e}")
            traceback.print_exc()
            return Response(
                {
                    "error": "An unexpected error occurred while cancelling the subscription."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        responses={
            200: OpenApiResponse(
                description="Payment history retrieved successfully",
                response=inline_serializer(
                    name="PaymentHistoryResponse",
                    fields={
                        "payments": serializers.ListField(
                            child=serializers.DictField(
                                child=serializers.CharField(),
                            )
                        ),
                    },
                ),
            ),
        },
        description="Retrieve payment history. Admins can see all users' payments.",
    )
    @action(detail=False, methods=["get"])
    def payment_history(self, request):
        """Retrieve payment history. Admins see all payments, users see only own."""
        if request.user.is_staff:
            # Admin users can see all payments
            queryset = (
                UserSubscription.objects.all()
                .exclude(stripe_payment_intent_id__isnull=True)
                .order_by("-start_date")
            )
        else:
            # Regular users can only see their own payments
            queryset = UserSubscription.objects.filter(
                user=request.user, stripe_payment_intent_id__isnull=False
            ).order_by("-start_date")

        payments = []
        for subscription in queryset:
            try:
                # Retrieve payment intent details from Stripe
                payment_intent = stripe.PaymentIntent.retrieve(
                    subscription.stripe_payment_intent_id
                )

                payment_data = {
                    "subscription_id": subscription.id,
                    "product_name": subscription.product.name,
                    "amount": payment_intent.amount / 100,  # Convert cents to dollars
                    "currency": payment_intent.currency,
                    "status": payment_intent.status,
                    "payment_date": subscription.start_date.isoformat(),
                    "payment_method": payment_intent.payment_method_types[0],
                }

                # Add user details only for admin users
                if request.user.is_staff:
                    payment_data.update(
                        {
                            "user_email": subscription.user.email,
                            "user_id": subscription.user.id,
                        }
                    )

                payments.append(payment_data)

            except stripe.error.StripeError as e:
                print(f"Error retrieving {subscription.stripe_payment_intent_id}: {e}")
                continue

        return Response({"payments": payments})

    @extend_schema(
        responses={
            200: OpenApiResponse(
                description="Active subscription timer details",
                response=inline_serializer(
                    name="SubscriptionTimerResponse",
                    fields={
                        "has_active_subscription": serializers.BooleanField(),
                        "product_name": serializers.CharField(allow_null=True),
                        "start_date": serializers.DateTimeField(allow_null=True),
                        "end_date": serializers.DateTimeField(allow_null=True),
                        "days_remaining": serializers.IntegerField(allow_null=True),
                        "total_days": serializers.IntegerField(allow_null=True),
                        "percentage_remaining": serializers.FloatField(allow_null=True),
                    },
                ),
            ),
        },
        description="Get details about the user's active subscription timer",
    )
    @action(detail=False, methods=["get"])
    def timer(self, request):
        """Get the remaining time for user's active subscription."""
        active_subscription = UserSubscription.objects.filter(
            user=request.user, status="active", end_date__gt=timezone.now()
        ).first()

        if not active_subscription:
            return Response(
                {
                    "has_active_subscription": False,
                    "product_name": None,
                    "start_date": None,
                    "end_date": None,
                    "days_remaining": None,
                    "total_days": None,
                    "percentage_remaining": None,
                }
            )

        # Calculate days remaining
        now = timezone.now()
        total_duration = (
            active_subscription.end_date - active_subscription.start_date
        ).days
        days_remaining = (active_subscription.end_date - now).days
        percentage_remaining = (
            (days_remaining / total_duration) * 100 if total_duration > 0 else 0
        )

        return Response(
            {
                "has_active_subscription": True,
                "product_name": active_subscription.product.name,
                "start_date": active_subscription.start_date,
                "end_date": active_subscription.end_date,
                "days_remaining": max(0, days_remaining),  # Ensure non-negative
                "total_days": total_duration,
                "percentage_remaining": round(
                    max(0, percentage_remaining), 2
                ),  # Round to 2 decimal places
            }
        )


@csrf_exempt
@require_POST
def stripe_webhook(request):
    """Handle Stripe webhook events."""
    payload = request.body
    sig_header = request.META.get("HTTP_STRIPE_SIGNATURE")

    try:
        # Verify the webhook signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        print(f"Invalid webhook payload: {e}")
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except stripe.error.SignatureVerificationError as e:
        print(f"Invalid webhook signature: {e}")
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    # Handle checkout.session.completed event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        print(f"Processing checkout.session.completed for session: {session.id}")

        # Find the corresponding subscription
        subscription = UserSubscription.objects.filter(
            stripe_checkout_session_id=session.id
        ).first()

        if subscription and session.payment_status == "paid":
            # Use the shared utility function to process the payment
            from .utils import process_subscription_payment

            process_subscription_payment(subscription, session)
        else:
            # Check if this might be an early access checkout
            try:
                from management.forms.earlyaccess_form.models import EarlyAccessForm

                from .utils import process_early_access_payment

                early_access = EarlyAccessForm.objects.filter(
                    stripe_checkout_session_id=session.id
                ).first()

                if early_access and session.payment_status == "paid":
                    # Process early access payment
                    process_early_access_payment(early_access, session)
                    print(
                        "Processed early access payment via subscription webhook: "
                        f"{early_access}"
                    )
                elif not subscription and not early_access:
                    print(f"Unknown checkout session in webhook: {session.id}")
            except ImportError:
                # Early access module not available
                print("Early access module not available")

    return Response({"status": "success"})


def checkout_success(request):
    """Handle successful checkout and redirect to appropriate page."""
    session_id = request.GET.get("session_id")
    print(f"\n\nSession ID: {session_id}\n\n")
    if not session_id:
        print("No session_id provided in checkout success")
        return redirect(settings.FRONTEND_URL)

    try:
        # Retrieve the Stripe checkout session
        session = stripe.checkout.Session.retrieve(session_id)

        # Find the corresponding subscription
        subscription = UserSubscription.objects.get(
            stripe_checkout_session_id=session.id,
            status="pending",
        )

        # Use the shared utility function to process the payment
        from .utils import process_subscription_payment

        process_subscription_payment(subscription, session)

        # Redirect to the frontend application
        return redirect(settings.FRONTEND_URL)

    except stripe.error.StripeError as e:
        print(f"Stripe error in checkout_success: {e}")
        traceback.print_exc()
        return redirect(f"{settings.FRONTEND_URL}/subscription/error")
    except UserSubscription.DoesNotExist:
        # Check if this might be an early access checkout
        try:
            from management.forms.earlyaccess_form.models import EarlyAccessForm

            early_access = EarlyAccessForm.objects.filter(
                stripe_checkout_session_id=session_id
            ).first()

            if early_access:
                # Redirect to early access success page
                return redirect(f"{settings.FRONTEND_URL}/early-access/success")
            else:
                print(
                    "No subscription or early access found for session ID: "
                    f"{session_id}"
                )
        except ImportError:
            # Early access module not available
            print("Early access module not available")

        return redirect(f"{settings.FRONTEND_URL}/subscription/error")
    except Exception as e:
        print(f"Unexpected error in checkout_success: {e}")
        traceback.print_exc()
        return redirect(f"{settings.FRONTEND_URL}/subscription/error")


def checkout_cancel(request):
    """Handle cancelled checkout and redirect to appropriate page."""
    session_id = request.GET.get("session_id")
    if session_id:
        UserSubscription.objects.filter(
            stripe_checkout_session_id=session_id, status="pending"
        ).delete()

    return redirect(f"{settings.FRONTEND_URL}/subscription")
