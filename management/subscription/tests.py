"""Tests for the subscription app."""

from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

from .models import SubscriptionProduct, UserSubscription

User = get_user_model()


class AffiliateReferralTests(TestCase):
    """Tests for the affiliate referral functionality."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpassword"
        )
        self.client.force_authenticate(user=self.user)

        # Create a subscription product
        self.product = SubscriptionProduct.objects.create(
            name="Test Product",
            description="Test Description",
            price=100.00,
            duration_months=1,
            regular_price=120.00,
            searches_per_month=100,
            active_projects=5,
            stripe_product_id="prod_test",
            stripe_price_id="price_test",
        )

    @patch("stripe.checkout.Session.create")
    def test_checkout_with_referral(self, mock_create):
        """Test creating a checkout session with a referral ID."""
        # Mock the Stripe checkout session
        mock_session = MagicMock()
        mock_session.id = "cs_test"
        mock_session.url = "https://checkout.stripe.com/test"
        mock_create.return_value = mock_session

        # Make the checkout request with a referral ID
        url = reverse("subscription:subscription-product-checkout")
        data = {"product_id": self.product.id, "ref": "ref_test123"}
        response = self.client.post(url, data, format="json")

        # Check the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["session_id"], "cs_test")

        # Check that the subscription was created with the referral ID
        subscription = UserSubscription.objects.get(
            user=self.user, product=self.product, stripe_checkout_session_id="cs_test"
        )
        self.assertEqual(subscription.referral_id, "ref_test123")

        # Check that the referral ID was included in the Stripe metadata
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        self.assertEqual(kwargs["metadata"]["ref"], "ref_test123")

    @patch("stripe.Webhook.construct_event")
    @patch("stripe.PaymentIntent.retrieve")
    @patch("requests.post")
    def test_webhook_with_referral(self, mock_post, mock_retrieve, mock_construct):
        """Test the webhook handler with a referral ID."""
        # Create a subscription with a referral ID
        subscription = UserSubscription.objects.create(
            user=self.user,
            product=self.product,
            stripe_checkout_session_id="cs_test",
            status="pending",
            referral_id="ref_test123",
        )

        # Mock the Stripe event
        mock_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_test",
                    "payment_status": "paid",
                    "payment_intent": "pi_test",
                }
            },
        }
        mock_construct.return_value = mock_event

        # Mock the payment intent
        mock_payment = MagicMock()
        mock_payment.amount = 10000  # $100.00 in cents
        mock_payment.currency = "usd"
        mock_payment.payment_method_types = ["card"]
        mock_retrieve.return_value = mock_payment

        # Mock the Rewardful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Make the webhook request
        client = Client()
        url = reverse("subscription:stripe-webhook")
        response = client.post(
            url,
            content_type="application/json",
            data="{}",
            HTTP_STRIPE_SIGNATURE="test_signature",
        )

        # Check the response
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check that the subscription was updated
        subscription.refresh_from_db()
        self.assertEqual(subscription.status, "active")

        # Check that the Rewardful API was called
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://api.getrewardful.com/v1/referrals")
        self.assertEqual(kwargs["json"]["referral_id"], "ref_test123")
        self.assertEqual(kwargs["json"]["invoice"]["commission_value"], 30)
