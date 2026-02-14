"""Utility functions for subscription and checkout handling."""

from datetime import timedelta

import requests
import stripe
from django.utils import timezone
from envs.env_loader import env_loader

from subscription.models import SubscriptionProduct

stripe.api_key = env_loader.stripe_secret_key


def create_checkout_session(
    product,
    success_url,
    cancel_url,
    metadata=None,
    customer_email=None,
    user=None,
    referral_id=None,
):
    """Create a Stripe checkout session for a product.

    Args:
        product: The SubscriptionProduct to purchase
        success_url: URL to redirect after successful payment
        cancel_url: URL to redirect after cancelled payment
        metadata: Additional metadata to store with the session
        customer_email: Email address of the customer (required if user is None)
        user: User object (optional, used for regular subscriptions)
        referral_id: Rewardful referral ID (optional)

    Returns:
        tuple: (checkout_session, subscription_obj or None)
    """
    if not metadata:
        metadata = {}

    # Add product ID to metadata
    metadata["product_id"] = (
        product.id if hasattr(product, "id") else product.stripe_product_id
    )

    # Add referral ID to metadata if available
    if referral_id:
        metadata["ref"] = referral_id

    # Ensure success_url and cancel_url are properly formatted
    # Use a simple string append for the session ID parameter to avoid formatting issues
    success_url_with_session = success_url + "?session_id={CHECKOUT_SESSION_ID}"
    cancel_url_with_session = cancel_url + "?session_id={CHECKOUT_SESSION_ID}"

    # Prepare checkout session parameters
    session_params = {
        "payment_method_types": ["card"],
        "line_items": [
            {
                "price": product.stripe_price_id,
                "quantity": 1,
            }
        ],
        "mode": "payment",
        "success_url": success_url_with_session,
        "cancel_url": cancel_url_with_session,
        "metadata": metadata,
    }

    # Add customer email
    if customer_email:
        session_params["customer_email"] = customer_email
    elif user:
        session_params["customer_email"] = user.email
        metadata["user_id"] = user.id

    # Create the checkout session
    checkout_session = stripe.checkout.Session.create(**session_params)

    # Create subscription record if user is provided (regular subscription flow)
    subscription_obj = None
    if user and hasattr(product, "id"):
        from .models import UserSubscription

        subscription_obj = UserSubscription.objects.create(
            user=user,
            product=product,
            stripe_checkout_session_id=checkout_session.id,
            status="pending",
            referral_id=referral_id,
        )

    return checkout_session, subscription_obj


def process_subscription_payment(subscription, session):
    """Process a successful subscription payment.

    Args:
        subscription: UserSubscription object
        session: Stripe checkout session
    """
    # Update subscription status and details
    subscription.status = "active"
    subscription.stripe_payment_intent_id = session.payment_intent
    subscription.start_date = timezone.now()

    # Ensure the duration calculation doesn't overflow
    try:
        days = min(
            subscription.product.duration_months * 30, 2147483647
        )  # Max C int value
        subscription.end_date = timezone.now() + timedelta(days=days)
    except OverflowError:
        # Fallback to a safe maximum (10 years)
        subscription.end_date = timezone.now() + timedelta(days=3650)

    # Handle referral ID from metadata if not already set
    if (
        session.metadata
        and session.metadata.get("ref")
        and not subscription.referral_id
    ):
        referral_id = session.metadata.get("ref")
        print(f"Found referral ID in webhook metadata: {referral_id}")
        subscription.referral_id = referral_id

    subscription.save()
    print(f"Subscription activated: {subscription}")

    # Process affiliate commission via Rewardful if there's a referral ID
    if (
        subscription.referral_id
        and hasattr(env_loader, "rewardful_api_key")
        and env_loader.rewardful_api_key
    ):
        affiliate_info = notify_rewardful(
            referral_id=subscription.referral_id,
            payment_intent_id=session.payment_intent,
            customer_email=subscription.user.email,
            customer_id=str(subscription.user.id),
            customer_name=subscription.user.get_full_name()
            or subscription.user.username,
            product_name=subscription.product.name,
            product_id=str(subscription.product.id),
        )

        # Save affiliate info to subscription metadata
        if affiliate_info:
            subscription.affiliate_name = affiliate_info["name"]
            subscription.affiliate_email = affiliate_info["email"]
            subscription.save()


def process_early_access_payment(early_access, session):
    """Process a successful early access payment.

    Args:
        early_access: EarlyAccessForm object
        session: Stripe checkout session
    """
    # Update early access form status
    early_access.payment_status = "paid"
    early_access.has_paid = True
    early_access.stripe_payment_intent_id = session.payment_intent
    early_access.payment_date = timezone.now()

    # Handle referral ID from metadata if not already set
    if (
        session.metadata
        and session.metadata.get("ref")
        and not hasattr(early_access, "referral_id")
    ):
        print(
            "Early access form doesn't have referral_id field. "
            "Consider adding it to the model."
        )
    elif (
        session.metadata
        and session.metadata.get("ref")
        and not early_access.referral_id
    ):
        referral_id = session.metadata.get("ref")
        print(f"Found referral ID in webhook metadata for early access: {referral_id}")
        early_access.referral_id = referral_id

    early_access.save()
    print(f"Early access payment completed: {early_access}")

    # Process affiliate commission via Rewardful if there's a referral ID
    if (
        hasattr(early_access, "referral_id")
        and early_access.referral_id
        and hasattr(env_loader, "rewardful_api_key")
        and env_loader.rewardful_api_key
    ):
        try:
            product = SubscriptionProduct.objects.get(
                stripe_product_id=early_access.product_id
            )

            affiliate_info = notify_rewardful(
                referral_id=early_access.referral_id,
                payment_intent_id=session.payment_intent,
                customer_email=early_access.email,
                customer_id=str(early_access.id),
                customer_name=early_access.email.split("@")[0],
                product_name=product.name,
                product_id=product.stripe_product_id,
            )

            # Save affiliate info to early access metadata
            if affiliate_info:
                early_access.affiliate_name = affiliate_info["name"]
                early_access.affiliate_email = affiliate_info["email"]
                early_access.save()

        except SubscriptionProduct.DoesNotExist:
            print(f"Product not found for early access: {early_access.product_id}")


def notify_rewardful(
    referral_id,
    payment_intent_id,
    customer_email,
    customer_id,
    customer_name,
    product_name,
    product_id,
):
    """Notify Rewardful about a successful payment and return affiliate info.

    Args:
        referral_id: Rewardful referral ID
        payment_intent_id: Stripe payment intent ID
        customer_email: Customer's email address
        customer_id: Customer's ID in our system
        customer_name: Customer's name
        product_name: Name of the purchased product
        product_id: ID of the purchased product

    Returns:
        dict: Affiliate information including name and email, or None if not found
    """
    try:
        # Get payment details from Stripe
        payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        amount = payment_intent.amount / 100  # Convert cents to dollars

        print(f"Processing Rewardful commission for referral ID: {referral_id}")

        # Get affiliate info first using Bearer auth
        headers = {
            "Authorization": f"Bearer {env_loader.rewardful_api_key}",
            "Content-Type": "application/json",
        }

        referral_response = requests.get(
            f"https://api.getrewardful.com/v1/referrals/{referral_id}",
            headers=headers,
            timeout=10,
        )

        affiliate_info = None
        if referral_response.status_code == 200:
            referral_data = referral_response.json()
            affiliate = referral_data.get("affiliate", {})
            if affiliate:
                name = (
                    f"{affiliate.get('first_name', '')} "
                    f"{affiliate.get('last_name', '')}".strip()
                )
                affiliate_info = {
                    "name": name,
                    "email": affiliate.get("email"),
                }

        # Prepare data payload for Rewardful commission
        rewardful_data = {
            "referral_id": referral_id,
            "invoice": {
                "external_id": payment_intent_id,
                "amount": amount,
                "currency": payment_intent.currency,
                "commission_type": "percentage",
                "commission_value": 30,  # 30% commission
                "customer": {
                    "email": customer_email,
                    "external_id": customer_id,
                    "name": customer_name,
                },
                "products": [
                    {"name": product_name, "external_id": product_id, "amount": amount}
                ],
            },
        }

        # Send the commission request to Rewardful API
        response = requests.post(
            "https://api.getrewardful.com/v1/commissions",
            json=rewardful_data,
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            print(f"Successfully notified Rewardful for referral: {referral_id}")
        else:
            print(f"Rewardful API error: {response.status_code} - {response.text}")

        return affiliate_info

    except Exception as e:
        print(f"Error in notify_rewardful: {e}")
        return None


def get_rewardful_affiliate_info(referral_id):
    """Get affiliate information from Rewardful API.

    First gets referral info, then finds the affiliate details.

    Args:
        referral_id: Rewardful referral ID

    Returns:
        dict: Affiliate information including name, or None if not found
    """
    if not (
        hasattr(env_loader, "rewardful_api_secret") and env_loader.rewardful_api_secret
    ):
        print("Rewardful API secret not configured")
        return None

    try:
        # Use HTTP Basic Auth with API secret as username and empty password
        auth = (env_loader.rewardful_api_secret, "")

        # First get referral information
        referral_response = requests.get(
            f"https://api.getrewardful.com/v1/referrals/{referral_id}",
            auth=auth,
            timeout=10,
        )

        # print(f"Rewardful API response: {referral_response.text}")

        if referral_response.status_code == 200:
            try:
                referral_data = referral_response.json()
                print(f"Rewardful API response: {referral_data}")
                affiliate = referral_data.get("affiliate", {})

                if affiliate:
                    name = (
                        f"{affiliate.get('first_name', '')} "
                        f"{affiliate.get('last_name', '')}".strip()
                    )
                    return {
                        "name": name,
                        "email": affiliate.get("email"),
                    }

                print("No affiliate found in referral data")
                return None

            except ValueError as e:
                print(f"Error parsing JSON response: {e}")
                return None

        print(f"Error: {referral_response.status_code} - {referral_response.text}")
        return None

    except requests.RequestException as e:
        print(f"Network error when contacting Rewardful: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error getting affiliate info: {e}")
        return None
