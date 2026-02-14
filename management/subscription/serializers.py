"""Serializers for subscription-related models."""

import stripe
from django.utils import timezone
from rest_framework import serializers

from .models import SubscriptionProduct, UserSubscription


class SubscriptionProductSerializer(serializers.ModelSerializer):
    """Serializer for subscription products."""

    monthly_price = serializers.SerializerMethodField()
    discount_percentage = serializers.SerializerMethodField()

    class Meta:
        """Meta options for SubscriptionProductSerializer."""

        model = SubscriptionProduct
        fields = [
            "id",
            "name",
            "description",
            "price",
            "duration_months",
            "monthly_price",
            "regular_price",
            "discount_percentage",
            "searches_per_month",
            "active_projects",
            "early_adopter_benefits",
            "community_access",
            "priority_support",
            "stripe_product_id",
            "stripe_price_id",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "stripe_product_id",
            "stripe_price_id",
            "created_at",
            "updated_at",
        ]

    def get_monthly_price(self, obj):
        """Calculate the approximate monthly price."""
        return (
            float(obj.price)
            if obj.duration_months <= 0
            else round(float(obj.price) / obj.duration_months, 2)
        )

    def get_discount_percentage(self, obj):
        """Calculate the discount percentage from regular price."""
        regular_total = float(obj.regular_price) * obj.duration_months
        if regular_total <= 0 or not obj.price:
            return 0
        return round(((regular_total - float(obj.price)) / regular_total) * 100)

    def validate(self, data):
        """Validate that the price is less than the regular price * duration."""
        price = data.get("price", 0)
        regular_price = data.get("regular_price", 0)
        duration_months = data.get("duration_months", 1)

        if (
            all([price, regular_price, duration_months])
            and float(price) > float(regular_price) * duration_months
        ):
            raise serializers.ValidationError(
                "Price cannot exceed regular price times duration."
            )
        return data


class UserSubscriptionSerializer(serializers.ModelSerializer):
    """Serializer for user subscriptions."""

    product_details = SubscriptionProductSerializer(source="product", read_only=True)
    user_email = serializers.EmailField(source="user.email", read_only=True)
    days_remaining = serializers.SerializerMethodField()

    class Meta:
        """Meta options for UserSubscriptionSerializer."""

        model = UserSubscription
        fields = [
            "id",
            "user",
            "user_email",
            "product",
            "product_details",
            "status",
            "start_date",
            "end_date",
            "days_remaining",
            "referral_id",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "user",
            "start_date",
            "end_date",
            "created_at",
            "updated_at",
        ]

    def get_days_remaining(self, obj):
        """Calculate days remaining in the subscription."""
        if obj.status != "active" or not obj.end_date:
            return 0

        now = timezone.now()
        if now > obj.end_date:
            return 0

        return (obj.end_date - now).days


class StripeProductSerializer(serializers.Serializer):
    """Serializer for Stripe products."""

    id = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField(allow_null=True)
    prices = serializers.SerializerMethodField()
    metadata = serializers.DictField(required=False)

    def get_prices(self, stripe_product):
        """Get all prices for the product."""
        try:
            # Get all active prices for this product (remove the limit=1)
            prices = stripe.Price.list(product=stripe_product.id, active=True)

            return [
                {
                    "id": price.id,
                    "unit_amount": price.unit_amount / 100,  # Convert cents to dollars
                    "currency": price.currency,
                    "recurring": {
                        "interval": price.recurring.interval,
                        "interval_count": price.recurring.interval_count,
                    }
                    if price.recurring
                    else None,
                }
                for price in prices.data
            ]
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return []

    def to_representation(self, stripe_product):
        """Convert Stripe product to serialized format."""
        metadata = stripe_product.metadata

        return {
            "id": stripe_product.id,
            "name": stripe_product.name,
            "description": stripe_product.description,
            "prices": self.get_prices(stripe_product),
            "duration_months": int(metadata.get("duration_months", 1)),
            "regular_price": float(metadata.get("regular_price", 0)),
            "searches_per_month": int(metadata.get("searches_per_month", 0)),
            "active_projects": int(metadata.get("active_projects", 0)),
            "early_adopter_benefits": metadata.get(
                "early_adopter_benefits", "false"
            ).lower()
            == "true",
            "community_access": metadata.get("community_access", "false").lower()
            == "true",
            "priority_support": metadata.get("priority_support", "false").lower()
            == "true",
        }
