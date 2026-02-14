"""Models for managing subscription products and user subscriptions."""

import stripe
from django.conf import settings
from django.db import models
from django.utils import timezone
from djstripe.models import Product as StripeProduct
from envs.env_loader import env_loader

stripe.api_key = env_loader.stripe_secret_key


class SubscriptionProduct(models.Model):
    """Model for subscription product offerings with pricing and features."""

    stripe_product = models.ForeignKey(
        StripeProduct,
        on_delete=models.SET_NULL,
        null=True,
        related_name="subscription_products",
        to_field="id",
    )
    name = models.CharField(max_length=100)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    duration_months = models.PositiveIntegerField()
    regular_price = models.DecimalField(max_digits=10, decimal_places=2)
    searches_per_month = models.PositiveIntegerField()
    active_projects = models.PositiveIntegerField()
    early_adopter_benefits = models.BooleanField(default=False)
    community_access = models.BooleanField(default=False)
    priority_support = models.BooleanField(default=False)

    # Only keep stripe_price_id
    stripe_price_id = models.CharField(max_length=100, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        """Override save to create or update Stripe product and price."""
        try:
            if not self.stripe_product:
                # Create Stripe product with all details in metadata
                stripe_product = stripe.Product.create(
                    name=self.name,
                    description=self.description,
                    metadata={
                        "duration_months": str(self.duration_months),
                        "regular_price": str(self.regular_price),
                        "searches_per_month": str(self.searches_per_month),
                        "active_projects": str(self.active_projects),
                        "early_adopter_benefits": str(self.early_adopter_benefits),
                        "community_access": str(self.community_access),
                        "priority_support": str(self.priority_support),
                        "django_product_id": str(self.id) if self.id else None,
                        "feature_searches": f"{self.searches_per_month} searches per month",
                        "feature_projects": f"{self.active_projects} active projects",
                        "feature_early_adopter": "Early adopter benefits"
                        if self.early_adopter_benefits
                        else "",
                        "feature_community": "Community access"
                        if self.community_access
                        else "",
                        "feature_support": "Priority support"
                        if self.priority_support
                        else "",
                    },
                )
                # Sync the product to dj-stripe and set the relationship
                from djstripe.models import Product

                self.stripe_product = Product.sync_from_stripe_data(stripe_product)

            if not self.stripe_price_id:
                # Create price in Stripe
                stripe_price = stripe.Price.create(
                    product=self.stripe_product.id,
                    unit_amount=int(self.price * 100),
                    currency="usd",
                    metadata={
                        "duration_months": str(self.duration_months),
                        "regular_price": str(self.regular_price),
                    },
                )
                self.stripe_price_id = stripe_price.id

            # If product exists but details changed, update it
            elif self.stripe_product:
                stripe.Product.modify(
                    self.stripe_product.id,
                    name=self.name,
                    description=self.description,
                    metadata={
                        "duration_months": str(self.duration_months),
                        "regular_price": str(self.regular_price),
                        "searches_per_month": str(self.searches_per_month),
                        "active_projects": str(self.active_projects),
                        "early_adopter_benefits": str(self.early_adopter_benefits),
                        "community_access": str(self.community_access),
                        "priority_support": str(self.priority_support),
                        "django_product_id": str(self.id),
                        "feature_searches": f"{self.searches_per_month} searches per month",
                        "feature_projects": f"{self.active_projects} active projects",
                        "feature_early_adopter": "Early adopter benefits"
                        if self.early_adopter_benefits
                        else "",
                        "feature_community": "Community access"
                        if self.community_access
                        else "",
                        "feature_support": "Priority support"
                        if self.priority_support
                        else "",
                    },
                )

                # If price changed, create a new price and update the ID
                current_price = stripe.Price.retrieve(self.stripe_price_id)
                if current_price.unit_amount != int(self.price * 100):
                    new_price = stripe.Price.create(
                        product=self.stripe_product.id,
                        unit_amount=int(self.price * 100),
                        currency="usd",
                        metadata={
                            "duration_months": str(self.duration_months),
                            "regular_price": str(self.regular_price),
                        },
                    )
                    stripe.Price.modify(self.stripe_price_id, active=False)
                    self.stripe_price_id = new_price.id

        except stripe.error.StripeError as e:
            print(f"Stripe error: {e}")
            raise Exception("Failed to create/update Stripe product and price") from e

        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Override delete to deactivate Stripe product."""
        try:
            if self.stripe_product:
                # Archive the product in Stripe instead of deleting
                stripe.Product.modify(
                    self.stripe_product.id,
                    active=False,
                    metadata={"archived_at": timezone.now().isoformat()},
                )
            if self.stripe_price_id:
                stripe.Price.modify(self.stripe_price_id, active=False)
        except stripe.error.StripeError as e:
            print(f"Stripe error: {e}")

        super().delete(*args, **kwargs)

    def __str__(self):
        """Return string representation of the subscription product."""
        return f"{self.name} - ${self.price} for {self.duration_months} months"

    @classmethod
    def get_or_sync_product(cls, stripe_id):
        """Get a product by Stripe ID (product or price) or sync it from Stripe."""
        try:
            # Check if the ID is a price ID
            if stripe_id.startswith("price_"):
                # Get price from Stripe
                price = stripe.Price.retrieve(stripe_id)
                stripe_product_id = price.product
            else:
                stripe_product_id = stripe_id

            # Try to get the product from our database first
            from djstripe.models import Product

            try:
                # Try to get our custom product first
                product = cls.objects.select_related("stripe_product").get(
                    stripe_product__id=stripe_product_id
                )
                if stripe_id.startswith("price_"):
                    # Update price if needed
                    product.stripe_price_id = stripe_id
                    product.price = price.unit_amount / 100
                    product.save()
                return product
            except cls.DoesNotExist:
                # Product not found in our database, fetch from Stripe
                try:
                    # Get the product data from Stripe
                    stripe_product_data = stripe.Product.retrieve(stripe_product_id)

                    # Sync it to dj-stripe
                    stripe_product = Product.sync_from_stripe_data(stripe_product_data)

                    # Get the active price for this product
                    if stripe_id.startswith("price_"):
                        prices = [price]
                    else:
                        prices = stripe.Price.list(
                            product=stripe_product_id, active=True, limit=1
                        ).data

                    if not prices:
                        raise ValueError(
                            f"No active price found for product {stripe_product_id}"
                        )

                    price = prices[0]
                    metadata = stripe_product_data.metadata

                    # Create our custom product
                    product = cls.objects.create(
                        stripe_product=stripe_product,
                        name=stripe_product_data.name,
                        description=stripe_product_data.description or "",
                        price=price.unit_amount / 100,  # Convert cents to dollars
                        stripe_price_id=price.id,
                        duration_months=int(metadata.get("duration_months", 1)),
                        regular_price=float(metadata.get("regular_price", 0)),
                        searches_per_month=int(metadata.get("searches_per_month", 0)),
                        active_projects=int(metadata.get("active_projects", 0)),
                        early_adopter_benefits=metadata.get(
                            "early_adopter_benefits", "false"
                        ).lower()
                        == "true",
                        community_access=metadata.get(
                            "community_access", "false"
                        ).lower()
                        == "true",
                        priority_support=metadata.get(
                            "priority_support", "false"
                        ).lower()
                        == "true",
                    )

                    print(
                        f"Created new product: {product.name} (ID: {product.stripe_product.id})"
                    )
                    return product

                except stripe.error.InvalidRequestError as e:
                    print(f"Invalid Stripe ID: {stripe_id}")
                    raise ValueError(f"Product/Price not found in Stripe: {str(e)}")
                except Exception as e:
                    print(f"Error creating product: {str(e)}")
                    raise

        except Exception as e:
            print(f"Error in get_or_sync_product: {str(e)}")
            raise ValueError(f"Failed to get or create product: {str(e)}")

    @classmethod
    def sync_from_stripe(cls):
        """Sync products from Stripe to database."""
        try:
            # Fetch all active products from Stripe
            stripe_products = stripe.Product.list(active=True)

            for stripe_product_data in stripe_products.data:
                try:
                    # First sync the Stripe product to dj-stripe
                    from djstripe.models import Product

                    stripe_product = Product.sync_from_stripe_data(stripe_product_data)

                    # Get the price for this product
                    prices = stripe.Price.list(
                        product=stripe_product_data.id, active=True, limit=1
                    )
                    if not prices.data:
                        continue

                    price = prices.data[0]
                    metadata = stripe_product_data.metadata

                    # Create or update product in database
                    product, created = cls.objects.update_or_create(
                        stripe_product=stripe_product,
                        defaults={
                            "name": stripe_product_data.name,
                            "description": stripe_product_data.description or "",
                            "price": price.unit_amount / 100,
                            "stripe_price_id": price.id,
                            "duration_months": int(metadata.get("duration_months", 1)),
                            "regular_price": float(metadata.get("regular_price", 0)),
                            "searches_per_month": int(
                                metadata.get("searches_per_month", 0)
                            ),
                            "active_projects": int(metadata.get("active_projects", 0)),
                            "early_adopter_benefits": metadata.get(
                                "early_adopter_benefits", "false"
                            ).lower()
                            == "true",
                            "community_access": metadata.get(
                                "community_access", "false"
                            ).lower()
                            == "true",
                            "priority_support": metadata.get(
                                "priority_support", "false"
                            ).lower()
                            == "true",
                        },
                    )
                    print(
                        f"Synced product: {product.name} (ID: {product.stripe_product.id})"
                    )

                except Exception as e:
                    print(f"Error syncing product {stripe_product_data.id}: {str(e)}")
                    continue

            return True
        except stripe.error.StripeError as e:
            print(f"Stripe sync error: {e}")
            return False


class UserSubscription(models.Model):
    """Model to track user subscriptions and their status."""

    STATUS_CHOICES = [
        ("active", "Active"),
        ("expired", "Expired"),
        ("cancelled", "Cancelled"),
        ("pending", "Pending"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="subscriptions"
    )
    product = models.ForeignKey(
        SubscriptionProduct, on_delete=models.CASCADE, related_name="user_subscriptions"
    )
    stripe_checkout_session_id = models.CharField(max_length=100, blank=True, null=True)
    stripe_payment_intent_id = models.CharField(max_length=100, blank=True, null=True)
    referral_id = models.CharField(
        max_length=100, blank=True, null=True, help_text="Rewardful referral ID"
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="pending", unique=False
    )
    start_date = models.DateTimeField(null=True, blank=True)
    end_date = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta options for UserSubscription model."""

        constraints = [
            models.UniqueConstraint(
                fields=["user"],
                condition=models.Q(status="active"),
                name="unique_active_subscription_per_user",
            )
        ]

    def __str__(self):
        """Return string representation of the user subscription."""
        return f"{self.user.username} - {self.product.name} ({self.status})"

    def save(self, *args, **kwargs):
        """Override save to update Stripe customer and subscription data."""
        try:
            if not self.stripe_checkout_session_id and self.status == "active":
                # Create or get Stripe customer
                customers = stripe.Customer.list(email=self.user.email, limit=1)
                if customers.data:
                    customer = customers.data[0]
                else:
                    customer = stripe.Customer.create(
                        email=self.user.email,
                        metadata={
                            "django_user_id": str(self.user.id),
                            "django_subscription_id": str(self.id) if self.id else None,
                        },
                    )

                # Store subscription details in Stripe customer metadata
                stripe.Customer.modify(
                    customer.id,
                    metadata={
                        "subscription_status": self.status,
                        "subscription_start": self.start_date.isoformat()
                        if self.start_date
                        else None,
                        "subscription_end": self.end_date.isoformat()
                        if self.end_date
                        else None,
                        "product_id": str(self.product.stripe_product.id),
                        "referral_id": self.referral_id if self.referral_id else None,
                    },
                )

        except stripe.error.StripeError as e:
            print(f"Stripe error while saving subscription: {e}")
            # Continue with save even if Stripe update fails

        super().save(*args, **kwargs)
