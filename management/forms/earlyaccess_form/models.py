"""Models for the early access form app."""

import random
import string
from datetime import datetime

from django.db import models


def generate_alphanumeric_token(length=32):
    """Generate a random alphanumeric token of specified length."""
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


class EarlyAccessForm(models.Model):
    """Model for early access registration form."""

    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Add new fields for payment tracking
    has_paid = models.BooleanField(default=False)
    product_id = models.CharField(max_length=200, blank=True, null=True)
    stripe_checkout_session_id = models.CharField(max_length=200, blank=True, null=True)
    stripe_payment_intent_id = models.CharField(max_length=200, blank=True, null=True)
    payment_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("paid", "Paid"),
            ("failed", "Failed"),
            ("cancelled", "Cancelled"),
        ],
        default="pending",
    )
    payment_amount = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True
    )
    payment_date = models.DateTimeField(null=True, blank=True)
    # New fields for email verification
    email_verification_token = models.CharField(max_length=64, null=True, blank=True)
    token_created_at = models.DateTimeField(null=True, blank=True)
    is_email_verified = models.BooleanField(default=False)
    # Field for affiliate program
    referral_id = models.CharField(
        max_length=100, blank=True, null=True, help_text="Rewardful referral ID"
    )

    class Meta:
        """Meta class for EarlyAccessForm."""

        verbose_name = "Early Access Form"
        verbose_name_plural = "Early Access Forms"
        ordering = ["-created_at"]

    def __str__(self):
        """Return string representation of the early access form."""
        return f"Early Access Request - {self.email} ({self.payment_status})"

    def generate_token(self):
        """Generate and save a new verification token."""
        self.email_verification_token = generate_alphanumeric_token()
        self.token_created_at = datetime.now()
        self.save()
        return self.email_verification_token
