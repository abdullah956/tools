"""Script to set up Stripe API keys in the database."""

from django.core.management.base import BaseCommand
from djstripe.models import APIKey
from envs.env_loader import env_loader

from subscription.models import SubscriptionProduct


class Command(BaseCommand):
    """Command to set up Stripe API keys in the database."""

    help = "Set up Stripe API keys in the database"

    def handle(self, *args, **options):
        """Handle the command."""
        try:
            # Get keys from environment using env_loader
            test_secret_key = env_loader.stripe_secret_key
            test_publishable_key = env_loader.stripe_publishable_key
            test_webhook_secret = env_loader.stripe_webhook_secret

            env_type = env_loader.env_type
            self.stdout.write(f"Setting up Stripe keys for {env_type} environment")

            if not test_secret_key or not test_publishable_key:
                self.stdout.write(
                    self.style.ERROR(
                        f"Error: Stripe API keys not found in {env_type} environment"
                    )
                )
                return

            # Set up test secret key
            test_secret, created = APIKey.objects.get_or_create(
                secret=test_secret_key,
                defaults={
                    "type": "secret",
                    "livemode": env_type == "production",
                    "name": f"Default {env_type.title()} Secret Key",
                },
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Secret Key: {'Added' if created else 'Already exists'}"
                )
            )

            # Set up test publishable key
            test_publishable, created = APIKey.objects.get_or_create(
                secret=test_publishable_key,
                defaults={
                    "type": "publishable",
                    "livemode": env_type == "production",
                    "name": f"Default {env_type.title()} Publishable Key",
                },
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Publishable Key: {'Added' if created else 'Already exists'}"
                )
            )

            # Set up test webhook secret
            if test_webhook_secret:
                test_webhook, created = APIKey.objects.get_or_create(
                    secret=test_webhook_secret,
                    defaults={
                        "type": "webhook",
                        "livemode": env_type == "production",
                        "name": f"Default {env_type.title()} Webhook Secret",
                    },
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Webhook Secret: {'Added' if created else 'Already exists'}"
                    )
                )

            # Verify keys in database
            self.stdout.write(
                f"\nCurrent API keys in database for {env_type} environment:"
            )
            all_keys = APIKey.objects.all()
            for key in all_keys:
                masked_key = f"{key.secret[:8]}...{key.secret[-4:]}"
                self.stdout.write(f"{key.name}: {masked_key}")

            # Sync subscription products
            SubscriptionProduct.sync_from_stripe()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully synced subscription products from Stripe for {env_type} environment"
                )
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error setting up Stripe keys: {str(e)}")
            )
