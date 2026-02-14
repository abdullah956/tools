"""Models for the onboarding questions app."""

from django.conf import settings
from django.db import models


class OnboardingQuestion(models.Model):
    """Model representing an onboarding question for a user."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    company_info = models.TextField(blank=True, null=True)
    website_or_linkedin = models.URLField(blank=True, null=True)
    experience_level = models.CharField(
        max_length=50,
        choices=[
            ("beginner", "Beginner"),
            ("intermediate", "Intermediate"),
            ("advanced", "Advanced"),
            ("expert", "Expert"),
        ],
        blank=True,
        null=True,
    )
    goals = models.TextField(blank=True, null=True)

    def __str__(self):
        """Return a string representation of the onboarding question."""
        return f"Onboarding for {self.user.username}"
