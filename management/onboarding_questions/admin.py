"""Admin configuration for the onboarding questions app."""

from django.contrib import admin

from .models import OnboardingQuestion

admin.site.register(OnboardingQuestion)

# Register your models here.
