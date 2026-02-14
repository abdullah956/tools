"""Admin configuration for the guest user app."""

from django.contrib import admin

from .models import GuestUser

admin.site.register(GuestUser)
