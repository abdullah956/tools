"""Admin configuration for the authentication app."""

from django.contrib import admin

from .models import CustomUser

admin.site.register(CustomUser)
