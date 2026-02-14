"""Admin configuration for the prompt optimization app."""

from django.contrib import admin

from .models import UserQuery

# Register your models here.
admin.site.register(UserQuery)
