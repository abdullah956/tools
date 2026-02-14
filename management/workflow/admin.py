"""Admin configuration for the workflow app."""

from django.contrib import admin

from .models import Edge, Node, Workflow

# Register your models here.
admin.site.register(Workflow)
admin.site.register(Node)
admin.site.register(Edge)
