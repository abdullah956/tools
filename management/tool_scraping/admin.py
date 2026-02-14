"""Admin configuration for tool scraping."""

from django.contrib import admin

from .models import CombinedText, ScrapingJob, SitePage, ToolSite


@admin.register(ScrapingJob)
class ScrapingJobAdmin(admin.ModelAdmin):
    """Admin for ScrapingJob."""

    list_display = ["id", "job_type", "status", "created_at", "finished_at"]
    list_filter = ["status", "job_type", "created_at"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(ToolSite)
class ToolSiteAdmin(admin.ModelAdmin):
    """Admin for ToolSite."""

    list_display = ["id", "title", "website", "status", "master_category"]
    list_filter = ["status", "master_category", "category"]
    search_fields = ["title", "website", "description"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(SitePage)
class SitePageAdmin(admin.ModelAdmin):
    """Admin for SitePage."""

    list_display = ["id", "site", "url", "status", "scraped_at"]
    list_filter = ["status", "discovered_at", "scraped_at"]
    search_fields = ["url"]
    readonly_fields = ["id", "discovered_at", "scraped_at"]


@admin.register(CombinedText)
class CombinedTextAdmin(admin.ModelAdmin):
    """Admin for CombinedText."""

    list_display = ["id", "site", "char_count", "created_at"]
    readonly_fields = ["id", "created_at", "updated_at"]
    search_fields = ["site__title", "site__website"]
