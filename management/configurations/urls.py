"""URL configuration for Artilence Backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/

Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')

Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')

Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.http import HttpResponse
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from rest_framework_simplejwt.views import TokenRefreshView

from .views import CustomTokenObtainPairView, CustomTokenVerifyView


def health_check(request):
    """Health check endpoint for ECS and ELB."""
    return HttpResponse("OK", status=200)


urlpatterns = [
    path("health/", health_check, name="health_check"),
    path("admin/", admin.site.urls),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/schema/swagger-ui/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
    path("api/guest-user/", include("guest_user.urls")),
    path("api/token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/token/verify/", CustomTokenVerifyView.as_view(), name="token_verify"),
    path("accounts/", include("authentication.urls")),
    path("onboarding-questions/", include("onboarding_questions.urls")),
    path("prompt-optimization/", include("prompt_optimization.urls")),
    path("workflow/", include("workflow.urls")),
    path("subscription/", include("subscription.urls")),
    path("early-access/", include("forms.earlyaccess_form.urls")),
    path("newsletter-subscriber/", include("forms.newslettersubscriber_form.urls")),
    path("submit-resource/", include("forms.submitresource_form.urls")),
    path("share-ideas/", include("forms.shareideas_form.urls")),
    path("invite-friends/", include("forms.invitefriends_form.urls")),
    path("search/", include("management.search.urls")),
    path("tools/", include("management.tools.urls")),
    path("chat/", include("management.chat.urls")),
    path("contractor/", include("management.contractor.urls")),
    path("ai-tool-recommender/", include("ai_tool_recommender.urls")),
    path("consultant-recommender/", include("consultant_recommender.urls")),
    path("tool-scraping/", include("management.tool_scraping.urls")),
]
