"""Configuration package for the Victor project."""

# This will make sure the app is always imported when
# Django starts so that shared_task will use this app.
try:
    from .celery import app as celery_app  # noqa: F401

    __all__ = ("celery_app",)
except ImportError:
    # Celery not installed yet, skip import
    __all__ = ()
