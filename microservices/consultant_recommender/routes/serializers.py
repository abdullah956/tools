"""Data serializers for consultant information."""

from datetime import datetime

from pydantic import BaseModel


class ConsultantData(BaseModel):
    """Data model for consultant information."""

    date: datetime
    time: datetime
    company_name: str
    country: str | None = None
    apps_included: str | None = None
    language: str | None = None
    phone: str | None = None
    website: str | None = None
    gmail: str | None = None
    about: str | None = None
    type_of_services: str | None = None
    countries_with_office_locations: str | None = None
