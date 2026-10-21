"""Legacy microservice model (all remaining functions have been distributed to profile_service, geo_service, and social_service)."""

from Model.common_base import ProfileBase, RegistrationBase, profile


class LegacyProfileService(ProfileBase):
    """Empty compatibility class; all remaining methods were distributed to ProfileService, GeoService, and SocialService."""
    pass
