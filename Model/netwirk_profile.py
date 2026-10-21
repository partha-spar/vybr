"""Combined profile model (imports all split microservice profile classes)."""

from Model.auth_service import AuthProfileService
from Model.common_base import (
    ALL_TAGS,
    CHECK_INTERVAL,
    COMM_COMPAT,
    COMM_STYLE_MAP,
    DEFAULT_WEIGHTS,
    INACTIVITY_TIMEOUT_SECONDS,
    INTENT_COMPAT,
    INTENT_MAP,
    IST,
    MATCHING_WEIGHTS_V2,
    MIN_INTERACTIONS_FOR_DYNAMIC_WEIGHTS,
    RECENCY_HALF_LIFE_DAYS,
    SKIP_PENALTY_WEIGHT,
    SUPPORTED_METROS,
    TAG_TYPE_MAP,
)
from Model.geo_service import GeoService
from Model.legacy_service import LegacyProfileService
from Model.profile_service import ProfileService
from Model.recsys_service import RecsysService
from Model.social_service import SocialService


class profile(
    AuthProfileService,
    ProfileService,
    GeoService,
    SocialService,
    RecsysService,
    LegacyProfileService,
):
    pass
