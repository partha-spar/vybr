"""Which URL paths each microservice owns.

Every path not listed here belongs to the "legacy" service.
"""

AUTH = "/registrationRoute"
PROFILE = "/netwirkProfileRoute"

SERVICES = {
    "auth": [
        f"{AUTH}/register_user",
        f"{AUTH}/check_email_exists",
        f"{AUTH}/verify_otp",
        f"{AUTH}/resend_otp",
        f"{AUTH}/send_phone_otp",
        f"{AUTH}/verify_phone_otp",
        f"{AUTH}/checkPhoneNumberExists",
        f"{AUTH}/user_login",
        f"{AUTH}/user_logout",
        f"{AUTH}/social_user_login",
        f"{AUTH}/forgot_password",
        f"{AUTH}/confirm_new_password",
        f"{AUTH}/set_new_password",
        f"{AUTH}/confirm_user_in_cognito",
        f"{AUTH}/register_apple_user",
        f"{AUTH}/get_apple_user",
        f"{AUTH}/update_authenticationId_using_userId",
        f"{PROFILE}/retrieve_userId_via_authenticationId",
    ],
    "profile": [
        f"{PROFILE}/create_profile_new",
        f"{PROFILE}/edit_profile_details_new",
        f"{PROFILE}/get_user_profile_details_new",
        f"{PROFILE}/upload_selfie_pic",
        f"{PROFILE}/user_pictures",
        f"{PROFILE}/allow_incognito_visibility",
        f"{PROFILE}/delete_account",
        f"{PROFILE}/get_vibe_list",
        f"{PROFILE}/get_vibepersonality_qns3",
        f"{PROFILE}/get_vibepersonality_qns4",
        f"{PROFILE}/get_country_list",
        f"{PROFILE}/get_state_list",
        f"{PROFILE}/get_city_list",
        f"{PROFILE}/update_devicetoken",
        f"{PROFILE}/get_list_of_all_notifications",
        f"{PROFILE}/clear_all_notifications",
        f"{PROFILE}/get_user_profile_details",
        f"{PROFILE}/receive_notifications",
        f"{PROFILE}/get_vibing_list",
        f"{PROFILE}/send_fcm_with_rest_new_user_dummysilent",
        f"{PROFILE}/add_to_waitlist",
    ],
    "geo": [
        f"{PROFILE}/fetch_locations_from_google",
        f"{PROFILE}/validate_location",
        f"{PROFILE}/userCheckin",
        f"{PROFILE}/userCheckin_radius",
        f"{PROFILE}/checkout_from_checkinlocation",
        f"{PROFILE}/checkout_from_checkinlocation_radius",
        f"{PROFILE}/user_logout_from_location",
        f"{PROFILE}/user_inactivity_checkout",
        f"{PROFILE}/user_live_location",
        f"{PROFILE}/update_user_radius_dynamic",
        f"{PROFILE}/search_within_100m",
        f"{PROFILE}/get_user_list_by_radius",
        f"{PROFILE}/get_user_list_by_industry",
        f"{PROFILE}/get_user_list_by_industry_radius",
        f"{PROFILE}/get_industry_user_count",
        f"{PROFILE}/get_industry_user_count_radius",
        f"{PROFILE}/usercheckinSendNotification",
        f"{PROFILE}/usercheckinSendNotification_radius",
        f"{PROFILE}/user_live_location_offline",
    ],
    "social": [
        f"{PROFILE}/handle_send_wave_new",
        f"{PROFILE}/handle_send_wave_near",
        f"{PROFILE}/confirmNetwirkAPI",
        f"{PROFILE}/get_list_of_accepted_users_netwirk",
        f"{PROFILE}/netwirk_Requests_Received",
        f"{PROFILE}/get_list_of_Requests_pending",
        f"{PROFILE}/get_count_of_accepted_users_and_requests_received",
        f"{PROFILE}/disconnect_user",
        f"{PROFILE}/block_user",
        f"{PROFILE}/unblock_user",
        f"{PROFILE}/get_Blocked_Users_List",
        f"{PROFILE}/who_blocked_me_list",
        f"{PROFILE}/report_user",
        f"{PROFILE}/get_liked_user_list",
        f"{PROFILE}/get_current_liked_user_list",
        f"{PROFILE}/get_current_liked_user_list_near",
        f"{PROFILE}/get_total_likes",
        f"{PROFILE}/delete_received_wave",
        f"{PROFILE}/delete_received_profile_likes",
        f"{PROFILE}/hide_user_from_list",
        f"{PROFILE}/count_of_vibe_notifications",
        f"{PROFILE}/get_liked_by_me_user_list",
    ],
    "recsys": [
        f"{PROFILE}/get_user_list_offline",
        f"{PROFILE}/hide_user_from_offline_list",
        f"{PROFILE}/get_later_remaining",
    ],
}

LEGACY = "legacy"
ALL_SERVICES = [*SERVICES, LEGACY]

COMMON_PATHS = {"/", "/health", "/healthz"}
COMMON_PATH_PREFIXES = ("/swagger", "/apispec", "/flasgger_static", "/static")

_OWNER = {path: name for name, paths in SERVICES.items() for path in paths}


def owner_of(path):
    return _OWNER.get(path, LEGACY)


def is_common(path):
    return path in COMMON_PATHS or path.startswith(COMMON_PATH_PREFIXES)


def serves(service, path):
    if service is None or is_common(path):
        return True
    return owner_of(path) == service
