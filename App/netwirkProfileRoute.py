import logging

from flask import Blueprint, request, jsonify
from flasgger import swag_from
from flask_jwt_extended import jwt_required
from Model.netwirk_profile import profile




netwirkProfileRoute = Blueprint('netwirkProfileRoute', __name__, url_prefix='/netwirkProfileRoute')


@netwirkProfileRoute.route('/create_profile_new', methods=['POST'])
@swag_from('netwirkProfile/create_profile_new.yml')
def create_profile_new():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.create_profile_new()
        return create
    except Exception as e:
        return "Error"

@netwirkProfileRoute.route('/fetch_locations_from_google', methods=['POST'])
@swag_from('netwirkProfile/fetch_locations_from_google.yml')
def fetch_google_locations():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.fetch_locations_from_google()
        return create
    except Exception:
        logging.exception("fetch_locations_from_google failed")
        return "Error"

# @netwirkProfileRoute.route('/fetch_location_from_overture', methods=['POST'])
# @swag_from('netwirkProfile/fetch_location_from_overture.yml')
# def fetch_overture_locations():
#     netwirkProfile = profile()
#     try:
#         create = netwirkProfile.fetch_location_from_overture()
#         return create
#     except:
#         return "Error"

@netwirkProfileRoute.route('/userCheckin', methods=['POST'])
@swag_from('netwirkProfile/userCheckin.yml')
def userCheckin_location():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_checkin()
        return create
    except:
        return "Error"  


@netwirkProfileRoute.route('/usercheckinSendNotification', methods=['POST'])
@swag_from('netwirkProfile/usercheckinSendNotification.yml')
def user_checkinSendNotification():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.usercheckinSendNotification()
        return create
    except:
        return "Error"


@netwirkProfileRoute.route('/user_logout_from_location', methods=['POST'])
@swag_from('netwirkProfile/user_logout_from_location.yml')
def logout_from_location():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_logout_from_location()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_user_list_by_industry', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_by_industry.yml')
def get_users_by_industry():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_by_industry()
        return create
    except:
        return "Error"


@netwirkProfileRoute.route('/get_industry_user_count', methods=['POST'])
@swag_from('netwirkProfile/get_industry_user_count.yml')
def fetch_industry_user_count():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_industry_user_count()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/confirmNetwirkAPI', methods=['POST'])
@swag_from('netwirkProfile/confirmNetwirkAPI.yml')
def confirmNetwirkAPInow():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.confirmNetwirkAPI()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_list_of_all_notifications', methods=['POST'])
@swag_from('netwirkProfile/get_list_of_all_notifications.yml')
def get_list_of_all_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_list_of_all_notifications()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/clear_all_notifications', methods=['POST'])
@swag_from('netwirkProfile/clear_all_notifications.yml')
def clear_all_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.clear_all_notifications()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_list_of_accepted_users_netwirk', methods=['POST'])
@swag_from('netwirkProfile/get_list_of_accepted_users_netwirk.yml')
def get_list_of_accepted_users_netwirk_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_list_of_accepted_users_netwirk()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/netwirk_Requests_Received', methods=['POST'])
@swag_from('netwirkProfile/netwirk_Requests_Received.yml')
def netwirk_Requests_Received_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.netwirk_Requests_Received()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_user_profile_details', methods=['POST'])
@swag_from('netwirkProfile/get_user_profile_details.yml')
def get_user_profile_details_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_profile_details()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_count_of_accepted_users_and_requests_received', methods=['POST'])
@swag_from('netwirkProfile/get_count_of_accepted_users_and_requests_received.yml')
def get_count_of_accepted_users_and_requests_received_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_count_of_accepted_users_and_requests_received()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/block_user', methods=['POST'])
@swag_from('netwirkProfile/block_user.yml')
def block_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.block_user()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/unblock_user', methods=['POST'])
@swag_from('netwirkProfile/unblock_user.yml')
def unblock_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.unblock_user()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_Blocked_Users_List', methods=['POST'])
@swag_from('netwirkProfile/get_Blocked_Users_List.yml')
def get_Blocked_Users_List_Now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_Blocked_Users_List()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/who_blocked_me_list', methods=['POST'])
@swag_from('netwirkProfile/who_blocked_me_list.yml')
def who_blocked_me_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.who_blocked_me_list()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/report_user', methods=['POST'])
@swag_from('netwirkProfile/report_user.yml')
def report_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.report_user()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/disconnect_user', methods=['POST'])
@swag_from('netwirkProfile/disconnect_user.yml')
def disconnect_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.disconnect_user()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/retrieve_userId_via_authenticationId', methods=['POST'])
@swag_from('netwirkProfile/retrieve_userId_via_authenticationId.yml')
def retrieve_userId_via_authenticationId_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.retrieve_userId_via_authenticationId()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/receive_notifications', methods=['POST'])
@swag_from('netwirkProfile/receive_notifications.yml')
def receive_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.receive_notifications()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/checkout_from_checkinlocation', methods=['POST'])
@swag_from('netwirkProfile/checkout_from_checkinlocation.yml')
def checkout_from_checkinlocation_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.checkout_from_checkinlocation()
        return create
    except:
        return "Error"

# @netwirkProfileRoute.route('/extract_designation_and_industry', methods=['POST'])
# @swag_from('netwirkProfile/extract_designation_and_industry.yml')
# def extract_designation_and_industry_now():
#     netwirkProfile = profile()
#     try:
#         create = netwirkProfile.extract_designation_and_industry()
#         return create
#     except:
#         return "Error"

# @netwirkProfileRoute.route('/matches_openai', methods=['POST'])
# @swag_from('netwirkProfile/matches_openai.yml')
# def matches_openai_now():
#     netwirkProfile = profile()
#     try:
#         create = netwirkProfile.matches_openai()
#         return create
#     except:
#         return "Error"

@netwirkProfileRoute.route('/delete_account', methods=['POST'])
@swag_from('netwirkProfile/delete_account.yml')
def delete_account_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.delete_account()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_vibe_list', methods=['POST'])
@swag_from('netwirkProfile/get_vibe_list.yml')
def get_vibe_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibe_list()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/get_vibing_list', methods=['POST'])
@swag_from('netwirkProfile/get_vibing_list.yml')
def get_vibing_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibing_list()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_user_list_by_industry_radius', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_by_industry_radius.yml')
def get_user_list_by_industry_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_by_industry_radius()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_industry_user_count_radius', methods=['POST'])
@swag_from('netwirkProfile/get_industry_user_count_radius.yml')
def get_industry_user_count_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_industry_user_count_radius()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/usercheckinSendNotification_radius', methods=['POST'])
@swag_from('netwirkProfile/usercheckinSendNotification_radius.yml')
def usercheckinSendNotification_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.usercheckinSendNotification_radius()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/userCheckin_radius', methods=['POST'])
@swag_from('netwirkProfile/userCheckin_radius.yml')
def userCheckin_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.userCheckin_radius()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/checkout_from_checkinlocation_radius', methods=['POST'])
@swag_from('netwirkProfile/checkout_from_checkinlocation_radius.yml')
def checkout_from_checkinlocation_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.checkout_from_checkinlocation_radius()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/get_user_list_by_radius', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_by_radius.yml')
def get_user_list_by_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_by_radius()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/user_pictures', methods=['POST'])
@swag_from('netwirkProfile/user_pictures.yml')
def upload_user_pictures():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_pictures()
        return create
    except:
        return "Error"
    
# @netwirkProfileRoute.route('/events/<int:userId>', methods=['GET'])
# @swag_from('netwirkProfile/events.yml')  
# def events(userId):
#     netwirkProfile = profile()
#     try:
#         return netwirkProfile.sse(userId)
#     except Exception as e:
#         return "Error"
		
@netwirkProfileRoute.route('/allow_incognito_visibility', methods=['POST'])
@swag_from('netwirkProfile/allow_incognito_visibility.yml')
def allow_incognito_visibility_for_users():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.allow_incognito_visibility()
        return create
    except:
        return "Error"
    
    
@netwirkProfileRoute.route('/get_country_list', methods=['POST'])
@swag_from('netwirkProfile/get_country_list.yml')
def get_country_from_list():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_country_list()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/get_city_list', methods=['POST'])
@swag_from('netwirkProfile/get_city_list.yml')
def get_city_from_list():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_city_list()
        return create
    except:
        return "Error"
    
        
@netwirkProfileRoute.route('/get_state_list', methods=['POST'])
@swag_from('netwirkProfile/get_state_list.yml')
def get_state_from_list():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_state_list()
        return create
    except:
        return "Error"
    

@netwirkProfileRoute.route('/user_inactivity_checkout', methods=['POST'])
@swag_from('netwirkProfile/user_inactivity_checkout.yml')
def userId_inactivity_checkout():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_inactivity_checkout()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/update_user_radius_dynamic', methods=['POST'])
@swag_from('netwirkProfile/update_user_radius_dynamic.yml')
def update_user_radius_dynamic_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.update_user_radius_dynamic()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/send_fcm_with_rest_new_user_dummysilent', methods=['POST'])
@swag_from('netwirkProfile/send_fcm_with_rest_new_user_dummysilent.yml')
def send_fcm_with_rest_new_user_dummysilent_now():
    netwirkProfile = profile()
    try:
        data = request.get_json()
        create = netwirkProfile.send_fcm_with_rest_new_user_dummysilent(data)
        # Always return a JSON response
        return create, 200
    except Exception as e:
        # Return JSON error with 500 status
        return {"status": "error", "message": str(e)}, 500

@netwirkProfileRoute.route('/handle_send_wave_new', methods=['POST'])
@swag_from('netwirkProfile/handle_send_wave_new.yml')
def handle_send_wave_new_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.handle_send_wave_new()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/upload_selfie_pic', methods=['POST'])
@swag_from('netwirkProfile/upload_selfie_pic.yml')
def upload_user_selfie_pic():
    netwirkProfile = profile()
    try:
        user_id_raw = request.form.get('userId', '')
        if not user_id_raw.isdigit():
            return "Invalid user ID", 400
        user_id = int(user_id_raw)
        create = netwirkProfile.upload_selfie_pic(user_id)
        return create
    except Exception as e:
        return "Error"

@netwirkProfileRoute.route('/user_live_location', methods=['POST'])
@swag_from('netwirkProfile/user_live_location.yml')
def user_live_location_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_live_location()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/get_vibepersonality_qns3', methods=['POST'])
@swag_from('netwirkProfile/get_vibepersonality_qns3.yml')
def get_vibepersonality_qns3_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibepersonality_qns3()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/get_vibepersonality_qns4', methods=['POST'])
@swag_from('netwirkProfile/get_vibepersonality_qns4.yml')
def get_vibepersonality_qns4_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibepersonality_qns4()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/get_user_profile_details_new', methods=['POST'])
@swag_from('netwirkProfile/get_user_profile_details_new.yml')
def get_user_profile_details_new_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_profile_details_new()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/edit_profile_details_new', methods=['POST'])
@swag_from('netwirkProfile/edit_profile_details_new.yml')
def edit_profile_details_now_new():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.edit_profile_details_new()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/get_liked_user_list', methods=['POST'])
@swag_from('netwirkProfile/get_liked_user_list.yml')
def get_liked_user_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_liked_user_list()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/get_current_liked_user_list', methods=['POST'])
@swag_from('netwirkProfile/get_current_liked_user_list.yml')
def get_current_liked_user_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_current_liked_user_list()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/delete_received_wave', methods=['POST'])
@swag_from('netwirkProfile/delete_received_wave.yml')
def delete_received_wave_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.delete_received_wave()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/get_total_likes', methods=['POST'])
@swag_from('netwirkProfile/get_total_likes.yml')
def get_total_likes_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_total_likes()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/update_devicetoken', methods=['POST'])
@swag_from('netwirkProfile/update_devicetoken.yml')
def update_devicetoken_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.update_devicetoken()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/handle_send_wave_near', methods=['POST'])
@swag_from('netwirkProfile/handle_send_wave_near.yml')
def handle_send_wave_near_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.handle_send_wave_near()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/get_current_liked_user_list_near', methods=['POST'])
@swag_from('netwirkProfile/get_current_liked_user_list_near.yml')
def get_current_liked_user_list_near_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_current_liked_user_list_near()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/delete_received_profile_likes', methods=['POST'])
@swag_from('netwirkProfile/delete_received_profile_likes.yml')
def delete_received_profile_likes_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.delete_received_profile_likes()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/get_list_of_Requests_pending', methods=['POST'])
@swag_from('netwirkProfile/get_list_of_Requests_pending.yml')
def get_list_of_Requests_pending_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_list_of_Requests_pending()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/count_of_vibe_notifications', methods=['POST'])
@swag_from('netwirkProfile/count_of_vibe_notifications.yml')
def count_of_vibe_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.count_of_vibe_notifications()
        return create
    except:
        return "Error"

@netwirkProfileRoute.route('/search_within_100m', methods=['POST'])
@swag_from('netwirkProfile/search_within_100m.yml')
def search_within_100m_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.search_within_100m()
        return create
    except:
        return "Error"   

@netwirkProfileRoute.route('/hide_user_from_list', methods=['POST'])
@swag_from('netwirkProfile/hide_user_from_list.yml')
def hide_user_from_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.hide_user_from_list()
        return create
    except:
        return "Error"  

@netwirkProfileRoute.route('/validate_location', methods=['POST'])
@swag_from('netwirkProfile/validate_location.yml')
def validate_location():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.validate_location()
        return create
    except:
        return "Error"  
    
@netwirkProfileRoute.route('/add_to_waitlist', methods=['POST'])
@swag_from('netwirkProfile/add_to_waitlist.yml')
def add_to_waitlist_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.add_to_waitlist()
        return create
    except:
        return "Error" 
@netwirkProfileRoute.route('/get_user_list_offline', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_offline.yml')
def get_user_list_offline_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_offline()
        return create
    except:
        return "Error"
    
@netwirkProfileRoute.route('/hide_user_from_offline_list', methods=['POST'])
@swag_from('netwirkProfile/hide_user_from_offline_list.yml')
def hide_user_from_offline_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.hide_user_from_offline_list()
        return create
    except:
        return "Error"  
    
@netwirkProfileRoute.route('/get_liked_by_me_user_list', methods=['POST'])
@swag_from('netwirkProfile/get_liked_by_me_user_list.yml')
def get_liked_by_me_user_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_liked_by_me_user_list()
        return create
    except Exception as e:
        return "Error"
    
@netwirkProfileRoute.route('/user_live_location_offline', methods=['POST'])
@swag_from('netwirkProfile/user_live_location_offline.yml')
def user_live_location_offline_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_live_location_offline()
        return create
    except:
        return "Error"    

@netwirkProfileRoute.route('/get_later_remaining', methods=['POST'])
@swag_from('netwirkProfile/get_later_remaining.yml')
def get_later_remaining_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_later_remaining()
        return create
    except:
        return "Error"
