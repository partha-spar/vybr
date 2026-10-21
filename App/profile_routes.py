"""2. Profile & Persona microservice routes."""
import logging
from flask import Blueprint, jsonify, request
from flasgger import swag_from
from Model.profile_service import ProfileService as profile

profileServiceRoute = Blueprint('profileServiceRoute', __name__, url_prefix='/netwirkProfileRoute')
BLUEPRINTS = (profileServiceRoute,)

@profileServiceRoute.route('/create_profile_new', methods=['POST'])
@swag_from('netwirkProfile/create_profile_new.yml')
def create_profile_new():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.create_profile_new()
        return create
    except Exception as e:
        return "Error"

@profileServiceRoute.route('/delete_account', methods=['POST'])
@swag_from('netwirkProfile/delete_account.yml')
def delete_account_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.delete_account()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_vibe_list', methods=['POST'])
@swag_from('netwirkProfile/get_vibe_list.yml')
def get_vibe_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibe_list()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/user_pictures', methods=['POST'])
@swag_from('netwirkProfile/user_pictures.yml')
def upload_user_pictures():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_pictures()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/allow_incognito_visibility', methods=['POST'])
@swag_from('netwirkProfile/allow_incognito_visibility.yml')
def allow_incognito_visibility_for_users():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.allow_incognito_visibility()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_country_list', methods=['POST'])
@swag_from('netwirkProfile/get_country_list.yml')
def get_country_from_list():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_country_list()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_city_list', methods=['POST'])
@swag_from('netwirkProfile/get_city_list.yml')
def get_city_from_list():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_city_list()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_state_list', methods=['POST'])
@swag_from('netwirkProfile/get_state_list.yml')
def get_state_from_list():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_state_list()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/upload_selfie_pic', methods=['POST'])
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

@profileServiceRoute.route('/get_vibepersonality_qns3', methods=['POST'])
@swag_from('netwirkProfile/get_vibepersonality_qns3.yml')
def get_vibepersonality_qns3_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibepersonality_qns3()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_vibepersonality_qns4', methods=['POST'])
@swag_from('netwirkProfile/get_vibepersonality_qns4.yml')
def get_vibepersonality_qns4_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibepersonality_qns4()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_user_profile_details_new', methods=['POST'])
@swag_from('netwirkProfile/get_user_profile_details_new.yml')
def get_user_profile_details_new_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_profile_details_new()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/edit_profile_details_new', methods=['POST'])
@swag_from('netwirkProfile/edit_profile_details_new.yml')
def edit_profile_details_now_new():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.edit_profile_details_new()
        return create
    except Exception as e:
        return "Error"

@profileServiceRoute.route('/update_devicetoken', methods=['POST'])
@swag_from('netwirkProfile/update_devicetoken.yml')
def update_devicetoken_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.update_devicetoken()
        return create
    except Exception as e:
        return "Error"

@profileServiceRoute.route('/get_list_of_all_notifications', methods=['POST'])
@swag_from('netwirkProfile/get_list_of_all_notifications.yml')
def get_list_of_all_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_list_of_all_notifications()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/clear_all_notifications', methods=['POST'])
@swag_from('netwirkProfile/clear_all_notifications.yml')
def clear_all_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.clear_all_notifications()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_user_profile_details', methods=['POST'])
@swag_from('netwirkProfile/get_user_profile_details.yml')
def get_user_profile_details_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_profile_details()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/receive_notifications', methods=['POST'])
@swag_from('netwirkProfile/receive_notifications.yml')
def receive_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.receive_notifications()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/get_vibing_list', methods=['POST'])
@swag_from('netwirkProfile/get_vibing_list.yml')
def get_vibing_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_vibing_list()
        return create
    except:
        return "Error"

@profileServiceRoute.route('/send_fcm_with_rest_new_user_dummysilent', methods=['POST'])
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

@profileServiceRoute.route('/add_to_waitlist', methods=['POST'])
@swag_from('netwirkProfile/add_to_waitlist.yml')
def add_to_waitlist_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.add_to_waitlist()
        return create
    except:
        return "Error" 
