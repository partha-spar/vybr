"""4. Social Graph microservice routes."""
import logging
from flask import Blueprint, jsonify, request
from flasgger import swag_from
from Model.social_service import SocialService as profile

socialServiceRoute = Blueprint('socialServiceRoute', __name__, url_prefix='/netwirkProfileRoute')
BLUEPRINTS = (socialServiceRoute,)

@socialServiceRoute.route('/confirmNetwirkAPI', methods=['POST'])
@swag_from('netwirkProfile/confirmNetwirkAPI.yml')
def confirmNetwirkAPInow():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.confirmNetwirkAPI()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/get_list_of_accepted_users_netwirk', methods=['POST'])
@swag_from('netwirkProfile/get_list_of_accepted_users_netwirk.yml')
def get_list_of_accepted_users_netwirk_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_list_of_accepted_users_netwirk()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/netwirk_Requests_Received', methods=['POST'])
@swag_from('netwirkProfile/netwirk_Requests_Received.yml')
def netwirk_Requests_Received_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.netwirk_Requests_Received()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/get_count_of_accepted_users_and_requests_received', methods=['POST'])
@swag_from('netwirkProfile/get_count_of_accepted_users_and_requests_received.yml')
def get_count_of_accepted_users_and_requests_received_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_count_of_accepted_users_and_requests_received()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/block_user', methods=['POST'])
@swag_from('netwirkProfile/block_user.yml')
def block_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.block_user()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/unblock_user', methods=['POST'])
@swag_from('netwirkProfile/unblock_user.yml')
def unblock_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.unblock_user()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/get_Blocked_Users_List', methods=['POST'])
@swag_from('netwirkProfile/get_Blocked_Users_List.yml')
def get_Blocked_Users_List_Now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_Blocked_Users_List()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/who_blocked_me_list', methods=['POST'])
@swag_from('netwirkProfile/who_blocked_me_list.yml')
def who_blocked_me_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.who_blocked_me_list()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/report_user', methods=['POST'])
@swag_from('netwirkProfile/report_user.yml')
def report_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.report_user()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/disconnect_user', methods=['POST'])
@swag_from('netwirkProfile/disconnect_user.yml')
def disconnect_user_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.disconnect_user()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/handle_send_wave_new', methods=['POST'])
@swag_from('netwirkProfile/handle_send_wave_new.yml')
def handle_send_wave_new_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.handle_send_wave_new()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/get_liked_user_list', methods=['POST'])
@swag_from('netwirkProfile/get_liked_user_list.yml')
def get_liked_user_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_liked_user_list()
        return create
    except Exception as e:
        return "Error"

@socialServiceRoute.route('/get_current_liked_user_list', methods=['POST'])
@swag_from('netwirkProfile/get_current_liked_user_list.yml')
def get_current_liked_user_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_current_liked_user_list()
        return create
    except Exception as e:
        return "Error"

@socialServiceRoute.route('/delete_received_wave', methods=['POST'])
@swag_from('netwirkProfile/delete_received_wave.yml')
def delete_received_wave_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.delete_received_wave()
        return create
    except Exception as e:
        return "Error"

@socialServiceRoute.route('/get_total_likes', methods=['POST'])
@swag_from('netwirkProfile/get_total_likes.yml')
def get_total_likes_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_total_likes()
        return create
    except Exception as e:
        return "Error"

@socialServiceRoute.route('/handle_send_wave_near', methods=['POST'])
@swag_from('netwirkProfile/handle_send_wave_near.yml')
def handle_send_wave_near_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.handle_send_wave_near()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/get_current_liked_user_list_near', methods=['POST'])
@swag_from('netwirkProfile/get_current_liked_user_list_near.yml')
def get_current_liked_user_list_near_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_current_liked_user_list_near()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/delete_received_profile_likes', methods=['POST'])
@swag_from('netwirkProfile/delete_received_profile_likes.yml')
def delete_received_profile_likes_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.delete_received_profile_likes()
        return create
    except Exception as e:
        return "Error"

@socialServiceRoute.route('/get_list_of_Requests_pending', methods=['POST'])
@swag_from('netwirkProfile/get_list_of_Requests_pending.yml')
def get_list_of_Requests_pending_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_list_of_Requests_pending()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/hide_user_from_list', methods=['POST'])
@swag_from('netwirkProfile/hide_user_from_list.yml')
def hide_user_from_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.hide_user_from_list()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/count_of_vibe_notifications', methods=['POST'])
@swag_from('netwirkProfile/count_of_vibe_notifications.yml')
def count_of_vibe_notifications_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.count_of_vibe_notifications()
        return create
    except:
        return "Error"

@socialServiceRoute.route('/get_liked_by_me_user_list', methods=['POST'])
@swag_from('netwirkProfile/get_liked_by_me_user_list.yml')
def get_liked_by_me_user_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_liked_by_me_user_list()
        return create
    except Exception as e:
        return "Error"
