"""3. Geo & Check-in microservice routes."""
import logging
from flask import Blueprint, jsonify, request
from flasgger import swag_from
from Model.geo_service import GeoService as profile

geoServiceRoute = Blueprint('geoServiceRoute', __name__, url_prefix='/netwirkProfileRoute')
BLUEPRINTS = (geoServiceRoute,)

@geoServiceRoute.route('/fetch_locations_from_google', methods=['POST'])
@swag_from('netwirkProfile/fetch_locations_from_google.yml')
def fetch_google_locations():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.fetch_locations_from_google()
        return create
    except Exception:
        logging.exception("fetch_locations_from_google failed")
        return "Error"

@geoServiceRoute.route('/userCheckin', methods=['POST'])
@swag_from('netwirkProfile/userCheckin.yml')
def userCheckin_location():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_checkin()
        return create
    except:
        return "Error"  

@geoServiceRoute.route('/user_logout_from_location', methods=['POST'])
@swag_from('netwirkProfile/user_logout_from_location.yml')
def logout_from_location():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_logout_from_location()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/get_user_list_by_industry', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_by_industry.yml')
def get_users_by_industry():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_by_industry()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/get_industry_user_count', methods=['POST'])
@swag_from('netwirkProfile/get_industry_user_count.yml')
def fetch_industry_user_count():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_industry_user_count()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/checkout_from_checkinlocation', methods=['POST'])
@swag_from('netwirkProfile/checkout_from_checkinlocation.yml')
def checkout_from_checkinlocation_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.checkout_from_checkinlocation()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/get_user_list_by_industry_radius', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_by_industry_radius.yml')
def get_user_list_by_industry_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_by_industry_radius()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/get_industry_user_count_radius', methods=['POST'])
@swag_from('netwirkProfile/get_industry_user_count_radius.yml')
def get_industry_user_count_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_industry_user_count_radius()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/userCheckin_radius', methods=['POST'])
@swag_from('netwirkProfile/userCheckin_radius.yml')
def userCheckin_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.userCheckin_radius()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/checkout_from_checkinlocation_radius', methods=['POST'])
@swag_from('netwirkProfile/checkout_from_checkinlocation_radius.yml')
def checkout_from_checkinlocation_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.checkout_from_checkinlocation_radius()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/get_user_list_by_radius', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_by_radius.yml')
def get_user_list_by_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_by_radius()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/user_inactivity_checkout', methods=['POST'])
@swag_from('netwirkProfile/user_inactivity_checkout.yml')
def userId_inactivity_checkout():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_inactivity_checkout()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/update_user_radius_dynamic', methods=['POST'])
@swag_from('netwirkProfile/update_user_radius_dynamic.yml')
def update_user_radius_dynamic_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.update_user_radius_dynamic()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/user_live_location', methods=['POST'])
@swag_from('netwirkProfile/user_live_location.yml')
def user_live_location_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_live_location()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/search_within_100m', methods=['POST'])
@swag_from('netwirkProfile/search_within_100m.yml')
def search_within_100m_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.search_within_100m()
        return create
    except:
        return "Error"   

@geoServiceRoute.route('/validate_location', methods=['POST'])
@swag_from('netwirkProfile/validate_location.yml')
def validate_location():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.validate_location()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/usercheckinSendNotification', methods=['POST'])
@swag_from('netwirkProfile/usercheckinSendNotification.yml')
def user_checkinSendNotification():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.usercheckinSendNotification()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/usercheckinSendNotification_radius', methods=['POST'])
@swag_from('netwirkProfile/usercheckinSendNotification_radius.yml')
def usercheckinSendNotification_radius_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.usercheckinSendNotification_radius()
        return create
    except:
        return "Error"

@geoServiceRoute.route('/user_live_location_offline', methods=['POST'])
@swag_from('netwirkProfile/user_live_location_offline.yml')
def user_live_location_offline_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.user_live_location_offline()
        return create
    except:
        return "Error"    
