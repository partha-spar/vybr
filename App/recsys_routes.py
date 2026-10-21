"""6. RecSys & AI microservice routes."""
import logging
from flask import Blueprint, jsonify, request
from flasgger import swag_from
from Model.recsys_service import RecsysService as profile

recsysServiceRoute = Blueprint('recsysServiceRoute', __name__, url_prefix='/netwirkProfileRoute')
BLUEPRINTS = (recsysServiceRoute,)

@recsysServiceRoute.route('/get_user_list_offline', methods=['POST'])
@swag_from('netwirkProfile/get_user_list_offline.yml')
def get_user_list_offline_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_user_list_offline()
        return create
    except:
        return "Error"

@recsysServiceRoute.route('/hide_user_from_offline_list', methods=['POST'])
@swag_from('netwirkProfile/hide_user_from_offline_list.yml')
def hide_user_from_offline_list_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.hide_user_from_offline_list()
        return create
    except:
        return "Error"  

@recsysServiceRoute.route('/get_later_remaining', methods=['POST'])
@swag_from('netwirkProfile/get_later_remaining.yml')
def get_later_remaining_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.get_later_remaining()
        return create
    except:
        return "Error"
