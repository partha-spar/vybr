import logging

from flask import Blueprint, jsonify
from flasgger import swag_from
from flask_jwt_extended import jwt_required
from Model.netwirk_registration import registration


registrationRoute = Blueprint('registrationRoute', __name__, url_prefix='/registrationRoute')


@registrationRoute.route('/register_user', methods=['POST'])
@swag_from('registration/register_user.yml')
def register_new_user():
    register = registration()
    try:
        create = register.register_user()
        return create
    except Exception as e:
        print(str(e))
        return "Error"

@registrationRoute.route('/check_email_exists', methods=['POST'])
@swag_from('registration/check_email_exists.yml')
def check_email_exists():
    register = registration()
    try:
        create = register.check_email_exists()
        return create
    except Exception as e:
        print(str(e))
        return "Error"


@registrationRoute.route('/verify_otp', methods=['POST'])
@swag_from('registration/verify_otp.yml')
def verify_user_otp():
    register = registration()
    try:
        create = register.verify_otp()
        return create
    except Exception as e:
        print(str(e))
        return "Error"


@registrationRoute.route('/resend_otp', methods=['POST'])
@swag_from('registration/resend_otp.yml')
def resend_user_otp():
    register = registration()
    try:
        return register.resend_otp()
    except Exception as e:
        print(str(e))
        return "Error"


@registrationRoute.route('/send_phone_otp', methods=['POST'])
@swag_from('registration/send_phone_otp.yml')
def send_phone_otp_now():
    register = registration()
    try:
        return register.send_phone_otp()
    except Exception as e:
        print(str(e))
        return "Error"


@registrationRoute.route('/verify_phone_otp', methods=['POST'])
@swag_from('registration/verify_phone_otp.yml')
def verify_phone_otp_now():
    register = registration()
    try:
        return register.verify_phone_otp()
    except Exception as e:
        print(str(e))
        return "Error"


@registrationRoute.route('/checkPhoneNumberExists', methods=['POST'])
@swag_from('registration/checkPhoneNumberExists.yml')
def checkPhoneNumber():
    register = registration()
    try:
        create = register.check_phone_number_exists()
        return create
    except Exception as e:
        print(str(e))
        return "Error"


@registrationRoute.route('/user_login', methods=['POST'])
@swag_from('registration/user_login.yml')
def user_signin():
    try:
        register = registration()
        create = register.user_login()
        if isinstance(create, dict):
            return jsonify(create)
        return create
    except Exception as e:
        logging.exception("user_login failed")
        return jsonify({"status": "Error", "message": str(e)})

@registrationRoute.route('/user_logout', methods=['POST'])
def user_signout():
    register = registration()
    try:
        create = register.user_logout()
        return create
    except Exception as e:
        return "Error"
    
@registrationRoute.route('/social_user_login', methods=['POST'])
@swag_from('registration/social_user_login.yml')
def social_user_signin():
    register = registration()
    try:
        create = register.social_user_login()
        return create
    except Exception as e:
        return "Error"

@registrationRoute.route('/forgot_password', methods=['POST'])
@swag_from('registration/forgot_password.yml')
def user_forgot_password():
    register = registration()
    try:
        create = register.forgot_password()
        return create
    except Exception as e:
        return "Error"


@registrationRoute.route('/confirm_new_password', methods=['POST'])
@swag_from('registration/confirm_new_password.yml')
def user_confirm_new_password():
    register = registration()
    try:
        create = register.confirm_new_password()
        return create
    except Exception as e:
        return "Error"

@registrationRoute.route('/confirm_user_in_cognito', methods=['POST'])
@swag_from('registration/confirm_user_in_cognito.yml')
def confirm_new_user_in_cognito():
    register = registration()
    try:
        create = register.confirm_user_in_cognito()
        return create
    except Exception as e:
        return "Error"

@registrationRoute.route('/update_authenticationId_using_userId', methods=['POST'])
@swag_from('registration/update_authenticationId_using_userId.yml')
def update_authenticationId_via_userId():
    register = registration()
    try:
        create = register.update_authenticationId_using_userId()
        return create
    except Exception as e:
        return "Error"


@registrationRoute.route('/set_new_password', methods=['POST'])
@swag_from('registration/set_new_password.yml')
def set_new_password_now():
    register = registration()
    try:
        create = register.set_new_password()
        return create
    except Exception as e:
        return "Error"

@registrationRoute.route('/register_apple_user', methods=['POST'])
@swag_from('registration/register_apple_user.yml')
def register_apple_user_now():
    register = registration()
    try:
        create = register.register_apple_user()
        return create
    except Exception as e:
        print(str(e))
        return "Error"
    
@registrationRoute.route('/get_apple_user', methods=['POST'])
@swag_from('registration/get_apple_user.yml')
def get_apple_user_now():
    register = registration()
    try:
        create = register.get_apple_user()
        return create
    except Exception as e:
        print(str(e))
        return "Error"
    
# @registrationRoute.route('/generate_hash_debug', methods=['POST'])
# @swag_from('registration/generate_hash_debug.yml')
# def generate_hash_debug_now():
#     register = registration()
#     try:
#         create = register.generate_hash_debug()
#         return create
#     except Exception as e:
#         print(str(e))
#         return "Error"
    
# @registrationRoute.route('/check_password_debug', methods=['POST'])
# @swag_from('registration/check_password_debug.yml')
# def check_password_debug_now():
#     register = registration()
#     try:
#         create = register.check_password_debug()
#         return create
#     except Exception as e:
#         print(str(e))
#         return "Error"