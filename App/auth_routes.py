"""1. Auth & Identity microservice routes."""
import logging
from flask import Blueprint, jsonify, request
from flasgger import swag_from
from Model.auth_service import AuthProfileService as profile, AuthRegistrationService as registration

authRegistrationRoute = Blueprint('registrationRoute', __name__, url_prefix='/registrationRoute')
authProfileRoute = Blueprint('authProfileRoute', __name__, url_prefix='/netwirkProfileRoute')
BLUEPRINTS = (authRegistrationRoute, authProfileRoute)

@authRegistrationRoute.route('/register_user', methods=['POST'])
@swag_from('registration/register_user.yml')
def register_new_user():
    register = registration()
    try:
        create = register.register_user()
        return create
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/check_email_exists', methods=['POST'])
@swag_from('registration/check_email_exists.yml')
def check_email_exists():
    register = registration()
    try:
        create = register.check_email_exists()
        return create
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/verify_otp', methods=['POST'])
@swag_from('registration/verify_otp.yml')
def verify_user_otp():
    register = registration()
    try:
        create = register.verify_otp()
        return create
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/resend_otp', methods=['POST'])
@swag_from('registration/resend_otp.yml')
def resend_user_otp():
    register = registration()
    try:
        return register.resend_otp()
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/send_phone_otp', methods=['POST'])
@swag_from('registration/send_phone_otp.yml')
def send_phone_otp_now():
    register = registration()
    try:
        return register.send_phone_otp()
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/verify_phone_otp', methods=['POST'])
@swag_from('registration/verify_phone_otp.yml')
def verify_phone_otp_now():
    register = registration()
    try:
        return register.verify_phone_otp()
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/checkPhoneNumberExists', methods=['POST'])
@swag_from('registration/checkPhoneNumberExists.yml')
def checkPhoneNumber():
    register = registration()
    try:
        create = register.check_phone_number_exists()
        return create
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/user_login', methods=['POST'])
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

@authRegistrationRoute.route('/user_logout', methods=['POST'])
def user_signout():
    register = registration()
    try:
        create = register.user_logout()
        return create
    except Exception as e:
        return "Error"

@authRegistrationRoute.route('/social_user_login', methods=['POST'])
@swag_from('registration/social_user_login.yml')
def social_user_signin():
    register = registration()
    try:
        create = register.social_user_login()
        return create
    except Exception as e:
        return "Error"

@authRegistrationRoute.route('/forgot_password', methods=['POST'])
@swag_from('registration/forgot_password.yml')
def user_forgot_password():
    register = registration()
    try:
        create = register.forgot_password()
        return create
    except Exception as e:
        return "Error"

@authRegistrationRoute.route('/confirm_new_password', methods=['POST'])
@swag_from('registration/confirm_new_password.yml')
def user_confirm_new_password():
    register = registration()
    try:
        create = register.confirm_new_password()
        return create
    except Exception as e:
        return "Error"

@authRegistrationRoute.route('/confirm_user_in_cognito', methods=['POST'])
@swag_from('registration/confirm_user_in_cognito.yml')
def confirm_new_user_in_cognito():
    register = registration()
    try:
        create = register.confirm_user_in_cognito()
        return create
    except Exception as e:
        return "Error"

@authRegistrationRoute.route('/update_authenticationId_using_userId', methods=['POST'])
@swag_from('registration/update_authenticationId_using_userId.yml')
def update_authenticationId_via_userId():
    register = registration()
    try:
        create = register.update_authenticationId_using_userId()
        return create
    except Exception as e:
        return "Error"

@authRegistrationRoute.route('/set_new_password', methods=['POST'])
@swag_from('registration/set_new_password.yml')
def set_new_password_now():
    register = registration()
    try:
        create = register.set_new_password()
        return create
    except Exception as e:
        return "Error"

@authRegistrationRoute.route('/register_apple_user', methods=['POST'])
@swag_from('registration/register_apple_user.yml')
def register_apple_user_now():
    register = registration()
    try:
        create = register.register_apple_user()
        return create
    except Exception as e:
        print(str(e))
        return "Error"

@authRegistrationRoute.route('/get_apple_user', methods=['POST'])
@swag_from('registration/get_apple_user.yml')
def get_apple_user_now():
    register = registration()
    try:
        create = register.get_apple_user()
        return create
    except Exception as e:
        print(str(e))
        return "Error"

@authProfileRoute.route('/retrieve_userId_via_authenticationId', methods=['POST'])
@swag_from('netwirkProfile/retrieve_userId_via_authenticationId.yml')
def retrieve_userId_via_authenticationId_now():
    netwirkProfile = profile()
    try:
        create = netwirkProfile.retrieve_userId_via_authenticationId()
        return create
    except:
        return "Error"
