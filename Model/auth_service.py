"""1. Auth & Identity microservice model."""

from flask import app, request, jsonify
from flask_jwt_extended import create_access_token
import requests
from datetime import datetime
import boto3
import hmac
import base64
import hashlib
from Model.netwirk_DB import DB
from botocore.exceptions import ClientError
import jwt
import time
import os
import json
import boto3
from flask import request, jsonify
import re
import logging
import bcrypt
import secrets
import random
import string
import uuid
import hashlib





import json
import time
from flask import app, jsonify, send_file, Response, session, request
from flask_jwt_extended import create_access_token
import requests
from datetime import datetime, timezone, timedelta
import boto3
import hmac
import base64
import hashlib
from Model.netwirk_DB import DB
import os
import qrcode
from PIL import Image, ImageOps, ImageStat, ImageFilter
import io
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import openai
import numpy as np
import ast
import threading
from collections import Counter
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import math
import mysql.connector
from collections import deque, defaultdict
import logging
import uuid 
from flask import Response, stream_with_context
import time
from collections import deque, defaultdict
import logging
import uuid
import queue  # Added for thread-safe SSE queues
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64 
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from datetime import datetime, date
from math import radians, cos, sin, asin, sqrt, atan2
from difflib import SequenceMatcher
import logging
# from sentence_transformers import SentenceTransformer
from Model.recsys.checkin_features import ALL_TAGS, primary_type_to_tag
from metrics import record_fallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Prompt embeddings are unused by the live feed. Leave this off so profile
# saves do not load or run all-MiniLM-L6-v2.
# model = SentenceTransformer('all-MiniLM-L6-v2') 

RECENCY_HALF_LIFE_DAYS = 14   # behaviour signal halves every 14 days
SKIP_PENALTY_WEIGHT    = 0.35  # how hard a hide penalises similar candidates
MIN_INTERACTIONS_FOR_DYNAMIC_WEIGHTS = 10
 
DEFAULT_WEIGHTS = {
    "prompt_sim":    0.28,
    "vibe":          0.20,
    "comm":          0.16,
    "intent":        0.16,
    "proximity":     0.10,
    "age":           0.05,
    "mutual_signal": 0.05,
}
 
COMM_STYLE_MAP = {
    "text-first": 0, "voice notes": 1, "in-person": 2, "mix of everything": 3
}
COMM_COMPAT = [
    [1.00, 0.80, 0.30, 0.70],
    [0.80, 1.00, 0.50, 0.75],
    [0.30, 0.50, 1.00, 0.65],
    [0.70, 0.75, 0.65, 1.00],
]
INTENT_MAP = {
    "long-term": 0, "long-term open to short": 1,
    "short-term": 2, "not sure yet": 3
}
INTENT_COMPAT = [
    [1.00, 0.85, 0.10, 0.60],
    [0.85, 1.00, 0.50, 0.70],
    [0.10, 0.50, 1.00, 0.55],
    [0.60, 0.70, 0.55, 1.00],
]

MATCHING_WEIGHTS_V2 = {
    "proximity": 0.25,
    "lifestyle": 0.20,
    "age": 0.15,
    "interest": 0.15,
    "intent": 0.10,
    "prompt": 0.10,
    "checkin_overlap": 0.05,
}


IST = ZoneInfo("Asia/Kolkata")
INACTIVITY_TIMEOUT_SECONDS = 180
CHECK_INTERVAL = 5  # how often to check for inactive users
SUPPORTED_METROS = [
    {"name": "new-york",    "displayName": "New York",    "lat": 40.7580, "lng": -73.9855,  "radiusKm": 10},
    {"name": "los-angeles", "displayName": "Los Angeles", "lat": 34.0522, "lng": -118.2437, "radiusKm": 10},
    {"name": "austin",      "displayName": "Austin",      "lat": 30.2672, "lng": -97.7431,  "radiusKm": 10},
    {"name": "vadodara",    "displayName": "Vadodara",    "lat": 22.3072, "lng": 73.1812,   "radiusKm": 10},
    # {"name": "frisco",      "displayName": "Frisco",      "lat": 33.1507, "lng": -96.8236,  "radiusKm": 10},
    {"name": "Chennai",      "displayName": "Chennai",    "lat": 12.9716, "lng": 80.2737,  "radiusKm": 10},
]

COMM_STYLE_MAP = {
    "Text-first — quick replies, low pressure":                             0,
    "Voice notes when typing feels like too much":                          1,
    "Call when it's important, text for everything else":                   2,
    "FaceTime / video calls once I'm comfortable":                          3,
    "In-person whenever possible":                                          4,
    "Flexible — I match the vibe and switch it up depending on the moment": 5,
}
 
INTENT_MAP = {
    "Long-term relationship":                      0,
    "Long-term relationship, open to short":        1,
    "Short-term relationship, open to long":        2,
    "Short-term relationship":                      3,
    "Figuring out my dating goals":                 4,
    "Dating that starts in real life, not in DMs":  5,
}
 
# Rows = my style, Cols = candidate's style (0.0 – 1.0)
COMM_COMPAT = [
#  Txt   Vce   Cal   Fce   Ipr   Flx
  [1.00, 0.80, 0.50, 0.30, 0.20, 0.85],  # Text-first
  [0.80, 1.00, 0.70, 0.50, 0.30, 0.85],  # Voice notes
  [0.50, 0.70, 1.00, 0.75, 0.50, 0.85],  # Call + text
  [0.30, 0.50, 0.75, 1.00, 0.70, 0.85],  # FaceTime
  [0.20, 0.30, 0.50, 0.70, 1.00, 0.85],  # In-person
  [0.85, 0.85, 0.85, 0.85, 0.85, 1.00],  # Flexible
]
 
# Rows = my intent, Cols = candidate's intent (0.0 – 1.0)
INTENT_COMPAT = [
#   LT    LTos  STol  ST    Fig   IRL
  [1.00, 0.85, 0.50, 0.10, 0.40, 0.60],  # Long-term
  [0.85, 1.00, 0.75, 0.40, 0.50, 0.65],  # Long-term open to short
  [0.50, 0.75, 1.00, 0.85, 0.50, 0.65],  # Short-term open to long
  [0.10, 0.40, 0.85, 1.00, 0.40, 0.55],  # Short-term
  [0.40, 0.50, 0.50, 0.40, 1.00, 0.50],  # Figuring out
  [0.60, 0.65, 0.65, 0.55, 0.50, 1.00],  # Real life dating
]

TAG_TYPE_MAP = {
    "cafe": "social", "bar": "social",
    "gym": "active", "park": "active",
    "restaurant": "foodie", "supermarket": "foodie", "grocery_or_supermarket": "foodie",
    "night_club": "nightlife", "lounge": "nightlife",
    "spa": "wellness", "yoga_studio": "wellness",
    "museum": "cultural", "art_gallery": "cultural",
    "hiking_area": "outdoor", "park_trail": "outdoor",
}

ALL_TAGS = ["social", "active", "foodie", "nightlife", "wellness", "cultural", "outdoor"]


from Model.common_base import ProfileBase, RegistrationBase, profile


class AuthRegistrationService(RegistrationBase):
    def _clear_device_token(self, cursor, user_id):
        cursor.execute(
            "UPDATE icebreaker.registration SET deviceToken = NULL WHERE userId = %s",
            (user_id,),
        )
        self.db.conn.commit()

    def register_user(self):
        try:
            data = request.get_json()
            # Input params
            full_name = data['fullName']
            email = self._normalize_email(data['email'])
            country_code = data.get('countryCode')
            phone_no = data.get('phoneNo')
            city = data.get('city')
            state = data.get('state')
            country = data.get('country')
            password = data['password']
            created_at = datetime.now()
            device_token = data['deviceToken'] 
            gender = data['Gender']
            preference = data.get('Preference')
            age_preference = data.get('age_preference')
            DOB = data['DOB']
            signupType = data['signupType']          

            cursor = self.db.cur()

            EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
  
            # Step 1: General syntax check
            if not EMAIL_REGEX.match(email):
                return {"Status": "Failure", "Message": "Invalid email format"}

            # Step 2: Domain-specific rule for Gmail
            domain = email.split("@")[1].lower()
            if domain.startswith("gmail.") and domain != "gmail.com":
                return {"Status": "Failure", "Message": "Invalid email format"}

            # Generating unique jwt token
            jwt_secret_key = self._jwt_secret_key()
            jwt_payload = {'email': email, 'timestamp': int(time.time())} 
            jwt_token = jwt.encode(jwt_payload, jwt_secret_key, algorithm='HS256')

            otp_sent = {"email": False, "phone": False}
            send_email = signupType != "social"
            send_phone = signupType != "social" and country_code and phone_no
            shared_otp = self._generate_otp() if (send_email and send_phone) else None

            if send_email:
                signup_result = self._register_cognito_user(email, password, otp=shared_otp)
                if signup_result.get("status") == "Failure":
                    return {
                        "Status": "Failure",
                        "Message": signup_result.get("message", "Cognito signup failed"),
                    }
                otp_sent["email"] = True

            phone_otp_info = None
            if send_phone:
                try:
                    phone_otp_info = self._send_phone_otp_internal(
                        email, country_code, phone_no, otp=shared_otp,
                    )
                    otp_sent["phone"] = True
                except Exception as otp_err:
                    logging.error("Phone OTP send failed for %s: %s", email, otp_err)

            if signupType == "social":
                if self._cognito_get_user_state(email) == "verified":
                    return {"Status": "Failure", "Message": "User already exists"}
                self.confirm_user_in_cognito(email)

            # Check if the email already exists
            cursor.execute("""
                SELECT userId FROM icebreaker.registration WHERE email = %s
            """, (email,))
            
            existing_user = cursor.fetchone()

            if device_token and str(device_token).strip():
                if existing_user:
                    self._detach_device_token_from_other_users(cursor, device_token, exclude_email=email)
                else:
                    self._detach_device_token_from_other_users(cursor, device_token)

            if existing_user:
                # Update existing record
                cursor.execute("""
                    UPDATE icebreaker.registration
                    SET fullName = %s, password = %s, createdAt = %s, deviceToken = %s, Gender = %s, Preference = %s, DOB = %s, countryCode = %s, phoneNo= %s, city = %s, state = %s, country = %s, signupType = %s
                    WHERE email = %s
                """, (full_name, password, created_at, device_token, gender, preference, DOB, country_code , phone_no, city, state, country, signupType, email))
                user_id = existing_user[0]
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO icebreaker.registration (fullName, email, password, createdAt, deviceToken, loginWebToken, Gender, Preference, countryCode , phoneNo, city, state, country, DOB, signupType)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (full_name, email, password, created_at, device_token, jwt_token, gender, preference, country_code, phone_no, city, state, country, DOB, signupType ))
            
            self.db.conn.commit()
            cursor.execute("""
                    SELECT userId, loginWebToken FROM icebreaker.registration WHERE email = %s 
            """,
            (email,))

            userId = cursor.fetchall()
                 
            user_id = userId[0][0]

            cursor.execute("""
                INSERT INTO icebreaker.profile (userId, notify_connection, background_notify, age_preference)
                SELECT %s, %s, %s, %s
                WHERE NOT EXISTS (SELECT 1 FROM icebreaker.profile WHERE userId = %s)
            """, (user_id, False, False, age_preference, user_id))
            
            self.db.conn.commit()

            loginWebToken = userId[0][1]

            self.db.conn.close()

            status_success = 'Success'
            message_success = 'User registered successfully'

            response = {
                'Status': status_success,
                'Message': message_success,
                'userId': user_id,
                'loginWebToken': loginWebToken,
                'otpSent': otp_sent,
            }
            if phone_otp_info:
                response['otpSent']['phoneOtp'] = phone_otp_info['otp']
                response['otpSent']['phoneE164'] = phone_otp_info['phone_e164']
            if otp_sent['email'] and otp_sent['phone']:
                response['otpSent']['sharedOtp'] = True
            if otp_sent['email']:
                response['otpSent']['emailTo'] = email
                if self._uses_ses_for_email_otp():
                    response['otpSent']['emailOtpNote'] = 'Sent via Amazon SES (check inbox)'
                else:
                    response['otpSent']['emailOtpNote'] = 'Sent via Cognito email (check inbox)'

            return response

        except Exception as e:
            message_failure = str(e)
            status_failure = 'Failure'

            return{'Status' : status_failure,  'Message' : message_failure}

    def check_email_exists(self):
        try:
            # === Load AWS credentials from JSON file ===
            aws_config = self._get_aws_config()

            cursor = self.db.cur()
            requestData = request.get_json() or {}
            email = requestData.get('email')
            country_code = requestData.get('countryCode')
            phone_no = requestData.get('phoneNo')

            def is_null(value):
                return value is None or (isinstance(value, str) and not value.strip())

            email_flag = 0
            message = "Email does not exist"
            signup_type = None

            if is_null(email):
                message = "Email is null"
            else:
                client = boto3.client(
                    'cognito-idp',
                    region_name=aws_config["aws_region"],
                    aws_access_key_id=aws_config["aws_access_key_id"],
                    aws_secret_access_key=aws_config["aws_secret_access_key"]
                )

                try:
                    response = client.admin_get_user(
                        UserPoolId=aws_config["userpool_id"],
                        Username=email
                    )

                    confirmation_status = None
                    for attr in response['UserAttributes']:
                        if attr['Name'] == 'email_verified':
                            confirmation_status = attr['Value']

                    user_status = response.get('UserStatus', '')
                    if confirmation_status == 'true' or user_status == 'CONFIRMED':
                        email_flag = 2
                        message = "Email already exists & verified"
                    elif confirmation_status == 'false':
                        email_flag = 1
                        message = "Email exists but not confirmed"
                except client.exceptions.UserNotFoundException:
                    pass  # Email does not exist in Cognito

                cursor.execute("""
                    SELECT signupType FROM icebreaker.registration WHERE email = %s
                """, (email,))
                result = cursor.fetchone()
                if result:
                    signup_type = result[0]

            phone_no_flag = 0
            phone_message = "Phone number does not exist"
            country_null = is_null(country_code)
            phone_null = is_null(phone_no)

            if country_null and phone_null:
                phone_message = "Country code and phone number are null"
            elif country_null:
                phone_message = "Country code is null"
            elif phone_null:
                phone_message = "Phone number is null"
            else:
                phone_no_flag, phone_message = self._get_phone_verification_status(
                    cursor, country_code, phone_no
                )

            return jsonify({
                'message': message,
                'email_flag': email_flag,
                'signupType': signup_type,
                'phoneNo_flag': phone_no_flag,
                'phone_message': phone_message,
            })

        except Exception as e:
            return jsonify({'message': str(e), 'status': 'Failure'})

    def _normalize_otp_input(self, value):
        if value is None:
            return None
        otp_value = str(value).strip()
        return otp_value if otp_value else None

    def _build_verification_message(self, phone_verified, email_verified, channel):
        if phone_verified and email_verified:
            return "Both phone and email are verified."
        if phone_verified and channel == 'both':
            return "Phone is verified but email is not verified."
        if email_verified and channel == 'both':
            return "Email is verified but phone is not verified."
        if phone_verified:
            return "Phone is verified."
        if email_verified:
            return "Email is verified."
        if channel == 'both':
            return "Phone and email are not verified."
        if channel == 'phone':
            return "Phone is not verified."
        return "Email is not verified."

    def _verification_response(
        self,
        cursor,
        email,
        phone_verified=False,
        email_verified=False,
        channel='both',
        status='Success',
        message=None,
        error_message='',
    ):
        if phone_verified:
            self._mark_registration_phone_verified(cursor, email)

        cursor.execute("SELECT userId FROM icebreaker.registration WHERE email = %s", (email,))
        result = cursor.fetchone()
        return jsonify({
            'status': status,
            'message': message or self._build_verification_message(
                phone_verified, email_verified, channel
            ),
            'error_message': error_message or '',
            'userId': result[0] if result else None,
        })

    def verify_otp(self):
        """Verify OTP by channel: phone, email (Cognito), or both."""
        phone_e164_on_miss = None
        phone_verified_email = None
        try:
            cfg = self._get_aws_config()
            cursor = self.db.cur()
            data = request.get_json() or {}
            email = self._normalize_email(data['email'])
            country_code = data.get('countryCode')
            phone_no = data.get('phoneNo')
            channel = (data.get('channel') or 'both').strip().lower()
            if channel == 'otp':
                channel = 'email'

            if channel not in ('phone', 'email', 'both'):
                return jsonify({
                    'status': 'Failure',
                    'message': 'Phone and email are not verified.',
                    'error_message': 'Invalid channel. Use phone, email, or both.',
                    'userId': None,
                }), 400

            phone_otp = self._normalize_otp_input(
                data.get('phoneOTP') or data.get('phoneOtp')
            )
            email_otp = self._normalize_otp_input(
                data.get('emailOTP') or data.get('emailOtp')
            )
            single_otp = self._normalize_otp_input(
                data.get('otp') or data.get('OTP')
            )
            if channel == 'both' and single_otp and not phone_otp and not email_otp:
                phone_otp = single_otp
                email_otp = single_otp
            elif channel == 'both' and phone_otp and not email_otp:
                email_otp = phone_otp
            elif channel == 'both' and email_otp and not phone_otp:
                phone_otp = email_otp

            verify_phone = channel in ('phone', 'both')
            verify_email = channel in ('email', 'both')

            if verify_phone and not phone_otp:
                return jsonify({
                    'status': 'Failure',
                    'message': self._build_verification_message(False, False, channel),
                    'error_message': 'phoneOTP is required for phone verification.',
                    'userId': None,
                }), 400
            if verify_email and not email_otp:
                return jsonify({
                    'status': 'Failure',
                    'message': self._build_verification_message(False, False, channel),
                    'error_message': 'emailOTP is required for email verification.',
                    'userId': None,
                }), 400

            if verify_email and channel == 'email' and self._cognito_is_confirmed(email):
                return self._verification_response(
                    cursor, email, phone_verified=False, email_verified=True, channel=channel,
                    message='Email is verified.',
                )

            if verify_phone:
                if not (country_code and phone_no):
                    cursor.execute(
                        "SELECT countryCode, phoneNo FROM icebreaker.registration WHERE email = %s",
                        (email,),
                    )
                    reg = cursor.fetchone()
                    if reg and reg[0] and reg[1]:
                        country_code, phone_no = reg[0], reg[1]

                phone_status, phone_result = self._attempt_phone_otp_verification(
                    email, country_code, phone_no, phone_otp, cursor
                )
                if phone_status == 'verified':
                    phone_verified_email = phone_result
                    if channel == 'phone':
                        return self._verification_response(
                            cursor, phone_result, phone_verified=True, email_verified=False,
                            channel=channel,
                        )
                elif phone_status == 'error':
                    error_body = phone_result.get_json() if hasattr(phone_result, 'get_json') else {}
                    return jsonify({
                        'status': 'Failure',
                        'message': self._build_verification_message(False, False, channel),
                        'error_message': error_body.get('message', 'Phone verification failed.'),
                        'userId': None,
                    }), phone_result.status_code if hasattr(phone_result, 'status_code') else 400
                else:
                    phone_e164_on_miss = phone_result
                    return jsonify({
                        'status': 'Failure',
                        'message': self._build_verification_message(False, False, channel),
                        'error_message': 'Phone OTP is invalid.',
                        'userId': None,
                    })

            if verify_email:
                if self._cognito_is_confirmed(email):
                    return self._verification_response(
                        cursor, email,
                        phone_verified=bool(phone_verified_email),
                        email_verified=True,
                        channel=channel,
                    )

                if self._uses_ses_for_email_otp():
                    ok, err = self._verify_email_otp_ses(cursor, email, email_otp)
                    if not ok:
                        return jsonify({
                            'status': 'Failure',
                            'message': 'Email is not verified.',
                            'error_message': err,
                            'userId': None,
                        })
                else:
                    client = self._get_cognito_client()
                    client.confirm_sign_up(
                        ClientId=cfg["app_client_id"],
                        SecretHash=self.get_secret_hash(email),
                        Username=email,
                        ConfirmationCode=email_otp,
                    )
                return self._verification_response(
                    cursor,
                    phone_verified_email or email,
                    phone_verified=bool(phone_verified_email),
                    email_verified=True,
                    channel=channel,
                )

            return jsonify({
                'status': 'Failure',
                'message': 'Phone and email are not verified.',
                'error_message': 'Invalid channel.',
                'userId': None,
            }), 400

        except ClientError as e:
            error_message = e.response.get("Error", {}).get("Message", "Verification failed.")
            req_data = request.get_json() or {}
            email = req_data.get("email")
            channel = (req_data.get('channel') or 'both').strip().lower()
            if channel == 'otp':
                channel = 'email'
            verify_email = channel in ('email', 'both')
            if (
                verify_email
                and email
                and e.response.get("Error", {}).get("Code", "") in ("NotAuthorizedException", "AliasExistsException")
                and self._cognito_is_confirmed(email)
            ):
                cursor = self.db.cur()
                return self._verification_response(
                    cursor, email,
                    phone_verified=bool(phone_verified_email),
                    email_verified=True,
                    channel=channel,
                )

            if phone_e164_on_miss:
                cursor = self.db.cur()
                self._ensure_phone_otp_table(cursor)
                cursor.execute(
                    "UPDATE icebreaker.phone_otp_store SET attempts = attempts + 1 WHERE phone_e164 = %s",
                    (phone_e164_on_miss,),
                )
                self.db.conn.commit()

            logging.error("verify_otp error: %s", e)

            if channel == 'phone':
                return jsonify({
                    'status': 'Failure',
                    'message': 'Phone is not verified.',
                    'error_message': 'Phone OTP is invalid.',
                    'userId': None,
                })
            if channel == 'email':
                return jsonify({
                    'status': 'Failure',
                    'message': 'Email is not verified.',
                    'error_message': error_message or 'Email OTP is invalid.',
                    'userId': None,
                })
            if channel == 'both' and phone_verified_email:
                cursor = self.db.cur()
                self._mark_registration_phone_verified(cursor, phone_verified_email)
                cursor.execute(
                    "SELECT userId FROM icebreaker.registration WHERE email = %s",
                    (phone_verified_email,),
                )
                user_row = cursor.fetchone()
                return jsonify({
                    'status': 'Failure',
                    'message': 'Phone is verified but email is not verified.',
                    'error_message': error_message or 'Email OTP is invalid.',
                    'userId': user_row[0] if user_row else None,
                })
            return jsonify({
                'status': 'Failure',
                'message': 'Phone and email are not verified.',
                'error_message': error_message or 'Verification failed. Please check your OTP.',
                'userId': None,
            })
        except Exception as e:
            logging.error("verify_otp error: %s", e)
            return jsonify({
                'status': 'Failure',
                'message': 'Phone and email are not verified.',
                'error_message': str(e),
                'userId': None,
            })

    def _uses_ses_for_email_otp(self) -> bool:
        return bool(self._get_aws_config().get("use_ses"))

    def _get_ses_client(self):
        cfg = self._get_aws_config()
        return boto3.client(
            'ses',
            region_name=cfg["aws_region"],
            aws_access_key_id=cfg["aws_access_key_id"],
            aws_secret_access_key=cfg["aws_secret_access_key"],
        )

    def _ses_sender_email(self):
        cfg = self._get_aws_config()
        sender = cfg.get("ses_sender_email") or cfg.get("email_from")
        if not sender:
            raise RuntimeError("ses_sender_email or email_from missing from registration config")
        return sender.strip()

    def _ses_source_address(self):
        cfg = self._get_aws_config()
        sender = self._ses_sender_email()
        from_name = (cfg.get("ses_from_name") or "VYB").strip()
        return f"{from_name} <{sender}>"

    def _ensure_email_otp_table(self, cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS icebreaker.otp_store (
                email VARCHAR(255) PRIMARY KEY,
                otp VARCHAR(6) NOT NULL,
                attempts INT NOT NULL DEFAULT 0,
                expires_at DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        self.db.conn.commit()

    def _send_otp_email(self, email: str, otp: str, purpose: str = "verification"):
        client = self._get_ses_client()
        to_email = self._normalize_email(email)
        expiry = self.OTP_EXPIRY_MINUTES
        if purpose == "password_reset":
            subject = "Your VYB password reset code"
            headline = "Reset your password"
            intro = "Use the code below to reset your password."
            text_body = (
                f"Your VYB password reset code is: {otp}\n"
                f"It expires in {expiry} minutes.\n"
                "Do not share this code with anyone."
            )
        else:
            subject = "Your VYB verification code"
            headline = "Verify your email"
            intro = "Use the code below to complete your registration."
            text_body = (
                f"Your VYB verification code is: {otp}\n"
                f"It expires in {expiry} minutes.\n"
                "Do not share this code with anyone."
            )

        html_body = f"""
            <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;">
                <h2>{headline}</h2>
                <p>{intro} It expires in <strong>{expiry} minutes</strong>.</p>
                <div style="font-size:36px;font-weight:bold;letter-spacing:8px;padding:16px 0;">
                    {otp}
                </div>
                <p style="color:#888;font-size:12px;">Do not share this code with anyone.</p>
            </div>
        """

        response = client.send_email(
            Source=self._ses_source_address(),
            Destination={'ToAddresses': [to_email]},
            ReplyToAddresses=[self._ses_sender_email()],
            Message={
                'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                'Body': {
                    'Text': {'Data': text_body, 'Charset': 'UTF-8'},
                    'Html': {'Data': html_body, 'Charset': 'UTF-8'},
                },
            },
        )
        message_id = response.get("MessageId")
        logging.info(
            "OTP email accepted by SES to=%s from=%s purpose=%s message_id=%s",
            to_email, self._ses_sender_email(), purpose, message_id,
        )
        return message_id

    def _store_and_send_email_otp(self, email: str, purpose: str = "verification", otp: str = None):
        otp = otp or self._generate_otp()
        cursor = self.db.cur()
        try:
            self._ensure_email_otp_table(cursor)
            cursor.execute("""
                INSERT INTO icebreaker.otp_store (email, otp, attempts, expires_at)
                VALUES (%s, %s, 0, NOW() + INTERVAL %s MINUTE)
                ON DUPLICATE KEY UPDATE
                    otp = VALUES(otp),
                    attempts = 0,
                    expires_at = VALUES(expires_at)
            """, (email, otp, self.OTP_EXPIRY_MINUTES))
            self.db.conn.commit()
        finally:
            cursor.close()

        self._send_otp_email(email, otp, purpose=purpose)

    def _validate_otp_from_store(self, cursor, email: str, otp: str):
        self._ensure_email_otp_table(cursor)
        cursor.execute("""
            SELECT otp, expires_at, attempts
            FROM icebreaker.otp_store
            WHERE email = %s
        """, (email,))
        result = cursor.fetchone()
        if not result:
            return False, 'OTP not found. Please request a new one.'

        stored_otp, expires_at, attempts = result
        if attempts >= self.MAX_OTP_ATTEMPTS:
            return False, 'Too many failed attempts. Please request a new OTP.'

        cursor.execute("SELECT NOW()")
        if cursor.fetchone()[0] > expires_at:
            return False, 'OTP has expired. Please request a new one.'

        if stored_otp != otp:
            cursor.execute(
                "UPDATE icebreaker.otp_store SET attempts = attempts + 1 WHERE email = %s",
                (email,),
            )
            self.db.conn.commit()
            return False, 'Invalid confirmation code. Please check and try again.'

        cursor.execute("DELETE FROM icebreaker.otp_store WHERE email = %s", (email,))
        self.db.conn.commit()
        return True, None

    def _verify_email_otp_ses(self, cursor, email: str, email_otp: str):
        ok, err = self._validate_otp_from_store(cursor, email, email_otp)
        if not ok:
            if err and 'Invalid confirmation code' in err:
                return False, 'Email OTP is invalid.'
            return ok, err
        self._cognito_admin_confirm(email)
        return True, None

    def _verify_password_reset_otp_ses(self, cursor, email: str, otp: str):
        return self._validate_otp_from_store(cursor, email, otp)

    def _cognito_set_user_password(self, email: str, password: str):
        cfg = self._get_aws_config()
        self._get_cognito_client().admin_set_user_password(
            UserPoolId=cfg["userpool_id"],
            Username=email,
            Password=password,
            Permanent=True,
        )

    def _cognito_get_user_state(self, email: str):
        """Return None, 'verified', or 'unverified' for a Cognito user."""
        cfg = self._get_aws_config()
        client = self._get_cognito_client()
        try:
            response = client.admin_get_user(
                UserPoolId=cfg["userpool_id"],
                Username=email,
            )
            if response.get("UserStatus") == "CONFIRMED":
                return "verified"
            for attr in response.get("UserAttributes", []):
                if attr.get("Name") == "email_verified" and attr.get("Value") == "true":
                    return "verified"
            return "unverified"
        except client.exceptions.UserNotFoundException:
            return None

    def _cognito_is_confirmed(self, email: str) -> bool:
        return self._cognito_get_user_state(email) == "verified"

    def _cognito_refresh_unverified_user(self, email, password, otp: str = None):
        """Update password and resend email OTP for an unverified Cognito user."""
        try:
            cfg = self._get_aws_config()
            client = self._get_cognito_client()
            client.admin_set_user_password(
                UserPoolId=cfg["userpool_id"],
                Username=email,
                Password=password,
                Permanent=True,
            )
            if self._uses_ses_for_email_otp():
                try:
                    self._store_and_send_email_otp(email, otp=otp)
                except Exception as e:
                    logging.error("SES OTP send failed for %s: %s", email, e)
                    return {"status": "Failure", "message": "Failed to send verification email."}
            else:
                client.resend_confirmation_code(
                    ClientId=cfg["app_client_id"],
                    SecretHash=self.get_secret_hash(email),
                    Username=email,
                )
            logging.info("Refreshed unverified Cognito user and resent OTP for %s", email)
            return {"status": "Success"}
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            message = e.response.get("Error", {}).get("Message", str(e))
            logging.error("Cognito refresh unverified user failed (%s): %s", code, message)
            return {"status": "Failure", "message": message, "code": code}
        except Exception as e:
            logging.error("Cognito refresh unverified user error: %s", e)
            return {"status": "Failure", "message": str(e)}

    def _register_cognito_user_ses(self, email, password, otp: str = None):
        """Create or refresh Cognito user without Cognito email; OTP sent via SES."""
        cognito_state = self._cognito_get_user_state(email)
        if cognito_state == "verified":
            return {"status": "Failure", "message": "User already exists"}
        if cognito_state == "unverified":
            return self._cognito_refresh_unverified_user(email, password, otp=otp)

        cfg = self._get_aws_config()
        client = self._get_cognito_client()
        try:
            client.admin_create_user(
                UserPoolId=cfg["userpool_id"],
                Username=email,
                UserAttributes=[{'Name': 'email', 'Value': email}],
                MessageAction='SUPPRESS',
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "UsernameExistsException":
                cognito_state = self._cognito_get_user_state(email)
                if cognito_state == "verified":
                    return {"status": "Failure", "message": "User already exists"}
                if cognito_state == "unverified":
                    return self._cognito_refresh_unverified_user(email, password, otp=otp)
            message = e.response.get("Error", {}).get("Message", str(e))
            return {"status": "Failure", "message": message, "code": code}

        client.admin_set_user_password(
            UserPoolId=cfg["userpool_id"],
            Username=email,
            Password=password,
            Permanent=True,
        )
        try:
            self._store_and_send_email_otp(email, otp=otp)
        except Exception as e:
            logging.error("SES OTP send failed for %s: %s", email, e)
            return {"status": "Failure", "message": "Failed to send verification email."}
        logging.info("Created unverified Cognito user and sent SES OTP for %s", email)
        return {"status": "Success"}

    def _register_cognito_user(self, email, password, otp: str = None):
        """Create or refresh Cognito signup for registration."""
        if self._uses_ses_for_email_otp():
            return self._register_cognito_user_ses(email, password, otp=otp)

        cognito_state = self._cognito_get_user_state(email)
        if cognito_state == "verified":
            return {"status": "Failure", "message": "User already exists"}

        if cognito_state == "unverified":
            return self._cognito_refresh_unverified_user(email, password, otp=otp)

        signup_result = self.sign_up_with_otp(email, password)
        if isinstance(signup_result, dict) and signup_result.get("status") == "Failure":
            if signup_result.get("code") == "UsernameExistsException":
                cognito_state = self._cognito_get_user_state(email)
                if cognito_state == "verified":
                    return {"status": "Failure", "message": "User already exists"}
                if cognito_state == "unverified":
                    return self._cognito_refresh_unverified_user(email, password, otp=otp)
            return signup_result

        return {"status": "Success"}

    def _cognito_admin_confirm(self, email: str):
        if self._cognito_is_confirmed(email):
            return
        cfg = self._get_aws_config()
        self._get_cognito_client().admin_confirm_sign_up(
            UserPoolId=cfg["userpool_id"],
            Username=email,
        )

    def _phone_lookup_digits(self, country_code, phone_no):
        cc_digits = re.sub(r"\D", "", country_code or "")
        pn_digits = re.sub(r"\D", "", phone_no or "")
        return cc_digits, pn_digits

    def _find_registrations_by_phone(self, cursor, country_code, phone_no):
        """Find registration rows matching countryCode + phoneNo (format-tolerant)."""
        cc_digits, pn_digits = self._phone_lookup_digits(country_code, phone_no)
        if not cc_digits or not pn_digits:
            return []

        lookup_pairs = []
        seen = set()
        for cc in {country_code, cc_digits, f"+{cc_digits}"}:
            if not cc:
                continue
            for pn in {phone_no, pn_digits}:
                if not pn:
                    continue
                key = (cc, pn)
                if key not in seen:
                    seen.add(key)
                    lookup_pairs.append(key)

        matches = []
        matched_emails = set()
        for cc, pn in lookup_pairs:
            cursor.execute("""
                SELECT email, countryCode, phoneNo, signupType, is_verified
                FROM icebreaker.registration
                WHERE countryCode = %s AND phoneNo = %s
            """, (cc, pn))
            for row in cursor.fetchall():
                if row[0] not in matched_emails:
                    matched_emails.add(row[0])
                    matches.append(row)

        if matches:
            return matches

        cursor.execute("""
            SELECT email, countryCode, phoneNo, signupType, is_verified
            FROM icebreaker.registration
            WHERE countryCode IS NOT NULL AND phoneNo IS NOT NULL
        """)
        return [
            row for row in cursor.fetchall()
            if self._phone_lookup_digits(row[1], row[2]) == (cc_digits, pn_digits)
        ]

    def _get_phone_verification_status(self, cursor, country_code, phone_no):
        """Return phoneNo_flag (0/1/2) and message for a countryCode + phoneNo pair.
        is_verified column uses 1 = verified, 0 = not verified.
        """
        rows = self._find_registrations_by_phone(cursor, country_code, phone_no)
        if not rows:
            return 0, "Phone number does not exist"
        if any(row[4] == 1 for row in rows):
            return 2, "Phone number already exists and verified"
        return 1, "Phone number exists but not verified"

    def _mark_registration_phone_verified(self, cursor, email):
        cursor.execute(
            "UPDATE icebreaker.registration SET is_verified = 1 WHERE email = %s",
            (email,),
        )
        self.db.conn.commit()

    def _generate_otp(self) -> str:
        return str(secrets.randbelow(1000000)).zfill(6)

    def _normalize_phone_e164(self, country_code: str, phone_no: str) -> str:
        cc = re.sub(r"\D", "", country_code or "")
        pn = re.sub(r"\D", "", phone_no or "")
        if not cc or not pn:
            raise ValueError("Invalid phone number")
        return f"+{cc}{pn}"

    def _ensure_phone_otp_table(self, cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS icebreaker.phone_otp_store (
                phone_e164 VARCHAR(20) PRIMARY KEY,
                otp VARCHAR(6) NOT NULL,
                email VARCHAR(255) NULL,
                attempts INT NOT NULL DEFAULT 0,
                expires_at DATETIME NOT NULL,
                verified TINYINT(1) NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = 'icebreaker' AND TABLE_NAME = 'phone_otp_store'
        """)
        existing_columns = {row[0] for row in cursor.fetchall()}
        migrations = {
            "email": "ALTER TABLE icebreaker.phone_otp_store ADD COLUMN email VARCHAR(255) NULL",
            "attempts": "ALTER TABLE icebreaker.phone_otp_store ADD COLUMN attempts INT NOT NULL DEFAULT 0",
            "verified": "ALTER TABLE icebreaker.phone_otp_store ADD COLUMN verified TINYINT(1) NOT NULL DEFAULT 0",
            "created_at": "ALTER TABLE icebreaker.phone_otp_store ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "ALTER TABLE icebreaker.phone_otp_store ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
        }
        for column, ddl in migrations.items():
            if column not in existing_columns:
                cursor.execute(ddl)
        self.db.conn.commit()

    def _send_otp_sms(self, phone_e164: str, otp: str):
        from Model.ringcentral_sms import RingCentralSMS

        RingCentralSMS().send_sms(
            phone_e164,
            f"Your VYB verification code is: {otp}. "
            f"It expires in {self.OTP_EXPIRY_MINUTES} minutes. Do not share this code.",
        )

    def _send_phone_otp_internal(self, email, country_code, phone_no, otp: str = None):
        phone_e164 = self._normalize_phone_e164(country_code, phone_no)
        otp = otp or self._generate_otp()
        cursor = self.db.cur()
        try:
            self._ensure_phone_otp_table(cursor)
            cursor.execute("""
                INSERT INTO icebreaker.phone_otp_store
                    (phone_e164, otp, email, attempts, expires_at, verified)
                VALUES (%s, %s, %s, 0, NOW() + INTERVAL %s MINUTE, 0)
                ON DUPLICATE KEY UPDATE
                    otp = VALUES(otp), email = VALUES(email),
                    attempts = 0, expires_at = VALUES(expires_at), verified = 0
            """, (phone_e164, otp, email, self.OTP_EXPIRY_MINUTES))
            self.db.conn.commit()
        finally:
            cursor.close()
        self._send_otp_sms(phone_e164, otp)
        logging.info(
            "SMS OTP sent to %s for %s — OTP: %s (expires in %s min)",
            phone_e164,
            email or "unknown",
            otp,
            self.OTP_EXPIRY_MINUTES,
        )
        return {"phone_e164": phone_e164, "otp": otp}

    def send_phone_otp(self):
        """Send RingCentral SMS OTP (separate from Cognito email code)."""
        try:
            data = request.get_json() or {}
            country_code = data.get("countryCode")
            phone_no = data.get("phoneNo")
            email = data.get("email")
            if not country_code or not phone_no:
                return jsonify({
                    "status": "Failure",
                    "message": "countryCode and phoneNo are required.",
                }), 400

            self._send_phone_otp_internal(email, country_code, phone_no)
            phone_e164 = self._normalize_phone_e164(country_code, phone_no)
            return jsonify({
                "status": "Success",
                "message": "Phone OTP sent via SMS.",
                "phone_e164": phone_e164,
            })
        except ValueError as e:
            return jsonify({"status": "Failure", "message": str(e)}), 400
        except FileNotFoundError:
            return jsonify({
                "status": "Failure",
                "message": "RingCentral is not configured.",
            }), 500
        except Exception as e:
            logging.error("send_phone_otp error: %s", e)
            return jsonify({"status": "Failure", "message": "Failed to send phone OTP."}), 500

    def _normalize_otp(self, otp) -> str:
        return str(otp or "").strip().zfill(6)

    def _fetch_phone_otp_row(self, cursor, email, country_code=None, phone_no=None):
        """Find latest phone OTP row by E.164 and/or email."""
        phone_e164 = None
        if country_code and phone_no:
            try:
                phone_e164 = self._normalize_phone_e164(country_code, phone_no)
                cursor.execute("""
                    SELECT phone_e164, otp, expires_at, attempts, verified, email
                    FROM icebreaker.phone_otp_store
                    WHERE phone_e164 = %s
                """, (phone_e164,))
                row = cursor.fetchone()
                if row:
                    return row
            except ValueError:
                pass

        if email:
            cursor.execute("""
                SELECT phone_e164, otp, expires_at, attempts, verified, email
                FROM icebreaker.phone_otp_store
                WHERE email = %s
                ORDER BY expires_at DESC
                LIMIT 1
            """, (email,))
            row = cursor.fetchone()
            if row:
                return row

        return None

    def _attempt_phone_otp_verification(self, email, country_code, phone_no, otp, cursor):
        """
        Try RingCentral phone OTP.
        Returns ('verified', link_email), ('try_email', phone_e164_or_None), or ('error', response).
        """
        self._ensure_phone_otp_table(cursor)
        row = self._fetch_phone_otp_row(cursor, email, country_code, phone_no)
        if not row:
            logging.info("Phone OTP verify: no record for email=%s", email)
            return ('try_email', None)

        phone_e164, stored_otp, expires_at, attempts, verified, stored_email = row
        link_email = email or stored_email
        otp_norm = self._normalize_otp(otp)
        stored_norm = self._normalize_otp(stored_otp)

        logging.info(
            "Phone OTP verify: email=%s phone=%s stored=%s provided=%s attempts=%s",
            link_email,
            phone_e164,
            stored_norm,
            otp_norm,
            attempts,
        )

        if verified and link_email and self._cognito_is_confirmed(link_email):
            return ('verified', link_email)

        cursor.execute(
            "SELECT expires_at IS NOT NULL AND expires_at <= NOW() AS expired "
            "FROM icebreaker.phone_otp_store WHERE phone_e164 = %s",
            (phone_e164,),
        )
        expired = cursor.fetchone()
        if expired and expired[0]:
            logging.info("Phone OTP expired for %s", phone_e164)
            return ('try_email', phone_e164)

        if attempts >= self.MAX_OTP_ATTEMPTS:
            logging.info("Phone OTP max attempts reached for %s", phone_e164)
            return ('try_email', phone_e164)

        if stored_norm != otp_norm:
            logging.info("Phone OTP mismatch for %s", phone_e164)
            return ('try_email', phone_e164)

        if not link_email:
            cursor.execute(
                "SELECT email FROM icebreaker.registration WHERE email = %s",
                (email,),
            )
            reg = cursor.fetchone()
            link_email = reg[0] if reg else None

        if not link_email:
            return ('error', jsonify({
                'message': 'email is required to complete AWS verification.',
                'status': 'Failure',
            }))

        cursor.execute("""
            UPDATE icebreaker.phone_otp_store
            SET verified = 1, attempts = 0, email = %s WHERE phone_e164 = %s
        """, (link_email, phone_e164))
        self.db.conn.commit()
        logging.info("Phone OTP verified for %s", link_email)
        return ('verified', link_email)

    def verify_phone_otp(self):
        """Alias for verify_otp (phone first, then email)."""
        return self.verify_otp()

    def resend_otp(self):
        """Resend OTP by channel: phone (SMS), email (Cognito), or both."""
        try:
            data = request.get_json() or {}
            email = self._normalize_email(data.get('email') or '')
            country_code = (data.get('countryCode') or '').strip()
            phone_no = (data.get('phoneNo') or '').strip()
            channel = (data.get('channel') or 'both').strip().lower()
            if channel == 'otp':
                channel = 'email'

            if channel not in ('phone', 'email', 'both'):
                return jsonify({
                    'status': 'Failure',
                    'message': 'Invalid channel. Use phone or email.',
                }), 400

            cursor = self.db.cur()
            try:
                if channel == 'phone':
                    rows = []
                    if country_code and phone_no:
                        rows = self._find_registrations_by_phone(cursor, country_code, phone_no)
                    if not rows and email:
                        cursor.execute(
                            """
                            SELECT email, countryCode, phoneNo, signupType, is_verified
                            FROM icebreaker.registration
                            WHERE email = %s
                            """,
                            (email,),
                        )
                        reg = cursor.fetchone()
                        if reg and reg[1] and reg[2]:
                            rows = [reg]

                    if not rows:
                        if not country_code or not phone_no:
                            if not email:
                                return jsonify({
                                    'status': 'Failure',
                                    'message': 'countryCode and phoneNo are required.',
                                }), 400
                        return jsonify({'status': 'Failure', 'message': 'Phone number not registered.'}), 404

                    reg = next((row for row in rows if row[4] != 1), None)
                    if reg is None:
                        return jsonify({'status': 'Failure', 'message': 'Phone is already verified.'}), 400
                    email = reg[0]
                    country_code = reg[1]
                    phone_no = reg[2]
                    signup_type = reg[3]
                    phone_verified = reg[4] == 1
                else:
                    email_regex = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
                    if not email_regex.match(email):
                        return jsonify({'status': 'Failure', 'message': 'Invalid email format.'}), 400

                    domain = email.split("@")[1].lower()
                    if domain.startswith("gmail.") and domain != "gmail.com":
                        return jsonify({'status': 'Failure', 'message': 'Invalid email format.'}), 400

                    cursor.execute(
                        """
                        SELECT email, countryCode, phoneNo, signupType, is_verified
                        FROM icebreaker.registration
                        WHERE email = %s
                        """,
                        (email,),
                    )
                    reg = cursor.fetchone()
                    if not reg:
                        return jsonify({'status': 'Failure', 'message': 'Email not registered.'}), 404

                    email = reg[0]
                    country_code = country_code or reg[1]
                    phone_no = phone_no or reg[2]
                    signup_type = reg[3]
                    phone_verified = reg[4] == 1
            finally:
                cursor.close()

            resend_email = channel in ('email', 'both')
            resend_phone = channel in ('phone', 'both')
            shared_otp = self._generate_otp() if (resend_email and resend_phone) else None

            email_sent = False
            sms_sent = False
            phone_e164 = None
            phone_otp = None

            if resend_email:
                if self._cognito_is_confirmed(email):
                    return jsonify({'status': 'Failure', 'message': 'Email is already verified.'}), 400

                if self._uses_ses_for_email_otp():
                    self._store_and_send_email_otp(email, otp=shared_otp)
                else:
                    cfg = self._get_aws_config()
                    self._get_cognito_client().resend_confirmation_code(
                        ClientId=cfg["app_client_id"],
                        SecretHash=self.get_secret_hash(email),
                        Username=email,
                    )
                email_sent = True
                logging.info("Email OTP resent to %s", email)

            if resend_phone:
                if phone_verified:
                    return jsonify({'status': 'Failure', 'message': 'Phone is already verified.'}), 400
                if signup_type == "social":
                    return jsonify({
                        'status': 'Failure',
                        'message': 'Phone OTP not available for social signup.',
                    }), 400
                if not country_code or not phone_no:
                    return jsonify({'status': 'Failure', 'message': 'Phone number not registered.'}), 400

                phone_otp_info = self._send_phone_otp_internal(
                    email, country_code, phone_no, otp=shared_otp,
                )
                phone_e164 = phone_otp_info['phone_e164']
                phone_otp = phone_otp_info['otp']
                sms_sent = True
                logging.info("SMS OTP resent to %s for %s", phone_e164, email)

            if email_sent and sms_sent:
                message = 'Email and SMS OTP resent successfully.'
            elif sms_sent:
                message = 'SMS OTP resent successfully.'
            else:
                message = 'Email OTP resent successfully.'

            response = {
                'status': 'Success',
                'message': message,
                'email_sent': email_sent,
                'sms_sent': sms_sent,
                'phoneotp': phone_otp,
            }
            if email_sent and sms_sent:
                response['sharedOtp'] = True

            return jsonify(response)

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            message = e.response.get("Error", {}).get("Message", str(e))
            logging.error("resend_otp Cognito error (%s): %s", code, message)
            if code in ("UserNotFoundException", "ResourceNotFoundException"):
                return jsonify({'status': 'Failure', 'message': 'Email not registered.'}), 404
            if code in ("InvalidParameterException", "NotAuthorizedException"):
                return jsonify({'status': 'Failure', 'message': 'Email is already verified.'}), 400
            if code == "LimitExceededException":
                return jsonify({
                    'status': 'Failure',
                    'message': 'Too many OTP requests. Please try again later.',
                }), 429
            return jsonify({'status': 'Failure', 'message': message}), 400
        except KeyError as e:
            return jsonify({'status': 'Failure', 'message': f'Missing field: {str(e)}'}), 400
        except Exception as e:
            logging.error("resend_otp error: %s", e)
            return jsonify({
                'status': 'Failure',
                'message': 'Failed to resend OTP. Please try again.',
            }), 500

    def check_phone_number_exists(self):
        try:
            cursor = self.db.cur()
            requestData = request.get_json()
            phoneNo = requestData['phoneNo']
            countryCode = requestData['countryCode']

            phoneNo_flag, message = self._get_phone_verification_status(
                cursor, countryCode, phoneNo
            )

            return jsonify({'message': message, 'phoneNo_flag': phoneNo_flag})

        except ValueError as e:
            return jsonify({'message': str(e), 'status': 'Failure'}), 400

        except Exception as e:
            return jsonify({'message': str(e), 'status': 'Failure'})

    def social_user_login(self):
        try:
            cursor = self.db.cur()
            request_data = request.get_json()
            email = self._normalize_email(request_data['email'])
            device_token = request_data['deviceToken']

            cursor.execute('SELECT userId FROM icebreaker.registration WHERE email = %s', (email,))
            user_id = cursor.fetchone()
            if user_id is None:
                return{'status':'error', 'message':'user not found'}
            
            cursor.execute('SELECT COUNT(*) FROM icebreaker.netwirkReports WHERE reportedUserId = %s', (user_id[0],))
            report_count = cursor.fetchone()[0]
            if report_count >= 5:
                return {
                    'status': 'Error',
                    'message': 'Your account has been temporarily suspended due to multiple reports. Please contact support.',
                    'report_count': report_count
                }

            cursor.execute('SELECT userId FROM icebreaker.location WHERE userId = %s', (user_id[0],))
            user_id_location = cursor.fetchone()

            if user_id_location:
                cursor.execute('DELETE FROM icebreaker.location WHERE userId = %s', (user_id[0],))
                self.db.conn.commit()

            cursor.execute('SELECT userId FROM icebreaker.radius WHERE userId = %s', (user_id[0],))
            user_id_radius = cursor.fetchone()

            if user_id_radius:
                cursor.execute('DELETE FROM icebreaker.radius WHERE userId = %s', (user_id[0],))
                self.db.conn.commit()


            # 1. Validate email existence
            if not self._login_check_email_exists(cursor, email):
                return {'status': 'Error', 'message': 'Account does not exist for this email. Please Sign Up to proceed.'}

    
            # 3. Generate JWT token
            jwt_token = self._login_generate_jwt(email)

            user_id = self._login_get_user_id(cursor, email)

            self._login_update_tokens(cursor, device_token, jwt_token, email)

            # 6. Get profile info
            profilePic, vibes, vibing, notify_connection, background_notify, answer1, answer2, answer3, answer4, height, answer5, answer6, age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer = self._login_get_profile(cursor, user_id)

            # 7. Get registration info
            user_data = self._login_get_user_data(cursor, email, profilePic, vibes, vibing, notify_connection, background_notify, answer1, answer2, answer3, answer4, height, answer5, answer6, age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer, jwt_token)

            # 8. Get pictures
            user_data["pictures"] = self._login_get_pictures(cursor, user_id)

            return {"message": "User logged in", "userData": user_data}

        except ClientError as e:
            if hasattr(e, 'response') and 'Error' in e.response:
                if e.response['Error']['Code'] == 'NotAuthorizedException':
                    return {'status': 'Error', 'message': self.INCORRECT_PASS_MSG}
            return {'status': 'Error', 'message': str(e)}

        except Exception as e:
            return {'status': 'Error', 'message': str(e)}

        finally:
            cursor.close()

    def user_login(self):
        cursor = None
        try:
            cursor = self.db.cur()
            request_data = request.get_json()
            if not request_data:
                return {'status': 'Error', 'message': 'Invalid or missing JSON body.'}
            email = self._normalize_email(request_data['email'])
            password = request_data['password']
            device_token = request_data['deviceToken']

            cursor.execute('SELECT userId FROM icebreaker.registration WHERE email = %s', (email,))
            user_id = cursor.fetchone()
            if user_id is None:
                return{'status':'error', 'message':'user not found'}

            cursor.execute('SELECT COUNT(*) FROM icebreaker.netwirkReports WHERE reportedUserId = %s', (user_id[0],))
            report_count = cursor.fetchone()[0]
            if report_count >= 5:
                return {
                    'status': 'Error',
                    'message': 'Your account has been temporarily suspended due to multiple reports. Please contact support.',
                    'report_count': report_count
                }

            if not self._login_check_email_exists(cursor, email):
                return {'status': 'Error', 'message': 'Account does not exist for this email. Please Sign Up to proceed.'}

            response = self.authenticate(email, password)
            if 'error' in response:
                return {'status': 'Error', 'message': response['error']}

            if response.get('ResponseMetadata', {}).get('HTTPStatusCode') != 200:
                return {
                    'status': 'Error',
                    'message': f"Failed to authenticate. HTTP status code: {response['ResponseMetadata']['HTTPStatusCode']}"
                }

            cursor.execute('SELECT userId FROM icebreaker.location WHERE userId = %s', (user_id[0],))
            user_id_location = cursor.fetchone()

            if user_id_location:
                cursor.execute('DELETE FROM icebreaker.location WHERE userId = %s', (user_id[0],))
                self.db.conn.commit()

            cursor.execute('SELECT userId FROM icebreaker.radius WHERE userId = %s', (user_id[0],))
            user_id_radius = cursor.fetchone()

            if user_id_radius:
                cursor.execute('DELETE FROM icebreaker.radius WHERE userId = %s', (user_id[0],))
                self.db.conn.commit()

            # 3. Generate JWT token
            jwt_token = self._login_generate_jwt(email)

            # 4. Get userId
            user_id = self._login_get_user_id(cursor, email)

            # 5. Update token in DB
            self._login_update_tokens(cursor, device_token, jwt_token, email)

            # 6. Get profile info
            profilePic, vibes, vibing, notify_connection, background_notify, answer1, answer2, answer3, answer4, height, answer5, answer6, age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer = self._login_get_profile(cursor, user_id)

            # 7. Get registration info
            user_data = self._login_get_user_data(cursor, email, profilePic, vibes, vibing, notify_connection, background_notify, answer1, answer2, answer3, answer4, height, answer5, answer6,age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer, jwt_token)

            # 8. Get pictures
            user_data["pictures"] = self._login_get_pictures(cursor, user_id)

            return {"message": "User logged in", "userData": user_data}

        except ClientError as e:
            if hasattr(e, 'response') and 'Error' in e.response:
                if e.response['Error']['Code'] == 'NotAuthorizedException':
                    return {'status': 'Error', 'message': self.INCORRECT_PASS_MSG}
            return {'status': 'Error', 'message': str(e)}

        except Exception as e:
            logging.exception("user_login failed")
            return {'status': 'Error', 'message': str(e)}

        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    logging.exception("user_login cursor close failed")

    def user_logout(self):
        cursor = None
        try:
            data = request.get_json()
            user_id = data['userId']
            cursor = self.db.cur()
            self._clear_device_token(cursor, user_id)
            return {'status': 'Success', 'message': 'User logged out successfully'}
        except Exception as e:
            return {'status': 'Failure', 'message': str(e)}
        finally:
            if cursor:
                cursor.close()

    def _login_check_email_exists(self, cursor, email):
        cursor.execute('SELECT email FROM icebreaker.registration WHERE email = %s', (email,))
        return cursor.fetchone() is not None

    def _login_generate_jwt(self, email):
        jwt_secret_key = self._jwt_secret_key()
        jwt_payload = {'email': email, 'timestamp': int(time.time())}
        return jwt.encode(jwt_payload, jwt_secret_key, algorithm='HS256')

    def _login_get_user_id(self, cursor, email):
        cursor.execute('SELECT userId FROM icebreaker.registration WHERE email = %s', (email,))
        uid = cursor.fetchone()
        return uid[0] if uid else None

    def _login_update_tokens(self, cursor, device_token, jwt_token, email):
        self._assign_device_token(cursor, device_token, jwt_token, email)

    def _login_get_profile(self, cursor, user_id):
        cursor.execute("SELECT profilePic, vibes, vibing, notify_connection, background_notify, answer1, answer2, answer3, answer4, height, answer5, answer6, age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer FROM icebreaker.profile WHERE userId = %s", (user_id,))
        row = cursor.fetchone()
        if row:
            return row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16]
        return "", "Coffee", "", 0, 0, "", "", "", "", "", "", "", "", "", "", "", ""

    def _login_get_user_data(self, cursor, email, profilePic, vibes, vibing, notify_connection, background_notify, answer1, answer2, answer3, answer4, height, answer5, answer6,age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer, jwt_token):
        cursor.execute("""SELECT userId, fullName, email, password, firebaseId, createdAt, signupType, deviceToken,
                                Gender, Preference, DOB, countryCode, phoneNo, city, state, country
                        FROM icebreaker.registration WHERE email = %s""", (email,))
        result = cursor.fetchone()
        if not result:
            return {}

        (user_id, full_name, email, password, firebase_id, created_at, signup_type, device_token, Gender,
        Preference, DOB, countryCode, phoneNo, city, state, country) = result

        formatted_dob = DOB.strftime('%m-%d-%Y') if DOB else None
        formatted_created_at = created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at else None

        return {
            "userId": user_id,
            "fullName": full_name,
            "email": email,
            "password": password,
            "firebaseId": firebase_id,
            "createdAt": formatted_created_at,
            "signupType": signup_type,
            "deviceToken": device_token,
            "profilePic": profilePic,
            "loginWebToken": jwt_token,
            "Gender": Gender,
            "Preference": Preference,
            "DOB": formatted_dob,
            "countryCode": countryCode,
            "phoneNo": phoneNo,
            "city": city,
            "state": state,
            "country": country,
            "vibe_I_bring": vibes,
            "vibing": vibing,
            "notify_connection":notify_connection,
            "background_notify":background_notify,
            "what_makes_my_friends_laugh_about_me":answer1,
            "weekend_usually_looks_like":answer2,
            "Describe_your_communication_style":answer3,
            "What_brings_you_to_Vyb":answer4,
            "What_kind_of_partner_do_you_vibe_with_most":answer5,
            "What_does_everyday_vyb_look_like":answer6,
            "Height":height,
            "age_preference":age_preference,
            "prompt1question": prompt1question,
            "prompt1answer": prompt1answer,
            "prompt2question": prompt2question,
            "prompt2answer": prompt2answer
        }

    def _login_get_pictures(self, cursor, user_id):
        cursor.execute("SELECT loginUserId, picture_urls, IsProfilePicture FROM icebreaker.userPictures WHERE loginUserId = %s", (user_id,))
        pics_data = cursor.fetchall()
        return [{"url": row[1], "isProfilePicture": bool(row[2])} for row in pics_data]

    def forgot_password(self):
        try:
            requestData = request.get_json() or {}
            email = self._normalize_email(requestData['email'])

            email_check = self.check_email_exists_forgot_password(email)
            if 'error' in email_check:
                return jsonify({'message': email_check['error'], 'status': 'Failure'})

            if not email_check['exists']:
                return jsonify({'message': self.USER_NOT_FOUND_MSG, 'status': 'Failure'})

            if not email_check['verified']:
                return jsonify({'message': 'Email exists but not confirmed.', 'status': 'Failure'})

            if self._uses_ses_for_email_otp():
                try:
                    self._store_and_send_email_otp(email, purpose='password_reset')
                except Exception as e:
                    logging.error("SES password reset OTP failed for %s: %s", email, e)
                    return jsonify({
                        'message': 'Failed to send password reset email. Please try again.',
                        'status': 'Failure',
                    })
                return jsonify({
                    'message': 'Password reset OTP sent to your email.',
                    'status': 'Success',
                })

            aws_config = self._get_aws_config()
            self.client = boto3.client(
                'cognito-idp',
                region_name=aws_config["aws_region"],
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"],
            )
            self.client.forgot_password(
                SecretHash=self.get_secret_hash(email),
                ClientId=aws_config["app_client_id"],
                Username=email,
            )

            return jsonify({
                'message': 'Password reset initiated. Please check your email for instructions.',
                'status': 'Success',
            })

        except ClientError as e:
            if e.response['Error']['Code'] == 'UserNotFoundException':
                return jsonify({'message': self.USER_NOT_FOUND_MSG, 'status': 'Failure'})
            message = f"An error occurred: {e.response['Error']['Message']}"
            return jsonify({'message': message, 'status': 'Failure'})

        except KeyError as e:
            return jsonify({'message': f'Missing field: {str(e)}', 'status': 'Failure'}), 400

        except Exception as e:
            logging.error("forgot_password error: %s", e)
            return jsonify({'message': str(e), 'status': 'Failure'})

    def confirm_new_password(self):
        try:
            request_data = request.get_json() or {}
            email = self._normalize_email(request_data['email'])
            confirmation_code = request_data['confirmation_code']
            new_password = request_data['new_password']

            if self._uses_ses_for_email_otp():
                cursor = self.db.cur()
                ok, err = self._verify_password_reset_otp_ses(cursor, email, confirmation_code)
                if not ok:
                    return jsonify({'message': err, 'status': 'Failure'})

                try:
                    self._cognito_set_user_password(email, new_password)
                    cursor.execute(
                        "UPDATE icebreaker.registration SET password = %s WHERE email = %s",
                        (new_password, email),
                    )
                    self.db.conn.commit()
                except Exception as db_error:
                    logging.error("confirm_new_password SES error for %s: %s", email, db_error)
                    return jsonify({
                        'message': 'Password successfully changed, but updating the database failed.',
                        'status': 'Partial Success',
                    })
                finally:
                    cursor.close()

                return jsonify({'message': 'Password successfully changed.', 'status': 'Success'})

            aws_config = self._get_aws_config()
            self.client = boto3.client(
                'cognito-idp',
                region_name=aws_config["aws_region"],
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"],
            )

            self.client.confirm_forgot_password(
                ClientId=aws_config["app_client_id"],
                SecretHash=self.get_secret_hash(email),
                Username=email,
                ConfirmationCode=confirmation_code,
                Password=new_password,
            )

            try:
                cursor = self.db.cur()
                cursor.execute(
                    "UPDATE icebreaker.registration SET password = %s WHERE email = %s",
                    (new_password, email),
                )
                self.db.conn.commit()
                cursor.close()

            except Exception as db_error:
                logging.error("confirm_new_password DB error for %s: %s", email, db_error)
                return jsonify({
                    'message': 'Password successfully changed, but updating the database failed.',
                    'status': 'Partial Success',
                })

            return jsonify({'message': 'Password successfully changed.', 'status': 'Success'})

        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', '')
            if code == 'UserNotFoundException':
                return jsonify({'message': self.USER_NOT_FOUND_MSG, 'status': 'Failure'})
            if code == 'CodeMismatchException':
                return jsonify({
                    'message': 'Invalid confirmation code. Please check and try again.',
                    'status': 'Failure',
                })
            message = e.response.get('Error', {}).get('Message', str(e))
            return jsonify({'message': message, 'status': 'Failure'})

        except KeyError as e:
            return jsonify({'message': f'Missing field: {str(e)}', 'status': 'Failure'}), 400

        except Exception as e:
            logging.error("confirm_new_password error: %s", e)
            return jsonify({'message': str(e), 'status': 'Failure'})

    def confirm_user_in_cognito(self,email):
        try:
            # === Load AWS credentials from JSON file ===
            aws_config = self._get_aws_config()

            AWS_REGION = aws_config["aws_region"]
            # APP_CLIENT_ID = aws_config["app_client_id"]
            USER_POOL_ID = aws_config["userpool_id"]

            self.client = boto3.client(
                'cognito-idp',
                region_name=AWS_REGION,
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"],
                # userpool_id=aws_config["userpool_id"],
                # app_client_id = aws_config["app_client_id"]              
            )
            

            # Confirm the user sign-up as an administrator
            self.client.admin_confirm_sign_up(
                UserPoolId=USER_POOL_ID,
                Username=email
            )

            # Confirmation successful
            return {'message': 'User account confirmed successfully.', 'status': 'Success'}
        
        except Exception as e:
            error_message = str(e)
            return {'message': error_message, 'status': 'Failure'}

    def update_authenticationId_using_userId(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            userId = data['userId'] 
            authentication_id = data['authenticationId']
            cursor.execute('UPDATE icebreaker.registration SET authenticationId = %s, firebaseId = %s WHERE userId = %s',(authentication_id, authentication_id, userId))
            self.db.conn.commit()       
            message = 'AuthenticationId Updated successfully'
            status = 'Success'

            cursor.execute('SELECT authenticationId, firebaseId FROM icebreaker.registration WHERE userId = %s', (userId,))
            result = cursor.fetchall()
            authId = result[0][0]
  
            return{"message":message, "status":status, "AuthenticationId":authId}

        except Exception as e:
            message = 'Failed to update authentication id'
            status = 'Failure'
            return{"message":str(e), "status":status}

    def check_email_exists_forgot_password(self, email):
        try:
            # === Load AWS credentials from JSON file ===
            aws_config = self._get_aws_config()

            userpoolId = aws_config["userpool_id"]
            AWS_REGION = aws_config["aws_region"]

            client = boto3.client(
                'cognito-idp',
                region_name=AWS_REGION,
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"]
            )

            response = client.admin_get_user(
                UserPoolId=userpoolId,
                Username=email
            )

            confirmation_status = None
            for attr in response['UserAttributes']:
                if attr['Name'] == 'email_verified':
                    confirmation_status = attr['Value']

            user_status = response.get('UserStatus', '')
            if confirmation_status == 'true' or user_status == 'CONFIRMED':
                return {'exists': True, 'verified': True}
            elif confirmation_status == 'false':
                return {'exists': True, 'verified': False}
            else:
                return {'exists': False, 'verified': False}

        except client.exceptions.UserNotFoundException:
            return {'exists': False, 'verified': False}

        except Exception as e:
            return {'error': str(e)}

    def set_new_password(self):
        try:
            cursor = self.db.cur()
            requestData = request.get_json()
            email = requestData['Email']
            old_password = requestData['OldPassword']
            new_password = requestData['NewPassword']

            # Authenticate user with old password to get access token
            access_token = self.change_user_password(email, old_password)

            # Fetch loginWebToken of user from Profile table
            cursor.execute('SELECT loginWebToken FROM icebreaker.registration WHERE email = %s',(email,))
            login_token = cursor.fetchall()
            loginWebToken = login_token[0][0]

            # Changing password using the new password and access token
            response = self.client.change_password(
                PreviousPassword=old_password,
                ProposedPassword=new_password,
                AccessToken=access_token
            )

            try:
                cursor = self.db.cur()
                cursor.execute("UPDATE icebreaker.registration SET password = %s where email = %s", (new_password, email))
                self.db.conn.commit()           
                self.db.conn.close()
            
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")
                return jsonify({'message': 'Password successfully changed, but updating the database failed.', 'status': 'Partial Success'})

            message = "Password changed successfully."
            return jsonify({'message': message, 'status': 'Success', "loginWebToken": loginWebToken})

        except Exception as e:
            print(str(e))
            return jsonify({'message': str(e), 'status': 'Failure'})

    def change_user_password(self, email, old_password):
        try:
            # === Load AWS credentials from JSON file ===
            aws_config = self._get_aws_config()

            AWS_REGION = aws_config["aws_region"]
            APP_CLIENT_ID = aws_config["app_client_id"]

            self.client = boto3.client(
                'cognito-idp',
                region_name=AWS_REGION,
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"]
            )

            # Authenticate user with old password
            auth_response = self.client.initiate_auth(
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': email,
                    'PASSWORD': old_password,
                    'SECRET_HASH': self.get_secret_hash(email)
                },
                ClientId=APP_CLIENT_ID
            )

            return auth_response['AuthenticationResult']['AccessToken']

        except self.client.exceptions.NotAuthorizedException:
            raise Exception("Invalid old password. Please provide the correct old password.")
        except self.client.exceptions.UserNotFoundException:
            raise Exception(self.USER_NOT_FOUND_MSG)

    def generate_relay_email(self):
        # UUID makes it globally unique, no DB check needed
        unique = uuid.uuid4().hex[:10]  # 10 char hex string
        return f"{unique}@privaterelay.appleid.com"

    def register_apple_user(self):
        try:
            data = request.get_json()

            apple_user_id = data.get("apple_user_id")
            email         = data.get("email")
            fullName      = data.get("fullName")
            created_at    = datetime.now()

            if not apple_user_id:
                return {"Status": "Failure", "Message": "appleUserId is required"}
            
            if not email:
                email = self.generate_relay_email()

            cursor = self.db.cur()

            # 1️⃣ Check if AppleUserID already exists
            cursor.execute("""
                SELECT email 
                FROM icebreaker.AppleRegistration
                WHERE AppleUserID = %s
            """, (apple_user_id,))
            existing = cursor.fetchone()

            # 2️⃣ If AppleUserID exists → return existing email
            if existing:
                return {
                    "Status": "Success",
                    "Message": "Existing user found",
                    "email": existing[0]
                }

            # 3️⃣ If AppleUserID does NOT exist → INSERT new record
            cursor.execute("""
                INSERT INTO icebreaker.AppleRegistration (AppleUserID, email, fullName, created_at)
                VALUES (%s, %s, %s, %s)
            """, (apple_user_id, email, fullName, created_at))

            self.db.conn.commit()

            return {
                "Status": "Success",
                "Message": "New user inserted successfully",
                "email": email
            }

        except Exception as e:
            return {"Status": "Failure", "Message": str(e)}

    def get_apple_user(self):
        try:
            data = request.get_json()
            apple_user_id = data.get("apple_user_id")

            cursor = self.db.cur()

            cursor.execute("""
                SELECT email, fullName
                FROM icebreaker.AppleRegistration
                WHERE AppleUserID = %s
            """, (apple_user_id,))

            result = cursor.fetchone()
            cursor.close()

            if result:
                return jsonify({
                    "Status": "Success",
                    "email": result[0],
                    "fullName": result[1]
                }), 200
            else:
                return jsonify({
                    "Status": "Failure",
                    "Message": "No record found"
                }), 404

        except Exception as e:
            return jsonify({"Status": "Failure", "Message": str(e)}), 500



class AuthProfileService(ProfileBase):
    def retrieve_userId_via_authenticationId(self):
        try :
            cursor = self.db.cur()
            data = request.get_json()
            authentication_id = data['AuthenticationId']
            cursor.execute('SELECT userId FROM icebreaker.registration WHERE authenticationId = %s', (authentication_id,))
            row = cursor.fetchone()
                        
            if row:
                result = row[0]
                return jsonify({'userId': result, 'Status': 'Success'})
            else:
                message = "No user present with give Authentication Id. Please check Authentication Id again"
                return jsonify({'Message': message, 'Status': 'Failure'})
                
        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error

