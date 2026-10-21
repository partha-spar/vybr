"""Shared base classes, constants, and cross-service helpers."""

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
from time import time
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



class RegistrationBase:
    VIBR_REGISTRATION_CONFIG = os.getenv(
        "VIBR_REGISTRATION_CONFIG_FILE",
        "/var/run/secrets/vyb/vibr_registration_config.json",
    )
    INCORRECT_PASS_MSG = 'Incorrect username or password.'
    USER_NOT_FOUND_MSG = 'User not found. Please check your email and try again.'
    RINGCENTRAL_CONFIG = "ringcentral_config.json"
    MAX_OTP_ATTEMPTS = 5
    OTP_EXPIRY_MINUTES = 10

    def __init__(self):
        self.db = DB()
        self._aws_config = None 

    def _get_runtime_config(self):
        if self._aws_config is None:
            with open(self.VIBR_REGISTRATION_CONFIG, "r", encoding="utf-8") as config_file:
                self._aws_config = json.load(config_file)
        return self._aws_config

    def _jwt_secret_key(self):
        jwt_secret_key = self._get_runtime_config().get("jwt_secret_key")
        if not jwt_secret_key:
            raise RuntimeError("jwt_secret_key missing from mounted runtime configuration")
        return jwt_secret_key

    def _detach_device_token_from_other_users(self, cursor, device_token, exclude_email=None, exclude_user_id=None):
        if not device_token or not str(device_token).strip():
            return
        if exclude_email:
            cursor.execute(
                "UPDATE icebreaker.registration SET deviceToken = NULL WHERE deviceToken = %s AND email != %s",
                (device_token, exclude_email),
            )
        elif exclude_user_id is not None:
            cursor.execute(
                "UPDATE icebreaker.registration SET deviceToken = NULL WHERE deviceToken = %s AND userId != %s",
                (device_token, exclude_user_id),
            )
        else:
            cursor.execute(
                "UPDATE icebreaker.registration SET deviceToken = NULL WHERE deviceToken = %s",
                (device_token,),
            )

    def _assign_device_token(self, cursor, device_token, jwt_token, email):
        if device_token and str(device_token).strip():
            self._detach_device_token_from_other_users(cursor, device_token, exclude_email=email)
        cursor.execute(
            "UPDATE icebreaker.registration SET deviceToken = %s, loginWebToken = %s WHERE email = %s",
            (device_token, jwt_token, email),
        )
        self.db.conn.commit()

    def sign_up_with_otp(self, email, password):
        try:
            # === Load AWS credentials from JSON file ===
            aws_config = self._get_aws_config()

            client = boto3.client(
                'cognito-idp',
                region_name=aws_config["aws_region"],
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"]
            )

            response = client.sign_up(
                ClientId=aws_config["app_client_id"],
                SecretHash=self.get_secret_hash(email),
                Username=email,
                Password=password,
                UserAttributes=[
                    {
                        'Name': 'email',
                        'Value': email
                    }
                ],
                ValidationData=[
                    {
                        'Name': 'email',
                        'Value': email
                    }
                ]
            )
            logging.info(f"Cognito Response: {response}")
            return response

        except Exception as e:
            logging.error(f"Cognito signup error: {str(e)}")
            error = getattr(e, "response", {}).get("Error", {})
            return {
                'message': error.get('Message', str(e)),
                'status': 'Failure',
                'code': error.get('Code', ''),
            }

    def get_secret_hash(self, email):
        # === Load AWS credentials from JSON file ===
        aws_config = self._get_aws_config()

        app_client_id = aws_config["app_client_id"]
        app_client_secret = aws_config["app_client_secret"]

        message = email + app_client_id
        digest = hmac.new(
            str(app_client_secret).encode('utf-8'),
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()

        return base64.b64encode(digest).decode()

    def _normalize_email(self, email: str) -> str:
        return (email or "").strip().lower()

    def _resolve_registration_config_path(self):
        candidates = [
            os.getenv("VIBR_REGISTRATION_CONFIG_FILE"),
            self.VIBR_REGISTRATION_CONFIG,
            os.path.join(os.getcwd(), "vibr_registration_config.json"),
        ]
        for config_path in candidates:
            if config_path and os.path.isfile(config_path):
                return config_path
        return candidates[1]

    def _get_aws_config(self):
        if self._aws_config is None:
            config_path = self._resolve_registration_config_path()
            with open(config_path, "r", encoding="utf-8") as f:
                self._aws_config = json.load(f)
        return self._aws_config

    def _get_cognito_client(self):
        cfg = self._get_aws_config()
        return boto3.client(
            'cognito-idp',
            region_name=cfg["aws_region"],
            aws_access_key_id=cfg["aws_access_key_id"],
            aws_secret_access_key=cfg["aws_secret_access_key"],
        )

    def authenticate(self,email,password): 
        try:
            aws_config = self._get_aws_config()
            client = self._get_cognito_client()
            response = client.initiate_auth(
                ClientId=aws_config["app_client_id"],
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': email,
                    'PASSWORD': password,
                    'SECRET_HASH': self.get_secret_hash(email),
                },
            )
            return response

        except ClientError as e:
            error = getattr(e, "response", {}).get("Error", {})
            code = error.get("Code", "")
            if code == "NotAuthorizedException":
                return {"error": self.INCORRECT_PASS_MSG}
            if code == "UserNotConfirmedException":
                return {"error": "Email is not verified. Please confirm your account before logging in."}
            if code == "UserNotFoundException":
                return {"error": "Account does not exist for this email. Please Sign Up to proceed."}
            return {"error": error.get("Message", str(e))}
        except Exception as e:
            return {"error": str(e)}



class ProfileBase:
    subscribers = []  # Class variable, shared across all instances
    CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
    CONTENT_TYPE = "application/json; UTF-8"
    LOGIN_WEBTOKEN_QUERY = 'SELECT loginWebToken FROM icebreaker.registration WHERE userId = %s'
    USER_NOT_FOUND_MESSAGE = "User not found"
    SUCCESSFUL_CHECKIN_MESSAGE = 'User successfully checked in the location.'
    RUNTIME_SECRET_DIR = os.getenv("VYB_RUNTIME_SECRET_DIR", "/var/run/secrets/vyb")
    FIREBASE_SERVICE_ACCOUNT_FILE = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT_FILE",
        os.path.join(RUNTIME_SECRET_DIR, "firebase-admin.json"),
    )
    FCM_SEND_URL = "https://fcm.googleapis.com/v1/projects/icebreakerfinal/messages:send" 
    SELECT_EVENT_ID_BY_USER = 'SELECT eventId FROM icebreaker.eventLocation WHERE userId = %s'
    SELECT_PLACE_ID_BY_USER = 'SELECT placeId FROM icebreaker.location WHERE userId = %s'
    USER_NOT_FOUND_MSG = "Logged-in user not found in the data"
    GET_USER_RADIUS_QUERY = "SELECT latitude, longitude FROM icebreaker.radius WHERE userId = %s"
    GET_USER_FULLNAME_QUERY = "SELECT fullName FROM icebreaker.registration WHERE userId = %s"
    GET_USER_LOGIN_INFO_QUERY = '''SELECT r.loginWebToken, r.fullName, p.currentDesignation
                           FROM icebreaker.registration r
                           JOIN icebreaker.profile p ON r.userId = p.userId
                           WHERE r.userId = %s
                           '''
    DELETE_USER_FROM_LOCATION = "DELETE FROM icebreaker.location WHERE userId = %s"
    BLOCKED_USER_CHECK_QUERY='SELECT userId, blockedId FROM icebreaker.blockedUsers where userId = %s AND blockedId = %s'
    CHECK_BLOCKED_USER_REVERSE_QUERY ='SELECT userId, blockedId FROM icebreaker.blockedUsers where blockedId = %s AND userId = %s'
    GET_USER_NOTIFICATION_PREFS_QUERY = 'Select deviceToken, fullName, receiveNotifications from icebreaker.registration where userId = %s'
    GET_BLOCKED_USER_QUERY= "SELECT userId, blockedId, blockedTime FROM icebreaker.blockedUsers WHERE userId= %s AND blockedId= %s"
    GET_PENDING_REQUEST_STATUS_QUERY = "SELECT status FROM netwirkRequestsPending WHERE userId = %s AND requestedUserId = %s"
    GET_ACCEPTED_REQUEST_STATUS_QUERY = "SELECT status FROM netwirkRequestsAccepted WHERE userId = %s AND requestedUserId = %s"
    GET_USER_PICTURES_QUERY = "SELECT loginUserId, picture_urls, IsProfilePicture FROM icebreaker.userPictures WHERE loginUserId = %s"
    CHECKIN_SUCCESS_MESSAGE = 'User check-in notifications sent successfully'
    SAME_SPOT_CHECKIN_METERS = 25
    VIBT_CONFIG = os.getenv(
        "VIBR_CONFIG_FILE",
        os.path.join(RUNTIME_SECRET_DIR, "vibr_config.json"),
    )
    UPDATE_PROFILE_QUERY = 'UPDATE icebreaker.profile set intro_line1 = %s, intro_line2 = %s, desiredIndustry = %s WHERE userId = %s'
    AI_MODEL = "gpt-3.5-turbo"
    FETCH_LAT_LONG_QUERY = 'SELECT latitude, longitude FROM icebreaker.radius WHERE userId = %s'
    SELECT_PENDING_REQUEST = 'SELECT * FROM netwirkRequestsPending WHERE userId = %s AND requestedUserId = %s'
    DELETE_PENDING_REQUEST = 'DELETE FROM netwirkRequestsPending WHERE userId = %s AND requestedUserID = %s'
    MAX_S3_IMAGE_BYTES = 500 * 1024
    _US_PLACES_PREFIX = "us_places/us_places/"
    _GEOAPIFY_CATEGORY_ALIASES = {
        "cafe": "catering.cafe",
        "coffee_shop": "catering.cafe",
        "restaurant": "catering.restaurant",
        "bar": "catering.bar",
        "hotel": "accommodation.hotel",
        "hospital": "healthcare.hospital",
        "school": "education.school",
        "park": "leisure.park",
        "bank": "service.financial",
        "gas_station": "service.vehicle.fuel",
        "supermarket": "commercial.supermarket",
    }
    _GEOAPIFY_SHARED_CATEGORIES = (
        "activity",
        "commercial",
        "catering",
        "education",
        "entertainment",
        "healthcare",
        "leisure",
        "office",
        "service",
        "religion",
        "parking",
    )
    _GEOAPIFY_SEPARATE_CATEGORIES = ("tourism", "sport", "accommodation")

    @staticmethod
    def _resolve_vibr_config_path():
        candidates = [
            os.getenv("VIBR_CONFIG_FILE"),
            os.path.join(
                os.getenv("VYB_RUNTIME_SECRET_DIR", "/var/run/secrets/vyb"),
                "vibr_config.json",
            ),
            os.path.join(os.getcwd(), "vibr_config.json"),
        ]
        for config_path in candidates:
            if config_path and os.path.isfile(config_path):
                return config_path
        return candidates[-1]

    @staticmethod
    def _load_vibr_config():
        config_path = profile._resolve_vibr_config_path()
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_openai_api_key():
        return profile._load_vibr_config().get("openai_api_key")

    def __init__(self):
        self.db = DB()

    def _load_runtime_config(self):
        return self._load_vibr_config()

    def _google_api_key(self):
        api_key = self._load_runtime_config().get("google_api_key")
        if not api_key:
            raise RuntimeError("google_api_key missing from mounted runtime configuration")
        return api_key

    @staticmethod
    def _compress_image_for_s3(path):
        """Return a JPEG at or under 500KB when the saved image is larger."""
        try:
            if os.path.getsize(path) <= profile.MAX_S3_IMAGE_BYTES:
                return path
        except OSError:
            return path

        try:
            with Image.open(path) as source:
                source.draft("RGB", (1600, 1600))
                oriented = ImageOps.exif_transpose(source)
                oriented.load()
                if oriented.mode in ("RGBA", "LA") or (
                    oriented.mode == "P" and "transparency" in oriented.info
                ):
                    rgba = oriented.convert("RGBA")
                    image = Image.new("RGB", rgba.size, (255, 255, 255))
                    image.paste(rgba, mask=rgba.split()[-1])
                elif oriented.mode != "RGB":
                    image = oriented.convert("RGB")
                else:
                    image = oriented.copy()
        except Exception:
            return path

        long_side = max(image.width, image.height)
        if long_side > 1600:
            ratio = 1600 / long_side
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                Image.Resampling.BILINEAR,
            )

        out_path = os.path.splitext(path)[0] + ".jpg"
        payload = b""
        for _ in range(4):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=75)
            payload = buffer.getvalue()
            if len(payload) <= profile.MAX_S3_IMAGE_BYTES:
                break
            image = image.resize(
                (max(1, int(image.width * 0.75)), max(1, int(image.height * 0.75))),
                Image.Resampling.BILINEAR,
            )

        with open(out_path, "wb") as handle:
            handle.write(payload)
        if os.path.normcase(os.path.abspath(out_path)) != os.path.normcase(os.path.abspath(path)):
            try:
                os.remove(path)
            except OSError:
                pass
        return out_path

    def _place_user_counts(self, cursor, place_ids):
        if not place_ids:
            return {}
        placeholders = ", ".join(["%s"] * len(place_ids))
        cursor.execute(
            f"SELECT placeId, COUNT(userId) FROM icebreaker.location "
            f"WHERE placeId IN ({placeholders}) GROUP BY placeId",
            tuple(place_ids),
        )
        return {place_id: count for place_id, count in cursor.fetchall()}

    def _get_login_token_google(self, cursor, user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
        row = cursor.fetchone()
        return row[0] if row else ""

    def _get_restricted_places(self, cursor, user_id):
        now = datetime.now()
        cursor.execute("""
            SELECT placeId, checkoutTime 
            FROM icebreaker.incognito_history
            WHERE userId = %s AND checkoutTime IS NOT NULL
        """, (user_id,))
        rows = cursor.fetchall()
        return {place_id for place_id, checkout_time in rows if now < checkout_time + timedelta(hours=6)}

    def send_fcm_with_rest_new_user(self, token, full_name):
        import traceback
        
        logger.info("=" * 60)
        logger.info(f"[FCM DEBUG] START send_fcm_with_rest_new_user")
        logger.info(f"[FCM DEBUG] full_name: {full_name}")
        logger.info(f"[FCM DEBUG] token type: {type(token)}")
        
        if isinstance(token, list):
            logger.info(f"[FCM DEBUG] token count: {len(token)}")
            for i, t in enumerate(token):
                logger.info(f"[FCM DEBUG] token[{i}]: '{t}' | len={len(str(t))} | type={type(t)}")
        else:
            logger.info(f"[FCM DEBUG] token value: '{token}' | len={len(str(token))} | type={type(token)}")
        
        title = f"{full_name} has arrived"
        new_user = "new_user"

        logger.info(f"[FCM DEBUG] notification title: '{title}'")
        logger.info(f"[FCM DEBUG] data type: '{new_user}'")

        try:
            # SERVICE_ACCOUNT_FILE = r"D:\icebreakerNew\vibrr\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            logger.info(f"[FCM DEBUG] SERVICE_ACCOUNT_FILE: {SERVICE_ACCOUNT_FILE}")
            
            import os
            if not os.path.exists(SERVICE_ACCOUNT_FILE):
                logger.error(f"[FCM DEBUG] ❌ Service account file NOT FOUND at: {SERVICE_ACCOUNT_FILE}")
                return {"status": "error", "message": "Service account file not found"}
            else:
                logger.info(f"[FCM DEBUG] ✅ Service account file exists")

            logger.info("[FCM DEBUG] Loading service account credentials...")
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE,
                scopes=[self.CLOUD_PLATFORM_SCOPE]
            )
            logger.info(f"[FCM DEBUG] credentials loaded | service_account_email: {credentials.service_account_email}")
            logger.info(f"[FCM DEBUG] scopes: {list(credentials.scopes)}")

            logger.info("[FCM DEBUG] Refreshing access token...")
            auth_request = Request()
            credentials.refresh(auth_request)
            logger.info(f"[FCM DEBUG] ✅ Token refreshed")
            logger.info(f"[FCM DEBUG] credentials.token (first 20 chars): {str(credentials.token)[:20]}...")
            logger.info(f"[FCM DEBUG] token valid: {credentials.valid}")
            logger.info(f"[FCM DEBUG] token expiry: {credentials.expiry}")

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }
            logger.info(f"[FCM DEBUG] headers Content-Type: {headers['Content-Type']}")
            logger.info(f"[FCM DEBUG] headers Authorization prefix: {headers['Authorization'][:30]}...")

            url = self.FCM_SEND_URL
            logger.info(f"[FCM DEBUG] FCM URL: {url}")

            def send_request(payload, target_token):
                try:
                    logger.info(f"[FCM DEBUG] ------ Sending to token: '{target_token}' ------")
                    logger.info(f"[FCM DEBUG] token length: {len(str(target_token))}")
                    logger.info(f"[FCM DEBUG] full payload:\n{json.dumps(payload, indent=2)}")

                    resp = requests.post(url, headers=headers, data=json.dumps(payload))

                    logger.info(f"[FCM DEBUG] Response status code: {resp.status_code}")
                    logger.info(f"[FCM DEBUG] Response headers: {dict(resp.headers)}")
                    logger.info(f"[FCM DEBUG] Response body: {resp.text}")

                    if resp.status_code == 200:
                        logger.info(f"[FCM DEBUG] ✅ SUCCESS for token: {target_token}")
                    elif resp.status_code == 400:
                        logger.error(f"[FCM DEBUG] ❌ 400 BAD REQUEST — Token is likely invalid/expired: '{target_token}'")
                        logger.error(f"[FCM DEBUG] ⚠️  Action needed: Remove this token from DB")
                    elif resp.status_code == 401:
                        logger.error(f"[FCM DEBUG] ❌ 401 UNAUTHORIZED — Check service account permissions")
                    elif resp.status_code == 403:
                        logger.error(f"[FCM DEBUG] ❌ 403 FORBIDDEN — FCM API may not be enabled in Firebase console")
                    elif resp.status_code == 404:
                        logger.error(f"[FCM DEBUG] ❌ 404 NOT FOUND — Token unregistered, app may be uninstalled: '{target_token}'")
                        logger.error(f"[FCM DEBUG] ⚠️  Action needed: Remove this token from DB")
                    elif resp.status_code == 429:
                        logger.error(f"[FCM DEBUG] ❌ 429 RATE LIMITED — Too many requests")
                    elif resp.status_code == 500:
                        logger.error(f"[FCM DEBUG] ❌ 500 FCM SERVER ERROR — Retry later")
                    else:
                        logger.warning(f"[FCM DEBUG] ⚠️  Unexpected status {resp.status_code} for token: {target_token}")

                    return resp

                except requests.exceptions.ConnectionError as ce:
                    logger.error(f"[FCM DEBUG] ❌ CONNECTION ERROR: {str(ce)}")
                    raise
                except requests.exceptions.Timeout as te:
                    logger.error(f"[FCM DEBUG] ❌ TIMEOUT ERROR: {str(te)}")
                    raise
                except Exception as inner_e:
                    logger.error(f"[FCM DEBUG] ❌ UNEXPECTED ERROR: {str(inner_e)}")
                    logger.error(traceback.format_exc())
                    raise

            def build_payload(t):
                return {
                    "message": {
                        "token": t,
                        "notification": {
                            "title": title,
                        },
                        "data": {
                            "type": new_user
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                    },
                                    "mutable-content": 1
                                }
                            }
                        }
                    }
                }

            if isinstance(token, list):
                logger.info(f"[FCM DEBUG] Mode: MULTI-TOKEN | count={len(token)}")
                responses = []
                for i, t in enumerate(token):
                    logger.info(f"[FCM DEBUG] Processing token {i+1}/{len(token)}: '{t}'")
                    payload = build_payload(t)
                    resp = send_request(payload, t)
                    responses.append(resp)
                    logger.info(f"[FCM DEBUG] Token {i+1} done | status={resp.status_code}")

                success = [r for r in responses if r.status_code == 200]
                failed  = [r for r in responses if r.status_code != 200]
                logger.info(f"[FCM DEBUG] MULTI RESULT: ✅ success={len(success)} | ❌ failed={len(failed)}")
                logger.info("=" * 60)
                return {"status": "success", "responses": [r.text for r in responses]}

            else:
                logger.info(f"[FCM DEBUG] Mode: SINGLE-TOKEN")
                payload = build_payload(token)
                resp = send_request(payload, token)
                logger.info(f"[FCM DEBUG] SINGLE RESULT: status={resp.status_code}")
                logger.info("=" * 60)
                return {"status": "success", "message": resp.text}

        except Exception as e:
            logger.error(f"[FCM DEBUG] ❌ TOP-LEVEL EXCEPTION: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("=" * 60)
            return {"status": "error", "message": str(e)}

    def _translate_preference_common(self, my_preference):
        if my_preference == "Women":
            return "Female"
        elif my_preference == "Men":
            return "Male"
        return None

    def _translate_age_preference_industry_v1(self, age_preference):
        if not age_preference:
            return None
        
        try:
            parts = age_preference.split("-")
            min_age = int(parts[0].strip())
            max_age = int(parts[1].strip())
            return [(min_age, max_age)]
        except (IndexError, ValueError):
            return None

    def _build_age_filter_sql(self, age_ranges):
        if not age_ranges:
            return None, []
        clauses = ["(TIMESTAMPDIFF(YEAR, r.DOB, CURDATE()) BETWEEN %s AND %s)" for _ in age_ranges]
        params  = [val for (lo, hi) in age_ranges for val in (lo, hi)]
        return f"({' OR '.join(clauses)})", params

    def _build_reverse_gender_filter_sql(self, my_gender):
        if my_gender is None:
            return None, []
        
        # Non-binary users only appear to people whose preference is "Everyone"
        if my_gender == "Non-binary":
            return (
                "(r.Preference IS NULL OR r.Preference = 'Everyone')",
                []
            )
        
        preference_value = "Women" if my_gender == "Female" else "Men"
        return (
            "(r.Preference IS NULL OR r.Preference = 'Everyone' OR r.Preference = %s)",
            [preference_value]
        )

    def _build_reverse_age_filter_sql(self, my_age):
        if my_age is None:
            return None, []
        
        # Show user if other person has no preference set, OR my_age falls within their range
        return (
            "(p.age_preference IS NULL OR ("
            "  CAST(SUBSTRING_INDEX(p.age_preference, '-', 1) AS UNSIGNED) <= %s"
            "  AND CAST(SUBSTRING_INDEX(p.age_preference, '-', -1) AS UNSIGNED) >= %s"
            "))",
            [my_age, my_age]
        )

    def _calculate_age(self, dob):
        from datetime import date
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    def _fetch_request_status_industry_v1(self, cursor, user_id, target_user_id, real_place_id):
        cursor.execute("SELECT status FROM netwirkRequestsAccepted WHERE userId = %s AND requestedUserId = %s", (user_id, target_user_id))
        accepted_status = cursor.fetchone()
        cursor.execute("SELECT status FROM netwirkRequestsPending WHERE userId = %s AND requestedUserId = %s", (user_id, target_user_id))
        pending_status = cursor.fetchone()
        cursor.execute("SELECT status FROM netwirkRequestsPending WHERE userId = %s AND requestedUserId = %s", (target_user_id, user_id))
        received_status = cursor.fetchone()
        return accepted_status, pending_status, received_status

    def _determine_status_industry_v1(self, accepted_status, pending_status, received_status):
        if accepted_status and accepted_status[0] == "Accepted":
            return "Connected"
        elif pending_status and pending_status[0] == "Pending":
            return "Request_Sent"
        elif received_status and received_status[0] == "Pending":
            return "Request_Received"
        return "No_Interaction"

    def _fetch_user_place(self, cursor, user_id):
        cursor.execute(self.SELECT_PLACE_ID_BY_USER, (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def _resolve_place_id(self, user_id, data, cursor=None):
        place_id = data.get('placeId') if data else None
        if not place_id and user_id is not None:
            own_cursor = cursor is None
            if own_cursor:
                cursor = self.db.cur()
            try:
                place_id = self._fetch_user_place(cursor, user_id)
            finally:
                if own_cursor:
                    cursor.close()
        return place_id

    def _get_interaction_status(self, cursor, login_user_id, requested_user_id):
        cursor.execute(self.GET_ACCEPTED_REQUEST_STATUS_QUERY, (login_user_id, requested_user_id))
        accepted_status = cursor.fetchone()

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (login_user_id, requested_user_id))
        pending_status = cursor.fetchone()

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (requested_user_id, login_user_id))
        received_status = cursor.fetchone()

        if accepted_status and accepted_status[0] == "Accepted":
            return "Connected"
        if pending_status and pending_status[0] == "Pending":
            return "Request_Sent"
        if received_status and received_status[0] == "Pending":
            return "Request_Received"
        return "No_Interaction"

    def get_bounding_box_common(self, lat, lon, input_radius):
        # Convert input radius from miles to degrees
        lat_delta = input_radius / 69.0  # 69 miles per degree latitude
        lon_delta = input_radius / (69.0 * math.cos(math.radians(lat)))  # Adjusted for latitude

        min_lat = lat - lat_delta
        max_lat = lat + lat_delta
        min_lon = lon - lon_delta
        max_lon = lon + lon_delta

        return min_lat, max_lat, min_lon, max_lon

    def get_bounding_box(self, lat, lon, input_radius):
        return self.get_bounding_box_common(lat, lon, input_radius)

    def get_connection_status_radius(self, cursor, user_id, other_id):
        cursor.execute(self.GET_ACCEPTED_REQUEST_STATUS_QUERY, (user_id, other_id))
        if (res := cursor.fetchone()) and res[0] == "Accepted":
            return "Connected"
        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (user_id, other_id))
        if (res := cursor.fetchone()) and res[0] == "Pending":
            return "Request_Sent"
        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (other_id, user_id))
        if (res := cursor.fetchone()) and res[0] == "Pending":
            return "Request_Received"
        return "No_Interaction"

    def _is_user_blocked(self, cursor, requester_id, target_user_id):
        cursor.execute(self.BLOCKED_USER_CHECK_QUERY, (requester_id, target_user_id))
        return cursor.fetchone()

    def _get_user_pictures(self, cursor, user_id):
        cursor.execute("""
            SELECT loginUserId, picture_urls, IsProfilePicture 
            FROM icebreaker.userPictures 
            WHERE loginUserId = %s
        """, (user_id,))
        pics = cursor.fetchall()
        return [{"url": pic[1], "isProfilePicture": bool(pic[2])} for pic in pics]

    def get_zodiac_sign(self, dob) -> str:
        """
        Returns the zodiac sign for a given date of birth (datetime.date or str).
        """
        # If dob is a string, convert it
        if isinstance(dob, str):
            dob = datetime.strptime(dob, "%Y-%m-%d").date()

        day, month = dob.day, dob.month

        zodiac_signs = {
            "capricorn":   ((12, 22), (1, 19)),
            "aquarius":    ((1, 20),  (2, 18)),
            "pisces":      ((2, 19),  (3, 20)),
            "aries":       ((3, 21),  (4, 19)),
            "taurus":      ((4, 20),  (5, 20)),
            "gemini":      ((5, 21),  (6, 20)),
            "cancer":      ((6, 21),  (7, 22)),
            "leo":         ((7, 23),  (8, 22)),
            "virgo":       ((8, 23),  (9, 22)),
            "libra":       ((9, 23),  (10, 22)),
            "scorpio":     ((10, 23), (11, 21)),
            "sagittarius": ((11, 22), (12, 21)),
        }

        for sign, ((start_month, start_day), (end_month, end_day)) in zodiac_signs.items():
            if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
                return sign

        return "capricorn"

    def get_wave_status_sender_here(self, cursor, loginUserId, waved_to_id, place_id):
        cursor = self.db.cur()

        # Get latest like from user_likes_near
        cursor.execute("""
            SELECT sender_id, receiver_id, created_at
            FROM user_likes
            WHERE (sender_id = %s AND receiver_id = %s)
            OR (sender_id = %s AND receiver_id = %s)
            AND place_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (
            loginUserId, waved_to_id,
            waved_to_id, loginUserId,place_id
        ))

        row = cursor.fetchone()

        if not row:
            return "no_wave"  # no like found

        sender, receiver, created_at = row

        # Check if mutual like exists in user_likes_near
        cursor.execute("""
            SELECT 1
            FROM user_likes
            WHERE sender_id = %s AND receiver_id = %s
            AND place_id = %s
            LIMIT 1
        """, (receiver, sender, place_id))

        mutual = cursor.fetchone()

        if mutual:
            return "waved"
        else:
            return "already_waved" if sender == loginUserId else "wave_back"

    def get_wave_status_sender_near(self, cursor, loginUserId, waved_to_id):
        try:
            # Get latest like from user_likes_near
            cursor.execute("""
                SELECT sender_id, receiver_id, created_at
                FROM icebreaker.user_likes_near
                WHERE (sender_id = %s AND receiver_id = %s)
                OR (sender_id = %s AND receiver_id = %s)
                ORDER BY created_at DESC
                LIMIT 1
            """, (
                loginUserId, waved_to_id,
                waved_to_id, loginUserId
            ))

            row = cursor.fetchone()

            if not row:
                return "no_wave"  # no like found

            sender, receiver, created_at = row

            # Check if mutual like exists in user_likes_near
            cursor.execute("""
                SELECT 1
                FROM icebreaker.user_likes_near
                WHERE sender_id = %s AND receiver_id = %s
                LIMIT 1
            """, (receiver, sender))

            mutual = cursor.fetchone()

            if mutual:
                return "waved"
            else:
                return "already_waved" if sender == loginUserId else "wave_back"

        except Exception:
            return "no_wave"

    def get_wave_status_sender(self, cursor, loginUserId, waved_to_id):
        cursor = self.db.cur()

        # Get latest like from either table
        cursor.execute("""
            SELECT 'user_likes' AS table_name, sender_id, receiver_id, created_at
            FROM user_likes
            WHERE (sender_id = %s AND receiver_id = %s)
            OR (sender_id = %s AND receiver_id = %s)
            UNION ALL
            SELECT 'user_likes_near' AS table_name, sender_id, receiver_id, created_at
            FROM user_likes_near
            WHERE (sender_id = %s AND receiver_id = %s)
            OR (sender_id = %s AND receiver_id = %s)
            ORDER BY created_at DESC
            LIMIT 1
        """, (
            loginUserId, waved_to_id,
            waved_to_id, loginUserId,
            loginUserId, waved_to_id,
            waved_to_id, loginUserId
        ))

        row = cursor.fetchone()

        if not row:
            return "no_wave"

        table_name, sender, receiver, created_at = row

        # Check if mutual like exists in the same table
        if table_name == 'user_likes':
            cursor.execute("""
                SELECT 1
                FROM user_likes
                WHERE sender_id = %s AND receiver_id = %s
                LIMIT 1
            """, (receiver, sender))
        else:  # user_likes_near
            cursor.execute("""
                SELECT 1
                FROM user_likes_near
                WHERE sender_id = %s AND receiver_id = %s
                LIMIT 1
            """, (receiver, sender))

        mutual = cursor.fetchone()

        if mutual:
            status = "waved"
        else:
            status = "already_waved" if sender == loginUserId else "wave_back"

        return status

    def _filter_liked_users_search(self, results, search_text):
        search_text = search_text.lower()
        def matches(user):
            return (
                search_text in user.get("fullName", "").lower()
            )
        return [user for user in results if matches(user)]

    def _load_embedding(self, raw):
        """Deserialize from DB (JSON string → list)."""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    @staticmethod
    def _invalidate_offline_feed_cache(viewer_id: int) -> None:
        try:
            from Model.recsys.feed_cache import get_offline_feed_cache
            get_offline_feed_cache().invalidate_viewer(int(viewer_id))
        except Exception:
            pass

    def _bp_load(self, cursor, user_id: int) -> dict:
        cursor.execute("""
            SELECT taste_vibes, taste_emb1, taste_emb2,
                   skip_vibes, skip_comm, skip_intent,
                   pref_age_min, pref_age_max,
                   interaction_counts, dynamic_weights
            FROM icebreaker.user_behaviour_profile
            WHERE userId = %s
        """, (user_id,))
        row = cursor.fetchone()
        if not row:
            return self._bp_empty()
        return {
            "_exists":            True,
            "taste_vibes":        self._j(row[0], {}),
            "taste_emb1":         self._j(row[1], None),
            "taste_emb2":         self._j(row[2], None),
            "skip_vibes":         self._j(row[3], {}),
            "skip_comm":          self._j(row[4], {}),
            "skip_intent":        self._j(row[5], {}),
            "pref_age_min":       row[6],
            "pref_age_max":       row[7],
            "interaction_counts": self._j(row[8], {"waves": 0, "hides": 0, "matches": 0}),
            "dynamic_weights":    self._j(row[9], None) or DEFAULT_WEIGHTS.copy(),
        }

    def _bp_save(self, cursor, user_id: int, profile: dict):
        cursor.execute("""
            INSERT INTO icebreaker.user_behaviour_profile
                (userId, taste_vibes, taste_emb1, taste_emb2,
                 skip_vibes, skip_comm, skip_intent,
                 pref_age_min, pref_age_max,
                 interaction_counts, dynamic_weights)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                taste_vibes        = VALUES(taste_vibes),
                taste_emb1         = VALUES(taste_emb1),
                taste_emb2         = VALUES(taste_emb2),
                skip_vibes         = VALUES(skip_vibes),
                skip_comm          = VALUES(skip_comm),
                skip_intent        = VALUES(skip_intent),
                pref_age_min       = VALUES(pref_age_min),
                pref_age_max       = VALUES(pref_age_max),
                interaction_counts = VALUES(interaction_counts),
                dynamic_weights    = VALUES(dynamic_weights),
                updated_at         = CURRENT_TIMESTAMP
        """, (
            user_id,
            json.dumps(profile.get("taste_vibes", {})),
            json.dumps(profile.get("taste_emb1")),
            json.dumps(profile.get("taste_emb2")),
            json.dumps(profile.get("skip_vibes", {})),
            json.dumps(profile.get("skip_comm", {})),
            json.dumps(profile.get("skip_intent", {})),
            profile.get("pref_age_min"),
            profile.get("pref_age_max"),
            json.dumps(profile.get("interaction_counts", {})),
            json.dumps(profile.get("dynamic_weights", DEFAULT_WEIGHTS)),
        ))

    def on_hide(self, cursor, user_id: int, hidden_user_id: int, recsys_action: str = "hide_1_day"):
        """
        Call this after inserting into hidden_users_1day or hidden_users_1month.
        hidden_user_id = hiddenUserId value just inserted.
        """
        profile = self._bp_load(cursor, user_id)
 
        cursor.execute("""
            SELECT p.vibes, p.answer3, p.answer4
            FROM icebreaker.profile p
            WHERE p.userId = %s
        """, (hidden_user_id,))
        row = cursor.fetchone()
        if not row:
            return
 
        vibes, ans3, ans4 = row
 
        if vibes:
            for v in vibes.lower().split(','):
                v = v.strip()
                profile["skip_vibes"][v] = profile["skip_vibes"].get(v, 0) + 1
 
        if ans3:
            k = ans3.strip().lower()
            profile["skip_comm"][k] = profile["skip_comm"].get(k, 0) + 1
 
        if ans4:
            k = ans4.strip().lower()
            profile["skip_intent"][k] = profile["skip_intent"].get(k, 0) + 1
 
        profile["interaction_counts"]["hides"] = \
            profile["interaction_counts"].get("hides", 0) + 1

        self._log_recsys_interaction(cursor, user_id, hidden_user_id, recsys_action, "on_hide")
        self._bp_save(cursor, user_id, profile)

    def _log_recsys_interaction(
        self, cursor, user_a_id: int, user_b_id: int, action: str, source_function: str,
    ) -> None:
        try:
            from Model.recsys.db_feedback import insert_interaction_row
            insert_interaction_row(cursor, user_a_id, user_b_id, action, source_function)
        except Exception:
            logger.debug("rec_interactions log skipped", exc_info=True)

    @staticmethod
    def _bp_empty() -> dict:
        return {
            "_exists":            False,
            "taste_vibes":        {},
            "taste_emb1":         None,
            "taste_emb2":         None,
            "skip_vibes":         {},
            "skip_comm":          {},
            "skip_intent":        {},
            "pref_age_min":       None,
            "pref_age_max":       None,
            "interaction_counts": {"waves": 0, "hides": 0, "matches": 0},
            "dynamic_weights":    DEFAULT_WEIGHTS.copy(),
        }

    @staticmethod
    def _age_from_dob(dob):
        if not dob:
            return None
        today = date.today()
        try:
            d = dob if isinstance(dob, date) else datetime.strptime(str(dob), "%Y-%m-%d").date()
            return today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        except Exception:
            return None

    @staticmethod
    def _j(val, default):
        if not val:
            return default
        try:
            return json.loads(val)
        except Exception:
            return default


# Alias used by static methods referencing profile.<attr>
profile = ProfileBase
