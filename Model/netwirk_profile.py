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

class profile:
    subscribers = []  # Class variable, shared across all instances
    # Defining constants
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
    # A later search counts as the same spot when it is within this many meters
    # of the search that preceded the check-in.
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

    def _retrieve_login_token(self, cursor, user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
        login_token = cursor.fetchall()
        return login_token[0][0] if login_token else ''

    def __init__(self):
        self.db = DB()

    def _load_runtime_config(self):
        return self._load_vibr_config()

    def _google_api_key(self):
        api_key = self._load_runtime_config().get("google_api_key")
        if not api_key:
            raise RuntimeError("google_api_key missing from mounted runtime configuration")
        return api_key

    def _open_places_api_key(self):
        api_key = self._load_runtime_config().get("open_places_api_key")
        if not api_key:
            raise RuntimeError("open_places_api_key missing from mounted runtime configuration")
        return api_key

    def _geoapify_api_key(self):
        return "9034a88edb1742a1a72d05ba28ab5c19"

    def create_profile(self):
        try:
            data = request.get_json()
            user_id = data.get('userId')
            vibe = data.get('vibe')
            vibing = data.get('vibing')
            notify_connection = data.get('notify_connection', False)   # default false
            background_notify = data.get('background_notify', False)
            
            cursor = self.db.cur()

            # Fetching DOB for assigning star signs
            cursor.execute("SELECT DOB FROM icebreaker.registration WHERE userId = %s", (user_id,))
            date_of_birth = cursor.fetchall()
            dob = date_of_birth[0][0]  
            if dob is None:
                dob = '2000-01-01'

            zodiac_sign = self.get_zodiac_sign(dob)

            cursor.execute("""
                SELECT * FROM icebreaker.profile WHERE userId = %s
            """,
            (user_id,))
            result = cursor.fetchall()
            print(result)
            if not result:
                # Insert profile data
                cursor.execute("""
                        INSERT INTO icebreaker.profile (userId,vibes,vibing,zodiac_sign,notify_connection, background_notify)
                        VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (user_id,vibe,vibing,zodiac_sign,notify_connection,background_notify))

            else:
                # Update profile data
                cursor.execute("""
                    UPDATE icebreaker.profile SET userId = %s, vibes = %s, vibing = %s, zodiac_sign = %s, notify_connection = %s, background_notify = %s  WHERE userId = %s
                """,
                (user_id,vibe,vibing,zodiac_sign,notify_connection,background_notify,user_id))

                    


            self.db.conn.commit()

            message = 'User profile created successfully'
            status = 'Success'
            return {'message': message, 'status': status}
        except Exception as e:
            self.db.conn.rollback()
            status = 'Failure'
            return{'Error' : str(e), 'status':status}
        
        finally:
            cursor.close()

    def create_profile_new(self):
        try:
            data = request.get_json()
            user_id = data.get('userId')
            height = data.get('height')
            vibe_I_bring = data.get('vibe_I_bring')
            what_makes_my_friends_laugh_about_me = data.get('what_makes_my_friends_laugh_about_me')
            weekend_usually_looks_like = data.get('weekend_usually_looks_like', None)
            Describe_your_communication_style = data.get('Describe_your_communication_style')
            What_brings_you_to_VYB = data.get('What_brings_you_to_VYB', None)
            What_kind_of_partner_do_you_vibe_with_most = data.get('What_kind_of_partner_do_you_vibe_with_most')
            # How_active_are_you_in_your_daily_life = data.get('How_active_are_you_in_your_daily_life')
            # How_do_you_usually_unwind_after_a_long_day = data.get('How_do_you_usually_unwind_after_a_long_day')
            # Do_you_prefer_planned_or_spontaneous_days = data.get('Do_you_prefer_planned_or_spontaneous_days')
            What_does_everyday_vyb_look_like = data.get('What_does_everyday_vyb_look_like')
            prompt1question = data.get('prompt1question')
            prompt1answer = data.get('prompt1answer')
            prompt2question = data.get('prompt2question')
            prompt2answer = data.get('prompt2answer')
            notify_connection = data.get('notify_connection', False)   
            background_notify = data.get('background_notify', False)
            
            cursor = self.db.cur()

            # Fetching DOB for assigning star signs
            cursor.execute("SELECT DOB FROM icebreaker.registration WHERE userId = %s", (user_id,))
            date_of_birth = cursor.fetchall()
            dob = date_of_birth[0][0]  
            if dob is None:
                dob = '2000-01-01'

            zodiac_sign = self.get_zodiac_sign(dob)

            cursor.execute("""
                SELECT * FROM icebreaker.profile WHERE userId = %s
            """,
            (user_id,))
            result = cursor.fetchall()
           
            if not result:
                # Insert profile data
                cursor.execute("""
                        INSERT INTO icebreaker.profile (userId,answer1,answer2,zodiac_sign,notify_connection, background_notify, height, answer3, answer4, vibes, answer5, answer6, prompt1question, prompt1answer, prompt2question, prompt2answer)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (user_id,what_makes_my_friends_laugh_about_me,weekend_usually_looks_like,zodiac_sign,notify_connection,background_notify,height, Describe_your_communication_style, What_brings_you_to_VYB, vibe_I_bring,
                What_kind_of_partner_do_you_vibe_with_most, What_does_everyday_vyb_look_like, prompt1question, prompt1answer, prompt2question, prompt2answer))

            else:
                # Update profile data
                cursor.execute("""
                    UPDATE icebreaker.profile SET userId = %s, answer1 = %s, answer2 = %s, zodiac_sign = %s, notify_connection = %s, background_notify = %s, height = %s, answer3 = %s, answer4 = %s, vibes = %s,
                            answer5 = %s , answer6 = %s, prompt1question = %s, prompt1answer = %s, prompt2question = %s, prompt2answer = %s WHERE userId = %s
                """,
                (user_id,what_makes_my_friends_laugh_about_me,weekend_usually_looks_like,zodiac_sign,notify_connection,background_notify, height, Describe_your_communication_style, What_brings_you_to_VYB, vibe_I_bring,
                 What_kind_of_partner_do_you_vibe_with_most, What_does_everyday_vyb_look_like, prompt1question, prompt1answer, prompt2question, prompt2answer, user_id))
            
            # self.store_embedding(user_id, data['prompt1answer'], data['prompt2answer'])


            self.db.conn.commit()

            message = 'User profile created successfully'
            status = 'Success'
            return {'message': message, 'status': status}
        except Exception as e:
            self.db.conn.rollback()
            status = 'Failure'
            return{'Error' : str(e), 'status':status}
        
        finally:
            cursor.close()


    def edit_profile_details(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            full_name = data['fullName']
            gender = data['Gender']
            preference = data.get('Preference')
            dob = data['DOB']
            vibes = data['vibes']
            vibing = data['vibing']
            country_code = data.get('countryCode')
            phone_no = data.get('phoneNo')
            city = data.get('city')
            state = data.get('state')
            country = data.get('country')
            notify_connection = data.get('notify_connection')
            background_notify = data.get('background_notify')
            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            # === Check if profile exists ===
            cursor.execute("SELECT COUNT(*) FROM icebreaker.profile WHERE userId = %s", (user_id,))
            profile_exists = cursor.fetchone()[0] 

            if profile_exists:
                # --- Update existing record ---
                cursor.execute("""
                            UPDATE icebreaker.profile p
                            JOIN icebreaker.registration r
                            ON p.userId = r.userId
                            SET r.fullName = %s, r.Gender = %s, r.Preference = %s, 
                                p.vibes = %s, p.vibing = %s, r.countryCode = %s, r.phoneNo = %s, 
                                r.city = %s, r.state = %s, r.country = %s, p.notify_connection = %s,p.background_notify = %s 
                            WHERE r.userId = %s""",
                            (full_name, gender, preference, vibes, vibing,
                            country_code, phone_no, city, state, country, notify_connection, background_notify, user_id))
            else:
                # --- Insert new profile record ---
                cursor.execute("""
                            INSERT INTO icebreaker.profile (userId, vibes, vibing, zodiac_sign, notify_connection, background_notify )
                            VALUES (%s, %s, %s, %s, %s, %s)""",
                            (user_id, vibes, vibing, notify_connection, background_notify, None))

                # Also update registration table with user details
                cursor.execute("""
                            UPDATE icebreaker.registration
                            SET fullName = %s, Gender = %s, Preference = %s,
                                countryCode = %s, phoneNo = %s, city = %s, state = %s, country = %s
                            WHERE userId = %s""",
                            (full_name, gender, preference, country_code,
                            phone_no, city, state, country, user_id))

            # Update designation rank if profile exists (new or updated)
            cursor.execute('''
                            UPDATE icebreaker.profile p
                            JOIN icebreaker.designationRank r
                            ON p.currentDesignation = r.designation
                            SET p.desRank = r.rank
                            WHERE p.userId = %s''',
                            (user_id,))
            
            self.db.conn.commit()

            # Fetch updated user details
            cursor.execute("""
                            SELECT r.Gender, r.Preference, r.DOB, p.vibes, p.vibing, 
                                r.fullName, r.countryCode, r.phoneNo, r.city, r.state, 
                                r.country, p.zodiac_sign, p.notify_connection, p.background_notify
                            FROM icebreaker.registration r
                            JOIN icebreaker.profile p
                            ON r.userId = p.userId
                            WHERE r.userId = %s""", (user_id,))
            row = cursor.fetchone()

            cursor.close()
            self.db.conn.close()

            if not row:
                return jsonify({
                    "error": self.USER_NOT_FOUND_MESSAGE,
                    "loginWebToken": login_web_token
                })
            
            keys = ["Gender", "Preference", "DOB", "vibes", "vibing", "fullName",
                    "countryCode", "phoneNo", "city", "state", "country", "zodiac_sign","notify_connection","background_notify"]

            def format_value(index):
                val = row[index]
                if val is None:
                    return ''
                if keys[index] == 'DOB':
                    return val.strftime('%m-%d-%Y') if hasattr(val, 'strftime') else ''
                return val

            user_dict = {key: format_value(i) for i, key in enumerate(keys)}
            user_dict["loginWebToken"] = login_web_token

            return jsonify(user_dict)
        
        except Exception as e:
            self.db.conn.rollback()
            return jsonify({'error': str(e)})
        

    def get_profile_details(self):
        try:
            data = request.get_json()
            user_id = data.get('userId')
            cursor = self.db.cur()

            login_web_token = self._fetch_login_token(cursor, user_id)
            user_row = self._fetch_user_details_profile(cursor, user_id)

            cursor.close()
            self.db.conn.close()

            if not user_row:
                return jsonify({"error":self.USER_NOT_FOUND_MESSAGE, "loginWebToken": login_web_token})

            user_dict = self._format_user_dict(user_row, login_web_token)
            return jsonify(user_dict)

        except Exception as e:
            self.db.conn.rollback()
            return jsonify({'error': str(e)})

    # ------------------ helper methods ------------------ #

    def _fetch_login_token(self, cursor, user_id):
        cursor.execute(
            self.LOGIN_WEBTOKEN_QUERY,
            (user_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else ""

    def _fetch_user_details_profile(self, cursor, user_id):
        cursor.execute("""
            SELECT r.fullName, r.cardEmail, r.countryCode, r.phoneNo,
                p.industry, p.currentDesignation, p.companyName,
                p.website, p.experience, p.education
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s
        """, (user_id,))
        return cursor.fetchone()

    def _format_user_dict(self, row, login_web_token):
        fields = [
            "fullName", "cardEmail", "countryCode", "phoneNo", "industry",
            "currentDesignation", "companyName", "website", "experience", "education"
        ]
        user_dict = {fields[i]: row[i] if row[i] is not None else '' for i in range(len(fields))}
        user_dict["loginWebToken"] = login_web_token
        return user_dict

        
    MAX_S3_IMAGE_BYTES = 500 * 1024

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

    def upload_profile_pic(self, user_id):
        try:
            # === Load AWS credentials from JSON file ===
            aws_config = self._load_vibr_config()

            s3 = boto3.client(
                's3',
                region_name=aws_config["region"],
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"]
            )

             
            # Set the bucket name and region
            bucket_name = 'sparoptui'
            region = 'ap-south-1'
            folder_name = 'netwirkUserPics/'
            file = request.files['image']
            original_filename = file.filename

            # sanitize filename to avoid path traversal or bad chars
            safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_filename)

            # Add current date and time to filename
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Netwirkuser_{user_id}_{now}_{safe_filename}"

            # build safe storage path
            save_dir = os.path.join(os.getcwd(), "profilepics")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            # only safe path is used
            file.save(save_path)
            save_path = self._compress_image_for_s3(save_path)
            filename = os.path.basename(save_path)

            # Upload to S3
            s3.upload_file(save_path, bucket_name, folder_name + filename,
                        ExtraArgs={'ContentType': 'image/jpeg'})
            
            # Get the URL of the file
            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{folder_name}{filename}"

            # === OpenAI Aesthetic Scoring ===
            openai.api_key = aws_config["openai_api_key"]

           
            with open(save_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            try:
                intent_response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Analyze the provided image of a human face (STRICTLY HUMAN FACE, no animals or statues, if not human face just say- 'NOT HUMAN')."
                                "Rate the face on a scale of 0 to 1 based on three factors: Facial Symmetry (0–0.33): Measure the balance between left and right facial features using landmark points (e.g., eyes, nose). "
                                "Feature Proportionality (0–0.33): Assess the harmony of facial features (e.g., eye size, nose width, skin clarity) relative to standard proportions. Face Alignment (0–0.33): Evaluate the positioning of facial features (e.g., eye level, nose centering). Provide a total score (sum of the three) and briefly explain each factor's score."
                                "Focus on technical metrics, not subjective opinions, for research purposes and people's upliftment."
                                "JUST RETURN 'total_score' in JSON format in response AND NO OTHER DATA, nothing else. NO Yapping."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Rate this profile picture between 1 (poor) and 10 (excellent)."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=10
                )

                gpt_reply = intent_response['choices'][0]['message']['content']
                match = re.search(r'\d+(\.\d+)?', gpt_reply)
                aesthetic_score = float(match.group()) if match else 0.2

            except Exception as e:
                print("Image scoring error:", str(e))
                aesthetic_score = 0.2


            # Inserting profile pic url into db table
            cursor = self.db.cur()
            cursor.execute("UPDATE icebreaker.profile SET profilePic = %s, aesthetic_score = %s where userId = %s", (url,aesthetic_score, user_id))
            cursor.execute("UPDATE icebreaker.userPictures SET picture_urls = %s WHERE loginUserId = %s AND IsProfilePicture = 1", (url,user_id))
            self.db.conn.commit()           
            cursor.close()

            message = 'Image uploaded successfully !'
            return jsonify({'image_url':url, 'status' : 'Success','message': message, 'userId':user_id})


        except Exception as e:
            message = 'Failed to upload image !'
            return jsonify({'status' : 'Failure','message': str(e)})


    # def fetch_locations_from_google(self):
    #     cursor = self.db.cur()
    #     GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    #     GOOGLE_API_KEY = self._google_api_key()

    #     data = request.get_json()
    #     input_latitude, input_longitude = data['latitude'], data['longitude']
    #     input_radius, user_id = data['radius'], data['userId']
    #     auto_checkout_radius = int(input_radius) + 150

    #     cursor.execute("""
    #         INSERT INTO icebreaker.user_latlong (user_id, latitude, longitude, radius, updated_at)
    #         VALUES (%s, %s, %s, %s, NOW())
    #     """, (user_id, input_latitude, input_longitude, input_radius))

    #     self.db.conn.commit()

    #     login_web_token = self._get_login_token_google(cursor, user_id)
    #     restricted_place_ids = self._get_restricted_places(cursor, user_id)

    #     all_results, total_user_count = self._fetch_places(
    #         cursor, GOOGLE_PLACES_URL, GOOGLE_API_KEY,
    #         input_latitude, input_longitude, input_radius, restricted_place_ids
    #     )

        # Geocoding disabled. Nearby Search is the only Google call.
        # current_result, compound_code, extra_user_count = self._fetch_geocode_result(
        #     cursor, GOOGLE_API_KEY, input_latitude, input_longitude, restricted_place_ids
        # )
        # all_results.extend(current_result)
        # total_user_count += extra_user_count

        # Sort results by user count
        # all_results = sorted(all_results, key=lambda x: x['user_count'], reverse=True)

        # return {
        #     "all_results": all_results,
        #     "address": "",
        #     "loginWebToken": login_web_token,
        #     "autoCheckoutRadius": auto_checkout_radius,
        #     "total_user_count": total_user_count
        # }


    def fetch_locations_from_google(self):
        cursor = self.db.cur()
        data = request.get_json()
        input_latitude, input_longitude = data['latitude'], data['longitude']
        input_radius, user_id = data['radius'], data['userId']
        auto_checkout_radius = int(input_radius) + 150
        radius_m = float(input_radius)

        cursor.execute("""
            INSERT INTO icebreaker.user_latlong (user_id, latitude, longitude, radius, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (user_id, input_latitude, input_longitude, input_radius))
        self.db.conn.commit()

        login_web_token = self._get_login_token_google(cursor, user_id)
        restricted_place_ids = self._get_restricted_places(cursor, user_id)
        places = self._fetch_us_places_file(
            input_latitude, input_longitude, radius_m
        )
        user_counts = self._place_user_counts(cursor, [place["id"] for _, place in places])

        ranked = []
        total_user_count = 0
        for meters, place in places:
            place_id = place["id"]
            user_count = user_counts.get(place_id, 0)
            total_user_count += user_count
            types = [value for value in (place["taxonomy"], place["category"]) if value]
            address = place["address"] or ""
            if place["locality"]:
                address = f"{address}, {place['locality']}" if address else place["locality"]
            ranked.append((meters, {
                "kind": "place",
                "place_id": place_id,
                "name": place["name"] or "",
                "type": types,
                "primary_type": types[0] if types else None,
                "user_count": user_count,
                "latitude": place["lat"],
                "longitude": place["lng"],
                "photo_url": "",
                "rating": 0,
                "vicinity": address,
                "incognito_checkin_allowed": place_id not in restricted_place_ids,
            }))

        history_place_ids = self._same_spot_checkin_place_ids(
            cursor, user_id, float(input_latitude), float(input_longitude)
        )
        ranked = self._rank_places_with_checkin_history(ranked, history_place_ids)
        all_results = [row for _, row in ranked[:60]]
        return {
            "all_results": all_results,
            "address": "",
            "loginWebToken": login_web_token,
            "autoCheckoutRadius": auto_checkout_radius,
            "total_user_count": total_user_count,
        }

    def _same_spot_checkin_place_ids(self, cursor, user_id, lat, lng):
        """Place ids checked in from this spot, most visits first.

        Every sessiontracker row is counted when the user_latlong search at or
        before that check-in is within SAME_SPOT_CHECKIN_METERS of the current
        search. Equal visit counts are ordered by the latest check-in.
        """
        cursor.execute("""
            SELECT st.placeid, ul.latitude, ul.longitude, st.checkintime
            FROM icebreaker.sessiontracker st
            INNER JOIN icebreaker.user_latlong ul
                ON ul.user_id = st.userid
               AND ul.updated_at = (
                    SELECT MAX(prev.updated_at)
                    FROM icebreaker.user_latlong prev
                    WHERE prev.user_id = st.userid
                      AND prev.updated_at <= st.checkintime
               )
            WHERE st.userid = %s
              AND st.placeid IS NOT NULL
              AND st.checkintime IS NOT NULL
            ORDER BY st.checkintime DESC
        """, (user_id,))

        stats = {}
        for place_id, search_lat, search_lng, checkin_time in cursor.fetchall():
            if search_lat is None or search_lng is None or checkin_time is None:
                continue
            meters = self._haversine_meters(lat, lng, float(search_lat), float(search_lng))
            if meters > self.SAME_SPOT_CHECKIN_METERS:
                continue
            entry = stats.setdefault(place_id, {"count": 0, "latest": checkin_time})
            entry["count"] += 1
            if checkin_time > entry["latest"]:
                entry["latest"] = checkin_time
        return sorted(
            stats,
            key=lambda place_id: (stats[place_id]["count"], stats[place_id]["latest"]),
            reverse=True,
        )

    @staticmethod
    def _rank_places_with_checkin_history(ranked, history_place_ids):
        ranked.sort(key=lambda item: item[0])
        if not history_place_ids:
            return ranked
        priority = {place_id: index for index, place_id in enumerate(history_place_ids)}
        tail = len(priority)
        ranked.sort(key=lambda item: (priority.get(item[1]["place_id"], tail), item[0]))
        return ranked


    _US_PLACES_PREFIX = "us_places/us_places/"
    # _US_PLACES_PREFIX = r"D:\us_places\us_places"

    @staticmethod
    def _spaces_settings():
        key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_DEFAULT_REGION", "nyc3")
        bucket = os.getenv("SPACES_BUCKET", "vybusplaces")
        endpoint_url = os.getenv(
            "SPACES_ENDPOINT",
            f"https://{bucket}.{region}.digitaloceanspaces.com",
        ).rstrip("/")
        if not key_id or not secret:
            raise RuntimeError("DigitalOcean Spaces credentials are missing")

        endpoint_host = urlparse(endpoint_url).netloc or endpoint_url
        bucket_prefix = f"{bucket}."
        if endpoint_host.startswith(bucket_prefix):
            endpoint_host = endpoint_host[len(bucket_prefix):]
        regional_endpoint_url = f"https://{endpoint_host}"
        return key_id, secret, region, bucket, regional_endpoint_url, endpoint_host

    def _us_places_client(self):
        key_id, secret, region, _, endpoint_url, _ = self._spaces_settings()
        return boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )

    def _us_place_tile_keys(self, names):
        from botocore.exceptions import ClientError

        client = self._us_places_client()
        _, _, _, bucket, _, _ = self._spaces_settings()
        keys = []
        for name in names:
            key = self._US_PLACES_PREFIX + name
            try:
                client.head_object(Bucket=bucket, Key=key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey", "NotFound"):
                    continue
                raise
            keys.append(key)
        return keys

    def _us_place_tile_bounds(self):
        tiles = []

        def grid(x_start, x_end, y_start, y_end):
            for x in range(x_start, x_end, 5):
                for y in range(y_start, y_end, 5):
                    tiles.append((x, y, x + 5, y + 5))

        grid(-130, -60, 24, 54)
        grid(-180, -130, 50, 75)
        tiles.extend([
            (170, 50, 180, 56),
            (-162, 18, -154, 23),
            (-68, 17, -64, 19),
            (144, 13, 146, 21),
            (-171, -15, -168, -13),
        ])
        return tiles


    def _fetch_us_places_file(self, lat, lng, radius_m, limit=60):
        lat = float(lat)
        lng = float(lng)
        lat_delta = radius_m / 111320.0
        cos_lat = math.cos(math.radians(lat)) or 1.0
        lng_delta = radius_m / (111320.0 * cos_lat)
        lat0, lat1 = lat - lat_delta, lat + lat_delta
        lng0, lng1 = lng - lng_delta, lng + lng_delta
        pad = 0.02
        names = []
        for x0, y0, x1, y1 in self._us_place_tile_bounds():
            if x0 < lng1 + pad and x1 > lng0 - pad and y0 < lat1 + pad and y1 > lat0 - pad:
                names.append(f"tile_{x0}_{y0}.parquet")
        object_keys = self._us_place_tile_keys(names)
        if not object_keys:
            return []

        key_id, secret, region, bucket, _, endpoint_host = self._spaces_settings()
        listed = ", ".join("'s3://{}/{}'".format(bucket, key) for key in object_keys)
        # local_paths = []
        # for name in names:
        #     path = os.path.join(r"D:\us_places\us_places", name)
        #     if os.path.isfile(path):
        #         local_paths.append(path.replace("\\", "/"))
        # if not local_paths:
        #     return []
        # listed = ", ".join("'" + path.replace("'", "''") + "'" for path in local_paths)
        import duckdb

        con = duckdb.connect(":memory:")
        try:
            con.execute("LOAD httpfs")
            con.execute("SET memory_limit = '512MB'")
            con.execute("SET threads = 2")
            con.execute("SET enable_external_file_cache = false")
            con.execute("SET enable_object_cache = false")
            con.execute("SET parquet_metadata_cache = false")
            con.execute(
                """
                CREATE SECRET us_places (
                    TYPE s3,
                    PROVIDER config,
                    KEY_ID '{key_id}',
                    SECRET '{secret}',
                    REGION '{region}',
                    ENDPOINT '{endpoint}',
                    URL_STYLE 'vhost',
                    USE_SSL true,
                    SCOPE 's3://{bucket}/{prefix}'
                )
                """.format(
                    key_id=key_id.replace("'", "''"),
                    secret=secret.replace("'", "''"),
                    region=region.replace("'", "''"),
                    endpoint=endpoint_host.replace("'", "''"),
                    bucket=bucket,
                    prefix=self._US_PLACES_PREFIX,
                )
            )
            rows = con.execute(
                f"""
                SELECT id, name, category, taxonomy, address, locality, lat, lng
                FROM read_parquet([{listed}])
                WHERE lat BETWEEN ? AND ? AND lng BETWEEN ? AND ?
                """,
                [lat0, lat1, lng0, lng1],
            ).fetchall()
        finally:
            con.close()

        columns = ("id", "name", "category", "taxonomy", "address", "locality", "lat", "lng")
        nearby = []
        for row in rows:
            place = dict(zip(columns, row))
            meters = self._haversine_meters(lat, lng, place["lat"], place["lng"])
            if meters <= radius_m:
                nearby.append((meters, place))
        nearby.sort(key=lambda item: item[0])
        return nearby[:limit]


    _US_PLACES_PREFIX = "us_places/us_places/"

    @staticmethod
    def _spaces_settings():
        key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_DEFAULT_REGION", "nyc3")
        bucket = os.getenv("SPACES_BUCKET", "vybusplaces")
        endpoint_url = os.getenv(
            "SPACES_ENDPOINT",
            f"https://{bucket}.{region}.digitaloceanspaces.com",
        ).rstrip("/")
        if not key_id or not secret:
            raise RuntimeError("DigitalOcean Spaces credentials are missing")

        endpoint_host = urlparse(endpoint_url).netloc or endpoint_url
        bucket_prefix = f"{bucket}."
        if endpoint_host.startswith(bucket_prefix):
            endpoint_host = endpoint_host[len(bucket_prefix):]
        regional_endpoint_url = f"https://{endpoint_host}"
        return key_id, secret, region, bucket, regional_endpoint_url, endpoint_host

    def _us_places_client(self):
        key_id, secret, region, _, endpoint_url, _ = self._spaces_settings()
        return boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )

    def _us_place_tile_keys(self, names):
        from botocore.exceptions import ClientError

        client = self._us_places_client()
        _, _, _, bucket, _, _ = self._spaces_settings()
        keys = []
        for name in names:
            key = self._US_PLACES_PREFIX + name
            try:
                client.head_object(Bucket=bucket, Key=key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey", "NotFound"):
                    continue
                raise
            keys.append(key)
        return keys

    def _us_place_tile_bounds(self):
        tiles = []

        def grid(x_start, x_end, y_start, y_end):
            for x in range(x_start, x_end, 5):
                for y in range(y_start, y_end, 5):
                    tiles.append((x, y, x + 5, y + 5))

        grid(-130, -60, 24, 54)
        grid(-180, -130, 50, 75)
        tiles.extend([
            (170, 50, 180, 56),
            (-162, 18, -154, 23),
            (-68, 17, -64, 19),
            (144, 13, 146, 21),
            (-171, -15, -168, -13),
        ])
        return tiles


    def _fetch_us_places_file(self, lat, lng, radius_m, limit=60):
        lat = float(lat)
        lng = float(lng)
        lat_delta = radius_m / 111320.0
        cos_lat = math.cos(math.radians(lat)) or 1.0
        lng_delta = radius_m / (111320.0 * cos_lat)
        lat0, lat1 = lat - lat_delta, lat + lat_delta
        lng0, lng1 = lng - lng_delta, lng + lng_delta
        pad = 0.02
        names = []
        for x0, y0, x1, y1 in self._us_place_tile_bounds():
            if x0 < lng1 + pad and x1 > lng0 - pad and y0 < lat1 + pad and y1 > lat0 - pad:
                names.append(f"tile_{x0}_{y0}.parquet")
        object_keys = self._us_place_tile_keys(names)
        if not object_keys:
            return []

        key_id, secret, region, bucket, _, endpoint_host = self._spaces_settings()
        listed = ", ".join("'s3://{}/{}'".format(bucket, key) for key in object_keys)
        import duckdb

        con = duckdb.connect(":memory:")
        try:
            con.execute("LOAD httpfs")
            con.execute("SET memory_limit = '512MB'")
            con.execute("SET threads = 2")
            con.execute("SET enable_external_file_cache = false")
            con.execute("SET enable_object_cache = false")
            con.execute("SET parquet_metadata_cache = false")
            con.execute(
                """
                CREATE SECRET us_places (
                    TYPE s3,
                    PROVIDER config,
                    KEY_ID '{key_id}',
                    SECRET '{secret}',
                    REGION '{region}',
                    ENDPOINT '{endpoint}',
                    URL_STYLE 'vhost',
                    USE_SSL true,
                    SCOPE 's3://{bucket}/{prefix}'
                )
                """.format(
                    key_id=key_id.replace("'", "''"),
                    secret=secret.replace("'", "''"),
                    region=region.replace("'", "''"),
                    endpoint=endpoint_host.replace("'", "''"),
                    bucket=bucket,
                    prefix=self._US_PLACES_PREFIX,
                )
            )
            rows = con.execute(
                f"""
                SELECT id, name, category, taxonomy, address, locality, lat, lng
                FROM read_parquet([{listed}])
                WHERE lat BETWEEN ? AND ? AND lng BETWEEN ? AND ?
                """,
                [lat0, lat1, lng0, lng1],
            ).fetchall()
        finally:
            con.close()

        columns = ("id", "name", "category", "taxonomy", "address", "locality", "lat", "lng")
        nearby = []
        for row in rows:
            place = dict(zip(columns, row))
            meters = self._haversine_meters(lat, lng, place["lat"], place["lng"])
            if meters <= radius_m:
                nearby.append((meters, place))
        nearby.sort(key=lambda item: item[0])
        return nearby[:limit]


    def fetch_location_from_open_places(self):
        cursor = self.db.cur()
        data = request.get_json()
        input_latitude, input_longitude = data["latitude"], data["longitude"]
        input_radius, user_id = data["radius"], data["userId"]
        auto_checkout_radius = int(input_radius) + 150
        radius_m = float(input_radius)

        cursor.execute("""
            INSERT INTO icebreaker.user_latlong (user_id, latitude, longitude, radius, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (user_id, input_latitude, input_longitude, input_radius))
        self.db.conn.commit()

        login_web_token = self._get_login_token_google(cursor, user_id)
        restricted_place_ids = self._get_restricted_places(cursor, user_id)
        places = self._fetch_open_places(
            input_latitude,
            input_longitude,
            radius_m,
            data.get("query"),
            data.get("category"),
        )
        user_counts = self._place_user_counts(cursor, [place["id"] for place in places])

        all_results = []
        total_user_count = 0
        for place in places:
            place_id = place["id"]
            user_count = user_counts.get(place_id, 0)
            total_user_count += user_count
            types = [value for value in (place["taxonomy"], place["category"]) if value]
            address = place["address"] or ""
            if place["locality"]:
                address = f"{address}, {place['locality']}" if address else place["locality"]
            all_results.append({
                "place_id": place_id,
                "name": place["name"] or "",
                "type": types,
                "primary_type": types[0] if types else None,
                "user_count": user_count,
                "latitude": place["lat"],
                "longitude": place["lng"],
                "photo_url": "",
                "rating": 0,
                "vicinity": address,
                "incognito_checkin_allowed": place_id not in restricted_place_ids,
            })

        all_results = sorted(all_results, key=lambda row: row["user_count"], reverse=True)
        return {
            "all_results": all_results,
            "address": "",
            "loginWebToken": login_web_token,
            "autoCheckoutRadius": auto_checkout_radius,
            "total_user_count": total_user_count,
        }


    def fetch_location_from_geoapify(self):
        cursor = self.db.cur()
        data = request.get_json()
        input_latitude, input_longitude = data["latitude"], data["longitude"]
        input_radius, user_id = data["radius"], data["userId"]
        auto_checkout_radius = int(input_radius) + 150
        radius_m = float(input_radius)

        cursor.execute("""
            INSERT INTO icebreaker.user_latlong (user_id, latitude, longitude, radius, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (user_id, input_latitude, input_longitude, input_radius))
        self.db.conn.commit()

        login_web_token = self._get_login_token_google(cursor, user_id)
        restricted_place_ids = self._get_restricted_places(cursor, user_id)
        places = self._fetch_geoapify(
            input_latitude,
            input_longitude,
            radius_m,
            data.get("query"),
            data.get("category"),
        )
        user_counts = self._place_user_counts(cursor, [place["id"] for place in places])

        all_results = []
        total_user_count = 0
        for place in places:
            place_id = place["id"]
            user_count = user_counts.get(place_id, 0)
            total_user_count += user_count
            types = [value for value in (place["taxonomy"], place["category"]) if value]
            address = place["address"] or ""
            if place["locality"]:
                address = f"{address}, {place['locality']}" if address else place["locality"]
            all_results.append({
                "place_id": place_id,
                "name": place["name"] or "",
                "type": types,
                "primary_type": types[0] if types else None,
                "user_count": user_count,
                "latitude": place["lat"],
                "longitude": place["lng"],
                "photo_url": "",
                "rating": 0,
                "vicinity": address,
                "incognito_checkin_allowed": place_id not in restricted_place_ids,
            })

        all_results = sorted(all_results, key=lambda row: row["user_count"], reverse=True)
        return {
            "all_results": all_results,
            "address": "",
            "loginWebToken": login_web_token,
            "autoCheckoutRadius": auto_checkout_radius,
            "total_user_count": total_user_count,
        }


    def _fetch_open_places(self, lat, lng, radius_m, query=None, category=None):
        radius_mi = float(radius_m) / 1609.344
        radius_mi = min(max(radius_mi, 0.01), 50)
        params = {
            "lat": float(lat),
            "lon": float(lng),
            "radius_mi": radius_mi,
            "limit": 50 if category else 20,
        }
        if query:
            params["q"] = query
        if category:
            params["category"] = category
        response = requests.get(
            "https://api.openplacesapi.com/v1/places",
            params=params,
            headers={"Authorization": f"Bearer {self._open_places_api_key()}"},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Open Places API error {response.status_code}")
        results = response.json().get("results", [])
        places = []
        for place in results:
            address = place.get("address") or {}
            if isinstance(address, str):
                formatted = address
                locality = ""
            else:
                formatted = address.get("formatted") or address.get("street") or ""
                locality = address.get("locality") or ""
            categories = place.get("categories") or []
            primary = place.get("category") or (categories[0] if categories else None)
            places.append({
                "id": place.get("place_id"),
                "name": place.get("name") or "",
                "category": primary,
                "taxonomy": categories[1] if len(categories) > 1 else primary,
                "address": formatted,
                "locality": locality,
                "lat": place.get("lat"),
                "lng": place.get("lon"),
            })
        return places


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
    # tourism, sport, and accommodation each wipe out other categories when they
    # share a Geoapify request, so those three are queried on their own.
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

    def _geoapify_categories(self, category):
        key = str(category).strip()
        if "." in key:
            return key
        return self._GEOAPIFY_CATEGORY_ALIASES.get(key.lower(), key)

    def _geoapify_category_groups(self, category):
        if category:
            return [self._geoapify_categories(category)]
        return [
            ",".join(self._GEOAPIFY_SHARED_CATEGORIES),
            *self._GEOAPIFY_SEPARATE_CATEGORIES,
        ]

    def _fetch_geoapify(self, lat, lng, radius_m, query=None, category=None):
        seen = set()
        places = []
        for categories in self._geoapify_category_groups(category):
            for place in self._fetch_geoapify_group(lat, lng, radius_m, query, categories):
                place_id = place["id"]
                if place_id and place_id in seen:
                    continue
                if place_id:
                    seen.add(place_id)
                places.append(place)
        places.sort(key=lambda place: place["meters"] if place["meters"] is not None else 10**12)
        return places[:60]

    def _fetch_geoapify_group(self, lat, lng, radius_m, query, categories):
        radius_m = max(float(radius_m), 1.0)
        params = {
            "categories": categories,
            "filter": f"circle:{float(lng)},{float(lat)},{radius_m}",
            "bias": f"proximity:{float(lng)},{float(lat)}",
            "limit": 60,
            "apiKey": self._geoapify_api_key(),
        }
        if query:
            params["name"] = query
        response = requests.get(
            "https://api.geoapify.com/v2/places",
            params=params,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Geoapify API error {response.status_code}")
        places = []
        for feature in response.json().get("features", []):
            props = feature.get("properties") or {}
            feature_categories = props.get("categories") or []
            primary = feature_categories[0] if feature_categories else None
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or [None, None]
            lng_value = props.get("lon")
            lat_value = props.get("lat")
            if lng_value is None and len(coordinates) > 0:
                lng_value = coordinates[0]
            if lat_value is None and len(coordinates) > 1:
                lat_value = coordinates[1]
            places.append({
                "id": props.get("place_id"),
                "name": props.get("name") or props.get("address_line1") or "",
                "category": primary,
                "taxonomy": feature_categories[1] if len(feature_categories) > 1 else primary,
                "address": props.get("formatted") or props.get("address_line1") or "",
                "locality": props.get("city") or props.get("suburb") or "",
                "lat": lat_value,
                "lng": lng_value,
                "meters": props.get("distance"),
            })
        return places


    def _fetch_overture_places(self, cursor, lat, lng, radius_m, limit=60):
        lat = float(lat)
        lng = float(lng)
        lat_delta = radius_m / 111320.0
        cos_lat = math.cos(math.radians(lat)) or 1.0
        lng_delta = radius_m / (111320.0 * cos_lat)
        cursor.execute(
            """
            SELECT id, name, region, lat, lng
            FROM icebreaker.overture_places
            WHERE lat BETWEEN %s AND %s
              AND lng BETWEEN %s AND %s
            """,
            (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta),
        )
        columns = ("id", "name", "region", "lat", "lng")
        nearby = []
        for row in cursor.fetchall():
            place = dict(zip(columns, row))
            place["category"] = ""
            place["taxonomy"] = ""
            place["address"] = ""
            place["locality"] = ""
            meters = self._haversine_meters(lat, lng, place["lat"], place["lng"])
            if meters <= radius_m:
                nearby.append((meters, place))
        nearby.sort(key=lambda item: item[0])
        return nearby[:limit]


    def _fetch_overture_addresses(self, cursor, lat, lng, radius_m, limit=60):
        lat = float(lat)
        lng = float(lng)
        lat_delta = radius_m / 111320.0
        cos_lat = math.cos(math.radians(lat)) or 1.0
        lng_delta = radius_m / (111320.0 * cos_lat)
        try:
            cursor.execute(
                """
                SELECT id, number, street, unit, postcode, city, lat, lng
                FROM icebreaker.overture_addresses
                WHERE lat BETWEEN %s AND %s
                  AND lng BETWEEN %s AND %s
                """,
                (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta),
            )
        except mysql.connector.Error as exc:
            if getattr(exc, "errno", None) != 1146:
                raise
            logging.warning("overture_addresses is missing; returning places only")
            return []
        columns = ("id", "number", "street", "unit", "postcode", "city", "lat", "lng")
        nearby = []
        for row in cursor.fetchall():
            address = dict(zip(columns, row))
            meters = self._haversine_meters(lat, lng, address["lat"], address["lng"])
            if meters <= radius_m:
                nearby.append((meters, address))
        nearby.sort(key=lambda item: item[0])
        return nearby[:limit]


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


    def _haversine_meters(self, lat1, lng1, lat2, lng2):
        radius = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lng / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(a))


    def _fetch_open_places(self, lat, lng, radius_m, query=None, category=None):
        radius_mi = float(radius_m) / 1609.344
        radius_mi = min(max(radius_mi, 0.01), 50)
        params = {
            "lat": float(lat),
            "lon": float(lng),
            "radius_mi": radius_mi,
            "limit": 50 if category else 20,
        }
        if query:
            params["q"] = query
        if category:
            params["category"] = category
        response = requests.get(
            "https://api.openplacesapi.com/v1/places",
            params=params,
            headers={"Authorization": f"Bearer {self._open_places_api_key()}"},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Open Places API error {response.status_code}")
        results = response.json().get("results", [])
        places = []
        for place in results:
            address = place.get("address") or {}
            if isinstance(address, str):
                formatted = address
                locality = ""
            else:
                formatted = address.get("formatted") or address.get("street") or ""
                locality = address.get("locality") or ""
            categories = place.get("categories") or []
            primary = place.get("category") or (categories[0] if categories else None)
            places.append({
                "id": place.get("place_id"),
                "name": place.get("name") or "",
                "category": primary,
                "taxonomy": categories[1] if len(categories) > 1 else primary,
                "address": formatted,
                "locality": locality,
                "lat": place.get("lat"),
                "lng": place.get("lon"),
            })
        return places


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
    # tourism, sport, and accommodation each wipe out other categories when they
    # share a Geoapify request, so those three are queried on their own.
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

    def _geoapify_categories(self, category):
        key = str(category).strip()
        if "." in key:
            return key
        return self._GEOAPIFY_CATEGORY_ALIASES.get(key.lower(), key)

    def _geoapify_category_groups(self, category):
        if category:
            return [self._geoapify_categories(category)]
        return [
            ",".join(self._GEOAPIFY_SHARED_CATEGORIES),
            *self._GEOAPIFY_SEPARATE_CATEGORIES,
        ]

    def _fetch_geoapify(self, lat, lng, radius_m, query=None, category=None):
        seen = set()
        places = []
        for categories in self._geoapify_category_groups(category):
            for place in self._fetch_geoapify_group(lat, lng, radius_m, query, categories):
                place_id = place["id"]
                if place_id and place_id in seen:
                    continue
                if place_id:
                    seen.add(place_id)
                places.append(place)
        places.sort(key=lambda place: place["meters"] if place["meters"] is not None else 10**12)
        return places[:60]

    def _fetch_geoapify_group(self, lat, lng, radius_m, query, categories):
        radius_m = max(float(radius_m), 1.0)
        params = {
            "categories": categories,
            "filter": f"circle:{float(lng)},{float(lat)},{radius_m}",
            "bias": f"proximity:{float(lng)},{float(lat)}",
            "limit": 60,
            "apiKey": self._geoapify_api_key(),
        }
        if query:
            params["name"] = query
        response = requests.get(
            "https://api.geoapify.com/v2/places",
            params=params,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Geoapify API error {response.status_code}")
        places = []
        for feature in response.json().get("features", []):
            props = feature.get("properties") or {}
            feature_categories = props.get("categories") or []
            primary = feature_categories[0] if feature_categories else None
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or [None, None]
            lng_value = props.get("lon")
            lat_value = props.get("lat")
            if lng_value is None and len(coordinates) > 0:
                lng_value = coordinates[0]
            if lat_value is None and len(coordinates) > 1:
                lat_value = coordinates[1]
            places.append({
                "id": props.get("place_id"),
                "name": props.get("name") or props.get("address_line1") or "",
                "category": primary,
                "taxonomy": feature_categories[1] if len(feature_categories) > 1 else primary,
                "address": props.get("formatted") or props.get("address_line1") or "",
                "locality": props.get("city") or props.get("suburb") or "",
                "lat": lat_value,
                "lng": lng_value,
                "meters": props.get("distance"),
            })
        return places


    def _fetch_overture_places(self, cursor, lat, lng, radius_m, limit=60):
        lat = float(lat)
        lng = float(lng)
        lat_delta = radius_m / 111320.0
        cos_lat = math.cos(math.radians(lat)) or 1.0
        lng_delta = radius_m / (111320.0 * cos_lat)
        cursor.execute(
            """
            SELECT id, name, category, taxonomy, address, locality, lat, lng
            FROM icebreaker.overture_places
            WHERE lat BETWEEN %s AND %s
              AND lng BETWEEN %s AND %s
            """,
            (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta),
        )
        columns = ("id", "name", "category", "taxonomy", "address", "locality", "lat", "lng")
        nearby = []
        for row in cursor.fetchall():
            place = dict(zip(columns, row))
            meters = self._haversine_meters(lat, lng, place["lat"], place["lng"])
            if meters <= radius_m:
                nearby.append((meters, place))
        nearby.sort(key=lambda item: item[0])
        return nearby[:limit]


    def _fetch_overture_addresses(self, cursor, lat, lng, radius_m, limit=60):
        lat = float(lat)
        lng = float(lng)
        lat_delta = radius_m / 111320.0
        cos_lat = math.cos(math.radians(lat)) or 1.0
        lng_delta = radius_m / (111320.0 * cos_lat)
        cursor.execute(
            """
            SELECT id, number, street, unit, postcode, city, lat, lng
            FROM icebreaker.overture_addresses
            WHERE lat BETWEEN %s AND %s
              AND lng BETWEEN %s AND %s
            """,
            (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta),
        )
        columns = ("id", "number", "street", "unit", "postcode", "city", "lat", "lng")
        nearby = []
        for row in cursor.fetchall():
            address = dict(zip(columns, row))
            meters = self._haversine_meters(lat, lng, address["lat"], address["lng"])
            if meters <= radius_m:
                nearby.append((meters, address))
        nearby.sort(key=lambda item: item[0])
        return nearby[:limit]


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


    def _haversine_meters(self, lat1, lng1, lat2, lng2):
        radius = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lng / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(a))


    # ---------- Helper Methods ------------

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


    def _fetch_places(self, cursor, base_url, api_key, lat, lng, radius, restricted_place_ids):
        all_results, total_user_count = [], 0
        params = {"location": f"{lat},{lng}", "radius": radius, "key": api_key}

        while True:
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Google Places API Error: {response.text}")
                break

            places = response.json().get("results", [])
            for place in places:
                place_data, user_count = self._process_place(cursor, place, api_key, restricted_place_ids)
                all_results.append(place_data)
                total_user_count += user_count

            next_page_token = response.json().get("next_page_token")
            if next_page_token:
                params["pagetoken"] = next_page_token
                time.sleep(2)
            else:
                break
        return all_results, total_user_count


    def _process_place(self, cursor, place, api_key, restricted_place_ids):
        place_id = place.get('place_id')
        incognito_allowed = place_id not in restricted_place_ids

        cursor.execute('SELECT count(userId) FROM icebreaker.location WHERE placeId = %s', (place_id,))
        user_count = cursor.fetchone()[0]

        place_data = {
            'place_id': place_id,
            'name': place.get('name', ''),
            'type': place.get('types',''),
            'primary_type': place.get('types', [None])[0],
            'user_count': user_count,
            'latitude': place.get('geometry', {}).get('location', {}).get('lat'),
            'longitude': place.get('geometry', {}).get('location', {}).get('lng'),
            # Place Photo disabled. The image URL is a separate Google endpoint.
            # 'photo_url': self._get_photo_url(place, api_key),
            'photo_url': "",
            'rating': place.get('rating', 0),
            'vicinity': place.get('vicinity', ''),
            'incognito_checkin_allowed': incognito_allowed
        }
        return place_data, user_count


    def _get_photo_url(self, place, api_key):
        if 'photos' in place:
            photo_ref = place['photos'][0]['photo_reference']
            return f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=1000&photoreference={photo_ref}&key={api_key}"
        return ""


    def _fetch_geocode_result(self, cursor, api_key, lat, lng, restricted_place_ids):
        url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}'
        response = requests.get(url)
        data = response.json()

        if data.get('status') != 'OK':
            print('Geocoding failed:', data['status'])
            return [], "", 0

        compound_code = data.get('plus_code',{}).get('compound_code',{})
        result = data['results'][0]
        place_id, vicinity = result['place_id'], result['formatted_address']
        latitude, longitude = result['geometry']['location'].values()

        result_types = result.get("types", [])
        primary_type = result_types[0] if result_types else None

        cursor.execute('SELECT count(userId) FROM icebreaker.location WHERE placeId = %s', (place_id,))
        user_count = cursor.fetchone()[0]

        incognito_allowed = place_id not in restricted_place_ids
        current_result = [{
            "name": vicinity,
            "photo_url": "",
            "place_id": place_id,
            "types":result_types,
            "primary_type": primary_type, 
            "rating": 0,
            "vicinity": vicinity,
            "latitude": latitude,
            "longitude": longitude,
            "user_count": user_count,
            "incognito_checkin_allowed": incognito_allowed
        }]
        return current_result, compound_code, user_count

    

    # Helper function to query the Google Places Find Place API
    def find_place(self, query, pagetoken=None):
        GOOGLE_API_KEY = self._google_api_key()
        url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        url += f'?fields=formatted_address,name,geometry,place_id&input={query}&inputtype=textquery&key={GOOGLE_API_KEY}'

        # If there's a next_page_token, add it to the request URL
        if pagetoken:
            url += f'&pagetoken={pagetoken}'
        
        # Send the GET request to the Google Places API
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': response.json()}


    # API endpoint to search for a specific place using Google Find Place API
    def search_place(self):
        try:
            data = request.get_json()
            query = data['query']
            
            if not query:
                return jsonify({"error": "Query parameter is required"}), 400
            
            results = []

            # Initial search request (no location or radius)
            response_data = self.find_place(query)

            if 'candidates' not in response_data:
                return jsonify({"message": "No results found for your query"}), 404
            
            places = response_data['candidates']
            print(f"places: {places}")

            # Add the first set of results to the results list
            for place in places:
                results.append({
                    'placeid': place['place_id'],
                    'eventPlaceName': place['name'],
                    'latitude': place['geometry']['location']['lat'],
                    'longitude': place['geometry']['location']['lng'],
                    'eventPlaceAdd': place.get('formatted_address', 'Address not available')
                })

            # Check if there's a next page of results
            next_page_token = response_data.get('next_page_token')

            # If there is a next page token, we will need to call the API again
            while next_page_token:
                response_data = self.find_place(query, next_page_token)
                if 'candidates' in response_data:
                    places = response_data['candidates']
                    for place in places:
                        results.append({
                            'placeid': place['place_id'],
                            'eventPlaceName': place['name'],
                            'latitude': place['geometry']['location']['lat'],
                            'longitude': place['geometry']['location']['lng'],
                            'eventPlaceAdd': place.get('formatted_address', 'Address not available')
                        })
                next_page_token = response_data.get('next_page_token')

            return jsonify(results)
        
        except Exception as e:
            error = {'error': str(e)}
            print(error)
            return error

    def user_checkin(self):
        try:
            request_data = request.get_json()
            user_id = int(request_data['userId'])
            lat = request_data['latitude']
            long = request_data['longitude']
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = request_data.get('placeId')
            address = request_data.get('address')
            name = request_data.get('name')
            primary_type = request_data.get('primary_type')
            vicinity = request_data.get('vicinity')
            intro_line1 = request_data['intro_line1']
            intro_line2 = request_data['intro_line2']
            desired_industry = request_data['desiredIndustry']
            incognito = request_data.get('incognito', 'no').lower() == 'yes'
            checkin_time = datetime.now()

            cursor = self.db.cur()

            # --- Fetch user details ---
            cursor.execute("""
                SELECT r.loginWebToken, r.fullName, r.Gender, r.Preference
                FROM icebreaker.registration r
                JOIN icebreaker.profile p ON r.userId = p.userId
                WHERE r.userId = %s
            """, (user_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(self.USER_NOT_FOUND_MESSAGE)
            login_web_token, full_name, my_gender, my_preference = row
            full_name = full_name.strip().title()
            
            # --- Gender filter ---
            if my_preference == "Women":
                gender_filter = "Female"
            elif my_preference == "Men":
                gender_filter = "Male"
            else:
                gender_filter = None

            # --- Update or insert location & profile ---

            

            cursor.execute("SELECT userId FROM icebreaker.location WHERE userId = %s", (user_id,))
            if cursor.fetchone():
                
                cursor.execute("""
                    UPDATE icebreaker.location 
                    SET latitude=%s, longitude=%s, placeId=%s, checkinTime=%s, incognito=%s, cognito_time = DATE_ADD(%s, INTERVAL 10 MINUTE), address =%s, name =%s, primary_type = %s, vicinity =%s
                    WHERE userId=%s
                """, (lat, long, place_id, checkin_time, incognito, checkin_time, address, name, primary_type, vicinity, user_id))
            else:
               
                cursor.execute("""
                    INSERT INTO icebreaker.location (latitude, longitude, placeId, checkinTime, incognito,cognito_time, address, name, primary_type, vicinity, userId) 
                    VALUES (%s, %s, %s, %s, %s, DATE_ADD(%s, INTERVAL 10 MINUTE), %s, %s, %s, %s, %s)
                """, (lat, long, place_id, checkin_time, incognito,checkin_time, address, name, primary_type, vicinity, user_id))

            cursor.execute("""
                UPDATE icebreaker.profile 
                SET intro_line1=%s, intro_line2=%s, desiredIndustry=%s 
                WHERE userId=%s
            """, (intro_line1, intro_line2, desired_industry, user_id))


            self.db.conn.commit()

            # --- Handle incognito history ---
            is_incognito_exit = False
            if incognito:
                self._update_incognito_history(user_id, place_id, checkin_time)
            else:
                is_incognito_exit = self._checkout_incognito_history(user_id, place_id)

            age_pref_raw = self._fetch_age_preference_industry_v1(cursor, user_id)
            age_ranges   = self._translate_age_preference_industry_v1(age_pref_raw)
            my_dob       = self._fetch_user_dob_industry_v1(cursor, user_id)
            my_age       = self._calculate_age(my_dob) if my_dob else None


            # --- Notify other users ---
            self._notify_users(cursor, user_id, place_id, gender_filter, age_ranges, my_age, incognito, full_name, my_gender=my_gender)

            if not is_incognito_exit:

                cursor.execute("""
                    INSERT INTO icebreaker.sessiontracker (userid, placeid, checkintime, address, name, primary_type, vicinity)
                    VALUES (%s, %s, %s,%s, %s, %s, %s)
                """, (user_id, place_id, checkin_time, address, name, primary_type, vicinity))
            self.db.conn.commit()
            self.db.conn.close()
            return jsonify({
                'message': self.SUCCESSFUL_CHECKIN_MESSAGE,
                'status': 'Success',
                'loginWebToken': login_web_token
            })

        except Exception as e:
            self.db.conn.rollback()
            return {'error': str(e)}

    # ---------------- Helper Methods ---------------- #

    def _checkout_incognito_history(self, user_id, place_id):
        checkout_time = datetime.now()
        cur = self.db.conn.cursor()
        cur.execute("""
            UPDATE icebreaker.incognito_history
            SET checkoutTime = %s
            WHERE userId = %s 
            AND placeId = %s 
            AND checkoutTime IS NULL
        """, (checkout_time, user_id, place_id))
        rows_affected = cur.rowcount  # <-- tells us if a record was actually updated
        self.db.conn.commit()
        cur.close()
        return rows_affected > 0  # True = incognito exit, False = normal checkin

    def _update_incognito_history(self, user_id, place_id, checkin_time):
        """Insert or update incognito history table"""
        cur = self.db.conn.cursor()
        cur.execute("SELECT 1 FROM icebreaker.incognito_history WHERE userId=%s AND placeId=%s", (user_id, place_id))
        if cur.fetchone():
            cur.execute("""
                UPDATE icebreaker.incognito_history
                SET checkinTime=%s, checkoutTime=NULL
                WHERE userId=%s AND placeId=%s
            """, (checkin_time, user_id, place_id))
        else:
            cur.execute("""
                INSERT INTO icebreaker.incognito_history (userId, placeId, checkinTime)
                VALUES (%s, %s, %s)
            """, (user_id, place_id, checkin_time))
        self.db.conn.commit()
        cur.close()

    def _notify_users(self, cursor, user_id, place_id, gender_filter, age_ranges=None, my_age=None, incognito=False, full_name=None, my_gender=None):
        print(f"[NOTIFY DEBUG] ========== _notify_users START ==========")
        print(f"[NOTIFY DEBUG] user_id={user_id}, place_id={place_id}")
        print(f"[NOTIFY DEBUG] gender_filter={gender_filter}, my_gender={my_gender}")
        print(f"[NOTIFY DEBUG] age_ranges={age_ranges}, my_age={my_age}")
        print(f"[NOTIFY DEBUG] incognito={incognito}, full_name={full_name}")

        age_clause, age_params         = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender) 

        print(f"[NOTIFY DEBUG] age_clause={age_clause}, age_params={age_params}")
        print(f"[NOTIFY DEBUG] reverse_clause={reverse_clause}, reverse_params={reverse_params}")

        base_query = f"""
            SELECT l.userId
                FROM icebreaker.location l
                JOIN icebreaker.registration r 
                    ON l.userId = r.userId
                JOIN icebreaker.profile p ON l.userId = p.userId
                LEFT JOIN icebreaker.blockedUsers b1 
                    ON l.userId = b1.blockedId AND b1.userId = %s      
                LEFT JOIN icebreaker.blockedUsers b2 
                    ON l.userId = b2.userId AND b2.blockedId = %s      
                WHERE l.placeId = %s
                AND l.userId != %s 
                AND b1.blockedId IS NULL  
                AND b2.userId IS NULL
                AND r.receiveNotifications = '1'
                AND (%s IS NULL OR r.Gender = %s)
                {"AND " + age_clause            if age_clause            else ""}
                {"AND " + reverse_clause        if reverse_clause        else ""}
                {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
        """
        params = [
                user_id, user_id,
                place_id,
                user_id,
                gender_filter, gender_filter,
                *age_params,
                *reverse_params,
                *reverse_gender_params,
        ]


       

        print(f"[NOTIFY DEBUG] Final SQL:\n{base_query}")
        print(f"[NOTIFY DEBUG] SQL params: {params}")

        cursor.execute(base_query, params)
        user_id_list = [u[0] for u in cursor.fetchall()]

        print(f"[NOTIFY DEBUG] user_id_list from location query: {user_id_list}")

        if not user_id_list:
            print(f"[NOTIFY DEBUG] ❌ EARLY EXIT — No users found at place_id={place_id} matching filters")
            return

        cursor.execute("""
            SELECT visible_to_userId 
            FROM icebreaker.incognito_visibility_exceptions
            WHERE incognito_userId=%s AND placeId=%s
        """, (user_id, place_id))
        already_notified = {row[0] for row in cursor.fetchall()}

        print(f"[NOTIFY DEBUG] already_notified (incognito exceptions): {already_notified}")

        users_to_notify = [uid for uid in user_id_list if uid not in already_notified]

        print(f"[NOTIFY DEBUG] users_to_notify after incognito filter: {users_to_notify}")

        if not users_to_notify:
            print(f"[NOTIFY DEBUG] ❌ EARLY EXIT — All users already notified (incognito exceptions)")
            return

        placeholders = ','.join(['%s'] * len(users_to_notify))
        query = f"""
            SELECT deviceToken 
            FROM icebreaker.registration 
            WHERE userId IN ({placeholders}) 
            AND deviceToken IS NOT NULL 
            AND deviceToken != ''
            AND receiveNotifications = '1'
        """
        cursor.execute(query, tuple(users_to_notify))
        tokens = cursor.fetchall()

        print(f"[NOTIFY DEBUG] device tokens fetched: {[t[0][:15] + '...' for t in tokens] if tokens else '[]'}")

        if not tokens:
            print(f"[NOTIFY DEBUG] ❌ EARLY EXIT — No valid device tokens for users_to_notify={users_to_notify}")
            print(f"[NOTIFY DEBUG] ⚠️  Check: are deviceTokens NULL/empty in DB for these users?")
            print(f"[NOTIFY DEBUG] ⚠️  Check: do these users have receiveNotifications='1'?")
            logger.warning(f"[NOTIFY] No valid device tokens found for users_to_notify={users_to_notify}")
            return

        threads = []
        for token in tokens:
            name = "Someone" if incognito else full_name
            print(f"[NOTIFY DEBUG] Spawning thread for token={token[0][:15]}... name={name}")
            logger.info(f"[NOTIFY] Sending to token={token[0][:20]}...")
            t = threading.Thread(target=self.send_fcm_with_rest_new_user, args=(token[0], name))
            t.start()
            threads.append(t)

        print(f"[NOTIFY DEBUG] Threads started: {len(threads)}")

        for t in threads:
            t.join()

        print(f"[NOTIFY DEBUG] ✅ All threads joined. Total notified={len(threads)}")
        print(f"[NOTIFY DEBUG] ========== _notify_users END ==========")
        logger.info(f"[NOTIFY] All threads done. Total notified={len(threads)}")


    def getUserList(self):
        try:
            request_data = request.get_json()
            user_id = request_data['userId']
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            event_id = request_data.get('eventId')
            output_list = []
            device_token = []
            cursor = self.db.cur()
            place_id = self._resolve_place_id(user_id, request_data, cursor)

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            def build_user_dict(user):
                return {
                    "userId": user[0],
                    "companyName": user[1],
                    "currentDesignation": user[2],
                    "website": user[3],
                    "experience": user[4],
                    "industry": user[5],
                    "education": user[6],
                    "profilePic": user[7],
                    "fullName": user[8],
                    "email": user[9],
                    "countryCode": user[10],
                    "phoneNo": user[11],
                    "firebaseId": user[12],
                    "signupType": user[13],
                    "deviceToken": user[14],
                    "authenticationId": user[15],
                    "intro_line1": user[16],
                    "intro_line2": user[17]
                }

            if event_id:
                cursor.execute("""
                    SELECT p.userId, p.companyName, p.currentDesignation, p.website, p.experience, p.industry, p.education, 
                        p.profilePic, r.fullName, r.email, r.countryCode, r.phoneNo, r.firebaseId, r.signupType, r.deviceToken, r.authenticationId,
                        p.intro_line1, p.intro_line2
                    FROM icebreaker.profile p
                    JOIN icebreaker.eventLocation e ON p.userId = e.userId
                    JOIN icebreaker.registration r ON p.userId = r.userId
                    WHERE p.userId != %s
                    AND e.eventId = %s
                    ORDER BY e.checkinTime DESC
                """, (user_id, event_id))
            else:
                cursor.execute("""
                    SELECT p.userId, p.companyName, p.currentDesignation, p.website, p.experience, p.industry, p.education, 
                        p.profilePic, r.fullName, r.email, r.countryCode, r.phoneNo, r.firebaseId, r.signupType, r.deviceToken, r.authenticationId,
                        p.intro_line1, p.intro_line2
                    FROM icebreaker.profile p
                    JOIN icebreaker.location l ON p.userId = l.userId
                    JOIN icebreaker.registration r ON p.userId = r.userId
                    WHERE p.userId != %s
                    AND l.placeId = %s
                    ORDER BY p.userId DESC
                """, (user_id, place_id))

            row = cursor.fetchall()

            for user in row:
                user_dict = build_user_dict(user)
                output_list.append(user_dict)
                if user[14] is not None:
                    device_token.append([user[14]])

            for _ in device_token:
                # self.send_fcm_with_rest_new_user(token)
                print('a')

            if not output_list:
                response = {"List": [], "loginWebToken": login_web_token}
            else:
                response = {"List": output_list, "loginWebToken": login_web_token}

            cursor.close()
            self.db.conn.close()
            return jsonify(response)

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            print(error)
            return error


    
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
   
    
    def send_fcm_with_rest_new_user_connected_user(self, token, full_name):
        # Notification data
        title = "Your connection has arrived"
        body = f"{full_name}"
        new_user = "new_user"

        try:
            # Path to your service account key
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE

            # Create a credentials object from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])

            # Create a request object for fetching the access token
            auth_request = Request()

            # Refresh the token
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Prepare payload based on token type
            if isinstance(token, list):
                for tokens in token:
                    payload = {
                        "message": {
                            "tokens": tokens,
                            "notification": {
                                "title": title,
                                "body": body
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
                                            "body": body
                                        },
                                        "mutable-content": 1  # This enables Notification Service Extension for iOS
                                    }
                                }
                            }
                        }
                    }
                    # Send the request to FCM
                    resp = requests.post(url, headers=headers, data=json.dumps(payload))
            else:
                payload = {
                    "message": {
                        "token": token,
                        "notification": {
                            "title": title,
                            "body": body
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
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }

                # Send the request to FCM
                resp = requests.post(url, headers=headers, data=json.dumps(payload))

            return {"status": "success", "message": resp.text}

        except Exception as e:
            message = str(e)  # Convert exception to string
            return {"status": "error", "message": message}

    def usercheckinSendNotification(self):
        try:
            data = request.get_json()
            user_id = int(data.get('userId'))
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'  # Default placeId or dynamic input

            cursor = self.db.cur()
            place_id = self._resolve_place_id(user_id, data, cursor)

            if self.db.conn is None or not self.db.conn.is_connected():
                self.db.__init__()  # Reinitialize the connection

            # Check if eventId is present
            event_id = data.get('eventId') or None
            if event_id:
                cursor.execute(self.SELECT_EVENT_ID_BY_USER, (user_id,))
                event_id = cursor.fetchone()
                cursor.execute("SELECT userId FROM icebreaker.eventLocation WHERE eventId = %s AND userId != %s", (event_id[0], user_id))
                user_id_list = cursor.fetchall()
            else:
                cursor.execute("SELECT userId FROM icebreaker.location WHERE placeId = %s AND userId != %s", (place_id, user_id))
                user_id_list = cursor.fetchall()

            user_id_list = [user[0] for user in user_id_list]  # Extract user IDs

        # Fetch user details for notification
            cursor.execute('''SELECT r.loginWebToken, r.fullName, p.currentDesignation
                            FROM icebreaker.registration r
                            JOIN icebreaker.profile p ON r.userId = p.userId
                            WHERE r.userId = %s''', (user_id,))
            login_token = cursor.fetchall()
            full_name = login_token[0][1]
            current_designation = login_token[0][2]

            # Close cursor after fetching login details
            cursor.close()

            # Prepare list of users to notify
            users_to_notify = user_id_list

            # Create the SSE event stream (keeps the connection open indefinitely)
            def event_stream():
                cursor = self.db.conn.cursor()
                
                try:
                        if users_to_notify:
                            placeholders = ','.join(['%s'] * len(users_to_notify))
                            query = f"SELECT deviceToken FROM icebreaker.registration WHERE userId IN ({placeholders}) AND deviceToken IS NOT NULL AND deviceToken != ''"
                            cursor.execute(query, tuple(users_to_notify))
                            tokens = cursor.fetchall()

                            for _ in tokens:
                                # Construct the notification message
                                message = f"{full_name} ({current_designation}) is nearby at {place_id}"
                                # Yield the message in SSE format (this sends the message to the client)
                                yield f"data: {message}\n\n"

                        # Sleep for a while before checking again for updates
                        time.sleep(10)  # You can adjust this interval as needed

                except Exception as e:
                    # Handle exceptions if any
                    print(f"Error in SSE stream: {e}")
                    yield f"data: Error in SSE stream: {str(e)}\n\n"

                finally:
                    cursor.close()  # Ensure cursor is closed when stream ends

            # Return the SSE stream as a response, keeping the connection open indefinitely
            return Response(event_stream(), content_type='text/event-stream')

        except mysql.connector.Error as db_error:
            # Rollback in case of database error
            self.db.conn.rollback()
            error = {'error': f"Database Error: {str(db_error)}"}
            return jsonify({"Status": "Failure", "message": error}), 500

        except Exception as e:
            # Rollback in case of general error
            self.db.conn.rollback()
            error = {'error': str(e)}
            return jsonify({"Status": "Failure", "message": error}), 500

    # def usercheckinSendNotification(self):
    #     return jsonify({"status": "ok"}), 200
        
    def send_user_checkin_notification(self, user_id, event_id):
        try:

            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.GET_USER_LOGIN_INFO_QUERY, (user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]
            full_name = login_token[0][1]
            current_designation = login_token[0][2]

            user_rows, place_id = self._get_user_rows_and_place(cursor, user_id, event_id)
            if not user_rows:
                return {
                    "Status": "Success",
                    "message": self.CHECKIN_SUCCESS_MESSAGE,
                    "loginWebToken": login_web_token
                }

            user_id_list = [user[0] for user in user_rows]
            users_to_notify = self._filter_users_to_notify(user_id_list, user_id, place_id, event_id)

            self._send_notifications(users_to_notify, full_name, current_designation, cursor)

            self.db.conn.close()

            return {
                "Status": "Success",
                "message": self.CHECKIN_SUCCESS_MESSAGE,
                "loginWebToken": login_web_token
            }

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return {"Status": "Failure", "message": error}

    # --- Inline helpers to reduce complexity, using your same variable names ---

    def _get_user_rows_and_place(self, cursor, user_id, event_id):
        if event_id:
            cursor.execute(self.SELECT_EVENT_ID_BY_USER, (user_id,))
            event_id = cursor.fetchone()[0]
            cursor.execute("""
                SELECT e.userId 
                FROM icebreaker.eventLocation e 
                LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s
                LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s
                WHERE e.eventId = %s 
                AND e.userId != %s 
                AND b1.blockedId IS NULL 
                AND b2.blockedId IS NULL
            """, (user_id, user_id, event_id[0], user_id))
            return cursor.fetchall(), ""
        else:
            cursor.execute(self.SELECT_PLACE_ID_BY_USER, (user_id,))
            place_id = cursor.fetchone()[0]
            print(user_id)
            print(place_id)
            cursor.execute("""
                SELECT l.userId
                FROM icebreaker.location l
                LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s
                LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s
                WHERE l.placeId = %s
                AND l.userId != %s
                AND b1.blockedId IS NULL
                AND b2.blockedId IS NULL
            """, (user_id, user_id, place_id[0], user_id))
            return cursor.fetchall(), place_id

    def _filter_users_to_notify(self, user_id_list, user_id, place_id, event_id):
        users_to_notify = []
        for other_user_id in user_id_list:
            matched_users = self.best_matches_cosine_for_notifications(1, 5, place_id, event_id, other_user_id)

            # Reconnect DB if needed
            if self.db.conn is None or not self.db.conn.is_connected():
                self.db.__init__()

            if not matched_users or matched_users.get("Status") != "Success":
                continue

            best_matches_user_ids = [int(mid) for mid in matched_users["MatchIds"]]
            if user_id in best_matches_user_ids:
                users_to_notify.append(other_user_id)

        return users_to_notify

    def _send_notifications(self, users_to_notify, full_name, current_designation, cursor):
        if not users_to_notify:
            return

        placeholders = ','.join(['%s'] * len(users_to_notify))
        query = f"""
            SELECT deviceToken 
            FROM icebreaker.registration 
            WHERE userId IN ({placeholders}) 
            AND deviceToken IS NOT NULL 
            AND deviceToken != ''
        """
        cursor.execute(query, tuple(users_to_notify))
        tokens = cursor.fetchall()

        threads = []
        for token in tokens:
            t = threading.Thread(
                target=self.send_fcm_with_rest_new_user,
                args=(token[0], full_name)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


# This api is only for location checkins and not event checkins, please create a new api for event checkin notifications or modify this one for the same
    def send_user_checkin_notification_connected_users(self, user_id, place_id, event_id):
        try:
           
            # Re-establish connection after best_matches API call
            if self.db.conn is None or not self.db.conn.is_connected():
                self.db.__init__()  # Re-establish the connection
            cursor = self.db.cur()  # Recreate cursor after reconnection
            
            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.GET_USER_LOGIN_INFO_QUERY,(user_id,))
            login_token = cursor.fetchall()
            if login_token:
                login_web_token = login_token[0][0]
                full_name = login_token[0][1].title().strip()
                current_designation = login_token[0][2]
            else:
                login_web_token = full_name = current_designation = None

            if event_id:
                # Fetch accepted users list
                query_to_fetch_connected_users = """
                    SELECT e.userId
                    FROM icebreaker.eventLocation e
                    LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                    LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                    where e.eventId = %s
                    AND e.userId IN (
                    SELECT requestedUserId
                    FROM icebreaker.netwirkRequestsAccepted
                    where userId = %s
                    )
                    AND b1.blockedId IS NULL
                    AND b2.blockedId IS NULL"""
                cursor.execute(query_to_fetch_connected_users, (user_id, user_id, event_id, user_id))
                connected_user_ids = [row[0] for row in cursor.fetchall()]

            else:
                #Fetch accepted users list
                query_to_fetch_connected_users = """
                    SELECT l.userId
                    FROM icebreaker.location l
                    LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                    LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                    where l.placeId = %s
                    AND l.userId IN (
                    SELECT requestedUserId
                    FROM icebreaker.netwirkRequestsAccepted
                    where userId = %s
                    )
                    AND b1.blockedId IS NULL
                    AND b2.blockedId IS NULL"""
               
                cursor.execute(query_to_fetch_connected_users, (user_id, user_id, place_id[0], user_id))
                connected_user_ids = [row[0] for row in cursor.fetchall()]

            # Initialize list to store users to notify
            users_to_notify = connected_user_ids

            # Send notifications to users_to_notify
            if users_to_notify:
                placeholders = ','.join(['%s'] * len(users_to_notify))
                query = f"SELECT deviceToken FROM icebreaker.registration WHERE userId IN ({placeholders}) AND deviceToken IS NOT NULL AND deviceToken != ''"
                cursor.execute(query, tuple(users_to_notify))
                tokens = cursor.fetchall()

                # Parallel processing for sending notifications
                threads = []
                for token in tokens:
                    t = threading.Thread(target=self.send_fcm_with_rest_new_user_connected_user, args=(token[0], full_name))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

            self.db.conn.close()
            message = self.CHECKIN_SUCCESS_MESSAGE
            return {"Status": "Success", "message": message, "loginWebToken": login_web_token}

        except Exception as e:
            self.db.conn.rollback()
            print(str(e))
            error = {'error': str(e)}
            return {"Status": "Failure", "message": error}

    
    def user_logout_from_location(self):
        try: 
            request_data = request.get_json()
            user_id = request_data['userId']
            cursor=self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute("SELECT user_id FROM icebreaker.user_live_location WHERE user_id = %s", (user_id,))
            if cursor.fetchone():
                cursor.execute("DELETE FROM icebreaker.user_live_location WHERE user_id = %s", (user_id,))

            cursor.execute("Select userId from icebreaker.location where userId = %s",(user_id,))
            result = cursor.fetchone()

            if result is not None :
                cursor.execute(self.DELETE_USER_FROM_LOCATION,(user_id,))
           
            cursor.execute("SELECT userId from icebreaker.radius where userId = %s",(user_id,))
            resradius = cursor.fetchone()

            if resradius is not None:
                cursor.execute('Delete from icebreaker.radius WHERE userId = %s',(user_id,))
            

            self.db.conn.commit()

            cursor.close()
            self.db.conn.close()
            message = 'User successfully logged out of location'
            return jsonify({'message': message, 'status' : 'Success', "loginWebToken": login_web_token})
        
        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error

    
    def get_user_list_by_industry(self):
        try:
            data = request.get_json()
            user_id = int(data['userId'])     
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)

            search_text = data['searchText']
            page = int(data.get('page', 1))  # Default to page 1 if not provided
            per_page = int(data.get('perPage', 10))  # Default to 10 items per page if not provided
            offset = (page - 1) * per_page
            mode = data.get('mode')
            output_list = []
            cursor = self.db.cur()

            login_web_token = self._fetch_login_token_industry_v1(cursor, user_id)
            user_location_exists = self._check_user_location_industry_v1(cursor, user_id)
            if not user_location_exists:
                return jsonify({
                    "Status": "checkout",
                    "Message": "User has not checked in. Please check in to continue"
                })
            my_vibes = self._fetch_user_vibes_industry_v1(cursor, user_id)
            my_gender, my_preference = self._fetch_gender_preference_industry_v1(cursor, user_id)

            gender_filter = self._translate_preference_industry_v1(my_preference)

            age_pref_raw = self._fetch_age_preference_industry_v1(cursor, user_id)
            age_ranges   = self._translate_age_preference_industry_v1(age_pref_raw)

            my_dob       = self._fetch_user_dob_industry_v1(cursor, user_id)
            my_age       = self._calculate_age(my_dob) if my_dob else None

            total_users = self._fetch_total_users_industry_v1(cursor, user_id, place_id, gender_filter, age_ranges, my_age, my_gender)
            row = self._fetch_users_industry_v1(cursor, user_id, place_id, gender_filter, age_ranges,my_age, my_gender, per_page, offset)
            # print(row)

            likes_count = self._fetch_likes_count_industry_v1(cursor, user_id, place_id)

            current_likes_count = self._fetch_current_likes_count_industry_v1(cursor, user_id, place_id)
           

            if mode == 'recommend':
                return self._handle_recommend_mode_industry_v1(user_id, row, page, per_page, search_text, login_web_token, output_list, place_id)

            recommendations_new = self.best_matches_cosine_vibr(user_id, row, page, per_page, place_id)
            if recommendations_new.get("Status") == "Success":
                recommend_output_list_new = recommendations_new.get("Matches", [])
                total_users = len(output_list)
            else:
                
                recommend_output_list_new = [] 
                total_users = 0

            recommended_user_ids = {user['userId'] for user in recommend_output_list_new}
            output_list = self._build_user_dicts_industry_v1(cursor, row, user_id, my_vibes, recommended_user_ids, search_text, place_id)
            total_users = len(output_list)

            # output_list.sort(key=lambda x: not x["matching_vibes"])

            output_list.sort(
                key=lambda x: (
             

                    -max(x.get("wavedTime", 0), x.get("wavedbackTime", 0)),
                    # Then sort by matching vibes
                    not x.get("matching_vibes", False)
                )
            )


            for user in output_list:
                user.pop("matching_vibes", None)

            if not output_list:
                response = {"List": [],
                            "Status": 'Success',
                            "Page": page,
                            "PerPage": per_page,
                            "TotalUsers": total_users,
                            "TotalLikes":likes_count,
                            "CurrentLikes":current_likes_count,
                            "loginWebToken": login_web_token}
                return jsonify(response)
            else:
                response = {"List": output_list,
                            "Status": 'Success',
                            "Page": page,
                            "PerPage": per_page,
                            "TotalUsers": total_users,
                            "TotalLikes":likes_count,
                            "CurrentLikes":current_likes_count,
                            "loginWebToken": login_web_token}
            return jsonify(response)

        except Exception as e:
            return {'message': str(e), 'Status': "Failure"}
        

    def _fetch_login_token_industry_v1(self, cursor, user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
        login_token = cursor.fetchall()
        return login_token[0][0]

    def _check_user_location_industry_v1(self, cursor, user_id):
        query = "SELECT 1 FROM icebreaker.location WHERE userId = %s LIMIT 1"
        cursor.execute(query, (user_id,))
        return cursor.fetchone() is not None

    def _fetch_user_vibes_industry_v1(self, cursor, user_id):
        cursor.execute('SELECT vibes FROM icebreaker.profile WHERE userId = %s', (user_id,))
        my_vibes_result = cursor.fetchone()
        return my_vibes_result[0] if my_vibes_result else None

    def _fetch_gender_preference_industry_v1(self, cursor, user_id):
        cursor.execute('SELECT Gender, Preference FROM icebreaker.registration WHERE userId = %s', (user_id,))
        user_info = cursor.fetchone()
        return user_info if user_info else (None, None)
    
    def _translate_preference_common(self, my_preference):
        if my_preference == "Women":
            return "Female"
        elif my_preference == "Men":
            return "Male"
        return None
    
    def _translate_preference_industry_v1(self, my_preference):
        return self._translate_preference_common(my_preference)

    def _fetch_age_preference_industry_v1(self, cursor, user_id):
        cursor.execute(
            'SELECT age_preference FROM icebreaker.profile WHERE userId = %s', 
            (user_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

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
    
    
    def _fetch_current_likes_count_industry_v1(self, cursor, user_id, place_id):
        # SESSION_GAP = timedelta(minutes=3)

        # # 1️⃣ Fetch all sessions for this user at this place
        # cursor.execute("""
        #     SELECT checkintime, checkouttime
        #     FROM icebreaker.sessiontracker
        #     WHERE userid = %s AND placeid = %s
        #     ORDER BY checkintime
        # """, (user_id, place_id))
        # sessions = cursor.fetchall()

        # if not sessions:
        #     return 0

        # # 2️⃣ Merge sessions within SESSION_GAP
        # effective_checkin = None
        # last_checkout = None
        # for checkin, checkout in sessions:
        #     checkout_time = checkout or checkin
        #     if effective_checkin is None:
        #         effective_checkin = checkin
        #         last_checkout = checkout_time
        #         continue
        #     gap = checkin - last_checkout
        #     if gap <= SESSION_GAP:
        #         last_checkout = checkout_time
        #     else:
        #         effective_checkin = checkin
                # last_checkout = checkout_time

        # 3️⃣ Count likes after effective_checkin
        cursor.execute("""
            SELECT COUNT(*)
            FROM icebreaker.session_likes sl
            WHERE sl.wavedTouserId = %s
            AND sl.place_id = %s

            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b1
                WHERE b1.blockedId = sl.user_id AND b1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b2
                WHERE b2.userId = sl.user_id AND b2.blockedId = %s
            )

            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r1
                WHERE r1.reportedUserId = sl.user_id AND r1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r2
                WHERE r2.userId = sl.user_id AND r2.reportedUserId = %s
            )
        """, (user_id, place_id, user_id, user_id, user_id, user_id))

        result = cursor.fetchone()
        return result[0] if result else 0

    
    def _fetch_likes_count_industry_v1(self, cursor, user_id, place_id=None):
        cursor.execute("""
            SELECT COUNT(*)
            FROM icebreaker.user_likes ul
            WHERE ul.receiver_id = %s
            # AND ul.place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            AND (%s IS NULL OR ul.place_id = %s)
            
            -- Exclude blocked users (both directions)
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b1
                WHERE b1.blockedId = ul.sender_id AND b1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b2
                WHERE b2.userId = ul.sender_id AND b2.blockedId = %s
            )
            
            -- Exclude reported users (both directions)
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r1
                WHERE r1.reportedUserId = ul.sender_id AND r1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r2
                WHERE r2.userId = ul.sender_id AND r2.reportedUserId = %s
            )
        """, (user_id, place_id, place_id, user_id, user_id, user_id, user_id))

        result = cursor.fetchone()
        return result[0] if result else 0


    def _fetch_total_users_industry_v1(self, cursor, user_id, place_id, gender_filter, age_ranges=None, my_age=None, my_gender=None):
        age_clause, age_params = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params   = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)
        query = f'''SELECT COUNT(*)
            FROM icebreaker.profile p
            JOIN icebreaker.location l ON p.userId = l.userId
            JOIN icebreaker.registration r ON p.userId = r.userId
            LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s
            LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s
            LEFT JOIN icebreaker.netwirkReports r1 ON l.userId = r1.reportedUserId AND r1.userId = %s
            LEFT JOIN icebreaker.netwirkReports r2 ON l.userId = r2.userId AND r2.reportedUserId = %s
            LEFT JOIN icebreaker.hidden_users h  ON p.userId = h.hiddenUserId AND h.userId = %s
            LEFT JOIN icebreaker.hidden_users h2 ON p.userId = h2.userId AND h2.hiddenUserId = %s
            WHERE p.userId != %s
            AND l.placeId = %s
            AND b1.blockedId IS NULL 
            AND b2.blockedId IS NULL
            AND r1.reportedUserId IS NULL 
            AND r2.reportedUserId IS NULL
            AND h.hiddenUserId IS NULL
            AND h2.userId IS NULL 
            AND (%s IS NULL OR r.Gender = %s)
            {"AND " + age_clause if age_clause else ""}
            {"AND " + reverse_clause if reverse_clause else ""}
            {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
            '''
        cursor.execute(query, (user_id, user_id, user_id, user_id, user_id, user_id, user_id, place_id, gender_filter, gender_filter, *age_params, *reverse_params,*reverse_gender_params))
        return cursor.fetchone()[0]

    def _fetch_users_industry_v1(self, cursor, user_id, place_id, gender_filter, age_ranges=None, my_age=None, my_gender=None, per_page = 10, offset = 0):
        age_clause, age_params = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params   = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)
        query = f"""
            SELECT p.userId, p.profilePic, p.selfiePic, r.fullName, r.email, r.firebaseId, r.signupType,
                   r.authenticationId, r.Gender, r.Preference, r.DOB, p.vibes, p.vibing,
                   r.state, r.city, p.zodiac_sign, p.aesthetic_score, l.incognito
            FROM icebreaker.profile p
            JOIN icebreaker.location l ON p.userId = l.userId
            JOIN icebreaker.registration r ON p.userId = r.userId
                             
            LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s
            LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s
            LEFT JOIN icebreaker.netwirkReports r1 ON l.userId = r1.reportedUserId AND r1.userId = %s
            LEFT JOIN icebreaker.netwirkReports r2 ON l.userId = r2.userId AND r2.reportedUserId = %s
            LEFT JOIN icebreaker.hidden_users h  ON p.userId = h.hiddenUserId AND h.userId = %s
            LEFT JOIN icebreaker.hidden_users h2 ON p.userId = h2.userId AND h2.hiddenUserId = %s
            WHERE p.userId != %s
            AND l.placeId = %s
            AND b1.blockedId IS NULL
            AND b2.blockedId IS NULL
            AND r1.reportedUserId IS NULL 
            AND r2.reportedUserId IS NULL
            AND h.hiddenUserId IS NULL 
            AND h2.userId IS NULL
            AND (%s IS NULL OR r.Gender = %s)
            {"AND " + age_clause if age_clause else ""}
            {"AND " + reverse_clause if reverse_clause else ""}
            {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
          
            LIMIT %s OFFSET %s                
        """
        cursor.execute(query, (user_id, user_id, user_id,user_id, user_id, user_id, user_id, place_id,
                  gender_filter, gender_filter, *age_params,*reverse_params, *reverse_gender_params, per_page, offset))
        return cursor.fetchall()

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

    def _fetch_user_dob_industry_v1(self, cursor, user_id):
        cursor.execute('SELECT DOB FROM icebreaker.registration WHERE userId = %s', (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def _calculate_age(self, dob):
        from datetime import date
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    def _handle_recommend_mode_industry_v1(self, user_id, row, page, per_page, search_text, login_web_token, output_list, place_id=None):
        try:
            cursor = self.db.cur()
            recommendations = self.best_matches_cosine_vibr(user_id, row, page, per_page, place_id)
            if recommendations.get("Status") == "Success":
                recommend_output_list = recommendations.get("Matches", [])
                total_users = len(output_list)
            else:
                recommend_output_list = []
                total_users = 0

            if search_text:
                search_text_lower = search_text.lower()
                recommend_output_list = [user for user in recommend_output_list if search_text_lower in user.get('fullName', '').lower()]
                

            for user in recommend_output_list:
                target_user_id = user["userId"]
                is_incognito = user.get("incognito", False)

                # Fetch the "real" place ID between the users (same as normal mode)
                real_place_id = self._fetch_real_place_industry_v1(cursor, user_id, target_user_id)

                # Get connection statuses
                accepted_status, pending_status, received_status = self._fetch_request_status_industry_v1(
                    cursor, user_id, target_user_id, real_place_id
                )

                # Determine relationship status
                user["status"] = self._determine_status_industry_v1(
                    accepted_status, pending_status, received_status
                )

                # Reveal incognito users if connected or request received
                if is_incognito and self._is_connected_industry_v1(accepted_status, received_status):
                    user["incognito"] = False

              
                for user in recommend_output_list:
                    user["wavedTime"] = user.get("wavedTime", 0)
                    user["wavedbackTime"] = user.get("wavedbackTime", 0)

                # Sort by most recent activity (wave or waveback) first
                recommend_output_list.sort(
                    key=lambda u: max(u["wavedTime"], u["wavedbackTime"]),
                    reverse=True
                )


                # --- Sort by wavedTime descending (most recent waves first) ---
            # recommend_output_list.sort(key=lambda u: u.get("wavedTime", 0), reverse=True)

            total_users = len(recommend_output_list)

            response = {
                "List": recommend_output_list,
                "Status": "Success",
                "Page": 1,
                "PerPage": 3,
                "TotalUsers": total_users,
                "loginWebToken": login_web_token
            }
            return jsonify(response)
        except Exception as e:
            return {"Status": "Failure", "message": str(e)}
        finally:
            if cursor:
                cursor.close()  


    def _build_user_dicts_industry_v1(self, cursor, row, user_id, my_vibes, recommended_user_ids, search_text, place_id=None):
        output_list = []
        for user in row:
            waved_user_id = user[0]
            
            cursor.execute(self.BLOCKED_USER_CHECK_QUERY, (user_id, user[0]))
            res2 = cursor.fetchone()
            block_flag = 1 if res2 else 0

            is_incognito = user[17]
            user_dict = self._create_user_dict_industry_v1(cursor, user, block_flag, my_vibes, recommended_user_ids)

            real_place_id = self._fetch_real_place_industry_v1(cursor, user_id, user[0])
            accepted_status, pending_status, received_status = self._fetch_request_status_industry_v1(cursor, user_id, user[0], real_place_id)

            user_dict["status"] = self._determine_status_industry_v1(accepted_status, pending_status, received_status)
            if is_incognito and (self._is_connected_industry_v1(accepted_status, received_status)):
                user_dict["incognito"] = False


            # user_dict["wave_status"] = self.get_wave_status_sender_here(cursor, user_id, waved_user_id,place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ')
            user_dict["wave_status"] = self.get_wave_status_sender_here(cursor, user_id, waved_user_id, place_id=place_id)
            cursor.execute("""
                    SELECT created_at 
                    FROM icebreaker.user_likes
                    WHERE sender_id = %s 
                    AND receiver_id = %s 
                    # AND place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
                    AND (%s IS NULL OR place_id = %s)
                """, (waved_user_id, user_id, place_id, place_id))
            result  = cursor.fetchone()

            if result and result[0]:
                # Convert datetime to integer timestamp
                user_dict["wavedTime"] = int(result[0].timestamp())
            else:
                user_dict["wavedTime"] = 0
            output_list.append(user_dict)

            if search_text:
                search_text = search_text.lower()
                output_list = [user for user in output_list if search_text in user["fullName"].lower()]
        return output_list

    def _create_user_dict_industry_v1(self, cursor, user, block_flag, my_vibes, recommended_user_ids):
        user_id = user[0]
        is_incognito = bool(user[17])

        user_dict = {
            "userId": user_id,
            "profilePic": user[1],
            "selfiePic":user[2],
            "fullName": user[3],
            "email": user[4],
            "firebaseId": user[5],
            "signupType": user[6],
            "authenticationId": user[7],
            "Gender": user[8],
            "Preference": user[9],
            "DOB": user[10].strftime('%m-%d-%Y') if user[10] is not None else '',
            "vibes": user[11],
            "vibing": user[12],
            "block_flag": block_flag,
            "recommended": user[0] in recommended_user_ids,
            "user_aesthetic_score": user[16],
            "incognito": is_incognito,
            "matching_vibes": bool(set(user[11].split(',')) & set(my_vibes.split(','))) if user[11] and my_vibes else False
        }       

        # Include cognito_time only for incognito users
        if is_incognito:
            cursor.execute(
                "SELECT cognito_time FROM icebreaker.location WHERE userId = %s", 
                (user_id,)
            )
            row = cursor.fetchone()
            cognito_time = row[0] if row else None

            if cognito_time is not None:
                # Assign to dictionary as-is
                user_dict["cognito_time"] = cognito_time

        return user_dict
    

    def _fetch_real_place_industry_v1(self, cursor, user_id, target_user_id):
        cursor.execute(" select placeId from netwirkRequestsPending WHERE userId = %s AND requestedUserId = %s", (target_user_id, user_id))
        real_place_row = cursor.fetchone()
        real_place_id = real_place_row[0] if real_place_row else None
        if real_place_id is None:
            cursor.execute(" select placeId from netwirkRequestsAccepted WHERE userId = %s AND requestedUserId = %s", (target_user_id, user_id))
            real_place_row = cursor.fetchone()
            real_place_id = real_place_row[0] if real_place_row else None
        return real_place_id

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

    def _is_connected_industry_v1(self, accepted_status, received_status):
        return (accepted_status and accepted_status[0] == "Accepted") or (received_status and received_status[0] == "Pending")



    def get_industry_user_count(self):
        try:
            data = request.get_json()
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            user_id = data['userId']
            event_id = data['eventId']
            cursor = self.db.cur()
            place_id = self._resolve_place_id(user_id, data, cursor)

          
            # Check for eventId and query accordingly, excluding blocked users (both directions)
            if event_id:
                cursor.execute('''SELECT p.industry
                              FROM icebreaker.eventLocation e
                              JOIN icebreaker.profile p ON e.userId = p.userId
                              LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                              LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                              WHERE e.eventId = %s 
                              AND e.userId != %s 
                              AND b1.blockedId IS NULL 
                              AND b2.blockedId IS NULL''', 
                              (user_id, user_id, event_id, user_id))
                
            else:
                cursor.execute('''SELECT p.industry
                              FROM icebreaker.location l
                              JOIN icebreaker.profile p ON l.userId = p.userId
                              LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                              LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                              WHERE l.placeId = %s 
                              AND l.userId != %s 
                              AND b1.blockedId IS NULL 
                              AND b2.blockedId IS NULL''', 
                              (user_id, user_id, place_id, user_id))
                   
            industries = cursor.fetchall()

            # Flatten the list of tuples and split industries
            industry_list = [industry for sublist in industries for industry in sublist[0].split(',')]

            # Use Counter to count occurrences efficiently
            industry_counts = Counter(industry.strip() for industry in industry_list)

            # Convert Counter to list of dictionaries
            result = [{'industry': key, 'count': value} for key, value in industry_counts.items()]
            
            if industry_counts:
            # Convert dictionary entries to list of dictionaries 
                return result
            else:
                return{'industry': "", 'count': ""}

        except Exception as e:
            print(str(e))
            return{'message': str(e), 'status': 'Failure'}


    def confirmNetwirkAPI(self):
        try:
            cursor = self.db.cur()
            data = self._parse_request_data()
            login_web_token = self._get_login_web_token_confirm_api(cursor, data['loginUserId'])

            if data['status'] == 'Withdraw':
                return self._handle_withdraw(cursor, data, login_web_token)
            elif data['status'] == 'Reject':
                return self._handle_reject(cursor, data, login_web_token)
            elif data['status'] == 'Accepted':
                return self._handle_accept(cursor, data, login_web_token)
            else:
                return self._handle_send_request(cursor, data, login_web_token)

        except Exception as e:
            print(str(e))
            return {"status": 'failure', "message": str(e)}

    # --- Helper Methods Below ---

    def _parse_request_data(self):
        data = request.get_json()
        data['currentTime'] = datetime.now()
        data['formattedTime'] = data['currentTime'].strftime('%Y-%m-%d %H:%M:%S')
        data['requestTime'] = datetime.strptime(data['formattedTime'], '%Y-%m-%d %H:%M:%S')
        return data

    def _get_login_web_token_confirm_api(self, cursor, user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
        return cursor.fetchall()[0][0]

    def _handle_withdraw(self, cursor, data, token):
        login_id = data['loginUserId']
        req_id = data['requestedUserId']

        cursor.execute('SELECT * FROM netwirkRequestsPending WHERE userId = %s AND requestedUserId = %s', (login_id, req_id))
        row = cursor.fetchone()

        if not row:
            return self._failure('Request does not exist to withdraw', 1, token)

        cursor.execute('DELETE FROM netwirkRequestsPending WHERE userId = %s AND requestedUserID = %s', (login_id, req_id))
        self._log_recsys_interaction(cursor, login_id, req_id, "withdraw_request", "confirmNetwirkAPI:withdraw")
        self.db.conn.commit()
        return self._success('Request Withdrawn Successfully!', 0, token)

    def _handle_reject(self, cursor, data, token):
        login_id = data['loginUserId']
        req_id = data['requestedUserId']

        cursor.execute('SELECT placeId FROM icebreaker.netwirkRequestsPending WHERE requestedUserId = %s AND userId = %s', (login_id, req_id))
        place_id = cursor.fetchone()

        if not place_id:
            return self._failure('Request does not exist to reject', 1, token)

        cursor.execute(self.SELECT_PENDING_REQUEST, (req_id, login_id))
        row = cursor.fetchone()

        if not row:
            return self._failure('Request does not exist to reject', 1, token)

        cursor.execute(self.DELETE_PENDING_REQUEST, (req_id, login_id))
        self._log_recsys_interaction(cursor, login_id, req_id, "reject_request", "confirmNetwirkAPI:reject")
        self.db.conn.commit()
        return self._success('Request Rejected Successfully!', 0, token)

    def _handle_accept(self, cursor, data, token):
        login_id = data['loginUserId']
        req_id = data['requestedUserId']
        place_id = self._resolve_place_id(login_id, data, cursor)
        cursor.execute('SELECT * FROM netwirkRequestsAccepted WHERE UserId = %s AND requestedUserId = %s', (login_id, req_id))
        row = cursor.fetchone()

        if row and row[4] == 'Accepted':
            return self._success('Request Already Accepted', 1, token)

        # Withdrawn?
        cursor.execute('SELECT placeId FROM netwirkRequestsPending WHERE requestedUserId = %s AND userId = %s', (login_id, req_id))
        new_place_id = cursor.fetchone()
        if not new_place_id:
            return self._failure('Sorry! The request was withdrawn. You can no longer accept it.', 1, token)

        cursor.execute(self.SELECT_PENDING_REQUEST, (req_id, login_id))
        is_withdrawn = cursor.fetchone()

        if is_withdrawn is None:
            return self._failure('Sorry! The request was withdrawn. You can no longer accept it.', 1, token)

        # can_accept, accepted_in_window = self._check_acceptted_request_cooldown(cursor, login_id, new_place_id[0])
        # if not can_accept:
        #     return self._failure('You have reached the limit of 3 accepts. Try again after cooldown.', 1, token)

        # remaining = 3 - (accepted_in_window + 1)

        # Insert tracker and update accepted table
        request_time = datetime.now()
        cursor.execute('''INSERT INTO icebreaker.acceptRequestTracker (userid, requesteduserid, placeid, requestcount, requestTime) 
                        VALUES (%s, %s, %s, %s, %s)''', (login_id, req_id, new_place_id[0], 1, request_time))

        if row:
            cursor.execute('UPDATE netwirkRequestsAccepted SET status = %s, requestTime = %s WHERE userId = %s AND requestedUserId = %s AND placeId = %s',
                        ('Accepted', request_time, login_id, req_id, place_id))
        else:
            cursor.execute('INSERT INTO netwirkRequestsAccepted (status, userId, requestedUserId, requestTime, placeId) VALUES (%s, %s, %s, %s, %s)',
                        ('Accepted', login_id, req_id, request_time, place_id))
            cursor.execute('INSERT INTO netwirkRequestsAccepted (status, userId, requestedUserId, requestTime, placeId) VALUES (%s, %s, %s, %s, %s)',
                        ('Accepted', req_id, login_id, request_time, place_id))
            cursor.execute(self.DELETE_PENDING_REQUEST, (req_id, login_id))
            cursor.execute(self.DELETE_PENDING_REQUEST, (login_id, req_id))

        self._log_recsys_interaction(cursor, login_id, req_id, "accept_request", "confirmNetwirkAPI:accept")
        self.db.conn.commit()

        # Notification
        result, token_data = self._get_user_profiles(cursor, login_id, req_id)
        if token_data[4] == 1:
            self.send_fcm_with_rest_request_accepted_notification_netwirk(token_data[2], result[1], req_id, login_id, result[0])
        
        self.db.conn.close()
        message = f"Request Accepted Successfully!"
        return jsonify({
            'message': message,
            'status': 'success',
            'flag': 0,
            'fullName': result[1],
            'loginUserId': login_id,
            'profile_pic': result[0],
            'requestedUserfullName': token_data[1],
            'requestedUserId': req_id,
            'requestedUserprofile_pic': token_data[0],
            'loginWebToken': token
        })

    def _handle_send_request(self, cursor, data, token):
        login_id = data['loginUserId']
        req_id = data['requestedUserId']
        place_id = self._resolve_place_id(login_id, data, cursor)
        req_type = data.get('type')

        cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (req_id,))
        tokens = cursor.fetchone()
        receiver_name = tokens[1]
        cursor.execute('SELECT r.deviceToken, r.fullName, p.profilePic FROM icebreaker.registration r JOIN icebreaker.profile p ON r.userId = p.userId WHERE r.userId = %s', (login_id,))
        res = cursor.fetchone()

        cursor.execute('''
            SELECT count(netwirkRequestsId)
            FROM icebreaker.netwirkRequestsPending
            WHERE requestedUserId = %s
        ''', (req_id,))

        count = cursor.fetchone()[0] or 0

        # If any row exists → one of them blocked the other
        if count >= 5:
            self.send_fcm_with_rest_request_full_notification(tokens[0], req_id, login_id)
            receiver_name = tokens[1]
            message = f"{receiver_name} is maxed out on requests! Come back shortly."
            return self._failure(message, 1, token)

        cursor.execute('''
            SELECT 1 
            FROM icebreaker.blockedUsers
            WHERE (userId = %s AND blockedId = %s)
            OR (userId = %s AND blockedId = %s)
        ''', (login_id, req_id, req_id, login_id))

        block = cursor.fetchone()

        # If any row exists → one of them blocked the other
        if block:
            return self._failure('You cannot send a request because  the user has blocked the other.', 1, token)


        cursor.execute(self.SELECT_PENDING_REQUEST, (login_id, req_id))
        row = cursor.fetchone()
        if row and row[4] == 'Pending':
            return self._success('Request Already Sent', 1, token)
        
        is_first_request = False

        if req_type == 'here':
            can_send, sent_in_window, is_first_request, remaining_seconds  = self._check_sent_request_cooldown(cursor, login_id, place_id)
            if not can_send:
                # return self._failure('You have reached the limit of 3 requests. Try again after cooldown.', 1, token)
                hours = remaining_seconds // 3600
                minutes = (remaining_seconds % 3600) // 60
                seconds = remaining_seconds % 60

                if hours > 0:
                    time_msg = f"{hours}h {minutes}m"
                elif minutes > 0:
                    time_msg = f"{minutes}m {seconds}s"
                else:
                    time_msg = f"{seconds}s"

                return self._failure(
                    f"You’ve reached the 3 request limit. Try again in {time_msg}.",
                    1,
                    token
                )
            remaining = 3 - (sent_in_window + 1)

        else:
            can_send = True
            sent_in_window = 0
            remaining = 'Unlimited'
       

        is_incognito = self._is_incognito_enabled_requestsending(cursor, login_id)
        incognito_flag = 1 if is_incognito else 0

        request_time = datetime.now()
        cursor.execute(
            'INSERT INTO icebreaker.requestTracker (userid, requesteduserid, placeid, requestcount, requestTime, incognito, type) VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (login_id, req_id, place_id, 1, request_time, incognito_flag, req_type)
        )

        cursor.execute('INSERT INTO netwirkRequestsPending (status, userId, requestedUserId, requestTime, placeId) VALUES (%s, %s, %s, %s, %s)',
                    ('Pending', login_id, req_id, request_time, place_id))

        # First incognito "here" request in 2hr window → extend visibility by 20 minutes
        if is_incognito and req_type == 'here' and is_first_request:
            self._extend_cognito_time_for_first_incognito_request(cursor, login_id, place_id)

        self._log_recsys_interaction(cursor, login_id, req_id, "send_request", "confirmNetwirkAPI:send")
        self.db.conn.commit()

        cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (req_id,))
        tokens = cursor.fetchone()
        cursor.execute('SELECT r.deviceToken, r.fullName, p.profilePic FROM icebreaker.registration r JOIN icebreaker.profile p ON r.userId = p.userId WHERE r.userId = %s', (login_id,))
        res = cursor.fetchone()
        

        if tokens[2] == 1:
            self.send_fcm_with_rest_request_recieved_notification_netwirk(tokens[0], res[1], req_id, login_id, res[2])

            if is_incognito:
                cursor.execute('''
                    SELECT r.Gender, r.fullName, p.currentDesignation
                    FROM icebreaker.registration r
                    JOIN icebreaker.profile p ON r.userId = p.userId
                    WHERE r.userId = %s
                ''', (login_id,))
                user_info = cursor.fetchone()
                if user_info:
                    gender_filter = user_info[0]
                    full_name = user_info[1].strip().title()
                    current_designation = user_info[2]

                    # Send silent notifications using helper
                    self._notify_users_silent(
                        cursor=cursor,
                        user_id=login_id,
                        place_id=place_id,
                        incognito=True,
                        full_name=full_name,
                        current_designation=current_designation
                    )


        message = f"Request Sent Successfully! You have {remaining} request(s) remaining in this 2-hour window."
        self.db.conn.close()
        response_dict = {'message': message, 'status': 'success', 'flag': 0, 'loginWebToken': token}
        response_dict['is_first_request'] = bool(is_first_request) if req_type == 'here' else False
        return jsonify(response_dict)

    # --- Utility Methods ---

    
    def _notify_users_silent(self, cursor, user_id, place_id, incognito, full_name, current_designation):

        print(f"\n🔍 Checking nearby users for incognito notifications...")
        print(f"UserID: {user_id}, PlaceID: {place_id}, Incognito: {incognito}")

        """Find and notify nearby users (incognito or not)"""
        cursor.execute("""
            SELECT l.userId FROM icebreaker.location l
            JOIN icebreaker.registration r ON l.userId = r.userId
            WHERE l.placeId=%s AND l.userId!=%s
        """, (place_id, user_id))
        user_id_list = [u[0] for u in cursor.fetchall()]

        # print(f" Found {len(user_id_list)} users in the same place: {user_id_list}")


        if not user_id_list:
            print("⚠️ No users found to notify.")
            return

        placeholders = ','.join(['%s'] * len(user_id_list))
        query = f"""
            SELECT deviceToken 
            FROM icebreaker.registration 
            WHERE userId IN ({placeholders}) AND deviceToken IS NOT NULL AND deviceToken!=''
        """
        cursor.execute(query, tuple(user_id_list))
        tokens = cursor.fetchall()

        threads = []
        for token in tokens:
            print(f" Sending silent notification to device token: {token[0]} (recipient) from user_id: {user_id} (sender)")
            t = threading.Thread(target=self.send_fcm_with_rest_new_user_silent, args=(token[0],user_id))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # print(f" Finished sending notifications to {len(tokens)} users.\n")

    def send_fcm_with_rest_new_user_silent(self, token,user_id):
        refresh_type = "refresh_all_users"  # updated payload
        payload_data = {"type": refresh_type}  # data-only payload

        if not token:
            return {"status": "error", "message": "No device token provided."}

        try:
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            # SERVICE_ACCOUNT_FILE = r"D:\icebreakerNew\vibrr\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE]
            )
            auth_request = Request()
            credentials.refresh(auth_request)

            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            def send_payload(t):
                payload = {
                    "message": {
                        "token": t,
                        "data": payload_data,  # data-only, silent
                        "android": {
                            "priority": "high"  # ensures timely delivery on Android
                        },
                        "apns": {
                            "headers": {
                                "apns-priority": "10"  # immediate delivery on iOS
                            },
                            "payload": {
                                "aps": {
                                    "content-available": 1  # silent notification
                                }
                            }
                        }
                    }
                }
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                return {"status_code": resp.status_code, "response": resp.text}

            # Handle single token or list
            if isinstance(token, list):
                responses = [send_payload(t) for t in token]
                return {"status": "success", "messages": responses}
            else:
                resp = send_payload(token)
                return {"status": "success", "message": resp}

        except Exception as e:
            return {"status": "error", "message": str(e)}

            

    def _is_incognito_enabled_requestsending(self, cursor, login_id):
        cursor.execute("""
            SELECT incognito FROM icebreaker.location 
            WHERE userId = %s
        """, (login_id,))
        row = cursor.fetchone()
        return 1 if row and row[0] == 1 else 0

    def _extend_cognito_time_for_first_incognito_request(self, cursor, login_id, place_id):
        """Add 20 minutes to cognito_time on first incognito 'here' request in the 2-hour window."""
        if not place_id:
            return

        cursor.execute(
            "SELECT cognito_time FROM icebreaker.location WHERE userId = %s",
            (login_id,),
        )
        row = cursor.fetchone()
        if not row:
            return

        base_time = row[0] or datetime.now()
        cognito_time_new = base_time + timedelta(minutes=20)

        cursor.execute(
            """
            UPDATE icebreaker.location
            SET cognito_time = %s
            WHERE userId = %s
            """,
            (cognito_time_new, login_id),
        )


    def _check_acceptted_request_cooldown(self, cursor, user_id, place_id):
        cursor.execute('''SELECT requestTime FROM icebreaker.acceptRequestTracker WHERE userid = %s AND placeid = %s ORDER BY requestTime DESC LIMIT 3''',
                    (user_id, place_id))
        recent_requests = cursor.fetchall() or []

        now = datetime.now()
        # Only keep requests made in the last 2 hours
        recent_requests = [r for r in recent_requests if (now - r[0]) < timedelta(hours=2)]
        accepted_count = len(recent_requests)

        if accepted_count < 3:
            return True, accepted_count
        time_diff = datetime.now() - recent_requests[0][0]
        can_accept = time_diff >= timedelta(hours=2)
        return can_accept, accepted_count
    
    # def _check_sent_request_cooldown(self, cursor, user_id, place_id):
    #     cursor.execute('''SELECT requestTime FROM icebreaker.requestTracker WHERE userid = %s AND placeid = %s ORDER BY requestTime DESC LIMIT 3''',
    #                 (user_id, place_id))
    #     recent_requests = cursor.fetchall() or []

    #     now = datetime.now()
    #     # Only keep requests made in the last 2 hours
    #     recent_requests = [r for r in recent_requests if (now - r[0]) < timedelta(hours=2)]
    #     sent_count = len(recent_requests)

    #     is_first_request = (sent_count == 0)

    #     if sent_count < 3:
    #         return True, sent_count, is_first_request
    #     time_diff = datetime.now() - recent_requests[0][0]
    #     can_send = time_diff >= timedelta(hours=2)
    #     return can_send, sent_count, is_first_request

    def _check_sent_request_cooldown(self, cursor, user_id, place_id):
        cursor.execute('''
            SELECT requestTime
            FROM icebreaker.requestTracker
            WHERE userid = %s
            AND placeid = %s
            AND type = 'here'
            AND requestTime >= NOW() - INTERVAL 2 HOUR
            ORDER BY requestTime DESC
        ''', (user_id, place_id))

        rows = cursor.fetchall()
        sent_count = len(rows)

        # is_first_request should only consider incognito requests
        cursor.execute('''
            SELECT COUNT(*)
            FROM icebreaker.requestTracker
            WHERE userid = %s
            AND placeid = %s
            AND type = 'here'
            AND incognito = 1
            AND requestTime >= NOW() - INTERVAL 2 HOUR
        ''', (user_id, place_id))
        incognito_count = cursor.fetchone()[0]
        is_first_request = (incognito_count == 0)

        # If under limit → can send immediately
        if sent_count < 3:
            return True, sent_count, is_first_request, 0

        # User hit limit → calculate remaining cooldown
        latest_request_time  = rows[0][0] 
        next_allowed_time = latest_request_time  + timedelta(hours=2)
        now = datetime.now()

        remaining_seconds = max(0, int((next_allowed_time - now).total_seconds()))

        return False, sent_count, is_first_request, remaining_seconds

    def _get_user_profiles(self, cursor, login_user_id, requested_user_id):
        cursor.execute('SELECT p.profilePic, r.fullName FROM icebreaker.profile p JOIN icebreaker.registration r ON p.userId = r.userId WHERE p.userId = %s', (login_user_id,))
        result = cursor.fetchone()
        cursor.execute('SELECT p.profilePic, r.fullName, r.deviceToken, r.authenticationId, r.receiveNotifications FROM icebreaker.profile p JOIN icebreaker.registration r ON p.userId = r.userId WHERE p.userId = %s', (requested_user_id,))
        token = cursor.fetchone()

        if result:
            result = (
                result[0],                      # profilePic
                (result[1] or "").strip().title()  # fullName
            )

        if token:
            token = (
                token[0],                        # profilePic
                (token[1] or "").strip().title(),# fullName
                token[2],                        # deviceToken
                token[3],                        # authenticationId
                token[4]                         # receiveNotifications
            )
        return result, token

    def _success(self, message, flag, token):
        return jsonify({'message': message, 'status': 'success', 'flag': flag, 'loginWebToken': token})

    def _failure(self, message, flag, token):
        return jsonify({'message': message, 'status': 'failure', 'flag': flag, 'loginWebToken': token})

    def make_db_entry_of_notification(self, user_id, requested_user_id, request_status, description, profile_pic):
        try:
            current_time = datetime.now(timezone.utc)
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            notification_time = formatted_time
            cursor = self.db.cur()
            # Convert sets to single values if needed
            if isinstance(user_id, set):
                user_id = next(iter(user_id))  # take the first value from the set

            if isinstance(requested_user_id, set):
                requested_user_id = next(iter(requested_user_id))  # take the first value from the set

            # Insert the details of the notification sent into the notifications table
            cursor.execute('''INSERT INTO icebreaker.notifications (userId, requestedUserId, notificationTime, requestStatus, description, profilePic)
                            VALUES (%s, %s, %s, %s, %s, %s)''',
                            (user_id, requested_user_id, notification_time, request_status, description, profile_pic))
            self.db.conn.commit()
            message = "Notification details inserted successfully in database"

            return {"status": 'success', "message": message}

        except Exception as e:
            print(str(e))
            message = str(e)
            return {"status": 'failure', "message": message}
            
    def send_fcm_with_rest_request_withdrawn_notification_netwirk(self, tokens, full_name):

            # Notification data
            title = "Request Withdrawn"
            body = f'{full_name} has withdrawn a request sent to you'
            request_withdrawn = "request_withdrawn"

            # Path to your service account key
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE

            # Create a credentials object from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])

            # Create a request object for fetching the access token
            auth_request = Request()

            # Refresh the token
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Prepare payload based on tokens type
            if isinstance(tokens, list):
                payload = {
                    "message": {
                        "tokens": tokens,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_withdrawn
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            else:
                payload = {
                    "message": {
                        "token": tokens,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_withdrawn
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            try:
                # Send the request to FCM
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                return {"status": "success", "message": resp.text}

            except Exception as e:
                message = str(e)  # Convert exception to string
                return {"status": "error", "message": message}


    def send_fcm_with_rest_request_rejected_notification_netwirk(self, tokens, full_name):

            # Notification data
            title = "Request Rejected"
            body = f'{full_name} has rejected a request you sent'
            request_rejected = "request_rejected"

            # Path to your service account key
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE

            # Create a credentials object from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])

            # Create a request object for fetching the access token
            auth_request = Request()

            # Refresh the token
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Prepare payload based on tokens type
            if isinstance(tokens, list):
                payload = {
                    "message": {
                        "tokens": tokens,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_rejected
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            else:
                payload = {
                    "message": {
                        "token": tokens,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_rejected
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                return {"status": "success", "message": resp.text}

            except Exception as e:
                message = str(e)  # Convert exception to string
                return {"status": "error", "message": message}
            
    def send_fcm_with_rest_request_full_notification(self, token, user_id, requested_user_id):
       
            # Notification data
            title = "Request Full"
            body = "Someone’s interested! Make room by Accepting / Deleting a request to receive it."
            request_full = "request_full"
            user_id = {user_id}
            requested_user_id = {requested_user_id}

            # Path to your service account key
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"


            
            # Create a credentials object from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])
            # print("Service account credentials loaded successfully.")

            # Create a request object for fetching the access token
            auth_request = Request()

            # Refresh the token
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Prepare payload based on token type
            if isinstance(token, list):
                payload = {
                    "message": {
                        "tokens": token,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_full,
                            "user_id": str(user_id),
                            "requested_user_id": str(requested_user_id)
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            else:
                payload = {
                    "message": {
                        "token": token,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_full,
                            "user_id": str(user_id),
                            "requested_user_id": str(requested_user_id)
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            
            try:
            #     # Send the request to FCM
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                # results = self.make_db_entry_of_notification(user_id, requested_user_id, request_recieved, body, profile_pic)
                # if results["status"] == 'success':
                #     print("Database entry success for request_recieved")
                # else:
                #     print("Database entry failure for request_recieved")
                # return {"status": "success", "message": resp.text}

            except Exception as e:
                message = str(e)  # Convert exception to string
                return {"status": "error", "message": message}


    def send_fcm_with_rest_request_recieved_notification_netwirk(self, token, full_name, user_id, requested_user_id, profile_pic):
            full_name = full_name.strip().title()
            # Notification data
            title = "Request Received"
            body = f'{full_name} has sent you a request'
            request_recieved = "request_recieved"
            user_id = {user_id}
            requested_user_id = {requested_user_id}

            # Path to your service account key
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"


            
            # Create a credentials object from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])
            # print("Service account credentials loaded successfully.")

            # Create a request object for fetching the access token
            auth_request = Request()

            # Refresh the token
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Prepare payload based on token type
            if isinstance(token, list):
                payload = {
                    "message": {
                        "tokens": token,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_recieved,
                            "user_id": str(user_id),
                            "requested_user_id": str(requested_user_id)
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            else:
                payload = {
                    "message": {
                        "token": token,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_recieved,
                            "user_id": str(user_id),
                            "requested_user_id": str(requested_user_id)
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            
            try:
                # Send the request to FCM
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                results = self.make_db_entry_of_notification(user_id, requested_user_id, request_recieved, body, profile_pic)
                if results["status"] == 'success':
                    print("Database entry success for request_recieved")
                else:
                    print("Database entry failure for request_recieved")
                return {"status": "success", "message": resp.text}

            except Exception as e:
                message = str(e)  # Convert exception to string
                return {"status": "error", "message": message}
            
            
    def send_fcm_with_rest_request_accepted_notification_netwirk(self, tokens, full_name, user_id, requested_user_id, profile_pic):
        
            full_name = full_name.strip().title()
            # Notification data
            title = "Request Accepted"
            body = f'{full_name} has accepted your request'
            request_accepted = "request_accepted"
            user_id = {user_id}
            requested_user_id = {requested_user_id}

            # Path to your service account key
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"

            # Create a credentials object from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])

            # Create a request object for fetching the access token
            auth_request = Request()

            # Refresh the token
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Prepare payload based on token type
            if isinstance(tokens, list):
                payload = {
                    "message": {
                        "tokens": tokens,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_accepted,
                            "user_id": str(user_id),
                            "requested_user_id": str(requested_user_id)
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            else:
                payload = {
                    "message": {
                        "token": tokens,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": request_accepted,
                            "user_id": str(user_id),
                            "requested_user_id": str(requested_user_id)
                        },
                        "android": {
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
            try:
                # Send the request to FCM
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                results = self.make_db_entry_of_notification(user_id,requested_user_id, request_accepted, body, profile_pic)
                if results["status"] == 'success':
                    print("Database entry success for request_accepted")
                else:
                    print("Database entry failure for request_accepted")
                return {"status": "success", "message": resp.text}

            except Exception as e:
                message = str(e)  # Convert exception to string
                return {"status": "error", "message": message} 


    def get_list_of_all_notifications(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            page = int(data.get('page', 1))  # Default to page 1 if not provided
            per_page = int(data.get('perPage', 10))  # Default to 10 items per page if not provided
            offset = (page - 1) * per_page
            results = []
            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            # Get the count of all notifications
            cursor.execute("""
                           SELECT COUNT(*)
                           FROM icebreaker.notifications
                           WHERE userId = %s
                           """, (user_id,))
            total_notification_count = cursor.fetchone()[0]

            # Fetch the details of all notifications
            cursor.execute("""
                           SELECT requestedUserId, notificationTime, requestStatus, description, profilePic
                           FROM icebreaker.notifications
                           WHERE userId = %s
                           ORDER BY notificationTime DESC
                           LIMIT %s OFFSET %s
                           """, (user_id, per_page, offset))
            
            rows = cursor.fetchall()
            
            for resp in rows:

                if rows:
                    user_dict = {
                        "requestedUserId": resp[0],
                        "notificationTime": resp[1],
                        "requestStatus" : resp[2],
                        "description" : resp[3],
                        "profilePic" : resp[4]
                        }
                    results.append(user_dict)
                            
                else:        
                    print(f"No notifications found for userId: {user_id}")

            cursor.close()

            return jsonify({
                'List': results,
                'Status': 'Success',
                'Page' : page,
                'PerPage' : per_page,
                'TotalNotifications' : total_notification_count,
                "loginWebToken": login_web_token
                })

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error


    def clear_all_notifications(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            # Delete the notifications for the given userId
            cursor.execute("""
                           DELETE
                           FROM icebreaker.notifications
                           WHERE userId = %s
                           """, (user_id,))
            
            self.db.conn.commit()
            cursor.close()

            return jsonify({
                'Status': 'Success',
                "loginWebToken": login_web_token
                })

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error


    def get_list_of_accepted_users_netwirk(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            search_text = data['searchText']
            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page
            cursor = self.db.cur()

            login_web_token = self._fetch_login_token_users_accepted_user(cursor, user_id)
            logged_in_user_place_id = self._fetch_user_place(cursor, user_id)

            total_accepted_users = self._fetch_total_accepted_users(cursor, user_id)
            rows = self._fetch_paginated_accepted_users(cursor, user_id, per_page, offset)

            same_place_users, other_users = [], []
            for row in rows:
                user_dict = self._build_user_dict_new(cursor, row, logged_in_user_place_id)
                if user_dict:
                    if user_dict["isLive"] == 1:
                        same_place_users.append(user_dict)
                    else:
                        other_users.append(user_dict)

            results = same_place_users + other_users
            results = sorted(
                results,
                key=lambda x: (
                    0 if x["online_status"] == "online" else 1,
                    -(datetime.strptime(x["session_start"], "%Y-%m-%d %H:%M:%S").timestamp())
                        if x["online_status"] == "online" and x.get("session_start")
                        else -(datetime.strptime(x["last_seen"], "%Y-%m-%d %H:%M:%S").timestamp())
                        if x.get("last_seen")
                        else 0
                )
            )
            # Apply search filter if provided
            if search_text:
                results = self._filter_results_searchText(results, search_text)
                total_accepted_users = len(results)

            return jsonify({
                'List': results,
                'Status': 'Success',
                'Page': page,
                'PerPage': per_page,
                'TotalAcceptedUsers': total_accepted_users,
                "loginWebToken": login_web_token
            })

        except Exception as e:
            self.db.conn.rollback()
            return {'error': str(e)}

    # ---------------- Helper methods ---------------- #

    def _fetch_login_token_users_accepted_user(self, cursor, user_id):    
        return self._get_login_web_token_confirm_api(cursor, user_id)

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

    def _fetch_total_accepted_users(self, cursor, user_id):
        cursor.execute("""
           SELECT COUNT(*)
        FROM icebreaker.netwirkRequestsAccepted nra
        JOIN icebreaker.registration r ON r.userId = nra.requestedUserId
        WHERE nra.userId = %s

        AND NOT EXISTS (
        SELECT 1 FROM icebreaker.blockedUsers b1
        WHERE b1.blockedId = nra.requestedUserId AND b1.userId = %s
        )
        AND NOT EXISTS (
        SELECT 1 FROM icebreaker.blockedUsers b2
        WHERE b2.userId = nra.requestedUserId AND b2.blockedId = %s
        )

        AND NOT EXISTS (
        SELECT 1 FROM icebreaker.netwirkReports r1
        WHERE r1.reportedUserId = nra.requestedUserId AND r1.userId = %s
        )
        AND NOT EXISTS (
        SELECT 1 FROM icebreaker.netwirkReports r2
        WHERE r2.userId = nra.requestedUserId AND r2.reportedUserId = %s
        )
        """, (user_id, user_id, user_id, user_id, user_id))
        return cursor.fetchone()[0]

    def _fetch_paginated_accepted_users(self, cursor, user_id, per_page, offset):
        cursor.execute("""
            SELECT nra.requestTime, nra.userId, nra.requestedUserId, nra.status, nra.netwirkRequestsId
            FROM icebreaker.netwirkRequestsAccepted nra
            JOIN icebreaker.registration r ON r.userId = nra.requestedUserId
            WHERE nra.userId = %s

            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b1
                WHERE b1.blockedId = nra.requestedUserId AND b1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b2
                WHERE b2.userId = nra.requestedUserId AND b2.blockedId = %s
            )

            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r1
                WHERE r1.reportedUserId = nra.requestedUserId AND r1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r2
                WHERE r2.userId = nra.requestedUserId AND r2.reportedUserId = %s
            )

            ORDER BY nra.requestTime DESC
            LIMIT %s OFFSET %s
        """, (user_id, user_id, user_id, user_id, user_id, per_page, offset))
        return cursor.fetchall()

    def _build_user_dict_new(self, cursor, row, logged_in_user_place_id):
        request_time, user_id, requested_user_id, status, _ = row

        # Profile info
        cursor.execute("""
                SELECT p.profilePic, r.fullName, r.email, r.firebaseId, r.signupType, r.authenticationId, r.Gender, 
                       r.Preference, r.DOB, p.vibes, p.vibing, 
                       l.placeId
                       FROM icebreaker.profile p 
                       JOIN icebreaker.registration r ON p.userId = r.userId 
                       LEFT JOIN icebreaker.location l ON p.userId = l.userId 
                       WHERE r.userId = %s
        """, (requested_user_id,))
        res = cursor.fetchall()

        if not res:
            return None

        # Block check
        cursor.execute(self.CHECK_BLOCKED_USER_REVERSE_QUERY, (requested_user_id, user_id))
        block_flag = 1 if cursor.fetchone() else 0

        profile = res[0]
        user_place_id = profile[11]
        is_live = 1 if logged_in_user_place_id and user_place_id == logged_in_user_place_id else 0

        cursor.execute("""
            SELECT lastping, session_start
            FROM icebreaker.user_pings
            WHERE userId = %s
        """, (requested_user_id,))
        ping_result = cursor.fetchone()

        online_status = "offline"
        last_seen = ""
        session_start = ""

        if ping_result and ping_result[0]:
            last_ping = ping_result[0]

            # Convert string to datetime if needed
            if isinstance(last_ping, str):
                last_ping_dt = datetime.strptime(last_ping, "%Y-%m-%d %H:%M:%S")
            else:
                last_ping_dt = last_ping

            now = datetime.utcnow()
            # print("now", now)

            time_diff = (now - last_ping_dt).total_seconds()
            # print("time_diff", time_diff)

            if time_diff <= 30:   # change to 30 if you want 30 seconds
                online_status = "online"

            last_seen = last_ping_dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "RequestTime": request_time,
            "UserId": user_id,
            "RequestedUserID": requested_user_id,
            "Status": status,
            "profilePic": profile[0] or '',
            "fullName": profile[1] or '',
            "email": profile[2] or '',
            "firebaseId": profile[3] or '',
            "signupType": profile[4] or '',
            "authenticationId": profile[5] or '',
            "Gender": profile[6] or '',
            "Preference": profile[7] or '',
            "DOB": profile[8].strftime('%m-%d-%Y') if profile[8] else '',
            "vibes": profile[9] or '',
            "vibing": profile[10] or '',
            "block_flag": block_flag,
            "isLive": is_live,
            "online_status": online_status,
            "last_seen":last_seen
        }

    def _filter_results_searchText(self, results, search_text):
        search_text = search_text.lower()
        def matches(user):
            return (
                search_text in user.get("fullName", "").lower() or
                search_text in user.get("currentDesignation", "").lower() or
                search_text in user.get("companyName", "").lower()
            )
        return [user for user in results if matches(user)]

        
    def netwirk_Requests_Received(self):
            try:
                data = request.get_json()
                requested_user_id = data['RequestedUserId']
                page = int(data.get('Page', 1))
                per_page = int(data.get('PerPage', 10))
                offset = (page - 1) * per_page

                results = []
                cursor = self.db.cur()

                cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (requested_user_id,))
                login_token = cursor.fetchall()
                login_web_token = login_token[0][0] if login_token else ""

                cursor.execute('''
                    SELECT COUNT(*)
                    FROM netwirkRequestsPending
                    WHERE requestedUserId = %s
                    AND userId NOT IN (
                        SELECT reportedUserId
                        FROM icebreaker.netwirkReports
                        WHERE userId = %s
                    )
                ''', (requested_user_id, requested_user_id))
                total_requests = cursor.fetchone()[0]

                cursor.execute('''
                    SELECT requestTime, userId, requestedUserId, status, netwirkRequestsId
                    FROM netwirkRequestsPending
                    WHERE requestedUserId = %s
                    AND userId NOT IN (
                        SELECT reportedUserId
                        FROM icebreaker.netwirkReports
                        WHERE userId = %s
                    )
                    ORDER BY requestTime DESC
                    LIMIT %s OFFSET %s
                ''', (requested_user_id, requested_user_id, per_page, offset))
                rows = cursor.fetchall()

                for user_id in rows:
                    user_dict = self._process_user(cursor, user_id)
                    if user_dict:
                        results.append(user_dict)
                    else:
                        print(f"No data found for userId {user_id[2]}")

                return jsonify({
                    'List': results,
                    'Status': 'Success',
                    'Page': page,
                    'PerPage': per_page,
                    'TotalRequests': total_requests,
                    'max_visible_req' : "5",
                    "loginWebToken": login_web_token
                })

            except Exception as e:
                self.db.conn.rollback()
                return {'error': str(e)}

    def _process_user(self, cursor, user_id):
        cursor.execute("""
                SELECT p.profilePic, r.fullName, r.email, r.firebaseId, r.signupType, 
                    r.authenticationId, r.Gender, r.Preference, r.DOB, p.vibes, p.vibing
                FROM icebreaker.profile p
                JOIN icebreaker.registration r ON p.userId = r.userId
                WHERE r.userId = %s
            """, (user_id[1],))
        res = cursor.fetchall()

        cursor.execute(self.CHECK_BLOCKED_USER_REVERSE_QUERY, (user_id[1], user_id[2]))
        result = cursor.fetchone()
        block_flag = 1 if result else 0

        if not res:
            return None

        user_dict = {
                "RequestTime": user_id[0],
                "UserId": user_id[1],
                "RequestedUserID": user_id[2],
                "Status": user_id[3],
                "profilePic": res[0][0] if res[0][0] is not None else '',
                "fullName": res[0][1] if res[0][1] is not None else '',
                "email": res[0][2] if res[0][2] is not None else '',
                "firebaseId": res[0][3] if res[0][3] is not None else '',
                "signupType": res[0][4] if res[0][4] is not None else '',
                "authenticationId": res[0][5] if res[0][5] is not None else '',
                "Gender": res[0][6] if res[0][6] is not None else '',
                "Preference": res[0][7] if res[0][7] is not None else '',
                "DOB": res[0][8].strftime('%m-%d-%Y') if res[0][8] is not None else '',
                "vibes": res[0][9] if res[0][9] is not None else '',
                "vibing": res[0][10] if res[0][10] is not None else '',
                "block_flag": block_flag,
            }

        cursor.execute(self.GET_USER_PICTURES_QUERY, (user_id[1],))
        pics_data = cursor.fetchall()

        pictures = []
        for row in pics_data:
            pic_url = row[1]
            is_profile = bool(row[2])
            pictures.append({
                "url": pic_url,
                "isProfilePicture": is_profile
            })

        user_dict["pictures"] = pictures
        return user_dict

            
    def get_user_profile_details(self):
        try:
            request_data = request.get_json()
            login_user_id = request_data['loginUserId']
            requested_user_id = request_data['requestedUserId']

            cursor = self.db.cur()
            place_id = self._resolve_place_id(login_user_id, request_data, cursor)

            login_web_token = self._get_login_token_google(cursor, login_user_id)
            print(login_web_token)
            block_flag = self._get_block_status(cursor, login_user_id, requested_user_id)

            user_dict = self._get_user_details_profile(cursor, requested_user_id, block_flag, login_user_id, place_id)
            if not user_dict:
                return self._user_not_found_response(cursor)

            user_dict["status"] = self._get_interaction_status(cursor, login_user_id, requested_user_id)
            user_dict["loginWebToken"] = login_web_token
            user_dict["pictures"] = self._get_user_pictures(cursor, requested_user_id)

            cursor.close()
            self.db.conn.close()
            return jsonify(user_dict)

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            print(error)
            return jsonify(error)

    # ----------------- HELPER METHODS -------------------

    def _get_block_status(self, cursor, login_user_id, requested_user_id):
        cursor.execute(
            'SELECT 1 FROM icebreaker.blockedUsers WHERE blockedId = %s AND userId = %s',
            (requested_user_id, login_user_id)
        )
        return 1 if cursor.fetchone() else 0

    def _get_user_details_profile(self, cursor, requested_user_id, block_flag, login_user_id, place_id=None):
        cursor.execute("""
                SELECT r.userId, 
            p.profilePic,
            p.selfiePic, 
            r.fullName, 
            r.email, 
            r.firebaseId, 
            r.signupType, 
            r.deviceToken, 
            r.authenticationId, 
            r.Gender, 
            r.Preference, 
            r.DOB, 
            p.vibes, 
            p.vibing, 
            r.countryCode, 
            r.phoneNo, 
            r.city, 
            r.state, 
            r.country, 
            p.zodiac_sign
        FROM icebreaker.registration r
        LEFT JOIN icebreaker.profile p ON r.userId = p.userId
        WHERE r.userId = %s
        """, (requested_user_id,))
        row = cursor.fetchone()

        if not row:
            return None

        cursor.execute("""
            SELECT 1 
            FROM icebreaker.wave
            WHERE userId = %s 
            AND wavedToUserId = %s 
            # AND placeId = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            AND (%s IS NULL OR placeId = %s)
        """, (requested_user_id, login_user_id, place_id, place_id))
        wave_status = cursor.fetchone() is not None

        cursor.execute("""
            SELECT 1 
            FROM icebreaker.wave
            WHERE userId = %s 
            AND wavedToUserId = %s 
            # AND placeId = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            AND (%s IS NULL OR placeId = %s)
        """, (login_user_id, requested_user_id, place_id, place_id))
        wave_status_sender = cursor.fetchone() is not None

        cursor.execute("""
            SELECT 1 
            FROM icebreaker.waveback
            WHERE userId = %s 
            AND wavedToUserId = %s 
            # AND placeId = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            AND (%s IS NULL OR placeId = %s)
        """, (login_user_id, requested_user_id, place_id, place_id))
        waveback_exists = cursor.fetchone() is not None

        # --- Determine wave_status based on conditions ---
        if waveback_exists:
            wave_status = "waved"
        elif wave_status:
            wave_status = "wave_back"
        elif wave_status_sender:
            wave_status = "already_waved"                     
        else:
            wave_status = "no_wave"
           

        return {
            "userId": row[0],
            "profilePic": row[1],
            "selfiePic": row[2],
            "fullName": row[3],
            "email": row[4],
            "firebaseId": row[5],
            "signupType": row[6],
            "deviceToken": row[7],
            "authenticationId": row[8],
            "Gender": row[9],
            "Preference": row[10],
            "DOB": row[11].strftime('%m-%d-%Y') if row[11] else None,
            "vibes": row[12],
            "vibing": row[13],
            "countryCode": row[14],
            "phoneNo": row[15],
            "city": row[16],
            "state": row[17],
            "country": row[18],
            "zodiac_sign": row[19],
            "block_flag": block_flag,
            "wave_status": wave_status
        }

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

    def _get_user_pictures(self, cursor, user_id):
        cursor.execute(
            "SELECT picture_urls, IsProfilePicture FROM icebreaker.userPictures WHERE loginUserId = %s",
            (user_id,)
        )
        pics_data = cursor.fetchall()
        return [
            {"url": pic_url, "isProfilePicture": bool(is_profile)}
            for pic_url, is_profile in pics_data
        ]

    def _user_not_found_response(self, cursor):
        cursor.close()
        self.db.conn.close()
        return jsonify({"error": self.USER_NOT_FOUND_MESSAGE})


    def get_count_of_accepted_users_and_requests_received(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute('''SELECT count(*)
                           FROM icebreaker.netwirkRequestsAccepted
                           WHERE userId = %s
                           AND requestedUserId NOT IN
                           (SELECT reportedUserId
                           FROM icebreaker.netwirkReports
                           WHERE userId = %s)''', 
                           (user_id, user_id))
            total_connections = cursor.fetchone()[0]
            
            cursor.execute('''SELECT count(*)
                           FROM netwirkRequestsPending
                           WHERE requestedUserId = %s
                           AND userId NOT IN
                           (SELECT reportedUserId
                           FROM icebreaker.netwirkReports
                           WHERE userId = %s)''', 
                           (user_id, user_id))
            total_requests = cursor.fetchone()[0]

            response = {'total_connections' : total_connections,
                        'total_requests' : total_requests,
                        "loginWebToken": login_web_token
                        }
            return jsonify(response)

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error
        

    # def send_fcm_with_rest_logout_notification(self, tokens, full_name):
    #     # Notification data
    #     title = "User Left"
    #     body = f"{full_name} left the place"
    #     logout_user = "logout_user"


    #     try:
    #         # Path to your service account key
    #         SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
    #         # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"

    #         # Create a credentials object from the service account file
    #         credentials = service_account.Credentials.from_service_account_file(
    #             SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])

    #         # Create a request object for fetching the access token
    #         auth_request = Request()

    #         # Refresh the token
    #         credentials.refresh(auth_request)

    #         headers = {
    #             'Authorization': f'Bearer {credentials.token}',
    #             'Content-Type': self.CONTENT_TYPE,
    #         }

    #         url = self.FCM_SEND_URL

    #         if isinstance(tokens, list):
    #             for token in tokens:
    #                 payload = {
    #                     "message": {
    #                         "token": token,
    #                         "notification": {
    #                             "title": title,
    #                             "body": body
    #                         },
    #                         "data": {
    #                             "type": logout_user
    #                         },
    #                         "android": {
    #                             "collapse_key": "type_a"
    #                         },
    #                         "apns": {
    #                             "payload": {
    #                                 "aps": {
    #                                     "alert": {
    #                                         "title": title,
    #                                         "body": body
    #                                     },
    #                                     "mutable-content": 1  # iOS Notification Service Extension
    #                                 }
    #                             }
    #                         }
    #                     }
    #                 }
    #                 # Make the request to FCM
    #                 resp = requests.post(url, headers=headers, data=json.dumps(payload))
    #         else:
    #             payload = {
    #                 "message": {
    #                     "token": tokens,
    #                     "notification": {
    #                         "title": title,
    #                         "body": body
    #                     },
    #                     "data": {
    #                         "type": logout_user
    #                     },
    #                     "android": {
    #                         "collapse_key": "type_a"
    #                     },
    #                     "apns": {
    #                         "payload": {
    #                             "aps": {
    #                                 "alert": {
    #                                     "title": title,
    #                                     "body": body
    #                                 },
    #                                 "mutable-content": 1  # iOS Notification Service Extension
    #                             }
    #                         }
    #                     }
    #                 }
    #             }

    #             # Make the request to FCM
    #             resp = requests.post(url, headers=headers, data=json.dumps(payload))

    #         return {"status": "success", "message": resp.text}

    #     except Exception as e:
    #         message = str(e)  # Convert exception to string
    #         return {"status": "error", "message": message}

    def send_fcm_with_rest_logout_notification(self, tokens, full_name):
        title = "User Left"
        body = f"{full_name} left the place"

        try:
            # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE]
            )
            credentials.refresh(Request())

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            def build_payload(t: str) -> dict:
                return {
                    "message": {
                        "token": t,
                        "notification": {
                            "title": title,
                            "body": body
                        },
                        "data": {
                            "type": "logout_user"
                        },
                        "android": {
                            "priority": "high",
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "headers": {
                                "apns-priority": "10"
                            },
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1,
                                    "sound": "default"
                                }
                            }
                        }
                    }
                }

            def send_to_token(t: str) -> dict:
                if credentials.expiry and credentials.expired:
                    credentials.refresh(Request())
                    headers['Authorization'] = f'Bearer {credentials.token}'

                resp = requests.post(
                    self.FCM_SEND_URL,
                    headers=headers,
                    data=json.dumps(build_payload(t))
                )
                print(f"FCM logout response for token {t[:20]}...: "
                    f"status={resp.status_code}, body={resp.text}")

                return {
                    "token": t,
                    "success": resp.status_code == 200,
                    "status_code": resp.status_code,
                    "response": resp.text
                }

            token_list = tokens if isinstance(tokens, list) else [tokens]
            results = [send_to_token(t) for t in token_list]
            any_success = any(r["success"] for r in results)

            for r in results:
                if not r["success"]:
                    print(f"FCM logout FAILED: token={r['token'][:20]}..., "
                        f"code={r['status_code']}, reason={r['response']}")

            return {
                "status": "success" if any_success else "error",
                "results": results if isinstance(tokens, list) else results[0]
            }

        except Exception as e:
            print(f"FCM logout Exception: {str(e)}")
            return {"status": "error", "message": str(e)}


    def block_user(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            user_id = data['userId']
            blocked_id = data['blockedId']
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            blocked__time = formatted_time

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]
            
            # Check if blocked_id exists in registration
            cursor.execute("SELECT 1 FROM icebreaker.registration WHERE userId = %s", (blocked_id,))
            blocked_exists = cursor.fetchone()
            if not blocked_exists:
                cursor.close()
                return {"status": "Failure", "message": f"User Not Found"}


            cursor.execute(self.GET_BLOCKED_USER_QUERY , (user_id, blocked_id))
            row = cursor.fetchone()

            if not row:
                cursor.execute("INSERT INTO icebreaker.blockedUsers (userId, blockedId, blockedTime) VALUES (%s, %s, %s)", (user_id, blocked_id, blocked__time))
                self.db.conn.commit()
                

                # Delete existing connections (accepted requests) both directions
                cursor.execute("""
                    DELETE FROM icebreaker.netwirkRequestsAccepted
                    WHERE (userId = %s AND requestedUserId = %s)
                    OR (userId = %s AND requestedUserId = %s)
                """, (user_id, blocked_id, blocked_id, user_id))

                cursor.execute("""
                    DELETE FROM icebreaker.netwirkRequestsPending
                    WHERE (userId = %s AND requestedUserId = %s)
                    OR (userId = %s AND requestedUserId = %s)
                """, (user_id, blocked_id, blocked_id, user_id))


                # ✅ Delete session likes
                cursor.execute("""
                    DELETE FROM icebreaker.session_likes
                    WHERE (user_id = %s AND wavedTouserId = %s)
                    OR (user_id = %s AND wavedTouserId = %s)
                """, (user_id, blocked_id, blocked_id, user_id))


                # ✅ Delete session likes near
                cursor.execute("""
                    DELETE FROM icebreaker.session_likes_near
                    WHERE (user_id = %s AND wavedTouserId = %s)
                    OR (user_id = %s AND wavedTouserId = %s)
                """, (user_id, blocked_id, blocked_id, user_id))


                # ✅ Delete user likes
                cursor.execute("""
                    DELETE FROM icebreaker.user_likes
                    WHERE (sender_id = %s AND receiver_id = %s)
                    OR (sender_id = %s AND receiver_id = %s)
                """, (user_id, blocked_id, blocked_id, user_id))


                # ✅ Delete user likes near
                cursor.execute("""
                    DELETE FROM icebreaker.user_likes_near
                    WHERE (sender_id = %s AND receiver_id = %s)
                    OR (sender_id = %s AND receiver_id = %s)
                """, (user_id, blocked_id, blocked_id, user_id))

                self.db.conn.commit()
                self._invalidate_offline_feed_cache(user_id)
                message = f"User {user_id} blocked user {blocked_id}"
         
            else:
                message = "This user is already blocked"

            cursor.close() 
            return {"status": "Success", "message": message, "loginWebToken": login_web_token}

        except Exception as e:
            message = str(e) 
            return {"status": "Failure", "message": message}

    
    def unblock_user(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            user_id = data['userId']
            blocked_id = data['blockedId']

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute(self.GET_BLOCKED_USER_QUERY , (user_id, blocked_id))
            row = cursor.fetchone()

            if row:
                cursor.execute("DELETE FROM icebreaker.blockedUsers WHERE userId= %s AND blockedId = %s ", (user_id, blocked_id))
                self.db.conn.commit()
                message = 'User unblocked successfully'

            else:
                message = "This user is not present in the blocked list"
            
            cursor.close() 
            return {"status": "Success", "message": message, "loginWebToken": login_web_token}

        except Exception as e:
            message = str(e)
            return {"status": "Failure", "message": message}


    def get_Blocked_Users_List(self):
        try:
            data = request.get_json()
            user_id = data['userId'] 

            results = [] 
            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute('''SELECT  blockedId, blockedTime 
                            FROM icebreaker.blockedUsers 
                            WHERE userId = %s
                            ''', 
                            (user_id,))
 
            rows = cursor.fetchall()
            
            for blocked_user_id, blocked_time in rows:

                    cursor.execute('''
                                   SELECT r.fullName, r.email, r.authenticationId, p.profilePic
                                   FROM icebreaker.profile p
                                   JOIN icebreaker.registration r ON p.userId = r.userId
                                   WHERE r.userId = %s'''
                                   , (blocked_user_id,))
                    res = cursor.fetchall()

                    full_name =''
                    email=''
                    authentication_id=''
                    Profile_pic=''

                    if res:
                        full_name = res[0][0] if res[0][0] is not None else ''
                        email = res[0][1] if res[0][1] is not None else ''
                        authentication_id = res[0][2] if res[0][2] is not None else ''
                        Profile_pic = res[0][3] if res[0][3] is not None else ''                        

                   
                    user_dict = {
                        "BlockedUserId": blocked_user_id,
                        "BlockedTime" : blocked_time,
                        "FullName" :full_name,
                        "Email" :email,
                        "AuthenticationId" : authentication_id,
                        "Profile_pic" : Profile_pic
                    }
                    results.append(user_dict)

            return jsonify({'List': results, 'Status': 'Success', "loginWebToken": login_web_token})

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error


    def who_blocked_me_list(self):
        try:
            data = request.get_json()
            user_id = data['userId'] 

            results = [] 
            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute('SELECT userId, blockedTime FROM icebreaker.blockedUsers WHERE blockedId = %s', (user_id,))
 
            rows = cursor.fetchall()
            
            for blocked_user_id, blocked_time in rows:

                    cursor.execute('''
                                   SELECT fullName, email, authenticationId
                                   FROM icebreaker.registration
                                   WHERE userId = %s'''
                                   , (blocked_user_id,))
                    res = cursor.fetchall() 
                   
                    user_dict = {
                        "BlockerUserId": blocked_user_id,
                        "BlockedTime" : blocked_time,
                        "FullName" : res[0][0],
                        "Email" : res[0][1],
                        "AuthenticationId" : res[0][2],
                    }
                    results.append(user_dict)

            return jsonify({'List': results, 'Status': 'Success', "loginWebToken": login_web_token})
        
        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error


    def report_user(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            user_id = data['userId']
            report_title = data['report_title']
            report_text = data['report_text']
            reported_user_id = data['reported_userId']
            report_time = datetime.now()
            formatted_time = report_time.strftime('%Y-%m-%d %H:%M') 

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute("SELECT userId, reportedUserId FROM icebreaker.netwirkReports WHERE userId = %s AND reportedUserId = %s", (user_id, reported_user_id)) 
            result = cursor.fetchall()

            if result:
                user_reported_message = 'You have already reported this user.'
                user_reported_status = 'Success'
                return{'message': user_reported_message, 'status': user_reported_status, "loginWebToken": login_web_token}   

            else:             
                cursor.execute('INSERT INTO icebreaker.netwirkReports (userId, reportTitle, reportText, reportedUserId, reportTime) values(%s, %s, %s, %s, %s)', (user_id, report_title, report_text, reported_user_id, formatted_time))
                self.db.conn.commit()
                
                status = 'Success'
                message = 'User report submitted'

                # Block the user as well
                # self.block_user_for_report_user(user_id, reported_user_id)

                self.db.conn.close()

                # Send email notification
                email_subject = 'icebreaker - New User Report'
                email_body = f"""
Report Summary:

User {user_id} reported user {reported_user_id}.
Report Title: {report_title}
Report Text: {report_text}
Reported at: {formatted_time}"""
                to_email = 'devanshu@sparinfosys.com'  # recipient's email address 
                self.send_email(email_subject, email_body, to_email)

            return{'message': message, 'status': status, "loginWebToken": login_web_token}
        except Exception as e:
            failure = 'Failure'
            return{'message': str(e), 'status': failure}
        

    def block_user_for_report_user(self, user_id, blocked_id):
        try:
            cursor = self.db.cur()
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            blocked_time = formatted_time

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute(self.GET_BLOCKED_USER_QUERY , (user_id, blocked_id))
            row = cursor.fetchone()

            if not row:
                cursor.execute("INSERT INTO icebreaker.blockedUsers (userId, blockedId, blockedTime) VALUES (%s, %s, %s)", (user_id, blocked_id, blocked_time))
                

                cursor.execute("""
                    DELETE FROM icebreaker.netwirkRequestsAccepted
                    WHERE (userId = %s AND requestedUserId = %s)
                    OR (userId = %s AND requestedUserId = %s)
                """, (user_id, blocked_id, blocked_id, user_id))

                self.db.conn.commit()
                message = f"User {user_id} blocked user {blocked_id}"
         
            else:
                message = "This user is already blocked"

            cursor.close() 
            return {"status": "Success", "message": message, "loginWebToken": login_web_token}

        except Exception as e:
            message = str(e) 
            return {"status": "Failure", "message": message}

    
    def send_email(self, subject, body, to_email):
        # Create a new SES resource
 
          # === Load AWS credentials from JSON file ===
        aws_config = self._load_vibr_config()

        # Create SES client
        ses_client = boto3.client(
            'ses',
            aws_access_key_id=aws_config["aws_access_key_id"],
            aws_secret_access_key=aws_config["aws_secret_access_key"],
            region_name=aws_config["region"]
        )
        
        # Send the email
        ses_client.send_email(
            Source= 'devanshu@sparinfosys.com',   # Sender's email id
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body}}
            }
        )


    def disconnect_user(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            user_id = data['userId']
            disconnect_id = data['disconnectId']

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            # Check if both the users are connected
            cursor.execute('SELECT userId FROM icebreaker.netwirkRequestsAccepted WHERE userId = %s AND requestedUserId = %s', (user_id, disconnect_id))
            connect = cursor.fetchone()

            # Proceed with deleting the connection
            if connect:
                cursor.execute('DELETE FROM icebreaker.netwirkRequestsAccepted WHERE userId = %s AND requestedUserId = %s', (user_id, disconnect_id))
                cursor.execute('DELETE FROM icebreaker.netwirkRequestsAccepted WHERE userId = %s AND requestedUserId = %s', (disconnect_id, user_id))
                self.db.conn.commit()
                message = f"User {user_id} & {disconnect_id} are disconnected"
            
            else:
                message = f"User {user_id} & {disconnect_id} do not have an existing connection"

            cursor.close() 
            return {"status": "Success", "message": message, "loginWebToken": login_web_token}

        except Exception as e:
            message = str(e) 
            return {"status": "Failure", "message": message}


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
        
    
    def receive_notifications(self):
        try: 
            request_data = request.get_json()
            user_id = request_data['userId']
            receive_notifications = request_data['receiveNotifications']
            # Convert string to boolean by comparing lowercase value
            if receive_notifications.lower() == 'true':
                receive_notifications = True
            else:
                receive_notifications = False
            cursor=self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute("Update icebreaker.registration set receiveNotifications = %s where userId = %s", (receive_notifications,user_id))
            
            self.db.conn.commit()

            cursor.close()
            self.db.conn.close()
            message = 'Receive Notification preference updated successfully'
            return jsonify({'message': message, 'status' : 'Success', "loginWebToken": login_web_token})
        
        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            print(error)
            return error
        

   # Contains SSE
    def checkout_from_checkinlocation(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            user_id = data['userId']
            # place_id_incognito = data.get('placeId') or 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id_incognito = self._resolve_place_id(user_id, data, cursor)
            place_id = place_id_incognito
            is_incognito = data.get('incognito', False)
            event_id = data.get('eventId') or None
            checkout_time = datetime.now()

            
            user_info = self.fetch_user_info(cursor, user_id)

            if not user_info:
                raise ValueError(f"No user found with userId: {user_id}")

            login_token, full_name, preference, my_gender  = user_info
            full_name = full_name.strip().title()
            gender_filter = self.get_gender_filter(preference)
            age_pref_raw = self._fetch_age_preference_industry_v1(cursor, user_id)
            age_ranges   = self._translate_age_preference_industry_v1(age_pref_raw)
            my_dob       = self._fetch_user_dob_industry_v1(cursor, user_id)
            my_age       = self._calculate_age(my_dob) if my_dob else None

            # Non-incognito logic
            if not is_incognito:
                cursor.execute(self.SELECT_PLACE_ID_BY_USER, (user_id,))
                place_row = cursor.fetchone()
                cursor.execute(self.DELETE_USER_FROM_LOCATION, (user_id,))
                self.db.conn.commit()

                if place_row:
                    place_id = place_row[0]
                    cursor.execute('''
                        SELECT l.userId 
                        FROM icebreaker.location l
                        LEFT JOIN icebreaker.blockedUsers b1 
                            ON l.userId = b1.blockedId AND b1.userId = %s
                        LEFT JOIN icebreaker.blockedUsers b2 
                            ON l.userId = b2.userId AND b2.blockedId = %s 
                        WHERE l.placeId = %s AND l.userId != %s
                        AND b1.blockedId IS NULL
                        AND b2.userId IS NULL
                    ''', (user_id, user_id, place_id, user_id))
                    nearby_users = cursor.fetchall()

                    user_ids = [uid[0] for uid in nearby_users]
                    if user_ids:
                        self.notify_users(cursor, user_ids, full_name=full_name, gender_filter=gender_filter, place_id=place_row[0], age_ranges=age_ranges, my_age=my_age, my_gender=my_gender)
            

            # Incognito user checkout process
            if is_incognito and place_id_incognito:
                self.handle_incognito_checkout(cursor, user_id, place_id_incognito, checkout_time)

                # Notify visible users
                visible_user_ids = self.get_visible_user_ids(cursor, user_id)
                self.send_notifications_to_visible_users(cursor, visible_user_ids, full_name)

                # Notify others nearby excluding visible users
                all_nearby_ids = self.get_nearby_users(cursor, user_id, place_id)
                filtered_ids = [uid for uid in all_nearby_ids if uid not in visible_user_ids]
                if filtered_ids:
                    self.notify_users(cursor, filtered_ids, full_name="Someone", gender_filter=gender_filter, place_id=place_id, age_ranges=age_ranges, my_age=my_age, my_gender=my_gender)
                    
                

            # Clean up user session info
            self.cleanup_user_visibility(cursor, user_id)

            self.delete_selfie_pic(cursor, user_id)
            # To be used later
            # self.cleanup_pending_requests(cursor, user_id)
   
            self.cleanup_session_likes(cursor, user_id)
            self.cleanup_hidden_users(cursor, user_id)
            cursor.execute('''
                UPDATE icebreaker.sessiontracker
                SET checkouttime = %s
                WHERE userid = %s AND placeid = %s AND checkouttime IS NULL
            ''', (checkout_time, user_id, place_id))
            self.db.conn.commit()
            cursor.close()

            return {"Status": "Success", "message": "User checked out from Checkin location", "loginWebToken": login_token}

        except Exception as e:
            self.db.conn.rollback()
            return {"Status": "Failure", "message": {"error": str(e)}}

    def fetch_user_info(self, cursor, user_id):
        cursor.execute("""
            SELECT loginWebToken, fullName, Preference, Gender 
            FROM icebreaker.registration 
            WHERE userId = %s
        """, (user_id,))
        return cursor.fetchone()
    
    
    def process_event_checkout(self, cursor, user_id, event_id, is_incognito):
        if not event_id:
            return None

        cursor.execute("SELECT eventId FROM icebreaker.eventLocation WHERE userId = %s", (user_id,))
        event_row = cursor.fetchone()
        cursor.execute("DELETE FROM icebreaker.eventLocation WHERE userId = %s", (user_id,))
        self.db.conn.commit()

        if event_row and not is_incognito:
            cursor.execute("SELECT userId FROM icebreaker.eventLocation WHERE eventId = %s", (event_row[0],))
            _ = cursor.fetchall()  # Possibly unused
        return event_row
        
    def get_gender_filter(self, preference):
        return {
            "Women": "Female",
            "Men": "Male"
        }.get(preference, None)


    def handle_incognito_checkout(self, cursor, user_id,current_place_id, checkout_time):
        cursor.execute(self.SELECT_PLACE_ID_BY_USER, (user_id,))
        current_place_id = cursor.fetchone()[0]
        cursor.execute(self.DELETE_USER_FROM_LOCATION, (user_id,))
        self.db.conn.commit()

        cursor.execute('''
            SELECT userId FROM icebreaker.incognito_history
            WHERE userId = %s AND placeId = %s AND checkoutTime IS NULL
        ''', (user_id, current_place_id))

        if cursor.fetchone():
            cursor.execute('''
                UPDATE icebreaker.incognito_history
                SET checkoutTime = %s
                WHERE userId = %s AND placeId = %s AND checkoutTime IS NULL
            ''', (checkout_time, user_id, current_place_id))
        else:
            print("No active incognito session found for this user at this place.")
            
        self.db.conn.commit()

    def get_visible_user_ids(self, cursor, user_id):
        cursor.execute('''
            SELECT visible_to_userId 
            FROM icebreaker.incognito_visibility_exceptions 
            WHERE incognito_userId = %s
        ''', (user_id,))
        return [row[0] for row in cursor.fetchall()]

    def send_notifications_to_visible_users(self, cursor, user_ids, full_name):
        if not user_ids:
            return
        placeholders = ','.join(['%s'] * len(user_ids))
        cursor.execute(f'''
            SELECT deviceToken 
            FROM icebreaker.registration r
            WHERE userId IN ({placeholders}) AND deviceToken IS NOT NULL AND deviceToken != ''
            AND r.receiveNotifications = '1'
        ''', tuple(user_ids))
        tokens = cursor.fetchall()
        threads = []
        for token in tokens:
            t = threading.Thread(target=self.send_fcm_with_rest_logout_notification, args=(token[0], full_name))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def get_nearby_users(self, cursor, user_id, place_id):
        cursor.execute('''
            SELECT l.userId 
            FROM icebreaker.location l
            JOIN icebreaker.registration r
                ON l.userId = r.userId
            LEFT JOIN icebreaker.blockedUsers b1 
                ON l.userId = b1.blockedId AND b1.userId = %s
            LEFT JOIN icebreaker.blockedUsers b2 
                ON l.userId = b2.userId AND b2.blockedId = %s 
            JOIN icebreaker.registration r 
                ON r.userId = l.userId
            WHERE l.placeId = %s AND l.userId != %s
            AND b1.blockedId IS NULL
            AND b2.userId IS NULL
            AND r.receiveNotifications = '1'
        ''', (user_id, user_id, place_id, user_id))
        return [row[0] for row in cursor.fetchall()]

    def notify_users(self, cursor, user_ids, full_name, gender_filter, place_id, age_ranges=None, my_age=None, my_gender=None):
        if not user_ids:
            return

        age_clause, age_params           = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params   = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)

       
        placeholders = ','.join(['%s'] * len(user_ids))

        query = f"""
            SELECT r.deviceToken
            FROM icebreaker.registration r
            JOIN icebreaker.location l ON r.userId = l.userId
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId IN ({placeholders})
            AND l.placeId = %s
            AND r.deviceToken IS NOT NULL
            AND r.deviceToken != ''
            AND r.receiveNotifications = '1'
            AND (%s IS NULL OR r.Gender = %s)
            {"AND " + age_clause            if age_clause            else ""}
            {"AND " + reverse_clause        if reverse_clause        else ""}
            {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
        """
        params = [
            *user_ids,
            place_id,
            gender_filter, gender_filter,
            *age_params,
            *reverse_params,
            *reverse_gender_params,
        ]
       

        cursor.execute(query, params)
        tokens = cursor.fetchall()

        threads = []
        for token in tokens:
            t = threading.Thread(target=self.send_fcm_with_rest_logout_notification, args=(token[0], full_name))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def cleanup_user_visibility(self, cursor, user_id):
        cursor.execute('DELETE FROM icebreaker.incognito_visibility_exceptions WHERE incognito_userId = %s', (user_id,))
        self.db.conn.commit()

    def cleanup_pending_requests(self, cursor, user_id):
        cursor.execute('DELETE FROM icebreaker.netwirkRequestsPending WHERE userId = %s', (user_id,))
        self.db.conn.commit()

    def delete_selfie_pic(self, cursor, user_id):
        cursor.execute('UPDATE icebreaker.profile SET selfiePic = NULL WHERE userId = %s',(user_id,))
        self.db.conn.commit()
    
    def cleanup_session_likes(self, cursor, user_id):
        cursor.execute('DELETE FROM icebreaker.session_likes WHERE wavedToUserId = %s', (user_id,))
        self.db.conn.commit()

    def cleanup_hidden_users(self, cursor, user_id):
        cursor.execute('DELETE FROM icebreaker.hidden_users WHERE userId = %s', (user_id,))
        self.db.conn.commit()



    def create_update_event(self):
        try:
            data = request.get_json()
            event_id = data['eventId']
            user_id = data['userId']
            event_name = data['eventName']
            is_private = data['isPrivate']
            location = data['location']
            event_start_time = datetime.fromisoformat(data['eventStartTime'])
            event_end_time = datetime.fromisoformat(data['eventEndTime'])
            description = data['description']
            latitude = data['latitude']
            longitude = data['longitude']
            auto_checkout_radius = data['autoCheckoutRadius']
            created_at = datetime.now()

            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            if event_id:
                # Check if event exists
                cursor.execute("SELECT * FROM icebreaker.events WHERE eventId = %s", (event_id,))
                result = cursor.fetchone()
                if result:
                    # Event exists, update the event
                    cursor.execute("""
                        UPDATE icebreaker.events
                        SET userId = %s, eventName = %s, eventStartTime = %s, eventEndTime = %s, description = %s, updatedAt = %s, location = %s,
                        latitude = %s, longitude = %s, autoCheckoutRadius = %s
                        WHERE eventId = %s
                    """, (user_id, event_name, event_start_time, event_end_time, description, created_at, location, latitude, longitude, auto_checkout_radius, event_id))
                else:
                    # Event does not exist, return error
                    return jsonify({"Status": "Failure", "message": "Event ID does not exist", "loginWebToken": login_web_token}), 400
            else:
                # Create new event
                cursor.execute("""
                    INSERT INTO icebreaker.events (userId, eventName, eventStartTime, eventEndTime, description, createdAt, isPrivate, location, latitude,
                    longitude, autoCheckoutRadius)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (user_id, event_name, event_start_time, event_end_time, description, created_at, is_private, location, latitude, longitude, auto_checkout_radius))
                event_id = cursor.lastrowid

            # Commit changes to the database
            self.db.conn.commit()

            # Send eventId in QR code
            event_url = f"https://icebreaker.io/{event_id}"
            qr_data = (event_url)

            # Create a QR code instance
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )

            # Add data to the QR code
            qr.add_data(qr_data)
            qr.make(fit=True)

            # Create an image from the QR code instance
            img = qr.make_image(fill_color="black", back_color="white")

            # Save the image to a file
            filename = f"QRCode_{event_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            cwd = os.getcwd()
            img_path = os.path.join(cwd, 'qrcodes', filename)
            img.save(img_path)

            # Upload QR code to S3
            # === Load AWS credentials from JSON file ===
            aws_config = self._load_vibr_config()

            s3 = boto3.client(
                's3',
                region_name=aws_config["region"],
                aws_access_key_id=aws_config["aws_access_key_id"],
                aws_secret_access_key=aws_config["aws_secret_access_key"]
            )


            # Set the bucket name and region
            bucket_name = 'sparoptui'
            region = 'ap-south-1'
            folder_name = 'netwirkQrCodes/'

          
            s3.upload_file(img_path, bucket_name, folder_name + filename, ExtraArgs={'ContentType': 'image/png'})

            # Get the URL of the file
            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{folder_name}{filename}"

            # Update the event record in DB with the QR code URL
            cursor.execute("UPDATE icebreaker.events SET qrCode = %s, eventURL = %s WHERE eventId = %s", (url, event_url, event_id))
            self.db.conn.commit()
            cursor.close()

            return jsonify({"Status": "Success",
                            "message": "Event created/updated successfully",
                            "eventId": event_id,
                            "qrCode": url,
                            "eventURL": event_url,
                            "loginWebToken": login_web_token})

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return jsonify({"Status": "Failure", "message": error})
        
    def get_list_of_events_created(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            search_text = data['searchText']
            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page
            results = []
            cursor = self.db.cur()

            login_web_token = self._get_login_web_token(cursor, user_id)
            full_name = self._get_user_full_name(cursor, user_id)
            total_events = self._get_total_event_count(cursor, user_id)

            rows = self._get_event_rows(cursor, user_id, per_page, offset)

            for event in rows:
                user_dict = self._build_user_dict(event, full_name)
                results.append(user_dict)

            if search_text:
                search_text = search_text.lower()
                results = self._filter_by_search_text(results, search_text)
                total_events = len(results)

            return jsonify({
                'List': results,
                'Status': 'Success',
                'Page': page,
                'PerPage': per_page,
                'TotalEvents': total_events,
                "loginWebToken": login_web_token
            })

        except Exception as e:
            self.db.conn.rollback()
            return {'error': str(e)}

    # --- Helper Methods (no variable names changed) ---

    def _get_login_web_token(self, cursor, user_id):
        return self._retrieve_login_token(cursor, user_id)

    def _get_user_full_name(self, cursor, user_id):
        cursor.execute(self.GET_USER_FULLNAME_QUERY, (user_id,))
        res = cursor.fetchone()
        return res[0] if res else ''

    def _get_total_event_count(self, cursor, user_id):
        cursor.execute("""
            SELECT COUNT(*)
            FROM icebreaker.events
            WHERE userId = %s
            ORDER BY eventStartTime DESC
        """, (user_id,))
        return cursor.fetchone()[0]

    def _get_event_rows(self, cursor, user_id, per_page, offset):
        cursor.execute("""
            SELECT eventId, userId, eventName, eventStartTime, eventEndTime, description, 
                qrCode, location, createdAt, updatedAt, isPrivate, eventURL, 
                latitude, longitude, autoCheckoutRadius
            FROM icebreaker.events
            WHERE userId = %s
            ORDER BY eventStartTime DESC
            LIMIT %s OFFSET %s
        """, (user_id, per_page, offset))
        return cursor.fetchall()

    def _build_user_dict(self, event, full_name):
        return {
            "eventId": event[0] if event[0] is not None else '',
            "userId": event[1] if event[1] is not None else '',
            "eventName": event[2] if event[2] is not None else '',
            "eventStartTime": event[3] if event[3] is not None else '',
            "eventEndTime": event[4] if event[4] is not None else '',
            "description": event[5] if event[5] is not None else '',
            "qrCode": event[6] if event[6] is not None else '',
            "location": event[7] if event[7] is not None else '',
            "createdAt": event[8] if event[8] is not None else '',
            "updatedAt": event[9] if event[9] is not None else '',
            "isPrivate": event[10] if event[10] is not None else '',
            "eventURL": event[11] if event[11] is not None else '',
            "latitude": event[12] if event[12] is not None else '',
            "longitude": event[13] if event[13] is not None else '',
            "autoCheckoutRadius": event[14] if event[14] is not None else '',
            "fullName": full_name
        }

    def _filter_by_search_text(self, results, search_text):
        def matches(user):
            return (
                search_text in user["eventName"].lower() or
                search_text in user["description"].lower()
            )
        return [user for user in results if matches(user)]

        

    def get_event_details(self):
        try:
            data = request.get_json()
            event_id = data['eventId']
            user_id = data['userId']

            cursor = self.db.cur()

            # Step 1: Get login token
            login_web_token = self._get_login_token_event(cursor, user_id)

            # Step 2: Get event row
            row = self._get_event_row_data(cursor, event_id)

            if not row:
                return self._event_not_found_response(cursor, login_web_token)

            # Step 3: Get event-related details
            accepted_status = self._get_accepted_status_for_event(cursor, event_id, user_id)
            full_name = self._get_user_fullname(cursor, row[1])

            # Step 4: Build response
            user_dict = self._build_event_response(row, accepted_status, full_name, login_web_token)

            self._close_connection(cursor)
            return jsonify(user_dict)

        except Exception as e:
            self.db.conn.rollback()
            return jsonify({'error': str(e)})


    # ---------------- Helper methods ---------------- #

    def _get_login_token_event(self, cursor, user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
        result = cursor.fetchone()
        return result[0] if result else ''


    def _get_event_row_data(self, cursor, event_id):
        cursor.execute("""
            SELECT eventId, userId, eventName, eventStartTime, eventEndTime, description, qrCode,
                location, createdAt, updatedAt, isPrivate, eventURL, latitude, longitude, autoCheckoutRadius
            FROM icebreaker.events
            WHERE eventId = %s
        """, (event_id,))
        return cursor.fetchone()


    def _get_accepted_status_for_event(self, cursor, event_id, user_id):
        cursor.execute(
            "SELECT status FROM icebreaker.eventsAccepted WHERE eventId = %s AND acceptedUserId = %s",
            (event_id, user_id)
        )
        return 1 if cursor.fetchone() else 0


    def _get_user_fullname(self, cursor, user_id):
        cursor.execute(self.GET_USER_FULLNAME_QUERY, (user_id,))
        result = cursor.fetchone()
        return result[0] if result else ''


    def _build_event_response(self, row, accepted_status, full_name, login_web_token):
        keys = [
            "eventId", "userId", "eventName", "eventStartTime", "eventEndTime", "description",
            "qrCode", "location", "createdAt", "updatedAt", "isPrivate", "eventURL",
            "latitude", "longitude", "autoCheckoutRadius"
        ]

        response = {key: (row[i] if row[i] is not None else '') for i, key in enumerate(keys)}

        response.update({
            "accepted_status": accepted_status,
            "fullName": full_name,
            "loginWebToken": login_web_token
        })
        return response


    def _event_not_found_response(self, cursor, login_web_token):
        self._close_connection(cursor)
        return jsonify({"error": "Event not found", "loginWebToken": login_web_token})


    def _close_connection(self, cursor):
        cursor.close()
        self.db.conn.close()

        

    def event_userCheckin(self):
        try:
            request_data = request.get_json()
            user_id = request_data['userId']
            event_id = request_data['eventId']
            intro_line1 = request_data['intro_line1']
            intro_line2 = request_data['intro_line2']
            desired_industry = request_data['desiredIndustry']
            checkin_time = datetime.now()

            cursor = self.db.cur()

            login_web_token = self._fetch_login_token_event_userCheckin(cursor, user_id)
            event_dict = self._fetch_event_details(cursor, event_id)
            self._update_or_insert_event_location(cursor, user_id, event_id, checkin_time, intro_line1, intro_line2, desired_industry)

            self.db.conn.commit()
            cursor.close()
            self.db.conn.close()

            return jsonify({
                'message': 'User successfully checked in to the event.',
                'status': 'Success',
                'event details': event_dict,
                "loginWebToken": login_web_token
            })

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            print(error)
            return error


    # ----------------- Helper Methods ----------------- #

    def _fetch_login_token_event_userCheckin(self, cursor, user_id):
        return self._retrieve_login_token(cursor, user_id)


    def _fetch_event_details(self, cursor, event_id):
        cursor.execute("""
            SELECT eventId, userId, eventName, eventStartTime, eventEndTime, description, qrCode, 
                location, createdAt, updatedAt, isPrivate, eventURL
            FROM icebreaker.events
            WHERE eventId = %s
        """, (event_id,))
        row = cursor.fetchone()

        if not row:
            return {}

        cursor.execute(self.GET_USER_FULLNAME_QUERY, (row[1],))
        res = cursor.fetchone()
        full_name = res[0] if res else ''

        return {
            "eventId": row[0] or '',
            "userId": row[1] or '',
            "eventName": row[2] or '',
            "eventStartTime": row[3] or '',
            "eventEndTime": row[4] or '',
            "description": row[5] or '',
            "qrCode": row[6] or '',
            "location": row[7] or '',
            "createdAt": row[8] or '',
            "updatedAt": row[9] or '',
            "isPrivate": row[10] or '',
            "eventURL": row[11] or '',
            "fullName": full_name
        }


    def _update_or_insert_event_location(self, cursor, user_id, event_id, checkin_time, intro_line1, intro_line2, desired_industry):
        cursor.execute("SELECT userId FROM icebreaker.eventLocation WHERE userId = %s", (user_id,))
        result = cursor.fetchone()

        if result:
            cursor.execute(
                "UPDATE icebreaker.eventLocation SET eventId = %s, checkinTime = %s WHERE userId = %s",
                (event_id, checkin_time, user_id)
            )
        else:
            cursor.execute(
                "INSERT INTO icebreaker.eventLocation(eventId, userId, checkinTime) VALUES(%s, %s, %s)",
                (event_id, user_id, checkin_time)
            )

        cursor.execute(self.UPDATE_PROFILE_QUERY, (intro_line1, intro_line2, desired_industry, user_id))

        

    def accept_event(self):
        try:
            cursor=self.db.cur()
            request_data = request.get_json()
            login_user_id = request_data['loginUserId']
            event_id = request_data['eventId']
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            request_time = datetime.strptime(formatted_time, '%Y-%m-%d %H:%M:%S')
            status = 'Accepted'

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(login_user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]
            
            cursor.execute('SELECT * from icebreaker.eventsAccepted WHERE eventId = %s and acceptedUserId = %s', (event_id, login_user_id))
            row = cursor.fetchone()
            if row is None:
                cursor.execute('INSERT INTO icebreaker.eventsAccepted (eventId, acceptedUserId, requestTime, status) VALUES (%s, %s, %s, %s)', (event_id, login_user_id, request_time, status))
                self.db.conn.commit()
                message = 'User accepted event successfully.'
                result = {'message': message, 'status': 'Success', "loginWebToken": login_web_token}
            else:
                message = 'Event already accepted by user.'
                result = {'message': message, 'status': 'Failure', "loginWebToken": login_web_token}

            cursor.close()
            return jsonify(result)
   
        except Exception as e:
            self.db.conn.rollback()
            print(str(e))
            message = str(e)  # Convert exception to string
            return jsonify({"message": message, "status": 'Failure'})
        

    def get_list_of_events_accepted(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            search_text = data['searchText']
            page = int(data.get('page', 1))  # Default to page 1 if not provided
            per_page = int(data.get('perPage', 10))  # Default to 10 items per page if not provided
            offset = (page - 1) * per_page

            cursor = self.db.cur()

            login_web_token = self._fetch_login_token(cursor, user_id)
            total_accepted_events = self._count_total_accepted(cursor, user_id)
            event_ids = self._fetch_accepted_event_ids(cursor, user_id, per_page, offset)

            if not event_ids:
                return self._empty_response(page, per_page)

            events = self._fetch_accepted_event_details(cursor, event_ids)
            results = self._build_event_results(cursor, events)

            if search_text:
                results = self._filter_events(results, search_text)
                total_accepted_events = len(results)

            return jsonify({
                'List': results,
                'Status': 'Success',
                'Page': page,
                'PerPage': per_page,
                'TotalEvents': total_accepted_events,
                "loginWebToken": login_web_token
            })

        except Exception as e:
            self.db.conn.rollback()
            return {'error': str(e)}


    # ---------------- Helper methods ---------------- #

    def _count_total_accepted(self, cursor, user_id):
        cursor.execute("""
            SELECT COUNT(*)
            FROM icebreaker.eventsAccepted
            WHERE acceptedUserId = %s
        """, (user_id,))
        return cursor.fetchone()[0]


    def _fetch_accepted_event_ids(self, cursor, user_id, per_page, offset):
        cursor.execute("""
            SELECT eventId
            FROM icebreaker.eventsAccepted
            WHERE acceptedUserId = %s
            ORDER BY requestTime DESC
            LIMIT %s OFFSET %s
        """, (user_id, per_page, offset))
        return [event[0] for event in cursor.fetchall()]


    def _fetch_accepted_event_details(self, cursor, event_ids):
        placeholders = ','.join(['%s'] * len(event_ids))
        query = f"""
            SELECT eventId, userId, eventName, eventStartTime, eventEndTime, 
                description, qrCode, location, createdAt, updatedAt, isPrivate, eventURL
            FROM icebreaker.events
            WHERE eventId IN ({placeholders})
            ORDER BY eventStartTime DESC
        """
        cursor.execute(query, event_ids)
        return cursor.fetchall()


    def _build_event_results(self, cursor, events):
        results = []
        for event in events:
            cursor.execute(self.GET_USER_FULLNAME_QUERY, (event[1],))
            res = cursor.fetchone()
            full_name = res[0] if res else ''

            results.append({
                "eventId": event[0] or '',
                "userId": event[1] or '',
                "eventName": event[2] or '',
                "eventStartTime": event[3] or '',
                "eventEndTime": event[4] or '',
                "description": event[5] or '',
                "qrCode": event[6] or '',
                "location": event[7] or '',
                "createdAt": event[8] or '',
                "updatedAt": event[9] or '',
                "isPrivate": event[10] or '',
                "eventURL": event[11] or '',
                "fullName": full_name
            })
        return results


    def _filter_events(self, results, search_text):
        search_text = search_text.lower()
        return [
            event for event in results
            if search_text in event["eventName"].lower() or search_text in event["description"].lower()
        ]


    def _empty_response(self, page, per_page):
        return jsonify({
            'List': [],
            'Status': 'Success',
            'Page': page,
            'PerPage': per_page,
            'TotalEvents': 0
        })

        

    def cancel_decline_event(self):
        try:
            cursor=self.db.cur()
            request_data = request.get_json()
            login_user_id = request_data['loginUserId']
            event_id = request_data['eventId']
            status = request_data['status']

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(login_user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            if status == 'cancel' :
                cursor.execute('DELETE from icebreaker.events WHERE eventId = %s and userId = %s', (event_id, login_user_id))
                cursor.execute('DELETE from icebreaker.eventsAccepted WHERE eventId = %s', (event_id,))
                message = 'Event cancelled successfully.'
                result = {'message': message, 'status': 'Success', "loginWebToken": login_web_token}

            elif status == 'decline' :
                cursor.execute('DELETE from icebreaker.eventsAccepted WHERE eventId = %s and acceptedUserId = %s', (event_id, login_user_id))
                message = 'Event declined successfully.'
                result = {'message': message, 'status': 'Success', "loginWebToken": login_web_token}
            
            else:
                message = 'Event already cancelled/declined.'
                result = {'message': message, 'status': 'Failure', "loginWebToken": login_web_token}

            self.db.conn.commit()
            cursor.close()
            self.db.conn.close()
            return jsonify(result)
   
        except Exception as e:
            self.db.conn.rollback()
            print(str(e))
            message = str(e)  # Convert exception to string
            return jsonify({"message": message, "status": 'Failure'})
        
# This is based on working logic just before designation and industry dropdown was implemented
    def best_matches_1(self, page, per_page, offset, input_text):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)
            event_id = data.get('eventId')

            # Step 1: Fetch Data
            df, error = self.get_data(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}
            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MSG}

            # Step 2: Extract desired inputs
            extracted_data = self.extract_designation_and_industry(input_text)
            if 'error' in extracted_data:
                return {"Status": "Failure", "Message": "Failed to extract designation and industry"}
            desired_industry = extracted_data['industry']

            # Step 3: Prepare features
            df_prepared, logged_in_user_index = self._prepare_features(df, user_id, desired_industry)

            # Step 4: Compute matches
            combined_match_scores = self._compute_matches(df_prepared, logged_in_user_index)
            total_matches = len(combined_match_scores)

            # Step 5: Apply pagination
            paginated_match_scores = combined_match_scores[offset:min(offset + per_page, total_matches)]
            match_ids = self._extract_match_ids(df_prepared, paginated_match_scores, user_id)

            # Step 6: Fetch user details
            matches_details = self.fetch_user_details(match_ids, user_id, "")
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": total_matches
            }

        except Exception as e:
            print(str(e))
            return {"Status": "Failure", "Message": str(e)}

    # ---------------- Helper functions ---------------- #

    def _prepare_features(self, df, user_id, desired_industry):
        """Encodes designation and industry features."""
        from sklearn.preprocessing import LabelEncoder

        le_designation = LabelEncoder()
        df['designation_encoded'] = le_designation.fit_transform(df['currentDesignation'])
        df['industry_list'] = df['industry'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

        logged_in_user_industries = df[df['userId'] == user_id]['industry_list'].values[0]
        relevant_industries = set(logged_in_user_industries + [desired_industry])

        df['is_relevant'] = df['industry_list'].apply(
            lambda x: any(ind in relevant_industries for ind in x) if isinstance(x, list) else False
        )
        df_relevant = df[df['is_relevant']].copy()
        df_other = df[~df['is_relevant']].copy()

        # Encode industries for both groups
        df_relevant = self._encode_industries_best_matches(df_relevant)
        df_other = self._encode_industries_best_matches(df_other)

        # Merge back for convenience
        df_prepared = pd.concat([df_relevant, df_other], axis=0)
        logged_in_user_index = df_prepared.index.get_loc(df_prepared[df_prepared['userId'] == user_id].index[0])

        return df_prepared, logged_in_user_index


    def _encode_industries_best_matches(self, df_subset):
        """Encodes industries into one-hot columns."""
        all_industries = set(ind for sublist in df_subset['industry_list'].tolist() for ind in sublist)
        industry_encoding = {industry: idx for idx, industry in enumerate(all_industries)}
        for industry in all_industries:
            col_name = f'industry_{industry_encoding[industry]}'
            df_subset[col_name] = df_subset['industry_list'].apply(lambda x, ind=industry: 1 if ind in x else 0)
        df_subset['industry_encoded'] = df_subset['industry_list'].apply(
            lambda x: [industry_encoding[ind] for ind in x]
        )
        return df_subset


    def _compute_matches(self, df, logged_in_user_index):
        """Finds nearest neighbors within relevant/other groups."""
        from sklearn.neighbors import NearestNeighbors

        # Split back
        df_relevant = df[df['is_relevant']]
        df_other = df[~df['is_relevant']]

        matches = []

        if not df_relevant.empty:
            industry_cols = [c for c in df_relevant.columns if c.startswith("industry_")]
            x_relevant = df_relevant[industry_cols + ['designation_encoded', 'experience']].values
            knn = NearestNeighbors(n_neighbors=len(df_relevant), metric='euclidean')
            knn.fit(x_relevant)
            distances, indices = knn.kneighbors([x_relevant[logged_in_user_index]], n_neighbors=len(df_relevant))
            matches.extend([(i, distances[0][j]) for j, i in enumerate(indices[0]) if i != logged_in_user_index])

        if not df_other.empty:
            industry_cols = [c for c in df_other.columns if c.startswith("industry_")]
            x_other = df_other[industry_cols + ['designation_encoded', 'experience']].values
            knn = NearestNeighbors(n_neighbors=len(df_other), metric='euclidean')
            knn.fit(x_other)
            distances, indices = knn.kneighbors(x_other, n_neighbors=1)
            matches.extend([(i, distances[j][0]) for j, i in enumerate(indices[:, 0])])

        return sorted(matches, key=lambda x: x[1])


    def _extract_match_ids(self, df, match_scores, user_id):
        """Extracts user IDs from matches, excluding logged-in user."""
        match_ids = [str(df.iloc[i[0]]['userId']) for i in match_scores]
        return [mid for mid in match_ids if int(mid) != user_id]

        

    def get_data(self, user_id, place_id, event_id):
        try:
            cursor = self.db.cur()

            #Fetch accepted users list
            query_to_fetch_connected_users = """
                SELECT requestedUserId
                FROM icebreaker.netwirkRequestsAccepted
                where userId = %s
            """
            cursor.execute(query_to_fetch_connected_users, (user_id,))
            connected_user_ids = [row[0] for row in cursor.fetchall()]

            if event_id:
                query = """
                        SELECT p.userId, p.industry, p.currentDesignation , p.experience, p.desRank
                        FROM icebreaker.profile p
                        JOIN icebreaker.eventLocation e ON p.userId = e.userId
                        JOIN icebreaker.registration r ON p.userId = r.userId
                        LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                        LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                        WHERE e.eventId = %s
                        AND b1.blockedId IS NULL
                        AND b2.blockedId IS NULL                
                        """
                params = [user_id, user_id, event_id]
            else:
                query = """
                        SELECT p.userId, p.industry, p.currentDesignation , p.experience, p.desRank
                        FROM icebreaker.profile p
                        JOIN icebreaker.location l ON p.userId = l.userId
                        JOIN icebreaker.registration r ON p.userId = r.userId
                        LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                        LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                        WHERE l.placeId = %s
                        AND b1.blockedId IS NULL
                        AND b2.blockedId IS NULL                
                        """
                params = [user_id, user_id, place_id]
            
            #Modify the query if there are connected user IDs
            if connected_user_ids:
                placeholders = ', '.join(['%s'] * len(connected_user_ids))
                query += f" AND p.userId NOT IN ({placeholders})"
                params.extend(tuple(connected_user_ids))

            cursor.execute(query, params)
                
            rows = cursor.fetchall()
            cursor.close()

            # Convert fetched data to DataFrame
            df = pd.DataFrame(rows, columns=["userId", "industry", "currentDesignation", "experience", "desRank"])
            return df, None
        
        except Exception as e:
            print(str(e))
            return pd.DataFrame(), str(e)
        

    def fetch_user_details(self, user_ids, logged_in_user_id, search_text):
        try:
            cursor = self.db.cur()
            user_ids_tuple = tuple(user_ids)
            placeholders = ', '.join(['%s'] * len(user_ids_tuple))

            query = f"""
                SELECT p.userId, p.industry, p.companyName, p.currentDesignation, p.website, p.experience, p.education, p.profilePic,
                    r.fullName, r.email, r.countryCode, r.phoneNo, r.firebaseId, r.signupType, r.deviceToken, r.authenticationId, 
                    p.intro_line1, p.intro_line2
                FROM icebreaker.profile p
                JOIN icebreaker.registration r ON p.userId = r.userId
                WHERE p.userId IN ({placeholders})           
            """
            cursor.execute(query, user_ids_tuple)
            rows = cursor.fetchall()

            user_details_dict = {user[0]: user for user in rows}  # Map by userId
            output_list = []

            for user_id in map(int, user_ids):
                user_data = user_details_dict.get(user_id)
                if not user_data:
                    continue

                user_dict = self._build_user_dict_details(cursor, logged_in_user_id, user_data)
                output_list.append(user_dict)

            if search_text:
                output_list = self._filter_users(output_list, search_text)

            cursor.close()
            self.db.conn.close()
            return output_list

        except Exception as e:
            return {'message': str(e), 'Status': "Failure"}


    def _build_user_dict_details(self, cursor, logged_in_user_id, user):
        """Build user dictionary with block status and connection status"""
        block_flag = 1 if self._is_user_blocked(cursor, logged_in_user_id, user[0]) else 0

        user_dict = {
            "userId": user[0],
            "industry": user[1],
            "companyName": user[2],
            "currentDesignation": user[3],
            "website": user[4],
            "experience": user[5],
            "education": user[6],
            "profilePic": user[7],
            "fullName": user[8],
            "email": user[9],
            "countryCode": user[10],
            "phoneNo": user[11],
            "firebaseId": user[12],
            "signupType": user[13],
            "deviceToken": user[14],
            "authenticationId": user[15],
            "intro_line1": user[16],
            "intro_line2": user[17],
            "block_flag": block_flag,
            "placeId/eventId": "",
            "status": self._get_connection_status_check(cursor, logged_in_user_id, user[0])
        }
        return user_dict


    def _is_user_blocked(self, cursor, logged_in_user_id, other_user_id):
        cursor.execute(self.BLOCKED_USER_CHECK_QUERY, (logged_in_user_id, other_user_id))
        return cursor.fetchone() is not None


    def _get_connection_status_check(self, cursor, logged_in_user_id, other_user_id):
        """Determine connection status between two users"""
        cursor.execute(self.GET_ACCEPTED_REQUEST_STATUS_QUERY, (logged_in_user_id, other_user_id))
        accepted_status = cursor.fetchone()

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (logged_in_user_id, other_user_id))
        pending_status = cursor.fetchone()

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (other_user_id, logged_in_user_id))
        received_status = cursor.fetchone()

        if accepted_status and accepted_status[0] == "Accepted":
            return "Connected"
        if pending_status and pending_status[0] == "Pending":
            return "Request_Sent"
        if received_status and received_status[0] == "Pending":
            return "Request_Received"
        return "No_Interaction"


    def _filter_users(self, users, search_text):
        """Filter users by search text"""
        search_text = search_text.lower()
        return [
            user for user in users
            if search_text in user["fullName"].lower()
            or search_text in user["currentDesignation"].lower()
            or search_text in user["companyName"].lower()
        ]


    def send_event_invites(self):
        try:
            cursor = self.db.cur()
            request_data = request.get_json()
            login_user_id = request_data['loginUserId']
            event_id = request_data['eventId']
            invitees_user_ids = request_data['inviteesUserIds'].split(',')
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            request_time = datetime.strptime(formatted_time, '%Y-%m-%d %H:%M:%S')
            status = 'Pending'

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(login_user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            cursor.execute('SELECT fullName FROM icebreaker.registration WHERE userId = %s', (login_user_id,))
            res = cursor.fetchone()
            full_name = res[0] if res else ''
            
            for pending_user_id in invitees_user_ids:
                pending_user_id = pending_user_id.strip()
                cursor.execute('SELECT * FROM icebreaker.eventsPending WHERE eventId = %s AND pendingUserId = %s', (event_id, pending_user_id))
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        'INSERT INTO icebreaker.eventsPending (eventId, pendingUserId, requestTime, status) VALUES (%s, %s, %s, %s)',
                        (event_id, pending_user_id, request_time, status)
                    )
                    
                    cursor.execute('SELECT deviceToken FROM icebreaker.registration WHERE userId = %s', (pending_user_id,))
                    resp = cursor.fetchone()
                    token = resp[0] if resp else ''
                    self.send_fcm_with_rest_invite_recieved_notification_netwirk(token, full_name)
                else:
                    print(f'Record already exists for eventId: {event_id} and pendingUserId: {pending_user_id}')
            
            self.db.conn.commit()
            message = 'Invite sent successfully.'
            result = {'message': message, 'status': 'Success', "loginWebToken": login_web_token}
            cursor.close()
            return jsonify(result)

        except Exception as e:
            self.db.conn.rollback()
            print(str(e))
            message = str(e)
            return jsonify({"message": message, "status": 'Failure'})


    def get_list_of_events_invited(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            search_text = data['searchText']
            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page

            cursor = self.db.cur()

            login_web_token = self._fetch_login_token_users(cursor, user_id)
            total_invited_events = self._count_invited_events(cursor, user_id)
            event_ids = self._get_invited_event_ids(cursor, user_id, per_page, offset)

            if not event_ids:
                return self._build_response([], page, per_page, 0, login_web_token)

            results = self._get_event_details(cursor, event_ids)

            if search_text:
                results = self._filter_results(results, search_text)
                total_invited_events = len(results)

            return self._build_response(results, page, per_page, total_invited_events, login_web_token)

        except Exception as e:
            self.db.conn.rollback()
            return {'error': str(e)}

# ---------- Helper methods ----------

    def _count_invited_events(self, cursor, user_id):
        cursor.execute(
            "SELECT COUNT(*) FROM icebreaker.eventsPending WHERE pendingUserId = %s",
            (user_id,)
        )
        return cursor.fetchone()[0]

    def _get_invited_event_ids(self, cursor, user_id, per_page, offset):
        cursor.execute(
            """
            SELECT eventId 
            FROM icebreaker.eventsPending
            WHERE pendingUserId = %s
            ORDER BY requestTime DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, per_page, offset),
        )
        return [event[0] for event in cursor.fetchall()]

    def _get_event_details(self, cursor, event_ids):
        placeholders = ','.join(['%s'] * len(event_ids))
        query = f"""
            SELECT eventId, userId, eventName, eventStartTime, eventEndTime, 
                description, qrCode, location, createdAt, updatedAt, isPrivate, eventURL
            FROM icebreaker.events
            WHERE eventId IN ({placeholders})
            ORDER BY eventStartTime DESC
        """
        cursor.execute(query, event_ids)
        rows = cursor.fetchall()

        results = []
        for event in rows:
            cursor.execute(self.GET_USER_FULLNAME_QUERY, (event[1],))
            res = cursor.fetchone()
            full_name = res[0] if res else ''
            results.append(self._map_event_to_dict(event, full_name))
        return results

    def _map_event_to_dict(self, event, full_name):
        return {
            "eventId": event[0] or '',
            "userId": event[1] or '',
            "eventName": event[2] or '',
            "eventStartTime": event[3] or '',
            "eventEndTime": event[4] or '',
            "description": event[5] or '',
            "qrCode": event[6] or '',
            "location": event[7] or '',
            "createdAt": event[8] or '',
            "updatedAt": event[9] or '',
            "isPrivate": event[10] or '',
            "eventURL": event[11] or '',
            "fullName": full_name,
        }

    def _filter_results(self, results, search_text):
        search_text = search_text.lower()
        return [
            r for r in results
            if search_text in r["eventName"].lower() or search_text in r["description"].lower()
        ]

    def _build_response(self, results, page, per_page, total, login_web_token):
        return jsonify({
            'List': results,
            'Status': 'Success',
            'Page': page,
            'PerPage': per_page,
            'TotalEvents': total,
            'loginWebToken': login_web_token
        })



    def send_fcm_with_rest_invite_recieved_notification_netwirk(self, token, full_name):
        # Notification data
        title = "Invite Received"
        body = f'{full_name} has sent you an event invite'
        invite_recieved = "invite_recieved"

        # Path to your service account key
        SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE

        # Create a credentials object from the service account file
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])

        # Create a request object for fetching the access token
        auth_request = Request()

        # Refresh the token
        credentials.refresh(auth_request)
        
        headers = {
            'Authorization': f'Bearer {credentials.token}',
            'Content-Type': self.CONTENT_TYPE,
        }

        url = self.FCM_SEND_URL

        # Prepare target field
        target_key = "tokens" if isinstance(token, list) else "token"

        payload = {
            "message": {
                target_key: token,
                "notification": {
                    "title": title,
                    "body": body
                },
                "data": {
                    "type": invite_recieved
                },
                "android": {
                    "collapse_key": "type_a"
                },
                "apns": {
                    "payload": {
                        "aps": {
                            "alert": {
                                "title": title,
                                "body": body
                            },
                            "mutable-content": 1  # iOS Notification Service Extension
                        }
                    }
                }
            }
        }

        try:
            # Send the request to FCM
            resp = requests.post(url, headers=headers, data=json.dumps(payload))
            return {"status": "success", "message": resp.text}

        except Exception as e:
            return {"status": "error", "message": str(e)}
            


    def extract_designation_and_industry(self, input_text):
        try:
            api_key = self._get_openai_api_key()
            if not api_key:
                return {"error": "OpenAI API key not configured"}
            openai.api_key = api_key

            # Define the messages for the OpenAI API
            messages = [
                {"role": "system", "content": "You are an assistant that extracts designation and industry from a given text."},
                {"role": "user", "content": f"Extract the desiredDesignation and industry from the following text:\n\n'{input_text}'\n\nOutput the result in JSON format with keys 'desiredDesignation' and 'industry'. Add industry at the end of industry value"}
            ]

            response = openai.ChatCompletion.create(
            model= self.AI_MODEL,
            messages=messages,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5
        )

            # Extract the generated text from the response
            output_text = response.choices[0].message['content'].strip()

            # Parse the output_text as JSON
            output_json = json.loads(output_text)
            
            return output_json
        
        except Exception as e:
            return {'message': str(e), 'Status': "Failure"}
            
    def best_matches_top(self, page, per_page, offset, input_text):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)
            event_id = data.get('eventId')

            df, error = self.get_data(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}

            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MSG}

            logged_in_user = df[df['userId'] == user_id].iloc[0]
            prompt = self._build_prompt_top(df, logged_in_user, input_text)

            # Get AI response
            response = self._get_ai_response(prompt)
            result = self._parse_ai_response(response)

            # Remove logged-in user from results
            if user_id in result:
                result.remove(user_id)

            # Paginate
            paginated_ids = result[offset:offset + per_page]
            match_ids = [str(uid) for uid in paginated_ids]

            # Fetch details
            matches_details = self.fetch_user_details(match_ids, user_id, "")
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": len(result)
            }

        except Exception as e:
            return {"Status": "Failure", "Message": str(e)}

    # ------------------ Helper Methods ------------------

    def _build_prompt_top(self, df, logged_in_user, input_text):
        user_data_str = df.to_string(index=False)
        logged_in_user_details = (
            f"UserId: {logged_in_user['userId']}, "
            f"Industry: {logged_in_user['industry']}, "
            f"Current Designation: {logged_in_user['currentDesignation']}, "
            f"Experience: {logged_in_user['experience']}"
        )
        return f"""
        Based on the following input text: "{input_text}", and the logged-in user details: "{logged_in_user_details}", 
        find the most relevant user matches from the provided list. Use the input text if available else use the logged-in user details.
        The list of users is:
        {user_data_str}

        Return list of top 5 user IDs in descending order of their relevancy in list format. No verbose in output.
        """

    def _get_ai_response(self, prompt):
        response = openai.ChatCompletion.create(
            model= self.AI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert matchmaker assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        return response['choices'][0]['message']['content'].strip()

    def _parse_ai_response(self, output_text):
        # Case 1: string response
        if isinstance(output_text, str):
            try:
                result = ast.literal_eval(output_text)
                if not isinstance(result, list) or not all(isinstance(i, int) for i in result):
                    raise ValueError("Invalid AI result format")
            except (ValueError, SyntaxError):
                result = [int(item) for item in output_text.split(',') if item.strip().isdigit()]
        # Case 2: already a list
        elif isinstance(output_text, list):
            result = [int(item) for item in output_text]
        else:
            raise ValueError("Unexpected AI response format")

        return result

            

    def best_matches_all(self, page, per_page, offset):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)
            event_id = data.get('eventId')

            # Step 1: Fetch Data
            df, error = self.get_data(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}
            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MSG}

            logged_in_user = df[df['userId'] == user_id].iloc[0]

            # Step 2: Build Prompt
            prompt = self._build_prompt(df, logged_in_user)

            # Step 3: Get OpenAI Response
            output_text = self._get_openai_response(prompt)

            # Step 4: Parse Response
            result = self._parse_response(output_text)

            # Step 5: Remove self user & Paginate
            if user_id in result:
                result.remove(user_id)
            paginated_match_scores = result[offset:offset + per_page]

            # Step 6: Fetch match details
            match_ids = [str(user_id) for user_id in paginated_match_scores]
            matches_details = self.fetch_user_details(match_ids, user_id, "")
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": len(result)
            }

        except Exception as e:
            return {"Status": "Failure", "Message": str(e)}

    # ---------------- Helper Functions ---------------- #

    def _build_prompt(self, df, logged_in_user):
        user_data_str = df.to_string(index=False)
        logged_in_user_details = (
            f"UserId: {logged_in_user['userId']}, "
            f"Industry: {logged_in_user['industry']}, "
            f"Current Designation: {logged_in_user['currentDesignation']}, "
            f"Experience: {logged_in_user['experience']}"
        )

        return f"""
        Based on the following logged-in user details: "{logged_in_user_details}", 
        find the most relevant user matches from the provided list as per the industry of the logged-in user.
        The list of users is:
        {user_data_str}

        Return list of top 25 user IDs in descending order of their relevancy in list format. No verbose in output.
        """

    def _get_openai_response(self, prompt):
        response = openai.ChatCompletion.create(
            model= self.AI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert-level matchmaker assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        return response['choices'][0]['message']['content'].strip()

    def _parse_response(self, output_text):
        import ast
        if isinstance(output_text, str):
            try:
                result = ast.literal_eval(output_text)
                if not isinstance(result, list) or not all(isinstance(i, int) for i in result):
                    raise ValueError("Invalid format")
            except (ValueError, SyntaxError):
                result = [int(item) for item in output_text.split(',') if item.strip().isdigit()]
        elif isinstance(output_text, list):
            result = [int(item) for item in output_text]
        else:
            raise ValueError("Unexpected output_text format")
        return result



    def best_matches_all_sorted_by_checkintime(self, page, per_page, offset, search_text):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)
            event_id = data.get('eventId')
            cursor = self.db.cur()

            cursor.execute("SELECT industry FROM icebreaker.profile WHERE userId = %s", (user_id,))
            logged_in_user_industry = cursor.fetchone()[0]
            logged_in_user_industries = [ind.strip() for ind in logged_in_user_industry.split(',')]
            
            df, error = self.get_data_all_users(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}

            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message":self.USER_NOT_FOUND_MSG}
            
            # Remove the logged-in user from the DataFrame
            df = df[df['userId'] != user_id]
            
            # Filter users with matching industries
            def has_matching_industry(user_industry):
                user_industries = [ind.strip() for ind in user_industry.split(',')]
                return any(ind in user_industries for ind in logged_in_user_industries)
            
            # Users with matching industries sorted by checkinTime
            matching_industry_users = df[df['industry'].apply(has_matching_industry)]
            matching_industry_users = matching_industry_users.sort_values(by='checkinTime', ascending=False)

            # Users with non-matching industries sorted by checkinTime
            non_matching_industry_users = df[~df['industry'].apply(has_matching_industry)]
            non_matching_industry_users = non_matching_industry_users.sort_values(by='checkinTime', ascending=False)

            # Concatenate the two dataframes, matching industries first
            sorted_users = pd.concat([matching_industry_users, non_matching_industry_users])
            

            total_matches = len(sorted_users)

            paginated_match_scores = sorted_users[offset:min(offset + per_page, total_matches)]

            # Extracting user IDs from paginated results
            match_ids = paginated_match_scores['userId'].tolist()

            matches_details = self.fetch_user_details(match_ids, user_id, search_text)
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": total_matches
            }

        except Exception as e:
            print(str(e))
            return {"Status": "Failure", "Message": str(e)}
        

    def get_data_all_users(self, user_id, place_id, event_id):
        try:
            cursor = self.db.cur()

            if event_id:
                query = """
                        SELECT p.userId, p.industry, e.checkinTime
                        FROM icebreaker.profile p
                        JOIN icebreaker.eventLocation e ON p.userId = e.userId
                        JOIN icebreaker.registration r ON p.userId = r.userId
                        LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                        LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                        WHERE e.eventId = %s
                        AND b1.blockedId IS NULL
                        AND b2.blockedId IS NULL                
                        """
                cursor.execute(query, (user_id, user_id, event_id))
            else:
                query = """
                        SELECT p.userId, p.industry, l.checkinTime
                        FROM icebreaker.profile p
                        JOIN icebreaker.location l ON p.userId = l.userId
                        JOIN icebreaker.registration r ON p.userId = r.userId
                        LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                        LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                        WHERE l.placeId = %s
                        AND b1.blockedId IS NULL
                        AND b2.blockedId IS NULL               
                        """
                cursor.execute(query, (user_id, user_id, place_id))
                
            rows = cursor.fetchall()
            cursor.close()

            # Convert fetched data to DataFrame
            df = pd.DataFrame(rows, columns=["userId", "industry", "checkinTime"])
            return df, None
        
        except Exception as e:
            return pd.DataFrame(), str(e)
        
    
    def delete_account(self):
        # === Load credentials from JSON file ===
        aws_config = self._load_vibr_config()


        # Define Cognito client
        self.client = boto3.client(
            'cognito-idp',
            region_name=aws_config["region"],
            aws_access_key_id=aws_config["aws_access_key_id"],
            aws_secret_access_key=aws_config["aws_secret_access_key"]
        )

        try:
            data = request.get_json()
            email = (data.get('email') or '').strip().lower()
            reason = data.get('reason', None)
            reason_text = data.get('reason_text', None)
            cursor = self.db.cur()
            now = datetime.now()

            # Get userId from email
            cursor.execute("SELECT userId, email, DOB, city, state, Gender FROM icebreaker.registration WHERE email = %s", (email,))
            row = cursor.fetchone()
            if not row:
                return jsonify({'message': self.USER_NOT_FOUND_MESSAGE, 'status': 'Failure'})
            user_id = row[0]
            email = row[1]
            dob = row[2]
            city = row[3]
            state = row[4]
            Gender = row[5]
            

            # Fetch loginWebToken
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            # Delete Cognito user when present. Social logins exist only in the DB,
            # so a missing Cognito user must not block account deletion.
            response = self.client.list_users(
                UserPoolId=aws_config["user_pool_id"],
                Filter=f'email="{email}"'
            )
            if response.get('Users'):
                cognito_username = response['Users'][0]['Username']
                self.client.admin_delete_user(
                    UserPoolId=aws_config["user_pool_id"],
                    Username=cognito_username
                )

            cursor.execute("""INSERT INTO icebreaker.account_deletion_reasons(userId, email, dob, city, state, Gender, reason, other_reason, created_at)
                            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (user_id, email, dob, city, state, Gender, reason, reason_text, now))

            # Delete local records after optional Cognito cleanup
            cursor.execute('DELETE from icebreaker.location WHERE userId = %s', (user_id,))
            cursor.execute('DELETE from icebreaker.radius WHERE userId = %s', (user_id,))
            cursor.execute('DELETE from icebreaker.netwirkRequestsPending WHERE userId = %s OR requestedUserId = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.netwirkRequestsAccepted WHERE userId = %s OR requestedUserId = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.blockedUsers WHERE userId = %s OR blockedId = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.blockedUsers WHERE userId = %s OR blockedId = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.hidden_users WHERE userId = %s OR hiddenUserId = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.user_likes WHERE sender_id = %s OR receiver_id = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.user_likes_near WHERE sender_id = %s OR receiver_id = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.session_likes WHERE user_id = %s OR wavedTouserId = %s', (user_id, user_id))
            cursor.execute('DELETE from icebreaker.session_likes_near WHERE user_id = %s OR wavedTouserId = %s', (user_id, user_id))


            cursor.execute("SELECT eventId FROM icebreaker.events WHERE userId = %s", (user_id,))
            rows = cursor.fetchall()
            for event in rows:
                event_id = event[0]
                cursor.execute('DELETE from icebreaker.eventsPending WHERE eventId = %s', (event_id,))
                cursor.execute('DELETE from icebreaker.eventsAccepted WHERE eventId = %s', (event_id,))
                cursor.execute('DELETE from icebreaker.eventLocation WHERE eventId = %s', (event_id,))
            cursor.execute('DELETE FROM icebreaker.events WHERE userId = %s', (user_id,))

            # Delete profile + registration
            cursor.execute('DELETE from icebreaker.profile WHERE userId = %s', (user_id,))
            cursor.execute('DELETE from icebreaker.registration WHERE userId = %s', (user_id,))

            self.db.conn.commit()
            cursor.close()

            return jsonify({'message': "Account deleted successfully", 'status': 'Success', "loginWebToken": login_web_token})

        except Exception as e:
            self.db.conn.rollback()  # Rollback if failure
            return jsonify({'message': f"Error: {str(e)}", 'status': 'Failure'})


    # def delete_account(self):
    #     try:
    #         data = request.get_json()
    #         email       = data['email']
    #         reason      = data.get('reason', None)
    #         reason_text = data.get('reason_text', None)

    #         cursor = self.db.cur()
    #         now = datetime.now()

    #         # ── Fetch user details ──
    #         cursor.execute("""
    #             SELECT userId, email, DOB, city, state, Gender 
    #             FROM icebreaker.registration WHERE email = %s
    #         """, (email,))
    #         row = cursor.fetchone()

    #         if not row:
    #             return jsonify({'message': 'User not found.', 'status': 'Failure'})

    #         user_id = row[0]
    #         email   = row[1]
    #         dob     = row[2]
    #         city    = row[3]
    #         state   = row[4]
    #         gender  = row[5]  # was row[4] in original — bug fixed

    #         # ── Fetch loginWebToken ──
    #         cursor.execute("""
    #             SELECT loginWebToken FROM icebreaker.registration WHERE userId = %s
    #         """, (user_id,))
    #         token_row = cursor.fetchone()
    #         login_web_token = token_row[0] if token_row else None

    #         # ── Log deletion reason before deleting ──
    #         cursor.execute("""
    #             INSERT INTO icebreaker.account_deletion_reasons
    #                 (userId, email, dob, city, state, Gender, reason, other_reason, created_at)
    #             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    #         """, (user_id, email, dob, city, state, gender, reason, reason_text, now))

    #         # ── Delete related records ──
    #         cursor.execute("DELETE FROM icebreaker.location WHERE userId = %s", (user_id,))
    #         cursor.execute("DELETE FROM icebreaker.radius WHERE userId = %s", (user_id,))
    #         cursor.execute("DELETE FROM icebreaker.netwirkRequestsPending WHERE userId = %s OR requestedUserId = %s", (user_id, user_id))
    #         cursor.execute("DELETE FROM icebreaker.netwirkRequestsAccepted WHERE userId = %s OR requestedUserId = %s", (user_id, user_id))
    #         cursor.execute("DELETE FROM icebreaker.blockedUsers WHERE userId = %s OR blockedId = %s", (user_id, user_id))

    #         # ── Clean up events ──
    #         cursor.execute("SELECT eventId FROM icebreaker.events WHERE userId = %s", (user_id,))
    #         events = cursor.fetchall()
    #         for event in events:
    #             event_id = event[0]
    #             cursor.execute("DELETE FROM icebreaker.eventsPending WHERE eventId = %s", (event_id,))
    #             cursor.execute("DELETE FROM icebreaker.eventsAccepted WHERE eventId = %s", (event_id,))
    #             cursor.execute("DELETE FROM icebreaker.eventLocation WHERE eventId = %s", (event_id,))
    #         cursor.execute("DELETE FROM icebreaker.events WHERE userId = %s", (user_id,))

    #         # ── Delete OTP store if exists ──
    #         cursor.execute("DELETE FROM icebreaker.otp_store WHERE email = %s", (email,))

    #         # ── Delete profile and registration last ──
    #         cursor.execute("DELETE FROM icebreaker.profile WHERE userId = %s", (user_id,))
    #         cursor.execute("DELETE FROM icebreaker.registration WHERE userId = %s", (user_id,))

    #         self.db.conn.commit()
    #         cursor.close()

    #         return jsonify({
    #             'message'       : 'Account deleted successfully.',
    #             'status'        : 'Success',
    #             'loginWebToken' : login_web_token
    #         })

    #     except KeyError as e:
    #         self.db.conn.rollback()
    #         return jsonify({'message': f'Missing required field: {str(e)}', 'status': 'Failure'})

    #     except Exception as e:
    #         self.db.conn.rollback()
    #         logging.error(f"delete_account error: {str(e)}")
    #         return jsonify({'message': 'Failed to delete account. Please try again.', 'status': 'Failure'})

    def get_vibe_list(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibe_I_bring')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1],
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        
    def get_vibing_list(self):
        try:
            cursor = self.db.cur()
            vibing_list = []

            cursor.execute('SELECT id,icon,title FROM icebreaker.vibing')
            vibings = cursor.fetchall()

            for vibing in vibings:
                vibing_dict= {
                    'id': vibing[0],
                    # 'icon': vibing[1],
                    'answer': vibing[2]
                }
                
                vibing_list.append(vibing_dict)

            return jsonify({'Status': 'Success', 'vibing List': vibing_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})


    def get_designations_list(self):
        try:
            data = request.get_json()
            industry = data['industry']  
            cursor = self.db.cur()
            designation_list = []

            cursor.execute('SELECT designation FROM icebreaker.designationRank WHERE industry = %s', (industry,))
            designations = cursor.fetchall()

            for designation in designations:
                designation_dict= {
                    'designation': designation[0]
                }
                
                designation_list.append(designation_dict)

            return jsonify({'Status': 'Success', 'Industry List': designation_list})

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})


    # Below code is based on designation mapping logic
    def best_matches_2(self, per_page, offset, desired_industry, intro_line2):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)
            event_id = data.get('eventId')

            # Step 1: Load Data
            df, error = self.get_data(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}
            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MSG}

            # Step 2: Prepare industries
            df['desRank'] = df['desRank'].astype(int, errors='ignore')
            logged_in_industry = df[df['userId'] == user_id]['industry'].values[0]

            if desired_industry:
                df_relevant = df[df['industry'] == desired_industry].copy()
            else:
                df_relevant = df[df['industry'] == logged_in_industry].copy()

            df_other = df.drop(df_relevant.index).copy()

            if user_id not in df_relevant['userId'].values:
                df_relevant = pd.concat([df_relevant, df[df['userId'] == user_id]], ignore_index=True)

            # Step 3: Build match scores
            combined_match_scores = self._get_combined_scores(
                df, df_relevant, df_other, user_id, desired_industry, intro_line2, logged_in_industry
            )

            # Step 4: Paginate
            paginated_scores = combined_match_scores[offset:offset + per_page]
            match_ids = self._extract_user_ids(df_relevant, df_other, paginated_scores, user_id)

            return match_ids

        except Exception as e:
            print(f"Exception occurred: {e}")
            return {"Status": "Failure", "Message": str(e)}


    # ---------------------------
    #  Helper Methods
    # ---------------------------

    def _apply_knn(self, df_subset, base_index=0):
        """Run k-NN on a dataframe subset."""
        if df_subset.empty:
            return []
        X = df_subset[['desRank', 'experience']].values
        knn = NearestNeighbors(n_neighbors=len(df_subset), metric='euclidean')
        knn.fit(X)
        distances, indices = knn.kneighbors([X[base_index]], n_neighbors=len(df_subset))
        return sorted([(i, distances[0][j]) for j, i in enumerate(indices[0])], key=lambda x: x[1])


    def _get_combined_scores(self, df, df_relevant, df_other, user_id, desired_industry, desired_designation, logged_in_industry):
        """Combine match scores depending on filters."""
        combined = []
        logged_in_index = df_relevant.index.get_loc(df_relevant[df_relevant['userId'] == user_id].index[0])

        # Relevant matches
        match_relevant = self._apply_knn(df_relevant, base_index=logged_in_index)
        match_other = self._apply_knn(df_other) if not df_other.empty else []

        if desired_industry and desired_designation:
            df_desired = df_relevant[df_relevant['designation'] == desired_designation]
            combined += self._apply_knn(df_desired) if not df_desired.empty else []
            combined += match_relevant + match_other
        elif desired_industry:
            combined += match_relevant + match_other
        else:
            df_logged = df[df['industry'] == logged_in_industry].copy()
            combined += self._apply_knn(df_logged) + match_other

        return combined


    def _extract_user_ids(self, df_relevant, df_other, paginated_scores, user_id):
        """Extract userIds from paginated scores."""
        if not paginated_scores:
            return []
        user_index = df_relevant[df_relevant['userId'] == user_id].index[0] if user_id in df_relevant['userId'].values else -1
        if not df_relevant.empty:
            return [str(df_relevant.iloc[i[0]]['userId']) for i in paginated_scores if i[0] != user_index]
        return [str(df_other.iloc[i[0]]['userId']) for i in paginated_scores]

        

    def get_data_2(self, place_id, event_id):
        try:
            cursor = self.db.cur()

            if event_id:
                query = """
                        SELECT p.userId, p.industry, p.desRank , p.experience
                        FROM icebreaker.profile p
                        JOIN icebreaker.eventLocation e ON p.userId = e.userId
                        JOIN icebreaker.registration r ON p.userId = r.userId
                        WHERE e.eventId = %s                
                        """
                cursor.execute(query, (event_id,))
            else:
                query = """
                        SELECT p.userId, p.industry, p.desRank , p.experience
                        FROM icebreaker.profile p
                        JOIN icebreaker.location l ON p.userId = l.userId
                        JOIN icebreaker.registration r ON p.userId = r.userId
                        WHERE l.placeId = %s               
                        """
                cursor.execute(query, (place_id,))
                
            rows = cursor.fetchall()
            cursor.close()

            # Convert fetched data to DataFrame
            df = pd.DataFrame(rows, columns=["userId", "industry", "desRank", "experience"])
            return df, None
        
        except Exception as e:
            return pd.DataFrame(), str(e)


# Below code if to fetech the recommended users based on both industry and designation dropdown selection
    def best_matches(self, page, per_page, offset, desired_industry, intro_line2, search_text):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)
            event_id = data.get('eventId')

            # Fetch and prepare data
            df, error = self.get_data(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}

            df = self._prepare_dataframe(df)
            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MSG}

            # Split into relevant/other and encode industries
            df_relevant, df_other, industry_cols_relevant, industry_cols_other = \
                self._split_and_encode_industries(df, user_id, desired_industry)

            # Handle case logic (3 big branches)
            combined_match_scores, match_ids = self._handle_cases(
                df_relevant, df_other, user_id,
                desired_industry, intro_line2,
                industry_cols_relevant, industry_cols_other,
                offset, per_page
            )

            # Fetch user details
            matches_details = self.fetch_user_details(match_ids, place_id, search_text)
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": len(combined_match_scores)
            }

        except Exception as e:
            print(f"Error during match calculation: {e}")
            return {"Status": "Failure", "Message": str(e)}


    # ----------------- Helper Methods -----------------

    def _prepare_dataframe(self, df):
        """Clean and prepare DataFrame for processing."""
        df = df.dropna(subset=['desRank'])
        df['industry'] = df['industry'].astype(str)
        df['industry_list'] = df['industry'].apply(lambda x: {i.strip() for i in x.split(',')})

        le_designation = LabelEncoder()
        df['designation_encoded'] = le_designation.fit_transform(df['currentDesignation'])
        self.le_designation = le_designation  # store encoder for later use
        return df


    def _split_and_encode_industries(self, df, user_id, desired_industry):
        """Split into relevant/other and encode industries."""
        logged_in_user_industries = df[df['userId'] == user_id]['industry'].values[0]
        relevant_industries = set(logged_in_user_industries)
        if desired_industry:
            relevant_industries.update({i.strip() for i in desired_industry.split(',')})

        df_relevant = df[df['industry_list'].apply(lambda x: len(x.intersection(relevant_industries)) > 0)].copy()
        df_other = df[~df.index.isin(df_relevant.index)].copy()

        # Ensure logged-in user is included
        for target_df in [df_relevant, df_other]:
            if user_id not in target_df['userId'].values:
                target_df = pd.concat([df[df['userId'] == user_id], target_df], ignore_index=True)

        industry_cols_relevant = self._encode_industries(df_relevant)
        industry_cols_other = self._encode_industries(df_other)

        return df_relevant, df_other, industry_cols_relevant, industry_cols_other


    def _encode_industries(self, df_subset):
        """One-hot encode industries for a DataFrame subset."""
        if df_subset.empty:
            return []
        all_industries = set.union(*df_subset['industry_list'])
        encoding = {industry: idx for idx, industry in enumerate(all_industries)}
        for industry in all_industries:
            df_subset[f'industry_{encoding[industry]}'] = df_subset['industry_list'].apply(
                lambda x, industry=industry: 1 if industry in x else 0
            )
        return [f'industry_{encoding[industry]}' for industry in all_industries]


    def _compute_knn_scores(self, df_subset, x_subset, user_id):
        """Compute k-NN scores for a subset of the DataFrame."""
        if df_subset.empty or user_id not in df_subset['userId'].values:
            return []
        logged_in_user_index = df_subset.index.get_loc(df_subset[df_subset['userId'] == user_id].index[0])
        knn = NearestNeighbors(n_neighbors=len(df_subset), metric='euclidean')
        knn.fit(x_subset)
        distances, indices = knn.kneighbors([x_subset[logged_in_user_index]], n_neighbors=len(df_subset))
        return sorted(
            [(i, distances[0][j]) for j, i in enumerate(indices[0]) if i != logged_in_user_index],
            key=lambda x: x[1]
        )


    def _handle_cases(self, df_relevant, df_other, user_id,
                    desired_industry, desired_designation,
                    industry_cols_relevant, industry_cols_other,
                    offset, per_page):
        """Branch into case-specific matching logic."""
        if desired_industry and desired_designation:
            return self._case_industry_and_designation(
                df_relevant, user_id, desired_industry, desired_designation,
                industry_cols_relevant, offset, per_page
            )
        elif desired_industry:
            return self._case_only_industry(
                df_relevant, user_id, desired_industry,
                industry_cols_relevant, offset, per_page
            )
        else:
            return self._case_default(
                df_relevant, df_other, user_id,
                industry_cols_relevant, industry_cols_other,
                offset, per_page
            )


    # -------- Case Implementations --------

    def _case_industry_and_designation(self, df_relevant, user_id, desired_industry,
                                    desired_designation, industry_cols_relevant,
                                    offset, per_page):
        """Case 1: Both industry and designation provided."""
        le_designation = self.le_designation
        df_desired = df_relevant[
            (df_relevant['designation_encoded'] == le_designation.transform([desired_designation])[0]) &
            (df_relevant['industry'] == desired_industry)
        ]
        if user_id not in df_desired['userId'].values:
            df_desired = pd.concat([df_desired, df_relevant[df_relevant['userId'] == user_id]])

        match_scores = []
        if not df_desired.empty:
            x_desired = df_desired[industry_cols_relevant + ['experience']].values
            match_scores = self._compute_knn_scores(df_desired, x_desired, user_id)

        sorted_scores = sorted(match_scores, key=lambda x: x[1])
        paginated = sorted_scores[offset:offset + per_page]
        match_ids = [str(df_desired.iloc[idx]['userId']) for idx, _ in paginated]

        return match_scores, match_ids


    def _case_only_industry(self, df_relevant, user_id, desired_industry,
                            industry_cols_relevant, offset, per_page):
        """Case 2: Only industry provided."""
        df_logged_in_user = df_relevant[df_relevant['userId'] == user_id]
        logged_in_user_des_rank = df_logged_in_user['desRank'].values[0]
        des_rank_range = range(max(1, int(logged_in_user_des_rank) - 2), int(logged_in_user_des_rank) + 2)

        df_desired = df_relevant[
            (df_relevant['industry'] == desired_industry) &
            (df_relevant['desRank'].isin(des_rank_range))
        ]
        if user_id not in df_desired['userId'].values:
            df_desired = pd.concat([df_desired, df_logged_in_user])

        match_scores = []
        if not df_desired.empty:
            x_desired = df_desired[industry_cols_relevant + ['experience']].values
            match_scores = self._compute_knn_scores(df_desired, x_desired, user_id)

        sorted_scores = sorted(match_scores, key=lambda x: x[1])
        paginated = sorted_scores[offset:offset + per_page]
        match_ids = [str(df_desired.iloc[idx]['userId']) for idx, _ in paginated]

        return match_scores, match_ids


    def _case_default(self, df_relevant, df_other, user_id,
                    industry_cols_relevant, industry_cols_other,
                    offset, per_page):
        """Case 3: Neither industry nor designation provided."""
        df_logged_in_user = df_relevant[df_relevant['userId'] == user_id]
        logged_in_user_des_rank = df_logged_in_user['desRank'].values[0]
        des_rank_range = range(max(1, int(logged_in_user_des_rank) - 2), int(logged_in_user_des_rank) + 2)

        df_relevant = df_relevant[df_relevant['desRank'].isin(des_rank_range)]
        match_scores_relevant = self._compute_knn_scores(
            df_relevant, df_relevant[industry_cols_relevant + ['experience']].values, user_id
        )

        match_scores_other = []
        if len(df_other) > 0:
            match_scores_other = self._compute_knn_scores(
                df_other, df_other[industry_cols_other + ['experience']].values, user_id
            )

        combined_scores = match_scores_relevant + match_scores_other
        sorted_scores = sorted(combined_scores, key=lambda x: x[1])

        paginated = sorted_scores[offset:offset + per_page]
        match_ids = []
        for idx, _ in paginated:
            if idx < len(df_relevant):
                match_ids.append(str(df_relevant.iloc[idx]['userId']))
            else:
                adjusted_idx = idx - len(df_relevant)
                match_ids.append(str(df_other.iloc[adjusted_idx]['userId']))

        match_ids = [mid for mid in match_ids if int(mid) != user_id]
        return combined_scores, match_ids



    def get_update_card(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            update = data['update']
            cursor = self.db.cur()

            login_web_token = self._get_login_token_user_list(cursor, user_id)

            if update:
                self._update_user_details(cursor, data, user_id)
                self._update_designation_rank(cursor, user_id)
                self.db.conn.commit()

            row = self._fetch_user_details(cursor, user_id)
            cursor.close()
            self.db.conn.close()

            if row:
                return jsonify(self._build_user_dict(row, login_web_token))
            else:
                return jsonify({"error": self.USER_NOT_FOUND_MESSAGE, "loginWebToken": login_web_token})

        except Exception as e:
            self.db.conn.rollback()
            return jsonify({"error": str(e)})

    # ---------------- Helper Methods ---------------- #


    def _update_user_details(self, cursor, data, user_id):
        cursor.execute("""
            UPDATE icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            SET r.fullName = %s, r.cardEmail = %s, r.countryCode = %s, r.phoneNo = %s,
                p.industry = %s, p.currentDesignation = %s, p.companyName = %s
            WHERE r.userId = %s""",
            (data.get('fullName'), data.get('cardEmail'), data.get('countryCode'),
            data.get('phoneNo'), data.get('industry'),
            data.get('currentDesignation'), data.get('companyName'), user_id))

    def _update_designation_rank(self, cursor, user_id):
        cursor.execute("""
            UPDATE icebreaker.profile p
            JOIN icebreaker.designationRank r ON p.currentDesignation = r.designation
            SET p.desRank = r.rank
            WHERE p.userId = %s""", (user_id,))

    def _fetch_user_details(self, cursor, user_id):
        cursor.execute("""
            SELECT r.fullName, r.cardEmail, r.countryCode, r.phoneNo,
                p.industry, p.currentDesignation, p.companyName
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s""", (user_id,))
        return cursor.fetchone()

    def _build_user_dict(self, row, login_web_token):
        return {
            "fullName": row[0] or '',
            "cardEmail": row[1] or '',
            "countryCode": row[2] or '',
            "phoneNo": row[3] or '',
            "industry": row[4] or '',
            "currentDesignation": row[5] or '',
            "companyName": row[6] or '',
            "loginWebToken": login_web_token
        }



    def get_list_of_all_users(self):
        try:
            request_data = request.get_json()
            admin_id = request_data['adminId']
            output_list = []
            cursor=self.db.cur()

            cursor.execute("""
            SELECT p.userId, p.companyName, p.currentDesignation ,p.website, p.experience, p.industry, p.education, 
                    p.profilePic, r.fullName, r.email, r.countryCode, r.phoneNo, r.firebaseId, r.signupType, r.deviceToken, r.authenticationId
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            WHERE r.adminId = %s
            """, (admin_id,))

            row = cursor.fetchall()          

            for user in row:
                user_dict = {
                            "userId": user[0],
                            "companyName": user[1],
                            "currentDesignation" : user[2],
                            "website" : user[3],
                            "experience" : user[4],
                            "industry" : user[5],
                            "education" : user[6], 
                            "profilePic" : user[7],
                            "fullName" : user[8],
                            "email" : user[9], 
                            "countryCode" : user[10],
                            "phoneNo" : user[11],
                            "firebaseId" : user[12],
                            "signupType" : user[13],
                            "deviceToken" : user[14],
                            "authenticationId" : user[15]
                            }
                output_list.append(user_dict)
        
            total_users = len(output_list)

            if not output_list:
                response = {"List": []}
                return jsonify(response)
            else:
                cursor.close()
                self.db.conn.close()
                response = {"Total Users": total_users, "List": output_list}
                return jsonify(response)

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            print(error)
            return error
        

    def get_education_list(self):
        try:
            cursor = self.db.cur()
            education_list = []

            cursor.execute('SELECT DISTINCT eduId, name FROM icebreaker.education')
            educations = cursor.fetchall()

            for edu in educations:
                edu_dict= {
                    'eduId': edu[0],
                    'name': edu[1]
                }
                
                education_list.append(edu_dict)

            return jsonify({'Status': 'Success', 'Education List': education_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        

    def get_title_list(self):
        try:
            title_list = [
                "Chief Executive Officer",
                "Chief Financial Officer",
                "Chief Marketing Officer",
                "Chief Operating Officer",
                "Chief Technology Officer",
                "President",
                "Vice President",
                "Director",
                "General Manager",
                "Operations Manager",
                "Assistant Manager",
                "Manager",
                "Project Manager",                
                "Product Manager",                
                "Event Planner",
                "Graphic Designer",
                "Logistics Manager",
                "Customer Service Manager",                
                "Quality Assurance Manager",                
                "Compliance Officer",                
                "Digital Marketing Specialist",
                "Media Planner",
                "Public Relations Specialist",                
                "Production Manager",
                "Design Engineer",
                "Manufacturing Engineer",                
                "Clinical Research Associate",
                "Research Scientist",
                "Biostatistician",
                "Clinical Data Manager",
                "Regulatory Compliance Officer",                
                "Sales Executive",
                "Financial Consultant",
                "IT Consultant",
                "Business Analyst",               
                "Legal Consultant",
                "Risk Manager",
                "Actuary",
                "Underwriter",
                "Nutritionist",
                "Physical Therapist",                
                "Sales Manager",                
                "Cloud Engineer",
                "Data Scientist",
                "Machine Learning Engineer",                
                "Software Engineer",
                "Senior Software Engineer",
                "Quality Engineer",
                "Senior Quality Engineer"
            ]

            # Format the title_list as a list of dictionaries
            formatted_titles = [{'name': title} for title in title_list]

            return jsonify({'Status': 'Success', 'Title List': formatted_titles })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        

    def best_matches_cosine_for_notifications(self, page, per_page, place_id, event_id, user_id):
        try:
           
            user_id = int(user_id)
            place_id = place_id[0] if isinstance(place_id, tuple) and len(place_id[0])>1 else place_id # Ensure placeId is not a tuple
            event_id = int(event_id) if event_id is not None else None
           
            # Fetch data
            df, error = self.get_data(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}
            
            # Handle NULL values in desRank
            df = df.dropna(subset=['desRank'])

            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MSG}

            # Extract logged-in user's data
            logged_in_user = df.loc[df['userId'] == user_id].iloc[0]
            logged_in_industries = set(logged_in_user['industry'].split(','))
            logged_in_des_rank = logged_in_user['desRank']
           

            # Step 1: Filter based on industry overlap and desRank range
            df['industry_set'] = df['industry'].apply(lambda x: set(x.split(',')))
            primary_df = df[
                (df['industry_set'].apply(lambda inds: len(logged_in_industries.intersection(inds)) > 0)) &
                (df['desRank'] >= logged_in_des_rank - 2) &
                (df['desRank'] <= logged_in_des_rank + 2)
            ]

            # Step 2: Prepare secondary_df from remaining data if primary_df < 5 matches
            secondary_df = df.drop(primary_df.index)  # Remove primary matches
            secondary_df = secondary_df[(secondary_df['desRank'] >= logged_in_des_rank - 2) & 
                                        (secondary_df['desRank'] <= logged_in_des_rank + 2)]

            # Step 3: Encode 'industry' and 'currentDesignation'
            mlb = MultiLabelBinarizer()
            industry_encoded = mlb.fit_transform(df['industry'].str.split(','))
            
            encoder = OneHotEncoder(sparse_output=False)
            designation_encoded = encoder.fit_transform(df[['currentDesignation']])

            # Concatenate weighted encoded columns with experience and desRank
            weight_industry = 0.5
            weight_designation = 0.25
            weight_experience = 0.25

            df_encoded = pd.concat([
                df[['experience']].reset_index(drop=True) * weight_experience,
                pd.DataFrame(industry_encoded * weight_industry, columns=mlb.classes_),
                pd.DataFrame(designation_encoded * weight_designation, columns=encoder.get_feature_names_out(['currentDesignation']))
            ], axis=1)

            # Step 4: Calculate cosine similarity
            cos_sim = pd.DataFrame(cosine_similarity(df_encoded), index=df['userId'], columns=df['userId'])

            # Step 5: Calculate similarities for primary and secondary users
            primary_similarities = cos_sim.loc[user_id, primary_df['userId']].drop(user_id, errors='ignore')
            secondary_similarities = cos_sim.loc[user_id, secondary_df['userId']].drop(user_id, errors='ignore')

            # Combine and sort by similarity, prioritizing primary matches
            combined_similarities = pd.concat([primary_similarities, secondary_similarities]).sort_values(ascending=False)

            # Convert combined similarities to a list of sorted user IDs
            match_ids = combined_similarities.index.tolist()[:5]

            # Return only match_ids now
            return {
                "Status": "Success",
                "MatchIds": match_ids,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": len(match_ids)
            }

        except Exception as e:
            print(f"Error during match calculation: {e}")
            return {"Status": "Failure", "Message": str(e)}


    def best_matches_cosine(self, page, per_page, search_text):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)
            event_id = data.get('eventId')

            # Fetch data
            df, error = self.get_data(user_id, place_id, event_id)
            if df.empty:
                return {"Status": "Failure", "Message": error}
            
            # Handle NULL values in desRank
            df = df.dropna(subset=['desRank'])

            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message":self.USER_NOT_FOUND_MSG}
            
            # Extract logged-in user's data
            logged_in_user = df.loc[df['userId'] == user_id].iloc[0]
            logged_in_industries = set(logged_in_user['industry'].split(','))
            logged_in_des_rank = logged_in_user['desRank']


            # Step 1: Filter based on industry overlap and desRank range
            df['industry_set'] = df['industry'].apply(lambda x: set(x.split(',')))
            primary_df = df[
                (df['industry_set'].apply(lambda inds: len(logged_in_industries.intersection(inds)) > 0)) &
                (df['desRank'] >= logged_in_des_rank - 2) &
                (df['desRank'] <= logged_in_des_rank + 2)
            ]

            # Step 2: Prepare secondary_df from remaining data if primary_df < 5 matches
            secondary_df = df.drop(primary_df.index)  # Remove primary matches
            secondary_df = secondary_df[(secondary_df['desRank'] >= logged_in_des_rank - 2) & 
                                        (secondary_df['desRank'] <= logged_in_des_rank + 2)]

            # Step 3: Encode 'industry' and 'currentDesignation'
            mlb = MultiLabelBinarizer()
            industry_encoded = mlb.fit_transform(df['industry'].str.split(','))
            
            encoder = OneHotEncoder(sparse_output=False)
            designation_encoded = encoder.fit_transform(df[['currentDesignation']])

            # Concatenate weighted encoded columns with experience and desRank
            weight_industry = 0.5
            weight_designation = 0.25
            weight_experience = 0.25

            df_encoded = pd.concat([
                df[['experience']].reset_index(drop=True) * weight_experience,
                pd.DataFrame(industry_encoded * weight_industry, columns=mlb.classes_),
                pd.DataFrame(designation_encoded * weight_designation, columns=encoder.get_feature_names_out(['currentDesignation']))
            ], axis=1)

            # Step 4: Calculate cosine similarity
            cos_sim = pd.DataFrame(cosine_similarity(df_encoded), index=df['userId'], columns=df['userId'])

            # Step 5: Calculate similarities for primary and secondary users
            primary_similarities = cos_sim.loc[user_id, primary_df['userId']].drop(user_id, errors='ignore')
            secondary_similarities = cos_sim.loc[user_id, secondary_df['userId']].drop(user_id, errors='ignore')

            # Combine and sort by similarity, prioritizing primary matches
            combined_similarities = pd.concat([primary_similarities, secondary_similarities]).sort_values(ascending=False)

            # Convert combined similarities to a list of sorted user IDs
         
            top_5_matches = combined_similarities.index.tolist()[:5]

            matches_details = self.fetch_user_details(top_5_matches, user_id, search_text)
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": len(top_5_matches)
            }

        except Exception as e:
            print(f"Error during match calculation: {e}")
            return {"Status": "Failure", "Message": str(e)}
        

    # Function to compute bounding box
    def get_bounding_box(self, lat, lon, input_radius):
        return self.get_bounding_box_common(lat, lon, input_radius)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Radius of Earth in meters
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c  # in meters
    
    def get_user_list_by_industry_radius(self):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            place_id = self._resolve_place_id(user_id, data)
            input_radius = data['inputRadius']
            input_radius = int(input_radius) if input_radius else None
            search_text = data['searchText']
            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page
            mode = data.get('mode')
            output_list = []

            cursor = self.db.cur()
            lat, lon = self.get_user_location_radius(cursor, user_id)
            if lat is None or lon is None:
                return jsonify({
                    "Status": "checkout",
                    "Message": "User has no nearby location. Please set your radius location to continue"
                })
            login_web_token = self.get_login_token_radius(cursor, user_id)
            my_gender, my_preference = self._fetch_gender_preference_industry_v1(cursor, user_id)
            gender_filter = self._translate_preference_industry_v1(my_preference)

            age_pref_raw = self._fetch_age_preference_industry_v1(cursor, user_id)
            age_ranges = self._translate_age_preference_industry_v1(age_pref_raw)

            my_dob = self._fetch_user_dob_industry_v1(cursor, user_id)
            my_age = self._calculate_age(my_dob) if my_dob else None

            min_lat, max_lat, min_lon, max_lon = self.get_bounding_box(lat, lon, input_radius)
            total_users = self.get_total_users_radius(cursor, user_id, lat, lon, input_radius, min_lat, max_lat, min_lon, max_lon, gender_filter, age_ranges, my_age, my_gender)

            users = self._dedupe_rows_by_user_id(
                self.get_filtered_users_radius(cursor, user_id, lat, lon, input_radius, min_lat, max_lat, min_lon, max_lon, gender_filter, age_ranges, my_age, my_gender, per_page, offset)
            )

            likes_count = self._fetch_likes_count_industry_v1(cursor, user_id, place_id)

            current_likes_count = self._fetch_current_likes_count_industry_near(cursor, user_id)

            if mode == 'recommend':
                response = self.handle_recommend_mode_radius(user_id, users, login_web_token, search_text, place_id)
                return jsonify(response)

            recommend_output_list_new = self.best_matches_cosine_vibr_near(user_id, users, page, per_page, place_id)
            recommended_user_ids = {
                user['userId'] for user in recommend_output_list_new.get("Matches", [])
            } if recommend_output_list_new.get("Status") == "Success" else set()

            for user in users:
                user_dict = self.build_user_dict_radius(cursor, user_id, user, lat, lon, recommended_user_ids)
                output_list.append(user_dict)

            def _to_ts(val):
                if isinstance(val, datetime):
                    return int(val.timestamp())
                return int(val or 0)

            output_list.sort(
                key=lambda x: (
                    -max(
                        _to_ts(x.get("wavedTime")),
                        _to_ts(x.get("wavedbackTime"))
                    ),
                    not x.get("matching_vibes", False)
                )
            )

            for user in output_list:
                user.pop("matching_vibes", None)


            if search_text:
                output_list = self.filter_by_search_text_radius(output_list, search_text)

            total_users = len(output_list)

            return jsonify({
                "List": output_list,
                "Status": "Success",
                "Page": page,
                "PerPage": per_page,
                "TotalUsers": total_users,
                "TotalLikes": likes_count,
                "CurrentLikes":current_likes_count,
                "loginWebToken": login_web_token
            })

        except Exception as e:
            return {'message': str(e), 'Status': "Failure"}


    # ----- HELPER FUNCTIONS -----
    def _fetch_current_likes_count_industry_near(self, cursor, user_id):

        cursor.execute("""
            SELECT COUNT(*)
            FROM icebreaker.session_likes_near sl
            WHERE sl.wavedTouserId = %s

            -- Exclude blocked users (both directions)
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b1
                WHERE b1.blockedId = sl.user_id AND b1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.blockedUsers b2
                WHERE b2.userId = sl.user_id AND b2.blockedId = %s
            )

            -- Exclude reported users (both directions)
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r1
                WHERE r1.reportedUserId = sl.user_id AND r1.userId = %s
            )
            AND NOT EXISTS (
                SELECT 1 FROM icebreaker.netwirkReports r2
                WHERE r2.userId = sl.user_id AND r2.reportedUserId = %s
            )
        """, (user_id, user_id, user_id, user_id, user_id))

        result = cursor.fetchone()
        return result[0] if result else 0

    def get_user_location_radius(self, cursor, user_id):
        cursor.execute(self.GET_USER_RADIUS_QUERY, (user_id,))
        co_ordinates = cursor.fetchone()
        if not co_ordinates or co_ordinates[0] is None or co_ordinates[1] is None:
            return None, None
        return co_ordinates[0], co_ordinates[1]

    def get_login_token_radius(self, cursor, user_id):
        try:
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
            return cursor.fetchone()[0]
        except (TypeError, IndexError):
            return ''

    def get_user_preference_radius(self, cursor, user_id):
        cursor.execute("SELECT Preference FROM icebreaker.registration WHERE userId = %s", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def translate_preference_radius(self, my_preference):
        return self._translate_preference_common(my_preference)
    
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


    def get_total_users_radius(self, cursor, user_id, lat, lon, input_radius, min_lat, max_lat, min_lon, max_lon, gender_filter, age_ranges=None, my_age=None, my_gender=None):

        age_clause, age_params = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)
        query = f"""
            SELECT COUNT(DISTINCT p.userId)
                FROM icebreaker.profile p
                JOIN icebreaker.radius l ON p.userId = l.userId
                AND l.checkinTime <=> (
                    SELECT MAX(l2.checkinTime)
                    FROM icebreaker.radius l2
                    WHERE l2.userId = p.userId
                )
                JOIN icebreaker.registration r ON p.userId = r.userId
                LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                LEFT JOIN icebreaker.netwirkReports r1 ON l.userId = r1.reportedUserId AND r1.userId = %s
                LEFT JOIN icebreaker.netwirkReports r2 ON l.userId = r2.userId AND r2.reportedUserId = %s
                LEFT JOIN icebreaker.hidden_users h  ON p.userId = h.hiddenUserId AND h.userId = %s
                LEFT JOIN icebreaker.hidden_users h2 ON p.userId = h2.userId AND h2.hiddenUserId = %s
                WHERE p.userId != %s
                AND l.latitude BETWEEN %s AND %s
                AND l.longitude BETWEEN %s AND %s
                AND (
                    3959 * acos(
                        cos(radians(%s)) * cos(radians(l.latitude)) *
                        cos(radians(l.longitude) - radians(%s)) +
                        sin(radians(%s)) * sin(radians(l.latitude))
                    )
                ) <= LEAST(l.radius, %s)
                AND b1.blockedId IS NULL 
                AND b2.blockedId IS NULL
                AND r1.reportedUserId IS NULL 
                AND r2.reportedUserId IS NULL
                AND h.hiddenUserId IS NULL
                AND h2.userId IS NULL  
                AND (%s IS NULL OR r.Gender = %s)
                {"AND " + age_clause if age_clause else ""}
                {"AND " + reverse_clause if reverse_clause else ""}
                {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
        """
        params = [user_id, user_id, user_id, user_id, user_id, user_id, user_id, min_lat, max_lat, min_lon, max_lon, lat, lon, lat, input_radius,
                        gender_filter, gender_filter, *age_params, *reverse_params, *reverse_gender_params
        ]
      
        cursor.execute(query, params)
        return cursor.fetchone()[0]

    def get_filtered_users_radius(self, cursor, user_id, lat, lon, input_radius, min_lat, max_lat, min_lon, max_lon, gender_filter, age_ranges=None, my_age=None, my_gender=None, per_page=10, offset=0):
        age_clause, age_params = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)

        query = f"""
            SELECT p.userId, p.profilePic, p.selfiePic, r.fullName, r.email, r.firebaseId, r.signupType, r.authenticationId, r.Gender, r.Preference, r.DOB, p.vibes, p.vibing, 
                    r.state, r.city,  p.zodiac_sign, p.aesthetic_score, l.placeId, r.phoneNo, r.deviceToken, l.latitude, l.longitude, l.radius
                FROM icebreaker.profile p
                JOIN icebreaker.radius l ON p.userId = l.userId
                AND l.checkinTime <=> (
                    SELECT MAX(l2.checkinTime)
                    FROM icebreaker.radius l2
                    WHERE l2.userId = p.userId
                )
                JOIN icebreaker.registration r ON p.userId = r.userId
                 
                LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                LEFT JOIN icebreaker.netwirkReports r1 ON l.userId = r1.reportedUserId AND r1.userId = %s
                LEFT JOIN icebreaker.netwirkReports r2 ON l.userId = r2.userId AND r2.reportedUserId = %s
                LEFT JOIN icebreaker.hidden_users h  ON p.userId = h.hiddenUserId AND h.userId = %s
                LEFT JOIN icebreaker.hidden_users h2 ON p.userId = h2.userId AND h2.hiddenUserId = %s
                WHERE p.userId != %s
                AND l.latitude BETWEEN %s AND %s
                AND l.longitude BETWEEN %s AND %s
                AND (
                    3959 * acos(
                        cos(radians(%s)) * cos(radians(l.latitude)) *
                        cos(radians(l.longitude) - radians(%s)) +
                        sin(radians(%s)) * sin(radians(l.latitude))
                    )
                ) <= LEAST(l.radius, %s)
                AND b1.blockedId IS NULL
                AND b2.blockedId IS NULL
                AND r1.reportedUserId IS NULL 
                AND r2.reportedUserId IS NULL
                AND h.hiddenUserId IS NULL
                AND h2.userId IS NULL  
                AND (%s IS NULL OR r.Gender = %s)
                {"AND " + age_clause if age_clause else ""}
                {"AND " + reverse_clause if reverse_clause else ""}
                {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
                LIMIT %s OFFSET %s          
            """
        params = [ user_id, user_id, user_id, user_id, user_id, user_id, user_id, min_lat, max_lat, min_lon, max_lon, lat, lon, lat, input_radius, gender_filter, gender_filter,
        *age_params, *reverse_params, *reverse_gender_params,
        per_page, offset
        ]
      
        cursor.execute(query, params)
        return self._dedupe_rows_by_user_id(cursor.fetchall())

    @staticmethod
    def _dedupe_rows_by_user_id(rows):
        seen = set()
        unique = []
        for row in rows or []:
            uid = row[0]
            if uid in seen:
                continue
            seen.add(uid)
            unique.append(row)
        return unique

    def handle_recommend_mode_radius(self, user_id, users, login_web_token, search_text, place_id=None):
        recommendations = self.best_matches_cosine_vibr_near(user_id, users, 1, 3, place_id)
        output_list = recommendations.get("Matches", []) if recommendations.get("Status") == "Success" else []

        # Apply search filter if present
        if search_text:
            search_text = search_text.lower()
            output_list = [user for user in output_list if search_text in user.get("fullName", "").lower()]

        output_list.sort(key=lambda u: u.get("wavedTime", 0), reverse=True)

        return {
                "List": output_list,
                "Status": "Success",
                "Page": 1,
                "PerPage": 3,
                "TotalUsers": len(output_list),
                "loginWebToken": login_web_token
            }

    def build_user_dict_radius(self, cursor, user_id, user, lat, lon, recommended_user_ids):
        waved_user_id = user[0]
        cursor.execute('SELECT userId, blockedId FROM icebreaker.blockedUsers WHERE userId = %s AND blockedId = %s', (user_id, user[0]))
        res2 = cursor.fetchone()
        block_flag = 1 if res2 else 0
        distance = self.haversine_distance(lat, lon, user[20], user[21])


        wave_status = self.get_wave_status_sender_near(cursor, user_id, waved_user_id)
        cursor.execute("""
                SELECT created_at 
                FROM icebreaker.user_likes_near
                WHERE sender_id = %s 
                AND receiver_id = %s 
            
            """, (waved_user_id, user_id))  # Added missing parameter
        result  = cursor.fetchone()

        if result and result[0]:
            # Convert datetime to integer timestamp
            waved_time_value = int(result[0].timestamp())
        else:
            waved_time_value = 0



        user_dict = {
            "userId": user[0],
            "profilePic": user[1],
            "selfiePic":user[2],
            "fullName": user[3],
            "email": user[4],
            "firebaseId": user[5],
            "signupType": user[6],
            "authenticationId": user[7],
            "Gender": user[8],
            "Preference": user[9],
            "DOB": user[10].strftime('%m-%d-%Y') if user[10] else '',
            "vibes": user[11],
            "vibing": user[12],
            "block_flag": block_flag,
            "placeId": user[17],
            "phoneNo": user[18],
            "deviceToken": user[19],
            "latitude": user[20],
            "longitude": user[21],
            "recommended": user[0] in recommended_user_ids,
            "distance": round(distance, 2),
            "wave_status": wave_status,
            "wavedTime": waved_time_value
           
        }
        user_dict["pictures"] = self.get_user_pictures_radius(cursor, user[0])
        user_dict["status"] = self.get_connection_status_radius(cursor, user_id, user[0])
        return user_dict

    def get_user_pictures_radius(self, cursor, user_id):
        cursor.execute("SELECT picture_urls, IsProfilePicture FROM icebreaker.userPictures WHERE loginUserId = %s", (user_id,))
        pics_data = cursor.fetchall()
        return [{"url": pic[0], "isProfilePicture": bool(pic[1])} for pic in pics_data]

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

    def filter_by_search_text_radius(self, output_list, search_text):
        search_text = search_text.lower()
        return [user for user in output_list if search_text in user["fullName"].lower()]

        
    def best_matches_cosine_radius(self, page, per_page, search_text, min_lat, max_lat, min_lon, max_lon):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
        

            # Fetch data
            df, error = self.get_data_radius(user_id, min_lat, max_lat, min_lon, max_lon)
            if df.empty:
                return {"Status": "Failure", "Message": error}
            
            # Handle NULL values in desRank
            df = df.dropna(subset=['desRank'])

            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message":self.USER_NOT_FOUND_MSG}
            
            # Extract logged-in user's data
            logged_in_user = df.loc[df['userId'] == user_id].iloc[0]
            logged_in_industries = set(logged_in_user['industry'].split(','))
            logged_in_des_rank = logged_in_user['desRank']

            # Step 1: Filter based on industry overlap and desRank range
            df['industry_set'] = df['industry'].apply(lambda x: set(x.split(',')))
            primary_df = df[
                (df['industry_set'].apply(lambda inds: len(logged_in_industries.intersection(inds)) > 0)) &
                (df['desRank'] >= logged_in_des_rank - 2) &
                (df['desRank'] <= logged_in_des_rank + 2)
            ]

            # Step 2: Prepare secondary_df from remaining data if primary_df < 5 matches
            secondary_df = df.drop(primary_df.index)  # Remove primary matches
            secondary_df = secondary_df[(secondary_df['desRank'] >= logged_in_des_rank - 2) & 
                                        (secondary_df['desRank'] <= logged_in_des_rank + 2)]

            # Step 3: Encode 'industry' and 'currentDesignation'
            mlb = MultiLabelBinarizer()
            industry_encoded = mlb.fit_transform(df['industry'].str.split(','))
            
            encoder = OneHotEncoder(sparse_output=False)
            designation_encoded = encoder.fit_transform(df[['currentDesignation']])

            # Concatenate weighted encoded columns with experience and desRank
            weight_industry = 0.5
            weight_designation = 0.25
            weight_experience = 0.25

            df_encoded = pd.concat([
                df[['experience']].reset_index(drop=True) * weight_experience,
                pd.DataFrame(industry_encoded * weight_industry, columns=mlb.classes_),
                pd.DataFrame(designation_encoded * weight_designation, columns=encoder.get_feature_names_out(['currentDesignation']))
            ], axis=1)

            # Step 4: Calculate cosine similarity
            cos_sim = pd.DataFrame(cosine_similarity(df_encoded), index=df['userId'], columns=df['userId'])

            # Step 5: Calculate similarities for primary and secondary users
            primary_similarities = cos_sim.loc[user_id, primary_df['userId']].drop(user_id, errors='ignore')
            secondary_similarities = cos_sim.loc[user_id, secondary_df['userId']].drop(user_id, errors='ignore')

            # Combine and sort by similarity, prioritizing primary matches
            combined_similarities = pd.concat([primary_similarities, secondary_similarities]).sort_values(ascending=False)

            # Convert combined similarities to a list of sorted user IDs
            top_5_matches = combined_similarities.index.tolist()[:5]

            matches_details = self.fetch_user_details(top_5_matches, user_id, search_text)
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": len(top_5_matches)
            }

        except Exception as e:
            print(f"Error during match calculation: {e}")
            return {"Status": "Failure", "Message": str(e)}
        

    def get_data_radius(self, user_id, min_lat, max_lat, min_lon, max_lon):
        try:
            cursor = self.db.cur()

            #Fetch accepted users list
            query_to_fetch_connected_users = """
                SELECT requestedUserId
                FROM icebreaker.netwirkRequestsAccepted
                where userId = %s
            """
            cursor.execute(query_to_fetch_connected_users, (user_id,))
            connected_user_ids = [row[0] for row in cursor.fetchall()]

            query = """
                    SELECT p.userId, p.industry, p.currentDesignation , p.experience, p.desRank
                    FROM icebreaker.profile p
                    JOIN icebreaker.radius l ON p.userId = l.userId
                    JOIN icebreaker.registration r ON p.userId = r.userId
                    LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                    LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                    WHERE l.latitude BETWEEN %s AND %s 
                    AND l.longitude BETWEEN %s AND %s
                    AND b1.blockedId IS NULL
                    AND b2.blockedId IS NULL                
                    """
            params = [user_id, user_id, min_lat, max_lat, min_lon, max_lon]
            
            #Modify the query if there are connected user IDs
            if connected_user_ids:
                placeholders = ', '.join(['%s'] * len(connected_user_ids))
                query += f" AND p.userId NOT IN ({placeholders})"
                params.extend(tuple(connected_user_ids))

            cursor.execute(query, params)
                
            rows = cursor.fetchall()
            cursor.close()

            # Convert fetched data to DataFrame
            df = pd.DataFrame(rows, columns=["userId", "industry", "currentDesignation", "experience", "desRank"])
            return df, None
        
        except Exception as e:
            print(str(e))
            return pd.DataFrame(), str(e)
        
        

    def best_matches_all_sorted_by_checkintime_radius(self, page, per_page, offset, search_text, min_lat, max_lat, min_lon, max_lon):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
           
            cursor = self.db.cur()

            cursor.execute("SELECT industry FROM icebreaker.profile WHERE userId = %s", (user_id,))
            logged_in_user_industry = cursor.fetchone()[0]
            logged_in_user_industries = [ind.strip() for ind in logged_in_user_industry.split(',')]
            
            df, error = self.get_data_all_users_radius(user_id, min_lat, max_lat, min_lon, max_lon)
            if df.empty:
                return {"Status": "Failure", "Message": error}

            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message":self.USER_NOT_FOUND_MSG}
            
            # Remove the logged-in user from the DataFrame
            df = df[df['userId'] != user_id]
            
            # Filter users with matching industries
            def has_matching_industry(user_industry):
                user_industries = [ind.strip() for ind in user_industry.split(',')]
                return any(ind in user_industries for ind in logged_in_user_industries)
            
            # Users with matching industries sorted by checkinTime
            matching_industry_users = df[df['industry'].apply(has_matching_industry)]
            matching_industry_users = matching_industry_users.sort_values(by='checkinTime', ascending=False)

            # Users with non-matching industries sorted by checkinTime
            non_matching_industry_users = df[~df['industry'].apply(has_matching_industry)]
            non_matching_industry_users = non_matching_industry_users.sort_values(by='checkinTime', ascending=False)

            # Concatenate the two dataframes, matching industries first
            sorted_users = pd.concat([matching_industry_users, non_matching_industry_users])

            total_matches = len(sorted_users)

            paginated_match_scores = sorted_users[offset:min(offset + per_page, total_matches)]

            # Extracting user IDs from paginated results
            match_ids = paginated_match_scores['userId'].tolist()


            matches_details = self.fetch_user_details(match_ids, user_id, search_text)
            if isinstance(matches_details, dict) and matches_details.get("Status") == "Failure":
                return {"Status": "Failure", "Message": matches_details.get("message")}

            return {
                "Status": "Success",
                "Matches": matches_details,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": total_matches
            }

        except Exception as e:
            print(str(e))
            return {"Status": "Failure", "Message": str(e)}
        

    def get_data_all_users_radius(self, user_id, min_lat, max_lat, min_lon, max_lon):
        try:
            cursor = self.db.cur()

            query = """
                    SELECT p.userId, p.industry, l.checkinTime
                    FROM icebreaker.profile p
                    JOIN icebreaker.radius l ON p.userId = l.userId
                    JOIN icebreaker.registration r ON p.userId = r.userId
                    LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                    LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                    WHERE l.latitude BETWEEN %s AND %s
                    AND l.longitude BETWEEN %s AND %s
                    AND b1.blockedId IS NULL
                    AND b2.blockedId IS NULL               
                    """
            cursor.execute(query, (user_id, user_id, min_lat, max_lat, min_lon, max_lon))
                
            rows = cursor.fetchall()
            cursor.close()

            # Convert fetched data to DataFrame
            df = pd.DataFrame(rows, columns=["userId", "industry", "checkinTime"])
            return df, None
        
        except Exception as e:
            return pd.DataFrame(), str(e)
        

    def get_industry_user_count_radius(self):
        try:
            data = request.get_json()
     
            user_id = data['userId']
            event_id = data['eventId']
            input_radius = data['inputRadius']
            if input_radius == "":
                input_radius = None
            else:
                input_radius = int(input_radius)
            cursor = self.db.cur()

            # Fetch the co-ordinates for the logged in user
            cursor.execute(self.FETCH_LAT_LONG_QUERY, (user_id,))
            co_ordinates = cursor.fetchall()
            lat = co_ordinates[0][0]
            lon = co_ordinates[0][1]

            # Get the bounding box
            min_lat, max_lat, min_lon, max_lon = self.get_bounding_box(lat, lon, input_radius)

            # Check for eventId and query accordingly, excluding blocked users (both directions)
            if event_id:
                # Query for eventLocation, excluding users who have blocked the logged-in user
                # AND excluding users who the logged-in user has blocked.
                cursor.execute('''SELECT p.industry
                              FROM icebreaker.eventLocation e
                              JOIN icebreaker.profile p ON e.userId = p.userId
                              LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                              LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                              WHERE e.eventId = %s 
                              AND e.userId != %s 
                              AND b1.blockedId IS NULL 
                              AND b2.blockedId IS NULL''', 
                              (user_id, user_id, event_id, user_id))
                
            else:
                # Query for location, excluding users who have blocked the logged-in user
                # AND excluding users who the logged-in user has blocked.
                cursor.execute('''SELECT p.industry
                              FROM icebreaker.radius l
                              JOIN icebreaker.profile p ON l.userId = p.userId
                              LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                              LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                              WHERE l.latitude BETWEEN %s AND %s
                              AND l.longitude BETWEEN %s AND %s 
                              AND l.userId != %s 
                              AND b1.blockedId IS NULL 
                              AND b2.blockedId IS NULL''', 
                              (user_id, user_id, min_lat, max_lat, min_lon, max_lon, user_id))
                   
            industries = cursor.fetchall()

            # Flatten the list of tuples and split industries
            industry_list = [industry for sublist in industries for industry in sublist[0].split(',')]

            # Use Counter to count occurrences efficiently
            industry_counts = Counter(industry.strip() for industry in industry_list)

            # Convert Counter to list of dictionaries
            result = [{'industry': key, 'count': value} for key, value in industry_counts.items()]

            
            if industry_counts:
            # Convert dictionary entries to list of dictionaries 
                return result
            else:
                return{'industry': "", 'count': ""}

        except Exception as e:
            print(str(e))
            return{'message': str(e), 'status': 'Failure'}
        

    def usercheckinSendNotification_radius(self):
        try:
            data = request.get_json()
            user_id = int(data.get('userId'))

            cursor=self.db.cur()

            # Check if eventId is present and not empty
            event_id = data.get('eventId')
            if event_id == "":
                event_id = None
            else:
                event_id = int(event_id)
            
            input_radius = data.get('inputRadius')
            if input_radius == "":
                input_radius = None
            else:
                input_radius = int(input_radius)

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.GET_USER_LOGIN_INFO_QUERY,(user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            # Fetch the co-ordinates for the logged in user
            cursor.execute(self.FETCH_LAT_LONG_QUERY, (user_id,))
            co_ordinates = cursor.fetchall()
            lat = co_ordinates[0][0]
            lon = co_ordinates[0][1]
          
            # Get the bounding box
            min_lat, max_lat, min_lon, max_lon = self.get_bounding_box(lat, lon, input_radius)

            if event_id:
                cursor.execute(self.SELECT_EVENT_ID_BY_USER, (user_id,))
                event_id = cursor.fetchone()
              
                cursor.execute("SELECT userId FROM icebreaker.eventLocation WHERE eventId = %s AND userId != %s", (event_id[0], user_id))
              
                # print(userId_list)
            
            else:
                
                cursor.execute("SELECT userId FROM icebreaker.radius WHERE latitude BETWEEN %s AND %s AND longitude BETWEEN %s AND %s AND userId != %s", (min_lat, max_lat, min_lon, max_lon, user_id))
          

            self.send_user_checkin_notification_radius(user_id, event_id, min_lat, max_lat, min_lon, max_lon)
            
            # Add error handling for connected users notification
            try:
                self.send_user_checkin_notification_connected_users_radius(user_id, event_id, min_lat, max_lat, min_lon, max_lon)
            except Exception as e:
                print(f"Error in send_user_checkin_notification_connected_users: {e}")
                # Optionally log this error or handle it as needed

            message = self.SUCCESSFUL_CHECKIN_MESSAGE
            return jsonify({'message': message, 'status' : 'Success', "loginWebToken": login_web_token})
        
        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            return error


    # This api is only for location checkins and not event checkins, please create a new api for event checkin notifications or modify this one for the same
    def send_user_checkin_notification_radius(self, user_id, event_id, min_lat, max_lat, min_lon, max_lon):
        try:
            cursor = self.db.cur()
            login_web_token, full_name, current_designation = self._get_user_profile(cursor, user_id)

            user_id_list, place_id, event_id = self._get_relevant_users(cursor, user_id, event_id, min_lat, max_lat, min_lon, max_lon)
            users_to_notify = self._find_users_to_notify(user_id, user_id_list, place_id, event_id, min_lat, max_lat, min_lon, max_lon)

            if users_to_notify:
                self._send_notifications(cursor, users_to_notify, full_name, current_designation)

            self.db.conn.close()
            return {"Status": "Success", "message": "User check-in notifications sent successfully", "loginWebToken": login_web_token}

        except Exception as e:
            self.db.conn.rollback()
            return {"Status": "Failure", "message": {"error": str(e)}}


    # ---------- Helper methods -------------

    def _get_user_profile(self, cursor, user_id):
        cursor.execute(
            '''SELECT r.loginWebToken, r.fullName, p.currentDesignation
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s''',
            (user_id,)
        )
        login_token = cursor.fetchone()
        return login_token[0], login_token[1], login_token[2]


    def _get_relevant_users(self, cursor, user_id, event_id, min_lat, max_lat, min_lon, max_lon):
        if event_id:
            cursor.execute(self.SELECT_EVENT_ID_BY_USER, (user_id,))
            event_id = cursor.fetchone()
            place_id = ""
            cursor.execute(
                """SELECT e.userId 
                FROM icebreaker.eventLocation e 
                LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s
                LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s
                WHERE e.eventId = %s 
                AND e.userId != %s 
                AND b1.blockedId IS NULL 
                AND b2.blockedId IS NULL""",
                (user_id, user_id, event_id[0], user_id)
            )
            user_id_list = cursor.fetchall()
            return [u[0] for u in user_id_list], place_id, event_id[0]

        # Radius search
        cursor.execute('SELECT placeId FROM icebreaker.radius WHERE userId = %s', (user_id,))
        place_id = cursor.fetchone()
        cursor.execute(
            """SELECT l.userId
            FROM icebreaker.radius l
            LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s
            LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s
            WHERE l.latitude BETWEEN %s AND %s
            AND l.longitude BETWEEN %s AND %s
            AND l.userId != %s
            AND b1.blockedId IS NULL
            AND b2.blockedId IS NULL""",
            (user_id, user_id, min_lat, max_lat, min_lon, max_lon, user_id)
        )
        user_id_list = cursor.fetchall()
        return [u[0] for u in user_id_list], place_id, None


    def _find_users_to_notify(self, user_id, user_id_list, place_id, event_id, min_lat, max_lat, min_lon, max_lon):
        users_to_notify = []
        for other_user_id in user_id_list:
            matched_users = self.best_matches_cosine_for_notifications_radius(
                1, 5, place_id, event_id, other_user_id, min_lat, max_lat, min_lon, max_lon
            )

            if self.db.conn is None or not self.db.conn.is_connected():
                self.db.__init__()
      

            if matched_users and matched_users["Status"] == "Success":
                if user_id in [int(mid) for mid in matched_users["MatchIds"]]:
                    users_to_notify.append(other_user_id)
        return users_to_notify


    def _send_notifications(self, cursor, users_to_notify, full_name, current_designation):
        placeholders = ','.join(['%s'] * len(users_to_notify))
        query = f"SELECT deviceToken FROM icebreaker.registration WHERE userId IN ({placeholders}) AND deviceToken IS NOT NULL AND deviceToken != ''"
        cursor.execute(query, tuple(users_to_notify))
        tokens = cursor.fetchall()

        threads = []
        for token in tokens:
            t = threading.Thread(target=self.send_fcm_with_rest_new_user, args=(token[0], full_name))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        

    def best_matches_cosine_for_notifications_radius(self, page, per_page, place_id, event_id, user_id, min_lat, max_lat, min_lon, max_lon):
        try:
        
            user_id = int(user_id)
            place_id = place_id[0] if isinstance(place_id, tuple) and len(place_id[0])>1 else place_id # Ensure placeId is not a tuple
            event_id = int(event_id) if event_id is not None else None
       
            
            # Fetch data
            df, error = self.get_data_radius(user_id, min_lat, max_lat, min_lon, max_lon)
            if df.empty:
                return {"Status": "Failure", "Message": error}
            
            # Handle NULL values in desRank
            df = df.dropna(subset=['desRank'])

            if user_id not in df['userId'].values:
                return {"Status": "Failure", "Message":self.USER_NOT_FOUND_MSG}

            # Extract logged-in user's data
            logged_in_user = df.loc[df['userId'] == user_id].iloc[0]
            logged_in_industries = set(logged_in_user['industry'].split(','))
            logged_in_des_rank = logged_in_user['desRank']
           

            # Step 1: Filter based on industry overlap and desRank range
            df['industry_set'] = df['industry'].apply(lambda x: set(x.split(',')))
            primary_df = df[
                (df['industry_set'].apply(lambda inds: len(logged_in_industries.intersection(inds)) > 0)) &
                (df['desRank'] >= logged_in_des_rank - 2) &
                (df['desRank'] <= logged_in_des_rank + 2)
            ]

            # Step 2: Prepare secondary_df from remaining data if primary_df < 5 matches
            secondary_df = df.drop(primary_df.index)  # Remove primary matches
            secondary_df = secondary_df[(secondary_df['desRank'] >= logged_in_des_rank - 2) & 
                                        (secondary_df['desRank'] <= logged_in_des_rank + 2)]

            # Step 3: Encode 'industry' and 'currentDesignation'
            mlb = MultiLabelBinarizer()
            industry_encoded = mlb.fit_transform(df['industry'].str.split(','))
            
            encoder = OneHotEncoder(sparse_output=False)
            designation_encoded = encoder.fit_transform(df[['currentDesignation']])

            # Concatenate weighted encoded columns with experience and desRank
            weight_industry = 0.5
            weight_designation = 0.25
            weight_experience = 0.25

            df_encoded = pd.concat([
                df[['experience']].reset_index(drop=True) * weight_experience,
                pd.DataFrame(industry_encoded * weight_industry, columns=mlb.classes_),
                pd.DataFrame(designation_encoded * weight_designation, columns=encoder.get_feature_names_out(['currentDesignation']))
            ], axis=1)

            # Step 4: Calculate cosine similarity
            cos_sim = pd.DataFrame(cosine_similarity(df_encoded), index=df['userId'], columns=df['userId'])

            # Step 5: Calculate similarities for primary and secondary users
            primary_similarities = cos_sim.loc[user_id, primary_df['userId']].drop(user_id, errors='ignore')
            secondary_similarities = cos_sim.loc[user_id, secondary_df['userId']].drop(user_id, errors='ignore')

            # Combine and sort by similarity, prioritizing primary matches
            combined_similarities = pd.concat([primary_similarities, secondary_similarities]).sort_values(ascending=False)

            # Convert combined similarities to a list of sorted user IDs
         
            match_ids = combined_similarities.index.tolist()[:5]


            # Return only match_ids now
            return {
                "Status": "Success",
                "MatchIds": match_ids,
                "Page": page,
                "PerPage": per_page,
                "TotalMatches": len(match_ids)
            }

        except Exception as e:
            print(f"Error during match calculation: {e}")
            return {"Status": "Failure", "Message": str(e)}


# This api is only for location checkins and not event checkins, please create a new api for event checkin notifications or modify this one for the same
    def send_user_checkin_notification_connected_users_radius(self, user_id, event_id, min_lat, max_lat, min_lon, max_lon):
        try:
            
            # Re-establish connection after best_matches API call
            if self.db.conn is None or not self.db.conn.is_connected():
                self.db.__init__()  # Re-establish the connection
            cursor = self.db.cur()  # Recreate cursor after reconnection
          

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.GET_USER_LOGIN_INFO_QUERY,(user_id,))
            login_token = cursor.fetchall()
            if login_token:
                login_web_token = login_token[0][0]
                full_name = login_token[0][1].title().strip()
            else:
                login_web_token = full_name  = None

            if event_id:
                # Fetch accepted users list
                query_to_fetch_connected_users = """
                    SELECT e.userId
                    FROM icebreaker.eventLocation e
                    LEFT JOIN icebreaker.blockedUsers b1 ON e.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                    LEFT JOIN icebreaker.blockedUsers b2 ON e.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                    where e.eventId = %s
                    AND e.userId IN (
                    SELECT requestedUserId
                    FROM icebreaker.netwirkRequestsAccepted
                    where userId = %s
                    )
                    AND b1.blockedId IS NULL
                    AND b2.blockedId IS NULL"""
               
                cursor.execute(query_to_fetch_connected_users, (user_id, user_id, event_id, user_id))
                connected_user_ids = [row[0] for row in cursor.fetchall()]

            else:
                #Fetch accepted users list
                query_to_fetch_connected_users = """
                    SELECT l.userId
                    FROM icebreaker.radius l
                    LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s  -- Blocked by logged-in user
                    LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s  -- Logged-in user blocked by others
                    WHERE l.latitude BETWEEN %s AND %s
                    AND l.longitude BETWEEN %s AND %s
                    AND l.userId IN (
                    SELECT requestedUserId
                    FROM icebreaker.netwirkRequestsAccepted
                    where userId = %s
                    )
                    AND b1.blockedId IS NULL
                    AND b2.blockedId IS NULL"""
                cursor.execute(query_to_fetch_connected_users, (user_id, user_id, min_lat, max_lat, min_lon, max_lon, user_id))
                connected_user_ids = [row[0] for row in cursor.fetchall()]

            # Initialize list to store users to notify
            users_to_notify = connected_user_ids

            # Send notifications to users_to_notify
            if users_to_notify:
                placeholders = ','.join(['%s'] * len(users_to_notify))
                query = f"SELECT deviceToken FROM icebreaker.registration WHERE userId IN ({placeholders}) AND deviceToken IS NOT NULL AND deviceToken != ''"
                cursor.execute(query, tuple(users_to_notify))
                tokens = cursor.fetchall()

                # Parallel processing for sending notifications
                threads = []
                for token in tokens:
                    t = threading.Thread(target=self.send_fcm_with_rest_new_user_connected_user, args=(token[0], full_name))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

            self.db.conn.close()
            message = self.CHECKIN_SUCCESS_MESSAGE
            return {"Status": "Success", "message": message, "loginWebToken": login_web_token}

        except Exception as e:
            self.db.conn.rollback()
            print(str(e))
            error = {'error': str(e)}
            return {"Status": "Failure", "message": error}


    # Contains SSE
    def userCheckin_radius(self):
        try:
            request_data = request.get_json()
            user_id, lat, long, input_radius, intro_line1, intro_line2, desired_industry, incognito, place_id = self._parse_reques_data(request_data)
            checkin_time = datetime.now()

            cursor = self.db.cur()
            cursor.execute("""SELECT picture_urls FROM icebreaker.userPictures
                            where loginUserId = %s and IsProfilePicture = 1 """, (user_id,))
            pic_url = cursor.fetchone()
            
            cursor.execute("""SELECT profilePic FROM icebreaker.profile
                            WHERE userId = %s""", (user_id,))
            profile_pic = cursor.fetchone()

            if profile_pic[0] is None or profile_pic[0] == '' :
                cursor.execute("""UPDATE icebreaker.profile
                                SET profilePic = %s WHERE userId = %s """,(pic_url[0], user_id)) 
                self.db.conn.commit()
            login_web_token, full_name, current_designation, my_preference, my_gender  = self._get_user_profile_data(cursor, user_id)
            gender_filter = self._map_preference_to_gender_data(my_preference)

            age_pref_raw = self._fetch_age_preference_industry_v1(cursor, user_id)
            age_ranges   = self._translate_age_preference_industry_v1(age_pref_raw)
            my_dob       = self._fetch_user_dob_industry_v1(cursor, user_id)
            my_age       = self._calculate_age(my_dob) if my_dob else None

            self._update_user_radius(cursor, user_id, lat, long, checkin_time, input_radius, place_id)
            self._update_user_profile(cursor, user_id, intro_line1, intro_line2, desired_industry)
            self.db.conn.commit()

            if not incognito:
                self._notify_nearby_users(cursor, user_id, lat, long, input_radius, gender_filter,my_gender, full_name, current_designation, login_web_token,age_ranges=age_ranges, my_age=my_age)

            return jsonify({
                'message': self.SUCCESSFUL_CHECKIN_MESSAGE,
                'status': 'Success',
                "loginWebToken": login_web_token
            })

        except Exception as e:
            return jsonify({'message': str(e), 'status': 'Error'})


    # === Helper Methods ===

    def _parse_reques_data(self, data):
        user_id = int(data['userId'])
        lat = data['latitude']
        long = data['longitude']
        input_radius = float(data['inputRadius'])
        intro_line1 = data['intro_line1']
        intro_line2 = data['intro_line2']
        desired_industry = data['desiredIndustry']
        incognito = data.get('incognito', 'no').lower() == 'yes'
        place_id = data.get('placeId')
        return user_id, lat, long, input_radius, intro_line1, intro_line2, desired_industry, incognito, place_id


    def _get_user_profile_data(self, cursor, user_id):
        cursor.execute('''SELECT r.loginWebToken, r.fullName, p.currentDesignation, r.preference, r.Gender
                        FROM icebreaker.registration r
                        JOIN icebreaker.profile p ON r.userId = p.userId
                        WHERE r.userId = %s''', (user_id,))
        row = cursor.fetchone()
        return row[0], row[1].title(), row[2], row[3], row[4]


    def _map_preference_to_gender_data(self, preference):
        if preference == "Women":
            return "Female"
        elif preference == "Men":
            return "Male"
        return None


    def _update_user_radius(self, cursor, user_id, lat, long, checkin_time, input_radius, place_id=None):
        cursor.execute("DELETE FROM icebreaker.radius WHERE userId = %s", (user_id,))
        cursor.execute(
            """INSERT INTO icebreaker.radius(latitude, longitude, placeId, checkinTime, userId, radius)
               VALUES(%s, %s, %s, %s, %s, %s)""",
            (lat, long, place_id, checkin_time, user_id, input_radius),
        )


    def _update_user_profile(self, cursor, user_id, intro_line1, intro_line2, desired_industry):
        cursor.execute("""UPDATE icebreaker.profile 
                        SET intro_line1 = %s, intro_line2 = %s, desiredIndustry = %s 
                        WHERE userId = %s""",
                    (intro_line1, intro_line2, desired_industry, user_id))


    def _notify_nearby_users(self, cursor, user_id, lat, long, input_radius, gender_filter, my_gender, full_name, current_designation, login_web_token, age_ranges=None, my_age=None):
        try:
            min_lat, max_lat, min_lon, max_lon = self.get_bounding_box(lat, long, input_radius)

            age_clause,            age_params            = self._build_age_filter_sql(age_ranges)
            reverse_clause,        reverse_params        = self._build_reverse_age_filter_sql(my_age)
            reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)

            query = f"""
                SELECT l.userId FROM icebreaker.radius l
                JOIN icebreaker.registration r ON l.userId = r.userId
                JOIN icebreaker.profile p ON l.userId = p.userId
                LEFT JOIN icebreaker.blockedUsers b1
                    ON l.userId = b1.blockedId AND b1.userId = %s
                LEFT JOIN icebreaker.blockedUsers b2
                    ON l.userId = b2.userId AND b2.blockedId = %s
                LEFT JOIN icebreaker.netwirkReports r1
                    ON l.userId = r1.reportedUserId AND r1.userId = %s
                LEFT JOIN icebreaker.netwirkReports r2
                    ON l.userId = r2.userId AND r2.reportedUserId = %s
                LEFT JOIN icebreaker.hidden_users h
                    ON l.userId = h.hiddenUserId AND h.userId = %s
                LEFT JOIN icebreaker.hidden_users h2
                    ON l.userId = h2.userId AND h2.hiddenUserId = %s
                WHERE l.latitude BETWEEN %s AND %s
                AND l.longitude BETWEEN %s AND %s
                AND l.userId != %s
                AND (
                    3959 * acos(
                        cos(radians(%s)) * cos(radians(l.latitude)) *
                        cos(radians(l.longitude) - radians(%s)) +
                        sin(radians(%s)) * sin(radians(l.latitude))
                    )
                ) <= LEAST(l.radius, %s)
                AND b1.blockedId IS NULL
                AND b2.blockedId IS NULL
                AND r1.reportedUserId IS NULL
                AND r2.reportedUserId IS NULL
                AND h.hiddenUserId IS NULL
                AND h2.userId IS NULL
                AND r.receiveNotifications = '1'
                AND (%s IS NULL OR r.Gender = %s)
                {"AND " + age_clause            if age_clause            else ""}
                {"AND " + reverse_clause        if reverse_clause        else ""}
                {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
            """
            params = [
                user_id, user_id, user_id, user_id, user_id, user_id,
                min_lat, max_lat, min_lon, max_lon,
                user_id,
                lat, long, lat, input_radius,
                gender_filter, gender_filter,
                *age_params,
                *reverse_params,
                *reverse_gender_params,
            ]

            cursor.execute(query, params)
            nearby_user_ids = list(dict.fromkeys(row[0] for row in cursor.fetchall()))

            if nearby_user_ids:
                placeholders = ','.join(['%s'] * len(nearby_user_ids))
                cursor.execute(f"""
                    SELECT deviceToken FROM icebreaker.registration
                    WHERE userId IN ({placeholders})
                    AND deviceToken IS NOT NULL
                    AND deviceToken != ''
                    AND receiveNotifications = '1'
                """, tuple(nearby_user_ids))
                tokens = cursor.fetchall()

                threads = [
                    threading.Thread(
                        target=self.send_fcm_with_rest_new_user,
                        args=(token[0], full_name)
                    )
                    for token in tokens
                ]
                for t in threads: t.start()
                for t in threads: t.join()

        except Exception as e:
            logger.error(f"[NOTIFY NEARBY] Failed to notify nearby users: {e}")

    def checkout_from_checkinlocation_radius(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            user_id = data['userId']
            event_id = data.get('eventId') or None
            input_radius = float(data['inputRadius']) if data.get('inputRadius') else None
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            place_id = self._resolve_place_id(user_id, data)

            login_web_token, full_name, preference, my_gender  = self._get_user_details(cursor, user_id)
            full_name = full_name.title()
            gender_filter = self._map_preference_to_gender_data(preference)

            # Fetch age preference and own age
            age_pref_raw = self._fetch_age_preference_industry_v1(cursor, user_id)
            age_ranges   = self._translate_age_preference_industry_v1(age_pref_raw)
            my_dob       = self._fetch_user_dob_industry_v1(cursor, user_id)
            my_age       = self._calculate_age(my_dob) if my_dob else None

            if event_id:
                user_id_list = self._handle_event_checkout(cursor, user_id, place_id)
            elif input_radius:
                user_id_list = self._handle_radius_checkout(cursor, user_id, input_radius)
            else:
                user_id_list = []

            user_ids_to_notify = self._reverse_check_within_radius(
                cursor, user_id, user_id_list, input_radius
            )

            # Delete after reverse check
            cursor.execute(
                "DELETE FROM icebreaker.session_likes_near WHERE wavedTouserId = %s", (user_id,)
            )
            cursor.execute(
                "DELETE FROM icebreaker.radius WHERE userId = %s", (user_id,)
            )
            cursor.execute(
                'DELETE FROM icebreaker.hidden_users WHERE userId = %s', (user_id,)
            )
            self.db.conn.commit()

            if user_ids_to_notify:
                self._notify_checkout_users(
                    cursor, user_id, user_ids_to_notify, gender_filter, my_gender,
                    full_name, age_ranges, my_age
                )

            self.db.conn.close()
            return {
                "Status": "Success",
                "message": "User checked out from Checkin location",
                "loginWebToken": login_web_token
            }

        except Exception as e:
            self.db.conn.rollback()
            return {"Status": "Failure", "message": {"error": str(e)}}


    # === Helper methods ===

    def _get_user_details(self, cursor, user_id):
        cursor.execute(
            'SELECT loginWebToken, fullName, Preference, Gender '
            'FROM icebreaker.registration WHERE userId = %s',
            (user_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No user found with userId: {user_id}")
        return row  # (loginWebToken, fullName, Preference, Gender)


    def _handle_event_checkout(self, cursor, user_id, place_id):
        cursor.execute(self.SELECT_EVENT_ID_BY_USER, (user_id,))
        event_id = cursor.fetchone()

        cursor.execute("DELETE FROM icebreaker.eventLocation WHERE userId = %s", (user_id,))
        self.db.conn.commit()

        query = """
            SELECT el.userId
            FROM icebreaker.eventLocation el
            JOIN icebreaker.radius r ON el.userId = r.userId
            WHERE r.placeId = %s AND el.eventId = %s AND el.userId != %s
        """
        cursor.execute(query, (place_id, event_id[0], user_id))
        return cursor.fetchall()


    def _handle_radius_checkout(self, cursor, user_id, input_radius):
        cursor.execute(self.GET_USER_RADIUS_QUERY, (user_id,))
        location = cursor.fetchone()
        if not location:
            raise ValueError(f"Location not found for userId: {user_id}")
        latitude, longitude = location

        min_lat, max_lat, min_lon, max_lon = self.get_bounding_box(latitude, longitude, input_radius)

        query = """
            SELECT l.userId
            FROM icebreaker.radius l
            LEFT JOIN icebreaker.blockedUsers b1
                ON l.userId = b1.blockedId AND b1.userId = %s
            LEFT JOIN icebreaker.blockedUsers b2
                ON l.userId = b2.userId AND b2.blockedId = %s
            LEFT JOIN icebreaker.netwirkReports r1
                ON l.userId = r1.reportedUserId AND r1.userId = %s
            LEFT JOIN icebreaker.netwirkReports r2
                ON l.userId = r2.userId AND r2.reportedUserId = %s
            LEFT JOIN icebreaker.hidden_users h
                ON l.userId = h.hiddenUserId AND h.userId = %s
            LEFT JOIN icebreaker.hidden_users h2
                ON l.userId = h2.userId AND h2.hiddenUserId = %s
            WHERE l.latitude BETWEEN %s AND %s
            AND l.longitude BETWEEN %s AND %s
            AND l.userId != %s
            AND (
                3959 * acos(
                    cos(radians(%s)) * cos(radians(l.latitude)) *
                    cos(radians(l.longitude) - radians(%s)) +
                    sin(radians(%s)) * sin(radians(l.latitude))
                )
            ) <= LEAST(l.radius, %s)
            AND b1.blockedId IS NULL
            AND b2.blockedId IS NULL
            AND r1.reportedUserId IS NULL
            AND r2.reportedUserId IS NULL
            AND h.hiddenUserId IS NULL
            AND h2.userId IS NULL
        """
        cursor.execute(
            query,
            (
                user_id, user_id, user_id, user_id, user_id, user_id,
                min_lat, max_lat, min_lon, max_lon, user_id,
                latitude, longitude, latitude, input_radius,
            ),
        )
        return cursor.fetchall()


    def _reverse_check_within_radius(self, cursor, user_id, user_id_list, input_radius):
        user_ids_to_notify = []
        cursor.execute(
            'SELECT latitude, longitude, radius FROM icebreaker.radius WHERE userId = %s',
            (user_id,),
        )
        user_location = cursor.fetchone()
        if not user_location:
            return user_ids_to_notify
        user_lat, user_lon, _user_radius = user_location

        for other_user in user_id_list:
            other_user_id = other_user[0]
            cursor.execute(
                'SELECT latitude, longitude, radius FROM icebreaker.radius WHERE userId = %s',
                (other_user_id,),
            )
            location = cursor.fetchone()
            if not location or location[2] is None:
                continue
            other_lat, other_lon, other_radius = location
            distance_miles = self.haversine_distance(
                user_lat, user_lon, other_lat, other_lon
            ) / 1609.344
            max_distance = float(other_radius)
            if input_radius is not None:
                max_distance = min(max_distance, float(input_radius))
            if distance_miles <= max_distance:
                user_ids_to_notify.append(other_user_id)

        return user_ids_to_notify


    def _notify_checkout_users(self, cursor, user_id, user_ids_to_notify, gender_filter, my_gender,
                            full_name, age_ranges, my_age):
        if not user_ids_to_notify:
            return

        age_clause,            age_params            = self._build_age_filter_sql(age_ranges)
        reverse_clause,        reverse_params        = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)

        placeholders = ','.join(['%s'] * len(user_ids_to_notify))

        query = f"""
            SELECT r.userId, r.deviceToken
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            LEFT JOIN icebreaker.netwirkReports r1
                ON r.userId = r1.reportedUserId AND r1.userId = %s
            LEFT JOIN icebreaker.netwirkReports r2
                ON r.userId = r2.userId AND r2.reportedUserId = %s
            LEFT JOIN icebreaker.hidden_users h2
                ON r.userId = h2.userId AND h2.hiddenUserId = %s
            WHERE r.userId IN ({placeholders})
            AND r.deviceToken IS NOT NULL
            AND r.deviceToken != ''
            AND r.receiveNotifications = '1'
            AND r1.reportedUserId IS NULL
            AND r2.reportedUserId IS NULL
            AND h2.userId IS NULL
            AND (%s IS NULL OR r.Gender = %s)
            {"AND " + age_clause            if age_clause            else ""}
            {"AND " + reverse_clause        if reverse_clause        else ""}
            {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
        """
        params = [
            user_id, user_id, user_id,
            *user_ids_to_notify,
            gender_filter, gender_filter,
            *age_params,
            *reverse_params,
            *reverse_gender_params,
        ]

        cursor.execute(query, params)
        tokens = cursor.fetchall()

        threads = []
        for _, device_token in tokens:
            t = threading.Thread(
                target=self.send_fcm_with_rest_logout_notification,
                args=(device_token, full_name)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()



    def get_user_list_by_radius(self):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            lat = data['latitude']
            lon = data['longitude']
            input_radius = int(data['inputRadius']) if data['inputRadius'] else None

            cursor = self.db.cur()

            login_web_token = self._get_login_token(cursor, user_id)
            my_gender, my_preference = self._fetch_gender_preference_industry_v1(cursor, user_id)
            gender_filter = self._translate_preference_industry_v1(my_preference)
            age_pref_raw = self._fetch_age_preference_industry_v1(cursor, user_id)
            age_ranges = self._translate_age_preference_industry_v1(age_pref_raw)
            my_dob = self._fetch_user_dob_industry_v1(cursor, user_id)
            my_age = self._calculate_age(my_dob) if my_dob else None
            min_lat, max_lat, min_lon, max_lon = self.get_bounding_box(lat, lon, input_radius)

            users = self._fetch_users_in_radius(
                cursor, user_id, min_lat, max_lat, min_lon, max_lon,
                gender_filter, lat, lon, input_radius,
                age_ranges, my_age, my_gender,
            )
            output_list = [self._build_user_dict_radius(cursor, user_id, user) for user in users]

            response = {
                "List": output_list,
                "Status": 'Success',
                "TotalUsers": len(output_list),
                "loginWebToken": login_web_token
            }

            return jsonify(response)

        except Exception as e:
            return {'message': str(e), 'Status': "Failure"}
        
    def _get_login_token(self, cursor, user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
        token = cursor.fetchone()
        return token[0] if token else None


    def _get_gender_filter(self, cursor, user_id):
        cursor.execute('SELECT Gender, Preference FROM icebreaker.registration WHERE userId = %s', (user_id,))
        _, preference = cursor.fetchone() or (None, None)

        if preference == "Women":
            return "Female"
        elif preference == "Men":
            return "Male"
        return None


    def _fetch_users_in_radius(
        self, cursor, user_id, min_lat, max_lat, min_lon, max_lon, gender_filter,
        lat, lon, input_radius, age_ranges=None, my_age=None, my_gender=None,
    ):
        age_clause, age_params = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)
        query = f"""
            SELECT p.userId, p.industry, l.placeId, p.companyName,
                p.currentDesignation, p.website, p.experience, p.education,
                p.profilePic, r.fullName, r.email, r.countryCode, r.phoneNo, r.firebaseId,
                r.signupType, r.deviceToken, r.authenticationId, p.intro_line1, p.intro_line2, l.latitude, l.longitude,
                r.DOB, r.Gender, r.Preference, p.vibes, p.vibing
            FROM icebreaker.profile p
            JOIN icebreaker.radius l ON p.userId = l.userId
            AND l.checkinTime <=> (
                SELECT MAX(l2.checkinTime)
                FROM icebreaker.radius l2
                WHERE l2.userId = p.userId
            )
            JOIN icebreaker.registration r ON p.userId = r.userId
            LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s
            LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s
            LEFT JOIN icebreaker.netwirkReports r1 ON l.userId = r1.reportedUserId AND r1.userId = %s
            LEFT JOIN icebreaker.netwirkReports r2 ON l.userId = r2.userId AND r2.reportedUserId = %s
            LEFT JOIN icebreaker.hidden_users h ON p.userId = h.hiddenUserId AND h.userId = %s
            LEFT JOIN icebreaker.hidden_users h2 ON p.userId = h2.userId AND h2.hiddenUserId = %s
            WHERE p.userId != %s
            AND l.latitude BETWEEN %s AND %s
            AND l.longitude BETWEEN %s AND %s
            AND (
                3959 * acos(
                    cos(radians(%s)) * cos(radians(l.latitude)) *
                    cos(radians(l.longitude) - radians(%s)) +
                    sin(radians(%s)) * sin(radians(l.latitude))
                )
            ) <= LEAST(l.radius, %s)
            AND b1.blockedId IS NULL
            AND b2.blockedId IS NULL
            AND r1.reportedUserId IS NULL
            AND r2.reportedUserId IS NULL
            AND h.hiddenUserId IS NULL
            AND h2.userId IS NULL
            AND (%s IS NULL OR r.Gender = %s)
            {"AND " + age_clause if age_clause else ""}
            {"AND " + reverse_clause if reverse_clause else ""}
            {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
        """
        params = [
            user_id, user_id, user_id, user_id, user_id, user_id, user_id,
            min_lat, max_lat, min_lon, max_lon, lat, lon, lat, input_radius,
            gender_filter, gender_filter,
            *age_params, *reverse_params, *reverse_gender_params,
        ]
        cursor.execute(query, params)
        return self._dedupe_rows_by_user_id(cursor.fetchall())

    def _build_user_dict_radius(self, cursor, requester_id, user):
        user_id = user[0]
        block_flag = 1 if self._is_user_blocked(cursor, requester_id, user_id) else 0

        user_dict = {
            "userId": user_id,
            "placeId/eventId": user[2],
            "profilePic": user[8],
            "fullName": user[9],
            "email": user[10],
            "phoneNo": user[12],
            "firebaseId": user[13],
            "signupType": user[14],
            "deviceToken": user[15],
            "authenticationId": user[16],
            "block_flag": block_flag,
            "latitude": user[19],
            "longitude": user[20],
            "DOB": user[21],
            "Gender": user[22],
            "Preference": user[23],
            "vibes": user[24],
            "vibing": user[25],
            "pictures": self._get_user_pictures(cursor, user_id),
            "status": self._get_connection_status(cursor, requester_id, user_id)
        }

        return user_dict
    
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


    def _get_connection_status(self, cursor, requester_id, target_user_id):
        cursor.execute(self.GET_ACCEPTED_REQUEST_STATUS_QUERY, (requester_id, target_user_id))
        accepted = cursor.fetchone()

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (requester_id, target_user_id))
        pending_sent = cursor.fetchone()

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY, (target_user_id, requester_id))
        pending_received = cursor.fetchone()

        if accepted and accepted[0] == "Accepted":
            return "Connected"
        if pending_sent and pending_sent[0] == "Pending":
            return "Request_Sent"
        if pending_received and pending_received[0] == "Pending":
            return "Request_Received"
        return "No_Interaction"
        
        
    def ask_get_user_profile_details(self, requested_user_id):
        try:
            request_data = request.get_json()
            login_user_id = request_data['loginUserId']
            cursor = self.db.cur()

            # Fetch loginWebToken
            login_web_token = self._get_login_webtoken(cursor, login_user_id)
            print(login_web_token)

            # Early exit if requested user is blocked
            block_flag = self._is_blocked(cursor, requested_user_id, login_user_id)

            # Fetch user profile
            row = self._get_user_profile(cursor, requested_user_id)
            if not row:
                return self._user_not_found(cursor)

            # Build response dictionary
            user_dict = self._build_user_dict(row, block_flag)

            # Add connection status
            user_dict["status"] = self._get_relationship_status(cursor, login_user_id, requested_user_id)

            # Add login token
            user_dict["loginWebToken"] = login_web_token

            cursor.close()
            self.db.conn.close()
            return user_dict

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            print(error)
            return jsonify(error)


    # ----------------- Helper Functions -----------------

    def _get_login_webtoken(self, cursor, login_user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (login_user_id,))
        login_token = cursor.fetchone()
        return login_token[0] if login_token else None


    def _is_blocked(self, cursor, requested_user_id, login_user_id):
        cursor.execute(
            'SELECT 1 FROM icebreaker.blockedUsers WHERE blockedId = %s AND userId = %s',
            (requested_user_id, login_user_id)
        )
        return 1 if cursor.fetchone() else 0


    def _get_user_profile(self, cursor, requested_user_id):
        cursor.execute("""
            SELECT p.userId, p.companyName, p.currentDesignation, p.industry, p.education, r.fullName
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            WHERE p.userId = %s
        """, (requested_user_id,))
        return cursor.fetchone()


    def _get_relationship_status(self, cursor, login_user_id, requested_user_id):
        cursor.execute(self.GET_ACCEPTED_REQUEST_STATUS_QUERY,
                    (login_user_id, requested_user_id))
        if (status := cursor.fetchone()):
            return "Connected" if status[0] == "Accepted" else "No_Interaction"

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY,
                    (login_user_id, requested_user_id))
        if (status := cursor.fetchone()):
            return "Request_Sent" if status[0] == "Pending" else "No_Interaction"

        cursor.execute(self.GET_PENDING_REQUEST_STATUS_QUERY,
                    (requested_user_id, login_user_id))
        if (status := cursor.fetchone()):
            return "Request_Received" if status[0] == "Pending" else "No_Interaction"

        return "No_Interaction"


    def _build_user_dict(self, row, block_flag):
        return {
            "userId": row[0],
            "companyName": row[1],
            "currentDesignation": row[2],
            "industry": row[3],
            "education": row[4],
            "fullName": row[5],
            "block_flag": block_flag
        }


    def _user_not_found(self, cursor):
        cursor.close()
        self.db.conn.close()
        message = self.USER_NOT_FOUND_MESSAGE
        return jsonify({"error": message})

    
    def ask(self):
        try:
              # === Load credentials from JSON file ===
            aws_config = self._load_vibr_config()
            openai.api_key = aws_config["openai_api_key"]

            if not openai.api_key:
                return jsonify({'error': 'OpenAI API key not found in environment variables'}), 500
            
            # Ensure the request data contains 'loginUserId' and 'requestedUserId'
            request_data = request.get_json()

            if 'loginUserId' not in request_data:
                return jsonify({'error': "'loginUserId' is missing from the request"}), 400
            if 'requestedUserId' not in request_data:
                return jsonify({'error': "'requestedUserId' is missing from the request"}), 400
            
     
            requested_user_id = request_data['requestedUserId']
            user_question = request_data.get('question')

            if not user_question:
                return jsonify({'error': 'Question is required'}), 400

            # Fetch user profile details using the requestedUserId
            user_profile = self.ask_get_user_profile_details(requested_user_id)  # Pass the requestedUserId

            if 'error' in user_profile:
                return jsonify(user_profile), 404

            # Construct the system message for user context
            system_message = f"""
                You are a helpful assistant that provides short and simple answers based on the user's profile.
                The user's details are:
                Name: {user_profile['fullName']}
                Company: {user_profile['companyName']}
                Designation: {user_profile['currentDesignation']}
                Education: {user_profile['education']}
                Industry: {user_profile['industry']}
                Provide three clear, actionable suggestions in response to the user's query,
                Avoid generic answers and provide specific advice related to the user's profile and industry.
                Do not mention or refer to LinkedIn or any other social media platforms like Linkedin in your response.
            
            """

            # Interact with OpenAI to get intent from the user's question
            intent_response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an intent classifier."},
                    {"role": "user", "content": f"Classify the intent of this user question: {user_question}"}
                ],
                max_tokens=50,
                temperature=0.5,
                n=1
            )

            # Extract intent from the response
            intent = intent_response['choices'][0]['message']['content'].strip()

            # Return just the intent without repeating the question or providing extra explanation
            intent_cleaned = intent.split(":")[0]  # This will cleanly grab just the core intent (e.g., "Course recommendation")


            # Use the detected intent to guide OpenAI's response to the user's question
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=150,
                temperature=0.7,
                n=4
            )

            # Extract response content
            answer = response['choices'][0]['message']['content']

            # Ensure answer is not empty or None before proceeding
            if not answer:
                return jsonify({'error': 'No response from OpenAI'}), 500

            # Split the response into suggestions (ensure only 3 suggestions)
            suggestions = answer.split("\n")
            suggestions = [s.strip() for s in suggestions if s.strip()]

            # Limit to exactly 3 suggestions (in case OpenAI returns more)
            suggestions = suggestions[:4]
            while len(suggestions) < 4:
                suggestions.append("")  # Add empty strings if there are fewer than 3 suggestions


            # Return the generated response as an array along with the detected intent
            return jsonify({
                "answer": suggestions,
                "intent": [intent_cleaned]
            })

        except Exception as e:
            import logging
            logging.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 500


    def user_pictures(self):
        try:
            user_id, is_profile_pic_flags, files = self._parse_request()
            if not self._validate_request(files, is_profile_pic_flags):
                return jsonify({'status': 'Failure', 'message': 'Invalid request data.'}), 400

            s3 = self._init_s3_client()
            bucket_name = 'sparoptui'
            region = 'ap-south-1'
            folder_name = 'VibrrUserPics/'

            uploaded_urls = self._process_and_upload_files(
                user_id, files, is_profile_pic_flags, s3, bucket_name, region, folder_name
            )

            return jsonify({
                'status': 'Success',
                'uploaded_urls': uploaded_urls,
                'message': 'Images uploaded successfully.'
            })

        except Exception as e:
            self.db.conn.rollback()
            print(f"Error: {str(e)}")
            return jsonify({'status': 'Failure', 'message': f'Error: {str(e)}'})


# ----------------- Helpers -----------------

    def _parse_request(self):
        user_id_raw = request.form.get('userId', '').strip()
        if not user_id_raw.isdigit():
            raise ValueError("Invalid user ID. Must be numeric.")
        user_id = int(user_id_raw)
        is_profile_pic_flags = request.form.get('IsProfilePicture', 'false').split(',')
        files = [request.files.get(f'image{i}') for i in range(1, 5) if request.files.get(f'image{i}')]
        return user_id, is_profile_pic_flags, files


    def _validate_request(self, files, flags):
        if not files or len(files) > 4:
            return False
        if len(flags) != len(files):
            return False
        return True


    def _init_s3_client(self):
        aws_config = self._load_vibr_config()
        return boto3.client(
            's3',
            region_name=aws_config["region"],
            aws_access_key_id=aws_config["aws_access_key_id"],
            aws_secret_access_key=aws_config["aws_secret_access_key"]
        )


    def _process_and_upload_files(self, user_id, files, flags, s3, bucket, region, folder):
        cursor = self.db.cur()
        cursor.execute("DELETE FROM icebreaker.userPictures WHERE loginUserId = %s", (user_id,))
        uploaded_urls = []

        for idx, file in enumerate(files, start=1):
            is_profile_pic = flags[idx - 1].strip().lower() == 'true'
            url, local_path = self._upload_single_file(user_id, idx, file, s3, bucket, region, folder)
            uploaded_urls.append(url)

            aesthetic_score = self._score_image(local_path) if is_profile_pic else None

            cursor.execute("""
                INSERT INTO icebreaker.userPictures (loginUserId, picture_urls, IsProfilePicture)
                VALUES (%s, %s, %s)
            """, (user_id, url, is_profile_pic))

            if is_profile_pic:
                cursor.execute("""
                    SELECT 1 FROM icebreaker.profile WHERE userId = %s LIMIT 1
                """, (user_id,))

                exists = cursor.fetchone()

                if exists:
                    cursor.execute("""
                        UPDATE icebreaker.profile
                        SET profilePic = %s, aesthetic_score = %s
                        WHERE userId = %s
                    """, (url, aesthetic_score, user_id))
                else:
                    cursor.execute("""
                        INSERT INTO icebreaker.profile (userId, profilePic, aesthetic_score)
                        VALUES (%s, %s, %s)
                    """, (user_id, url, aesthetic_score))

            os.remove(local_path)

        self.db.conn.commit()
        cursor.close()
        return uploaded_urls


    def _upload_single_file(self, user_id, idx, file, s3, bucket, region, folder):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_filename = f"Vibrruser_{user_id}_{idx}_{timestamp}_{file.filename}"
        local_dir = os.path.join(os.getcwd(), 'userPicsTemp')
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, final_filename)

        file.save(local_path)
        original_path = local_path
        local_path = self._compress_image_for_s3(local_path)
        final_filename = os.path.basename(local_path)
        content_type = file.content_type or "application/octet-stream"
        if os.path.normcase(os.path.abspath(local_path)) != os.path.normcase(os.path.abspath(original_path)):
            content_type = "image/jpeg"
        s3.upload_file(local_path, bucket, folder + final_filename, ExtraArgs={'ContentType': content_type})
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{folder}{final_filename}"
        return url, local_path


    def _score_image(self, local_path):
        try:
            aws_config = self._load_vibr_config()
            openai.api_key = aws_config["openai_api_key"]

            with open(local_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            intent_response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Analyze face and return total_score JSON only."},
                    {"role": "user", "content": f"data:image/jpeg;base64,{img_base64}"}
                ],
                max_tokens=10
            )

            gpt_reply = intent_response['choices'][0]['message']['content']
            match = re.search(r'\d+(\.\d+)?', gpt_reply)
            return float(match.group()) if match else 0.2

        except Exception as e:
            print("Image scoring error:", str(e))
            return 0.2


    def vibe_today(self):
        try:
            cursor = self.db.cur()
            data = request.get_json()
            title = data['title']

            # Check the block status for the user whose details are fetched
            cursor.execute('SELECT deviceToken FROM icebreaker.registration')
            tokens = cursor.fetchall()
            threads = []
            for token in tokens:
                t = threading.Thread(target=self.send_fcm_notification_to_all_users, args=(token[0], title))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            message = "User's vibes set for today!"
            status = 'Success'
            return {'message': message, 'status': status}

        except Exception as e:
            print(str(e))

    
    def send_fcm_notification_to_all_users(self,token, title):
        # Notification data
        new_user = "Today_Vibe"

        try:
            # Path to your service account key
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
          

            # Create a credentials object from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE])

            # Create a request object for fetching the access token
            auth_request = Request()

            # Refresh the token
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Prepare payload based on token type
            if isinstance(token, list):
                for tokens in token:
                    payload = {
                        "message": {
                            "tokens": tokens, # Correct key for multiple tokens
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
                                        "mutable-content": 1  # This enables Notification Service Extension for iOS
                                    }
                                }
                            }
                        }
                    }
                    # Send the request to FCM
                    resp = requests.post(url, headers=headers, data=json.dumps(payload))
            else:
                payload = {
                    "message": {
                        "token": token, # Correct key for a single token
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
                                    "mutable-content": 1  # Enables Notification Service Extension for iOS
                                }
                            }
                        }
                    }
                }
                # Send the request to FCM
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
            return {"status": "success", "message": resp.text}

        except Exception as e:
            message = str(e)  # Convert exception to string
            return {"status": "error", "message": message}


    def allow_incognito_visibility(self):
        try:
            data = request.get_json()
            incognito_user_id = data['incognito_userId']
            visible_to_user_ids = list(map(int, data['visible_to_userId'].split(',')))  
            # place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            cursor = self.db.cur()
            place_id = self._resolve_place_id(incognito_user_id, data, cursor)

            if not self._is_incognito_enabled(cursor, incognito_user_id):
                cursor.close()
                return jsonify({
                    "status": "Failed",
                    "message": "You are not using incognito mode"
                })

            self._insert_visibility_exceptions(cursor, incognito_user_id, visible_to_user_ids, place_id)
            self.db.conn.commit()

            users_to_notify = self._fetch_users_to_notify(cursor, visible_to_user_ids)
            if not users_to_notify:
                raise ValueError("No users found to notify")

            incognito_user_data = self._fetch_incognito_user(cursor, incognito_user_id)
            full_name, current_designation = incognito_user_data

            # -----------------
            # Send notifications (kept inline)
            placeholders = ','.join(['%s'] * len(users_to_notify))
            query = f"""
                SELECT deviceToken 
                FROM icebreaker.registration 
                WHERE userId IN ({placeholders}) 
                AND deviceToken IS NOT NULL 
                AND deviceToken != ''
            """
            cursor.execute(query, tuple(visible_to_user_ids))
            tokens = cursor.fetchall()

            threads = []
            for token in tokens:
                t = threading.Thread(
                    target=self.send_fcm_with_rest_new_user,
                    args=(token[0], full_name)
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()
            # -----------------

            cursor.close()
            return jsonify({
                "status": "Success",
                "message": "Visibility granted to selected users for specified placeId."
            })

        except Exception as e:       
            self.db.conn.rollback()
            return jsonify({"status": "Error", "message": str(e)})


        # ---------- HELPERS ----------
    def _is_incognito_enabled(self, cursor, incognito_user_id):
        cursor.execute("""
            SELECT incognito FROM icebreaker.location 
            WHERE userId = %s
        """, (incognito_user_id,))
        row = cursor.fetchone()
        return row and row[0] == 1


    def _insert_visibility_exceptions(self, cursor, incognito_user_id, visible_to_user_ids, place_id):
        for visible_to_user_id in visible_to_user_ids:
            cursor.execute("""
                INSERT IGNORE INTO icebreaker.incognito_visibility_exceptions 
                (incognito_userId, visible_to_userId, placeId)
                VALUES (%s, %s, %s)                 
            """, (incognito_user_id, visible_to_user_id, place_id))


    def _fetch_users_to_notify(self, cursor, visible_to_user_ids):
        users_to_notify = []
        for user in visible_to_user_ids:
            cursor.execute('''
                SELECT r.loginWebToken, r.fullName, p.currentDesignation
                FROM icebreaker.registration r
                JOIN icebreaker.profile p ON r.userId = p.userId
                WHERE r.userId = %s
            ''', (user,))
            user_info = cursor.fetchone()
            if user_info:
                users_to_notify.append(user_info)
        return users_to_notify


    def _fetch_incognito_user(self, cursor, incognito_user_id):
        cursor.execute('''
            SELECT r.fullName, p.currentDesignation
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s
        ''', (incognito_user_id,))
        return cursor.fetchone()
    
    def best_matches_cosine_vibr_near(self, user_id, row, page, per_page, place_id=None):
        try:
            cursor = self.db.cur()
            cursor.execute('''
                SELECT r.gender, r.Preference, p.vibes, p.vibing, r.state, r.city, p.zodiac_sign, p.aesthetic_score, r.DOB
                FROM icebreaker.registration r 
                JOIN icebreaker.profile p ON r.userId = p.userId 
                WHERE r.userId = %s''', (user_id,))
            user_row = cursor.fetchone()
          

            if not user_row:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MESSAGE}

            my_gender, my_preference, my_vibes, _, my_state, my_city, my_zodiac_sign, my_aesthetic_score, my_dob = user_row
            my_vibes_set = set(my_vibes.lower().split(',')) if my_vibes else set()
            compatible_signs = self.find_compatible_signs(my_zodiac_sign)

            prioritized_users = self._filter_by_gender_preference(row, my_gender, my_preference)
            max_users = 5

            cursor.execute('''
                SELECT requestedUserId 
                FROM icebreaker.netwirkRequestsPending 
                WHERE userId = %s 
                ORDER BY requestTime ASC 
                LIMIT 2''', (user_id,))
            past_sent = cursor.fetchall()

            if len(past_sent) < 2:
                return self._fallback_cosine_recommendation_near(
                    prioritized_users, my_gender, my_vibes_set,
                    my_state, my_city, compatible_signs, my_aesthetic_score,
                    max_users, page, per_page, user_id, place_id=place_id
                )
            else:
                return self._behavior_based_recommendation_near(
                    cursor, past_sent, prioritized_users,
                    my_aesthetic_score,
                    max_users, page, per_page, user_id, place_id=place_id
                )

        except Exception as e:
            return {"Status": "Failure", "Message": str(e)}
        
    def _fallback_cosine_recommendation_near(
        self, prioritized_users, my_gender, my_vibes_set,
        my_state, my_city, compatible_signs, my_aesthetic_score,
        max_users, page, per_page, user_id, place_id=None
    ):
        record_fallback()
        final_row = prioritized_users[:max_users]
        if not final_row:
            return {"Status": "Success", "Matches": [], "Page": page, "PerPage": per_page, "TotalMatches": 0}

        user_map, encoded_data = self._prepare_encoding_data_near(
            final_row, my_gender, my_vibes_set, my_state, my_city,
            compatible_signs, my_aesthetic_score, user_id, place_id=place_id
        )

        cos_sim = cosine_similarity(encoded_data)
        similarities = cos_sim[0][1:]  # Skip self
        sorted_uids = list(user_map.keys())
        sorted_scores = sorted(zip(sorted_uids, similarities), key=lambda x: x[1], reverse=True)
        top_matches = sorted_scores[:max_users]

        results = [
            {**user_map[uid], "similarity_score": round(score * 100, 2)}
            for uid, score in top_matches
        ]
        return {
            "Status": "Success",
            "Matches": results,
            "Page": page,
            "PerPage": per_page,
            "TotalMatches": len(sorted_scores)
        }
    
    

    def best_matches_cosine_vibr(self, user_id, row, page, per_page, place_id=None):
        try:
            cursor = self.db.cur()
            cursor.execute('''
                SELECT r.gender, r.Preference, p.vibes, p.vibing, r.state, r.city, p.zodiac_sign, p.aesthetic_score, r.DOB
                FROM icebreaker.registration r 
                JOIN icebreaker.profile p ON r.userId = p.userId 
                WHERE r.userId = %s''', (user_id,))
            user_row = cursor.fetchone()
          

            if not user_row:
                return {"Status": "Failure", "Message": self.USER_NOT_FOUND_MESSAGE}

            my_gender, my_preference, my_vibes, _, my_state, my_city, my_zodiac_sign, my_aesthetic_score, my_dob = user_row
            my_vibes_set = set(my_vibes.lower().split(',')) if my_vibes else set()
            compatible_signs = self.find_compatible_signs(my_zodiac_sign)

            prioritized_users = self._filter_by_gender_preference(row, my_gender, my_preference)
            max_users = 5

            cursor.execute('''
                SELECT requestedUserId 
                FROM icebreaker.netwirkRequestsPending 
                WHERE userId = %s 
                ORDER BY requestTime ASC 
                LIMIT 2''', (user_id,))
            past_sent = cursor.fetchall()

            if len(past_sent) < 2:
                return self._fallback_cosine_recommendation(
                    prioritized_users, my_gender, my_vibes_set,
                    my_state, my_city, compatible_signs, my_aesthetic_score,
                    max_users, page, per_page,user_id, place_id=place_id
                )
            else:
                return self._behavior_based_recommendation(
                    cursor, past_sent, prioritized_users,
                    my_aesthetic_score,
                    max_users, page, per_page,user_id, place_id=place_id
                )

        except Exception as e:
            return {"Status": "Failure", "Message": str(e)}
        
    # ---------------- Helper Methods ------------------

    def _filter_by_gender_preference(self, row, my_gender, my_preference):
        preferred_gender_map = {
            ('Male', 'Women'): 'Female',
            ('Female', 'Men'): 'Male',
            ('Male', 'Men'): 'Male',
            ('Female', 'Women'): 'Female'
        }

        preferred_gender = preferred_gender_map.get((my_gender, my_preference))
        return [user for user in row if user[8] == preferred_gender] if preferred_gender else row


    def _fallback_cosine_recommendation(
        self, prioritized_users, my_gender, my_vibes_set,
        my_state, my_city, compatible_signs, my_aesthetic_score,
        max_users, page, per_page, user_id, place_id=None
    ):
        record_fallback()
        final_row = prioritized_users[:max_users]
        if not final_row:
            return {"Status": "Success", "Matches": [], "Page": page, "PerPage": per_page, "TotalMatches": 0}

        user_map, encoded_data = self._prepare_encoding_data(
            final_row, my_gender, my_vibes_set, my_state, my_city,
            compatible_signs, my_aesthetic_score, user_id, place_id=place_id
        )

        cos_sim = cosine_similarity(encoded_data)
        similarities = cos_sim[0][1:]  # Skip self
        sorted_uids = list(user_map.keys())
        sorted_scores = sorted(zip(sorted_uids, similarities), key=lambda x: x[1], reverse=True)
        top_matches = sorted_scores[:max_users]

        results = [
            {**user_map[uid], "similarity_score": round(score * 100, 2)}
            for uid, score in top_matches
        ]
        return {
            "Status": "Success",
            "Matches": results,
            "Page": page,
            "PerPage": per_page,
            "TotalMatches": len(sorted_scores)
        }
    
    def calculate_age(self,dob):
        """Converts DOB (datetime or string) to age in years"""
        if not dob:
            return 0
        if isinstance(dob, str):
            dob = datetime.strptime(dob, "%Y-%m-%d").date()
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    def _prepare_encoding_data_near(
            self, final_row, my_gender, my_vibes_set, my_state, my_city,
            compatible_signs, my_aesthetic_score, user_id, my_dob=None, place_id=None
        ):
        user_map = {}
        genders, vibes_list, states, cities = [], [], [], []
        zodiac_scores, aesthetics_score, ages = [], [], []
        answers_list = [[] for _ in range(6)]

         # --- Fetch current user's answers ---
        cursor = self.db.cur()
        cursor.execute("""
            SELECT answer1, answer2, answer3, answer4, answer5, answer6
            FROM icebreaker.profile
            WHERE userId = %s
        """, (user_id,))
        my_answers_row = cursor.fetchone()
        my_answers = [
            (str(my_answers_row[i]).strip().lower() if my_answers_row and my_answers_row[i] else '')
            for i in range(6)
        ]
        cursor.close()

        for user in final_row:
            uid = user[0]

            cognito_time = None
           
            cursor = self.db.cur()
            cursor.execute("""
                SELECT incognito FROM icebreaker.location 
                WHERE userId = %s
            """, (uid,))
            row = cursor.fetchone()
        
            if row and row[0] == 1:
            
                cursor = self.db.cur()
                cursor.execute(
                    "SELECT cognito_time FROM icebreaker.location WHERE userId = %s",
                    (uid,)
                )
                row2 = cursor.fetchone()
                if row2:
                    cognito_time = row2[0]

            # --- Calculate user age ---
            dob = user[10]  
            age = self.calculate_age(dob)
            ages.append(age)

            wave_status = self.get_wave_status_sender_near(cursor, user_id,uid)
  
            cursor.execute("""
                    SELECT created_at 
                    FROM icebreaker.user_likes_near
                    WHERE sender_id = %s 
                    AND receiver_id = %s 
                    # AND place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
                    AND (%s IS NULL OR place_id = %s)
                """, (user_id, uid, place_id, place_id))
            result  = cursor.fetchone()

            if result and result[0]:
                # Convert datetime to integer timestamp
               waved_time_value  = int(result[0].timestamp())
            else:
                waved_time_value  = 0

            # --- Fetch this user's answers ---
            cursor.execute("""
                SELECT answer1, answer2, answer3, answer4, answer5, answer6
                FROM icebreaker.profile
                WHERE userId = %s
            """, (uid,))
            ans_row = cursor.fetchone()
            user_answers = [
                (str(ans_row[i]).strip().lower() if ans_row and ans_row[i] else '')
                for i in range(6)
            ]
           
            cursor.close()

            for i in range(6):
                answers_list[i].append(user_answers[i])


            user_map[uid] = {
                "userId": uid,
                "selfiePic":user[2],
                "fullName": user[3],
                "email": user[4],
                "gender": user[8],
                "vibes": user[11],
                "profilePic": user[1],
                "firebaseId": user[5],
                "signupType": user[6],
                "authenticationId": user[7],
                "state": user[13],
                "city": user[14],
                "user_zodiac_sign": user[15],
                "user_aesthetic_score": user[16],
                "incognito": bool(user[17]),
                "cognito_time": cognito_time,
                "wave_status":wave_status,
                "wavedTime": waved_time_value,
            
            }

            vibes_list.append(user[11].lower().split(',') if user[11] else [])
            genders.append(user[8] or '')
            states.append(user[13] or '')
            cities.append(user[14] or '')
            zodiac_scores.append(1 if user[15] and user[15].lower() in compatible_signs else 0)
            aesthetics_score.append(user[16] if user[16] is not None else 0)


        # --- Add current user (self) data ---
        genders.insert(0, my_gender or '')
        vibes_list.insert(0, list(my_vibes_set))
        states.insert(0, my_state or '')
        cities.insert(0, my_city or '')
        zodiac_scores.insert(0, 1)
        aesthetics_score.insert(0, my_aesthetic_score if my_aesthetic_score is not None else 0)

        my_age = self.calculate_age(my_dob)
        ages.insert(0, my_age)

        for i in range(6):
            answers_list[i].insert(0, my_answers[i])

        # --- Encoding categorical & numeric features ---
        vibes_mlb = MultiLabelBinarizer()
        vibes_encoded = vibes_mlb.fit_transform(vibes_list)

        state_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        state_encoded = state_encoder.fit_transform(np.array(states).reshape(-1, 1))

        city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        city_encoded = city_encoder.fit_transform(np.array(cities).reshape(-1, 1))

        zodiac_encoded = np.array(zodiac_scores).reshape(-1, 1)
        aesthetics_encoded = np.array(aesthetics_score).reshape(-1, 1).astype(float)

        # --- Normalize age to 0–1 scale ---
        scaler = MinMaxScaler()
        age_encoded = scaler.fit_transform(np.array(ages).reshape(-1, 1))

        # --- Encode answers (one OneHotEncoder per answer column) ---
        answer_encoded_parts = []
        for i in range(6):
            col = np.array(answers_list[i]).reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            answer_encoded_parts.append(enc.fit_transform(col))

        # Concatenate all 6 answer encodings into one block
        answers_encoded = np.concatenate(answer_encoded_parts, axis=1)


        weight_vibes = 0.3
        weight_state = 0.2
        weight_city = 0.3
        weight_zodiac = 0.3
        weight_aesthetic = 0
        weight_age = 0.25
        weight_answers = 0.40   

        # --- Final Feature Concatenation ---
        encoded_features = np.concatenate([
            vibes_encoded * weight_vibes,
            state_encoded * weight_state,
            city_encoded * weight_city,
            zodiac_encoded * weight_zodiac,
            aesthetics_encoded * weight_aesthetic,
            age_encoded * weight_age,
            answers_encoded    * weight_answers,
        ], axis=1)

        return user_map, encoded_features
    
    def _behavior_based_recommendation_near(
        self, cursor, past_sent, prioritized_users, my_aesthetic_score,
        max_users, page, per_page, user_id, place_id=None
    ):
        requested_ids = [r[0] for r in past_sent]
        format_ids = ','.join(['%s'] * len(requested_ids))

        cursor.execute(f'''
            SELECT vibes, answer1, answer2, answer3, answer4, answer5, answer6
            FROM icebreaker.profile 
            WHERE userId IN ({format_ids})''', requested_ids)
        vibes_rows = cursor.fetchall()

        behavior_vibes = set()
        behavior_answers = [set() for _ in range(6)] 
        for v in vibes_rows:
            if v[0]:
                behavior_vibes.update(v[0].lower().split(','))
            for i in range(6):
                val = v[i + 1]  # answer1 is index 1, ..., answer6 is index 6
                if val:
                    behavior_answers[i].add(str(val).strip().lower())

        # --- Fetch current user's own answers for comparison ---
        cursor.execute("""
            SELECT answer1, answer2, answer3, answer4, answer5, answer6
            FROM icebreaker.profile
            WHERE userId = %s
        """, (user_id,))
        my_ans_row = cursor.fetchone()
        my_answers = [
            (str(my_ans_row[i]).strip().lower() if my_ans_row and my_ans_row[i] else '')
            for i in range(6)
        ]
                

        scored_users = []
        for user in prioritized_users:
            uid, full_name, email = user[0], user[3], user[4]
            gender, vibes = user[8], user[11]
            pic, selfie_pic, fb_id, signup_type = user[1], user[2], user[5], user[6]
            auth_id, state, city = user[7], user[13], user[14]
            user_zodiac_sign, user_aesthetic_score = user[15], user[16]
            is_incognito = bool(user[17])

            cognito_time = None
        
            if is_incognito:
                cursor.execute(
                    "SELECT cognito_time FROM icebreaker.location WHERE userId = %s",
                    (uid,)
                )
                row = cursor.fetchone()
                if row:
                    cognito_time = row[0]

            # Vibes score
            vibe_score = 0.0
            if vibes:
                candidate_set = set(vibes.lower().split(','))
                vibe_score = len(behavior_vibes & candidate_set) / max(1, len(behavior_vibes))

            # Aesthetic score
            aesthetic_sim = 0.0
            if user_aesthetic_score is not None and my_aesthetic_score is not None:
                try:
                    diff = abs(float(user_aesthetic_score) - float(my_aesthetic_score))
                    aesthetic_sim = 1.0 - min(1.0, diff)
                except Exception:
                    pass

            # --- Answer score ---
            # Fetch this candidate's answers
            cursor.execute("""
                SELECT answer1, answer2, answer3, answer4, answer5, answer6
                FROM icebreaker.profile
                WHERE userId = %s
            """, (uid,))
            ans_row = cursor.fetchone()
            candidate_answers = [
                (str(ans_row[i]).strip().lower() if ans_row and ans_row[i] else '')
                for i in range(6)
            ]

            # Score 1: how many answers match the behavior profile (past liked users)
            behavior_answer_score = 0.0
            for i in range(6):
                if candidate_answers[i] and candidate_answers[i] in behavior_answers[i]:
                    behavior_answer_score += 1.0
            behavior_answer_score /= 6.0  # Normalize to 0–1

            # Score 2: how many answers directly match the current user's own answers
            direct_answer_score = 0.0
            for i in range(6):
                if candidate_answers[i] and my_answers[i] and candidate_answers[i] == my_answers[i]:
                    direct_answer_score += 1.0
            direct_answer_score /= 6.0  # Normalize to 0–1

            # Blend both answer signals equally
            answer_score = 0.5 * behavior_answer_score + 0.5 * direct_answer_score

            final_score = (
                0.35 * vibe_score +
                0.35 * aesthetic_sim +
                0.30 * answer_score
            )

            wave_status = self.get_wave_status_sender_near(cursor, user_id,uid)
  
            cursor.execute("""
                    SELECT created_at 
                    FROM icebreaker.user_likes_near
                    WHERE sender_id = %s 
                    AND receiver_id = %s 
                """, (user_id, uid))  # Added missing parameter
            result  = cursor.fetchone()

            if result and result[0]:
                # Convert datetime to integer timestamp
               waved_time_value  = int(result[0].timestamp())
            else:
                waved_time_value  = 0
           
            scored_users.append((uid, final_score, {
                "userId": uid,
                "fullName": full_name,
                "email": email,
                "gender": gender,
                "vibes": vibes,
                "profilePic": pic,
                "selfiePic": selfie_pic,
                "firebaseId": fb_id,
                "signupType": signup_type,
                "authenticationId": auth_id,
                "state": state,
                "city": city,
                "user_zodiac_sign": user_zodiac_sign,
                "user_aesthetic_score": user_aesthetic_score,
                "incognito": is_incognito,
                "cognito_time": cognito_time,
                "wave_status":wave_status,
                "wavedTime":waved_time_value
            }))

        sorted_scores = sorted(scored_users, key=lambda x: x[1], reverse=True)
        top_matches = sorted_scores[:max_users]
        results = [
            {**profile, "similarity_score": round(score * 100, 2)}
            for uid, score, profile in top_matches
        ]
        return {
            "Status": "Success",
            "Matches": results,
            "Page": page,
            "PerPage": per_page,
            "TotalMatches": len(sorted_scores)
        }
    
    def _prepare_encoding_data(
            self, final_row, my_gender, my_vibes_set, my_state, my_city,
            compatible_signs, my_aesthetic_score, user_id, my_dob=None, place_id=None
        ):
        user_map = {}
        genders, vibes_list, states, cities = [], [], [], []
        zodiac_scores, aesthetics_score, ages = [], [], []
        answers_list = [[] for _ in range(6)]

         # --- Fetch current user's answers ---
        cursor = self.db.cur()
        cursor.execute("""
            SELECT answer1, answer2, answer3, answer4, answer5, answer6
            FROM icebreaker.profile
            WHERE userId = %s
        """, (user_id,))
        my_answers_row = cursor.fetchone()
        my_answers = [
            (str(my_answers_row[i]).strip().lower() if my_answers_row and my_answers_row[i] else '')
            for i in range(6)
        ]
        cursor.close()

        for user in final_row:
            uid = user[0]

            cognito_time = None
           
            cursor = self.db.cur()
            cursor.execute("""
                SELECT incognito FROM icebreaker.location 
                WHERE userId = %s
            """, (uid,))
            row = cursor.fetchone()
        
            if row and row[0] == 1:
            
                cursor = self.db.cur()
                cursor.execute(
                    "SELECT cognito_time FROM icebreaker.location WHERE userId = %s",
                    (uid,)
                )
                row2 = cursor.fetchone()
                if row2:
                    cognito_time = row2[0]

            # --- Calculate user age ---
            dob = user[10]  
            age = self.calculate_age(dob)
            ages.append(age)


            # wave_status = self.get_wave_status_sender_here(cursor, user_id,uid, place_id='ChIJRVR9ciHP5zsRtrFD4nhXuBQ')
            wave_status = self.get_wave_status_sender_here(cursor, user_id, uid, place_id=place_id)
  
            cursor.execute("""
                    SELECT created_at 
                    FROM icebreaker.user_likes
                    WHERE sender_id = %s 
                    AND receiver_id = %s 
                    # AND place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
                    AND (%s IS NULL OR place_id = %s)
                """, (user_id, uid, place_id, place_id))
            result  = cursor.fetchone()

            if result and result[0]:
                # Convert datetime to integer timestamp
               waved_time_value  = int(result[0].timestamp())
            else:
                waved_time_value  = 0

            # --- Fetch this user's answers ---
            cursor.execute("""
                SELECT answer1, answer2, answer3, answer4, answer5, answer6
                FROM icebreaker.profile
                WHERE userId = %s
            """, (uid,))
            ans_row = cursor.fetchone()
            user_answers = [
                (str(ans_row[i]).strip().lower() if ans_row and ans_row[i] else '')
                for i in range(6)
            ]
            
            cursor.close()

            for i in range(6):
                answers_list[i].append(user_answers[i])

            user_map[uid] = {
                "userId": uid,
                "selfiePic":user[2],
                "fullName": user[3],
                "email": user[4],
                "gender": user[8],
                "vibes": user[11],
                "profilePic": user[1],
                "firebaseId": user[5],
                "signupType": user[6],
                "authenticationId": user[7],
                "state": user[13],
                "city": user[14],
                "user_zodiac_sign": user[15],
                "user_aesthetic_score": user[16],
                "incognito": bool(user[17]),
                "cognito_time": cognito_time,
                "wave_status":wave_status,
                "wavedTime": waved_time_value,
              
            }

            vibes_list.append(user[11].lower().split(',') if user[11] else [])
            genders.append(user[8] or '')
            states.append(user[13] or '')
            cities.append(user[14] or '')
            zodiac_scores.append(1 if user[15] and user[15].lower() in compatible_signs else 0)
            aesthetics_score.append(user[16] if user[16] is not None else 0)


        # --- Add current user (self) data ---
        genders.insert(0, my_gender or '')
        vibes_list.insert(0, list(my_vibes_set))
        states.insert(0, my_state or '')
        cities.insert(0, my_city or '')
        zodiac_scores.insert(0, 1)
        aesthetics_score.insert(0, my_aesthetic_score if my_aesthetic_score is not None else 0)

        my_age = self.calculate_age(my_dob)
        ages.insert(0, my_age)

        for i in range(6):
            answers_list[i].insert(0, my_answers[i])

        # --- Encoding categorical & numeric features ---
        vibes_mlb = MultiLabelBinarizer()
        vibes_encoded = vibes_mlb.fit_transform(vibes_list)

        state_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        state_encoded = state_encoder.fit_transform(np.array(states).reshape(-1, 1))

        city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        city_encoded = city_encoder.fit_transform(np.array(cities).reshape(-1, 1))

        zodiac_encoded = np.array(zodiac_scores).reshape(-1, 1)
        aesthetics_encoded = np.array(aesthetics_score).reshape(-1, 1).astype(float)

        # --- Normalize age to 0–1 scale ---
        scaler = MinMaxScaler()
        age_encoded = scaler.fit_transform(np.array(ages).reshape(-1, 1))

        # --- Encode answers (one OneHotEncoder per answer column) ---
        answer_encoded_parts = []
        for i in range(6):
            col = np.array(answers_list[i]).reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            answer_encoded_parts.append(enc.fit_transform(col))

        # Concatenate all 6 answer encodings into one block
        answers_encoded = np.concatenate(answer_encoded_parts, axis=1)

        # --- Feature Weights ---
        weight_vibes = 0.3
        weight_state = 0.2
        weight_city = 0.3
        weight_zodiac = 0.3
        weight_aesthetic = 0
        weight_age = 0.25
        weight_answers = 0.40   

        # --- Final Feature Concatenation ---
        encoded_features = np.concatenate([
            vibes_encoded * weight_vibes,
            state_encoded * weight_state,
            city_encoded * weight_city,
            zodiac_encoded * weight_zodiac,
            aesthetics_encoded * weight_aesthetic,
            age_encoded * weight_age,
            answers_encoded    * weight_answers,
        ], axis=1)

        return user_map, encoded_features

    def _behavior_based_recommendation(
        self, cursor, past_sent, prioritized_users, my_aesthetic_score,
        max_users, page, per_page, user_id, place_id=None
    ):
        requested_ids = [r[0] for r in past_sent]
        format_ids = ','.join(['%s'] * len(requested_ids))

        cursor.execute(f'''
            SELECT vibes, answer1, answer2, answer3, answer4, answer5, answer6
            FROM icebreaker.profile 
            WHERE userId IN ({format_ids})''', requested_ids)
        vibes_rows = cursor.fetchall()

        behavior_vibes = set()
        behavior_answers = [set() for _ in range(6)] 
        for v in vibes_rows:
            if v[0]:
                behavior_vibes.update(v[0].lower().split(','))
            for i in range(6):
                val = v[i + 1]  # answer1 is index 1, ..., answer6 is index 6
                if val:
                    behavior_answers[i].add(str(val).strip().lower())

        # --- Fetch current user's own answers for comparison ---
        cursor.execute("""
            SELECT answer1, answer2, answer3, answer4, answer5, answer6
            FROM icebreaker.profile
            WHERE userId = %s
        """, (user_id,))
        my_ans_row = cursor.fetchone()
        my_answers = [
            (str(my_ans_row[i]).strip().lower() if my_ans_row and my_ans_row[i] else '')
            for i in range(6)
        ]

        scored_users = []
        for user in prioritized_users:
            uid, full_name, email = user[0], user[3], user[4]
            gender, vibes = user[8], user[11]
            pic, selfie_pic, fb_id, signup_type = user[1], user[2], user[5], user[6]
            auth_id, state, city = user[7], user[13], user[14]
            user_zodiac_sign, user_aesthetic_score = user[15], user[16]
            is_incognito = bool(user[17])

            cognito_time = None
        
            if is_incognito:
                cursor.execute(
                    "SELECT cognito_time FROM icebreaker.location WHERE userId = %s",
                    (uid,)
                )
                row = cursor.fetchone()
                if row:
                    cognito_time = row[0]

            # Vibes score
            vibe_score = 0.0
            if vibes:
                candidate_set = set(vibes.lower().split(','))
                vibe_score = len(behavior_vibes & candidate_set) / max(1, len(behavior_vibes))

            # Aesthetic score
            aesthetic_sim = 0.0
            if user_aesthetic_score is not None and my_aesthetic_score is not None:
                try:
                    diff = abs(float(user_aesthetic_score) - float(my_aesthetic_score))
                    aesthetic_sim = 1.0 - min(1.0, diff)
                except Exception:
                    pass

            # --- Answer score ---
            # Fetch this candidate's answers
            cursor.execute("""
                SELECT answer1, answer2, answer3, answer4, answer5, answer6
                FROM icebreaker.profile
                WHERE userId = %s
            """, (uid,))
            ans_row = cursor.fetchone()
            candidate_answers = [
                (str(ans_row[i]).strip().lower() if ans_row and ans_row[i] else '')
                for i in range(6)
            ]

            # Score 1: how many answers match the behavior profile (past liked users)
            behavior_answer_score = 0.0
            for i in range(6):
                if candidate_answers[i] and candidate_answers[i] in behavior_answers[i]:
                    behavior_answer_score += 1.0
            behavior_answer_score /= 6.0  # Normalize to 0–1

            # Score 2: how many answers directly match the current user's own answers
            direct_answer_score = 0.0
            for i in range(6):
                if candidate_answers[i] and my_answers[i] and candidate_answers[i] == my_answers[i]:
                    direct_answer_score += 1.0
            direct_answer_score /= 6.0  # Normalize to 0–1

            # Blend both answer signals equally
            answer_score = 0.5 * behavior_answer_score + 0.5 * direct_answer_score


            final_score = (
                0.35 * vibe_score +
                0.35 * aesthetic_sim +
                0.30 * answer_score
            )

            # wave_status = self.get_wave_status_sender_here(cursor, user_id,uid, place_id='ChIJRVR9ciHP5zsRtrFD4nhXuBQ')
            wave_status = self.get_wave_status_sender_here(cursor, user_id, uid, place_id=place_id)
  
            cursor.execute("""
                    SELECT created_at 
                    FROM icebreaker.user_likes
                    WHERE sender_id = %s 
                    AND receiver_id = %s 
                    # AND place_id = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
                    AND (%s IS NULL OR place_id = %s)
                """, (user_id, uid, place_id, place_id))
            result  = cursor.fetchone()

            if result and result[0]:
                # Convert datetime to integer timestamp
               waved_time_value  = int(result[0].timestamp())
            else:
                waved_time_value  = 0

            
            
            scored_users.append((uid, final_score, {
                "userId": uid,
                "fullName": full_name,
                "email": email,
                "gender": gender,
                "vibes": vibes,
                "profilePic": pic,
                "selfiePic": selfie_pic,
                "firebaseId": fb_id,
                "signupType": signup_type,
                "authenticationId": auth_id,
                "state": state,
                "city": city,
                "user_zodiac_sign": user_zodiac_sign,
                "user_aesthetic_score": user_aesthetic_score,
                "incognito": is_incognito,
                "cognito_time": cognito_time,
                "wave_status":wave_status,
                "wavedTime":waved_time_value
            }))

        sorted_scores = sorted(scored_users, key=lambda x: x[1], reverse=True)
        top_matches = sorted_scores[:max_users]
        results = [
            {**profile, "similarity_score": round(score * 100, 2)}
            for uid, score, profile in top_matches
        ]
        return {
            "Status": "Success",
            "Matches": results,
            "Page": page,
            "PerPage": per_page,
            "TotalMatches": len(sorted_scores)
        }

        
    def get_country_list(self):
        try:
            data = request.get_json()
            country = data['country']
            cursor = self.db.cur()

            cursor.execute("""
                SELECT country FROM icebreaker.Country WHERE country LIKE %s
            """, 
            ('%' + country + '%',))
            result = cursor.fetchall()
            print(result)
            country_list = [row[0] for row in result]  # Extract country names
            country_str = "" + ", ".join(country_list)      
            cursor.close()     
            
            return jsonify({'Status': 'Success', 'countries': country_str})

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})


    def get_city_list(self):
        try:
            data = request.get_json()
            country = data['country']
            state = data['state']
            city = data['city']
            cursor = self.db.cur()
            

            if len(city) < 3:
                return {"error": "Enter at least 3 characters for city field"}, 400
            elif len(state) < 4:
                return {"error": "Enter at least 4 characters for state field"}, 400
            else:
                # Initialize OpenAI client
                  # === Load credentials from JSON file ===
                aws_config = self._load_vibr_config()
                openai.api_key = aws_config["openai_api_key"]
                if not openai.api_key:
                    return {"error": "API key not found"}, 500
                

                # Create a prompt for OpenAI
                prompt = f"""
                Given the partial city '{city}', generate a list upto 5 cities (strictly actual cities, no fictional names or random words) that start with or closely match this input and strictly present within the state of '{state}' within '{country}'. 
                For example, if the input is 'San', and the user's state is California, country is United State of America, then the result shall be:
                - San Francisco
                Example 2, if the input is 'Bhavn', and the user's state is Gujarat, country is India, then the result shall be:
                - Bhavnagar
                Your focus should be finding cities which are STRICTLY present only in the input {country}' and '{state}'. If there no such city, return nothing.
                Return the response as a valid JSON array of strings, e.g., ["Title1", "Title2", ...].
                Return the response as a valid JSON array of strings, with no additional text, explanation, or formatting outside the JSON array.
                """

                # Call OpenAI API
                intent_response = openai.ChatCompletion.create( 
                model="gpt-4-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates professional job titles."},
                        {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.5,
                n=1
            )

                # Extract the recommendations
                recommendations = intent_response.choices[0].message.content
                cursor.close()

                # Parse the response as JSON
                try:
                    recommendations_list = json.loads(recommendations)
                    if not isinstance(recommendations_list, list):
                        raise ValueError("Response is not a list")
                    return {"recommendations": recommendations_list}, 200
                except ValueError:
                    return {"error": "Failed to parse recommendations"}, 500

        except openai.OpenAIError:
            return {"error": "Failed to generate recommendations due to API error"}, 500
        except Exception as e:       
            return {"error": str(e)}, 500


        # fetch state
    def get_state_list(self):
        try:
            data = request.get_json()
            country = data['country']
            state = data['state']
            cursor = self.db.cur()
                    

            if len(state) < 3:
                return {"error": "Enter at least 3 characters"}, 400
            else:
                # Initialize OpenAI clien
                  # === Load credentials from JSON file ===
                aws_config = self._load_vibr_config()
                openai.api_key = aws_config["openai_api_key"]
                if not openai.api_key:
                    return {"error": "API key not found"}, 500
                

                # Create a prompt for OpenAI
                prompt = f"""
                Given the partial state/province '{state}', generate a list upto 3 state/province (strictly actual states or provinces, no fictional names or random words) that start with or closely match this input and strictly present within '{country}'. 
                For example, if the input is 'Maha', and the user's country is India, then the result shall be:
                - Maharashtra
                Return the response as a valid JSON array of strings, e.g., ["Title1", "Title2", ...].
                Return the response as a valid JSON array of strings, with no additional text, explanation, or formatting outside the JSON array.
                """

                # Call OpenAI API
                intent_response = openai.ChatCompletion.create( 
                model="gpt-4-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates professional job titles."},
                        {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.5,
                n=1
            )

                # Extract the recommendations
                recommendations = intent_response.choices[0].message.content
                cursor.close()

                # Parse the response as JSON
                try:
                    recommendations_list = json.loads(recommendations)
                    if not isinstance(recommendations_list, list):
                        raise ValueError("Response is not a list")
                    return {"recommendations": recommendations_list}, 200
                except ValueError:
                    return {"error": "Failed to parse recommendations"}, 500

        except openai.OpenAIError:
            return {"error": "Failed to generate recommendations due to API error"}, 500
        except Exception as e:       
            return {"error": str(e)}, 500

    
    def find_compatible_signs(self, user_zodiac_sign):
        # Define compatibility based on traditional astrology
        if not user_zodiac_sign:
            user_zodiac_sign = 'cancer'
        compatibility = {
            'aries': ['leo', 'sagittarius', 'gemini', 'aquarius'],
            'taurus': ['virgo', 'capricorn', 'cancer', 'pisces'],
            'gemini': ['aries', 'libra', 'aquarius', 'leo'],
            'cancer': ['taurus', 'scorpio', 'pisces', 'virgo'],
            'leo': ['aries', 'sagittarius', 'gemini', 'libra'],
            'virgo': ['taurus', 'capricorn', 'cancer', 'scorpio'],
            'libra': ['gemini', 'aquarius', 'leo', 'sagittarius'],
            'scorpio': ['cancer', 'pisces', 'virgo', 'capricorn'],
            'sagittarius': ['aries', 'leo', 'libra', 'aquarius'],
            'capricorn': ['taurus', 'virgo', 'scorpio', 'pisces'],
            'aquarius': ['gemini', 'libra', 'aries', 'sagittarius'],
            'pisces': ['cancer', 'scorpio', 'taurus', 'capricorn']
        }
        
        user_zodiac_sign = user_zodiac_sign.lower()
        if user_zodiac_sign not in compatibility:
            return []
        
        return compatibility[user_zodiac_sign]


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



    def user_inactivity_checkout(self):
        try:
            data = request.get_json()
            userId = data['userId']
            cursor = self.db.cur()
            msg = ''
            isCheckIn = True

            # If user is in location, update last_active
            cursor.execute("""
                SELECT userId FROM icebreaker.location WHERE userId = %s
            """, (userId,))
            if cursor.fetchone():
                cursor.execute("""
                    UPDATE icebreaker.location SET last_active = NOW() WHERE userId = %s
                """, (userId,))
                msg = 'Activity refreshed in location'
                isCheckIn = True
            else:
                cursor.execute("""
                    SELECT userId FROM icebreaker.radius WHERE userId = %s
                """, (userId,))
                if cursor.fetchone():
                    cursor.execute("""
                        UPDATE icebreaker.radius SET last_active = NOW() WHERE userId = %s
                    """, (userId,))
                    msg = 'Activity refreshed in radius'
                    isCheckIn = True
                else:
                    msg = 'User is not checked-in at any place'
                    isCheckIn = False

            self.db.conn.commit()
            cursor.close()

            return {'Status': 'Success', 'Message': msg, 'isCheckIn': isCheckIn}

        except Exception as e:
            self.db.conn.rollback()
            return {'Status': 'Failure', 'Error': str(e)}

    def update_user_radius_dynamic(self):
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            old_radius = float(data['old_input_radius'])
            new_radius = float(data['new_input_radius'])

            cursor = self.db.cur()

            # Fetch user info for notifications
            cursor.execute("""
                SELECT fullName, p.currentDesignation, loginWebToken, preference
                FROM icebreaker.registration r
                JOIN icebreaker.profile p ON r.userId = p.userId
                WHERE r.userId = %s
            """, (user_id,))
            row = cursor.fetchone()
            if not row:
                return jsonify({'status': 'Error', 'message': 'User not found'})
            full_name, current_designation, login_web_token, preference = row
            gender_filter = self._map_preference_to_gender_data(preference)

            # Update user's location
            cursor.execute("""
                UPDATE icebreaker.radius
                SET latitude = %s, longitude = %s, radius = %s
                WHERE userId = %s
            """, (lat, lon, new_radius, user_id))
            self.db.conn.commit()

            # Helper function to get users in a radius
            def get_users_in_radius(radius, gender=None):
                min_lat, max_lat, min_lon, max_lon = self.get_bounding_box(lat, lon, radius)
                if gender:
                     cursor.execute("""
                        SELECT p.userId, l.placeId, 
                            p.profilePic,
                            l.latitude, l.longitude
                        FROM icebreaker.profile p
                        JOIN icebreaker.registration r ON p.userId = r.userId
                        JOIN icebreaker.radius l ON p.userId = l.userId
                        LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s
                        LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s
                        WHERE p.userId != %s
                        AND l.latitude BETWEEN %s AND %s
                        AND l.longitude BETWEEN %s AND %s
                        AND (
                            3959 * acos(
                                cos(radians(%s)) * cos(radians(l.latitude)) *
                                cos(radians(l.longitude) - radians(%s)) +
                                sin(radians(%s)) * sin(radians(l.latitude))
                            )
                        ) <= LEAST(l.radius, %s)
                        AND b1.blockedId IS NULL
                        AND b2.blockedId IS NULL
                        AND (%s IS NULL OR r.Gender = %s)
                    """, (user_id, user_id, user_id, min_lat, max_lat, min_lon, max_lon, lat, lon, lat, new_radius, gender_filter, gender_filter))
                else:
                     cursor.execute("""
                        SELECT p.userId, l.placeId, 
                            p.profilePic 
                            , l.latitude, l.longitude
                        FROM icebreaker.profile p
                        JOIN icebreaker.radius l ON p.userId = l.userId
                        LEFT JOIN icebreaker.blockedUsers b1 ON l.userId = b1.blockedId AND b1.userId = %s
                        LEFT JOIN icebreaker.blockedUsers b2 ON l.userId = b2.userId AND b2.blockedId = %s
                        WHERE p.userId != %s
                        AND l.latitude BETWEEN %s AND %s
                        AND l.longitude BETWEEN %s AND %s
                        AND (
                            3959 * acos(
                                cos(radians(%s)) * cos(radians(l.latitude)) *
                                cos(radians(l.longitude) - radians(%s)) +
                                sin(radians(%s)) * sin(radians(l.latitude))
                            )
                        ) <= LEAST(l.radius, %s)
                        AND b1.blockedId IS NULL
                        AND b2.blockedId IS NULL
                       
                    """, (user_id, user_id, user_id, min_lat, max_lat, min_lon, max_lon, lat, lon, lat, new_radius))
                return [row[0] for row in cursor.fetchall()]

            users_in_old_radius = get_users_in_radius(old_radius, gender=gender_filter)
            users_in_new_radius = get_users_in_radius(new_radius, gender=gender_filter)

            print("Users in OLD radius:", users_in_old_radius)
            print("Users in NEW radius:", users_in_new_radius)

            # Determine users to notify
            if new_radius > old_radius:
                users_to_notify = list(set(users_in_new_radius) - set(users_in_old_radius))
                notification_func = self.send_fcm_with_rest_new_user
            else:
                users_to_notify = list(set(users_in_old_radius) - set(users_in_new_radius))
                notification_func = self.send_fcm_with_rest_logout_notification

            print("Users to notify:", users_to_notify)

            # Send notifications
            if users_to_notify:
                placeholders = ','.join(['%s'] * len(users_to_notify))
                cursor.execute(f"""
                    SELECT deviceToken FROM icebreaker.registration
                    WHERE userId IN ({placeholders}) AND deviceToken IS NOT NULL AND deviceToken != ''
                """, tuple(users_to_notify))
                tokens = [t[0] for t in cursor.fetchall()]

                for token in tokens:
                    threading.Thread(
                        target=notification_func,
                        args=(token, full_name)
                    ).start()

            return jsonify({
                'status': 'Success',
                'message': f'Radius updated to {new_radius} miles. {"Notified " + str(len(users_to_notify)) + " users." if users_to_notify else ""}',
                'loginWebToken': login_web_token
            })

        except Exception as e:
            print(f"Error in update_user_radius_dynamic: {e}")
            return jsonify({'status': 'Error', 'message': str(e)})


    def send_fcm_with_rest_new_user_dummysilent(self, data):
       
        token = data.get("token")
        refresh_type = "refresh_all_users"  # updated payload
        payload_data = {"type": refresh_type}  # data-only payload

        if not token:
            return {"status": "error", "message": "No device token provided."}

        try:
            # SERVICE_ACCOUNT_FILE = r"D:\icebreakerNew\vibrr\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE]
            )
            auth_request = Request()
            credentials.refresh(auth_request)

            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            def send_payload(t):
                payload = {
                    "message": {
                        "token": t,
                        "data": payload_data,  # data-only, silent
                        "android": {
                            "priority": "high"  # ensures timely delivery on Android
                        },
                        "apns": {
                            "headers": {
                                "apns-priority": "10"  # immediate delivery on iOS
                            },
                            "payload": {
                                "aps": {
                                    "content-available": 1  # silent notification
                                }
                            }
                        }
                    }
                }
                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                return {"status_code": resp.status_code, "response": resp.text}

            # Handle single token or list
            if isinstance(token, list):
                responses = [send_payload(t) for t in token]
                return {"status": "success", "messages": responses}
            else:
                resp = send_payload(token)
                return {"status": "success", "message": resp}

        except Exception as e:
            return {"status": "error", "message": str(e)}


    def make_db_entry_of_wavesnotification(self, login_id, waved_to_user_id, wave_type, notification_body, profile_pic):
        try:
            current_time = datetime.now(timezone.utc)
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            notification_time = formatted_time
            cursor = self.db.cur()
            # Convert sets to single values if needed
            if isinstance(login_id, set):
                login_id = next(iter(login_id))  

            if isinstance(waved_to_user_id, set):
                waved_to_user_id = next(iter(waved_to_user_id))  

            # Insert the details of the notification sent into the notifications table
            cursor.execute('''INSERT INTO icebreaker.notifications (userId, requestedUserId, notificationTime, requestStatus, description, profilePic)
                            VALUES (%s, %s, %s, %s, %s, %s)''',
                            (waved_to_user_id, login_id, notification_time, wave_type, notification_body, profile_pic))
            self.db.conn.commit()
            message = "Notification details inserted successfully in database"

            return {"status": 'success', "message": message}

        except Exception as e:
            print(str(e))
            message = str(e)
            return {"status": 'failure', "message": message}

    def handle_send_wave_new(self):

        data = request.get_json()
        loginUserId = data['loginUserId']          
        waved_to_id = data['wavedToUserId']     
        status = data.get('status', 'wave')
        # place_id_static = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
        # place_id = place_id_static if place_id_real else 'None'

        cursor = self.db.cur()
        place_id = self._resolve_place_id(loginUserId, data, cursor)

        now = datetime.now()

        cursor.execute('''
            SELECT r.deviceToken, r.fullName, p.profilePic
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s
        ''', (loginUserId,))
        sender_info = cursor.fetchone()
        full_name = sender_info[1].title()

        # Check if like already exists
        cursor.execute("""
            SELECT id
            FROM user_likes
            WHERE sender_id = %s
            AND receiver_id = %s
            LIMIT 1
        """, (loginUserId, waved_to_id))

        existing_like = cursor.fetchone()

        if existing_like:
            # ---- UNLIKE ----
            cursor.execute("""
                DELETE FROM user_likes
                WHERE sender_id = %s
                AND receiver_id = %s
            """, (loginUserId, waved_to_id))

            cursor.execute("""
                DELETE FROM user_likes_near
                WHERE sender_id = %s
                AND receiver_id = %s
               
            """, (loginUserId, waved_to_id))

            # if place_id != "null":
            cursor.execute('''DELETE FROM icebreaker.session_likes
                            WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                        (loginUserId, waved_to_id))
            
            cursor.execute('''DELETE FROM icebreaker.session_likes_near
                            WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                        (loginUserId, waved_to_id))
                    
            cursor.execute('''DELETE FROM icebreaker.notifications
                                WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'wave_received' ''',
                            (loginUserId, waved_to_id))

            cursor.execute('''
                SELECT r.deviceToken, p.profilePic
                FROM icebreaker.registration r
                JOIN icebreaker.profile p ON r.userId = p.userId
                WHERE r.userId = %s
            ''', (waved_to_id,))
            receiver_info = cursor.fetchone()

            if receiver_info and receiver_info[0]:
                receiver_token = receiver_info[0]
              

                self.send_fcm_wave_deleted_silent(
                    token=receiver_token,
                    login_id=loginUserId,
                    waved_to_user_id=waved_to_id
                )

            message = f"User {loginUserId} unliked User {waved_to_id}"


        else:
            # ---- LIKE ----
            cursor.execute("""
                INSERT INTO user_likes (sender_id, receiver_id, place_id, created_at)
                VALUES (%s, %s, %s, %s)
            """, (loginUserId, waved_to_id, place_id, now))

            self.on_wave_sent(cursor, loginUserId, waved_to_id)

            if place_id != "null":
                cursor.execute('''INSERT INTO icebreaker.session_likes (user_id, wavedToUserId, place_id, like_type, wavedTime)
                                VALUES (%s, %s, %s, 'wave', %s)''',
                            (loginUserId, waved_to_id, place_id, now))

            if status == "waveback":
                notification_title = "Like Back"
                notification_body = f"{full_name} liked you back"
                wave_type = "like_back_received"
            else:
                notification_title = "Like Received"
                notification_body = f"{full_name} liked you"
                wave_type = "Like_received"

            cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (waved_to_id,))
            tokens = cursor.fetchone()

            if tokens and tokens[2] == 1:
                receiver_token = tokens[0]

                if isinstance(receiver_token, list):
                    receiver_token = [
                        t for t in receiver_token if t != sender_info[0]
                    ]

                self.send_fcm_with_wave_notification_new(
                    receiver_token,
                    sender_info[1],   # full name
                    loginUserId,        # sender ID
                    waved_to_id,      # receiver ID
                    sender_info[2],   # profile pic
                    notification_title,
                    notification_body,
                    wave_type
                )

            message = f"User {loginUserId} likes User {waved_to_id}"

        self.db.conn.commit()
        self.db.conn.close()

        return jsonify({
            "status": "Success",
            "message": message
        })
    
    def handle_send_wave_near(self):

        data = request.get_json()
        loginUserId = data['loginUserId']          # user who is waving
        waved_to_id = data['wavedToUserId']     # user who is being waved to
        status = data.get('status', 'wave')  

        cursor = self.db.cur()

        now = datetime.now()

        cursor.execute('''
            SELECT r.deviceToken, r.fullName, p.profilePic
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s
        ''', (loginUserId,))
        sender_info = cursor.fetchone()
        full_name = sender_info[1].title()

        # Check if like already exists
        cursor.execute("""
            SELECT id
            FROM user_likes_near
            WHERE sender_id = %s
            AND receiver_id = %s
            LIMIT 1
        """, (loginUserId, waved_to_id))

        existing_like = cursor.fetchone()

        if existing_like:
            # ---- UNLIKE ----
            cursor.execute("""
                DELETE FROM user_likes_near
                WHERE sender_id = %s
                AND receiver_id = %s
            
            """, (loginUserId, waved_to_id))

            cursor.execute("""
                DELETE FROM user_likes
                WHERE sender_id = %s
                AND receiver_id = %s
            
            """, (loginUserId, waved_to_id))

           
            cursor.execute('''DELETE FROM icebreaker.session_likes_near
                            WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                        (loginUserId, waved_to_id))

            cursor.execute('''DELETE FROM icebreaker.session_likes
                            WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                        (loginUserId, waved_to_id))
                    
            cursor.execute('''DELETE FROM icebreaker.notifications
                                WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'wave_received' ''',
                            (loginUserId, waved_to_id))
            
            cursor.execute('''
                SELECT r.deviceToken, p.profilePic
                FROM icebreaker.registration r
                JOIN icebreaker.profile p ON r.userId = p.userId
                WHERE r.userId = %s
            ''', (waved_to_id,))
            receiver_info = cursor.fetchone()

            if receiver_info and receiver_info[0]:
                receiver_token = receiver_info[0]
              

                self.send_fcm_wave_deleted_silent(
                    token=receiver_token,
                    login_id=loginUserId,
                    waved_to_user_id=waved_to_id
                )

            message = f"User {loginUserId} unliked User {waved_to_id}"


        else:
            # ---- LIKE ----
            cursor.execute("""
                INSERT INTO user_likes_near (sender_id, receiver_id, created_at)
                VALUES (%s, %s, %s)
            """, (loginUserId, waved_to_id, now))
            print("Inserted into user_likes_near")

            self.on_wave_sent(cursor, loginUserId, waved_to_id)  # target_id = receiver_id

            cursor.execute('''INSERT INTO icebreaker.session_likes_near (user_id, wavedToUserId, like_type, wavedTime)
                            VALUES (%s, %s, 'wave', %s)''',
                        (loginUserId, waved_to_id, now))

            if status == "waveback":
                notification_title = "Like Back"
                notification_body = f"{full_name} liked you back"
                wave_type = "like_back_received"
            else:
                notification_title = "Like Received"
                notification_body = f"{full_name} liked you"
                wave_type = "Like_received"

            cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (waved_to_id,))
            tokens = cursor.fetchone()

            if tokens and tokens[2] == 1:
                receiver_token = tokens[0]

                if isinstance(receiver_token, list):
                    receiver_token = [
                        t for t in receiver_token if t != sender_info[0]
                    ]

                self.send_fcm_with_wave_notification_new(
                    receiver_token,
                    sender_info[1],   # full name
                    loginUserId,        # sender ID
                    waved_to_id,      # receiver ID
                    sender_info[2],   # profile pic
                    notification_title,
                    notification_body,
                    wave_type
                )

            message = f"User {loginUserId} likes User {waved_to_id}"

        self.db.conn.commit()
        self.db.conn.close()

        return jsonify({
            "status": "Success",
            "message": message
        })
    
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
    
    def handle_send_wave_new_dummy(self):
        data = request.get_json()
        login_id = data['loginUserId']          
        waved_to_id = data['wavedToUserId']     
        status = data.get('status', 'wave')
        # place_id_static = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
        # place_id = place_id_static if place_id_real else 'None'

        cursor = self.db.cur()
        place_id = self._resolve_place_id(login_id, data, cursor)
        wave_time = datetime.now()

        # --- Fetch sender info ---
        cursor.execute('''
            SELECT r.deviceToken, r.fullName, p.profilePic
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s
        ''', (login_id,))
        sender_info = cursor.fetchone()
        full_name = sender_info[1].title()
        new_wave_inserted = False


        # Decide target table based on wave type
        if status == 'wave':
            # --- Check all tables first ---
            cursor.execute('''SELECT waveId FROM icebreaker.wave
                            WHERE userId = %s AND wavedToUserId = %s AND placeID = %s''',
                        (login_id, waved_to_id, place_id))
            existing_wave = cursor.fetchone()

            cursor.execute('''SELECT waveId FROM icebreaker.waveback
                            WHERE userId = %s AND wavedToUserId = %s AND placeID = %s''',
                        (login_id, waved_to_id, place_id))
            existing_waveback = cursor.fetchone()

            # cursor.execute('''SELECT waveId FROM icebreaker.waveNear
            #                 WHERE userId = %s AND wavedToUserId = %s''',
            #             (login_id, waved_to_id))
            # existing_wave_near = cursor.fetchone()

            # cursor.execute('''SELECT waveId FROM icebreaker.wavebackNear
            #                 WHERE userId = %s AND wavedToUserId = %s''',
            #             (login_id, waved_to_id))
            # existing_waveback_near = cursor.fetchone()

            actions_taken = []

            # --- Handle wave ---
            if existing_wave :
                cursor.execute('''DELETE FROM icebreaker.wave
                                WHERE userId = %s AND wavedToUserId = %s AND placeID = %s''',
                            (login_id, waved_to_id, place_id))
                cursor.execute('''DELETE FROM icebreaker.waveNear
                                WHERE userId = %s AND wavedToUserId = %s''',
                            (login_id, waved_to_id))
                if place_id != "null":
                    cursor.execute('''DELETE FROM icebreaker.session_likes
                                    WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                                (login_id, waved_to_id))
                    cursor.execute('''DELETE FROM icebreaker.session_likes_near
                                    WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                                (login_id, waved_to_id))
                cursor.execute('''DELETE FROM icebreaker.notifications
                                WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'wave_received' ''',
                            (waved_to_id, login_id))
                actions_taken.append("Wave removed")

            # --- Handle waveback ---
            if existing_waveback:

                cursor.execute('''SELECT placeID, wavedTime FROM icebreaker.waveback
                            WHERE userId = %s AND wavedToUserId = %s 
                            ORDER BY wavedTime DESC
                            LIMIT 1''',
                        (login_id, waved_to_id))
                row = cursor.fetchone()

                if row:
                    original_place_id, original_waved_time = row

                cursor.execute('''DELETE FROM icebreaker.waveback
                                WHERE ((userId = %s AND wavedToUserId = %s) OR (userId = %s AND wavedToUserId = %s))
                                AND placeID = %s''',
                            (login_id, waved_to_id, waved_to_id, login_id, place_id))
                
                waveback_deleted = cursor.rowcount
                if waveback_deleted > 0:
                    if place_id != "null":
                        cursor.execute('''DELETE FROM icebreaker.session_likes
                                        WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                                    (login_id, waved_to_id))
                    cursor.execute('''DELETE FROM icebreaker.notifications
                                    WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'waveback_received' ''',
                                (waved_to_id, login_id))
                    # Insert wave as a waveback conversion
                    cursor.execute('''INSERT INTO icebreaker.wave (userId, wavedToUserId, wavedTime, placeID, status)
                                    VALUES (%s, %s, %s, %s, 'wave')''',
                                (waved_to_id, login_id, original_waved_time, original_place_id))
                    actions_taken.append("Waveback converted to wave")

            # --- Handle wavebackNear ---
            # if existing_waveback_near:
                cursor.execute('''DELETE FROM icebreaker.wavebackNear
                                WHERE ((userId = %s AND wavedToUserId = %s) OR (userId = %s AND wavedToUserId = %s))
                                AND placeID = %s''',
                            (login_id, waved_to_id, waved_to_id, login_id, place_id))
                
                waveback_near_deleted = cursor.rowcount 

                if waveback_near_deleted > 0:

                    if place_id != "null":
                        cursor.execute('''DELETE FROM icebreaker.session_likes_near
                                        WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                                    (login_id, waved_to_id))
                    cursor.execute('''DELETE FROM icebreaker.notifications
                                    WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'waveback_received' ''',
                                (waved_to_id, login_id))
                    # Insert waveNear as wavebackNear conversion
                    cursor.execute('''INSERT INTO icebreaker.waveNear (userId, wavedToUserId, wavedTime, placeID, status)
                                    VALUES (%s, %s, %s, %s, 'wave')''',
                                (waved_to_id, login_id, wave_time, place_id))
                    actions_taken.append("WavebackNear converted to waveNear")

            # --- Insert new wave if none existed ---
            if not any([existing_wave, existing_waveback]):
                if place_id != "null":
                    cursor.execute('''INSERT INTO icebreaker.session_likes (user_id, wavedToUserId, place_id, like_type, wavedTime)
                                    VALUES (%s, %s, %s, 'wave', %s)''',
                                (login_id, waved_to_id, place_id, wave_time))
                cursor.execute('''INSERT INTO icebreaker.wave (userId, wavedToUserId, wavedTime, placeID, status)
                                VALUES (%s, %s, %s, %s, 'wave')''',
                            (login_id, waved_to_id, wave_time, place_id))
                actions_taken.append("Wave inserted")
                new_wave_inserted = True

            self.db.conn.commit()

            if new_wave_inserted:
                notification_title = "Like Received"
                notification_body = f"{full_name} liked you"
                wave_type = "wave_received"

                cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (waved_to_id,))
                tokens = cursor.fetchone()

                # --- Send notification if enabled ---
                if tokens and tokens[2] == 1:
                    receiver_token = tokens[0]
                    if isinstance(receiver_token, list):
                        receiver_token = [t for t in receiver_token if t != sender_info[0]]

                    self.send_fcm_with_wave_notification_new(
                        receiver_token,
                        sender_info[1],     # full name
                        login_id,           # sender ID
                        waved_to_id,        # receiver ID
                        sender_info[2],     # profile pic
                        notification_title,
                        notification_body,
                        wave_type
                    )

                # Close connection
                self.db.conn.close()
            

            return jsonify({"message": ", ".join(actions_taken), "status": "Success"})


        elif status == 'waveback':
            # --- Check if already waved back ---
            cursor.execute('''
                SELECT waveId FROM icebreaker.waveback
                WHERE userId = %s AND wavedToUserId = %s AND placeId = %s
            ''', (login_id, waved_to_id, place_id))
            existing_waveback = cursor.fetchone()

            if existing_waveback:
                message = "User has already waved back to this user"
                return jsonify({"message": message, "status": 'Success'})

            cursor.execute('''
                DELETE FROM icebreaker.wave 
                WHERE userId = %s AND wavedToUserId = %s AND placeId = %s
            ''', (waved_to_id,login_id, place_id))

            # --- Insert into waveback table ---
            cursor.execute('''
                INSERT INTO icebreaker.waveback (userId, wavedToUserId, wavedTime, placeId, status)
                VALUES (%s, %s, %s, %s, %s)
            ''', (login_id, waved_to_id, wave_time, place_id, 'waveback'))
            cursor.execute('''
                INSERT INTO icebreaker.waveback (userId, wavedToUserId, wavedTime, placeId, status)
                VALUES (%s, %s, %s, %s, %s)
            ''', (waved_to_id,login_id, wave_time, place_id, 'waveback'))
            notification_title = "Like Back "
            notification_body = f" {full_name} liked you back"
            wave_type = "waveback_received"

            if place_id:       
                cursor.execute('''
                    INSERT INTO icebreaker.session_likes (user_id, wavedTouserId, place_id, like_type, wavedTime)
                    VALUES (%s, %s, %s, 'wave', %s)
                ''', (login_id, waved_to_id, place_id, wave_time))
            # print(notification_body)

        else:
            return jsonify({"message": "Invalid status. Must be 'wave' or 'waveback'.", "status": "Error"})

        # Commit DB insert
        self.db.conn.commit()

        

        # --- Fetch receiver’s notification preferences ---
        cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (waved_to_id,))
        tokens = cursor.fetchone()

        # --- Send notification if enabled ---
        if tokens and tokens[2] == 1:
            receiver_token = tokens[0]
            if isinstance(receiver_token, list):
                receiver_token = [t for t in receiver_token if t != sender_info[0]]

            self.send_fcm_with_wave_notification_new(
                receiver_token,
                sender_info[1],     # full name
                login_id,           # sender ID
                waved_to_id,        # receiver ID
                sender_info[2],     # profile pic
                notification_title,
                notification_body,
                wave_type
            )

        # Close connection
        self.db.conn.close()

        message = f"User {login_id} performed '{status}' action to user {waved_to_id}."
        return jsonify({"message": message, "status": 'Success'})

    def send_fcm_wave_deleted_silent(self, token, login_id, waved_to_user_id):
        """Send a silent, data-only FCM notification when a wave is deleted."""
        if not token:
            return {"status": "error", "message": "No device token provided."}

        try:
            # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE,
                scopes=[self.CLOUD_PLATFORM_SCOPE]
            )
            credentials.refresh(Request())

            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": self.CONTENT_TYPE,
            }

            payload_data = {
                "type": "like_removed",
                "user_id": str(login_id),
                "waved_to_user_id": str(waved_to_user_id),
            }

            def build_message(t: str) -> dict:
                return {
                    "message": {
                        "token": t,
                        "data": payload_data,
                        "android": {
                            "priority": "high"
                        },
                        "apns": {
                            "headers": {
                                "apns-priority": "10"
                            },
                            "payload": {
                                "aps": {
                                    "content-available": 1
                                }
                            }
                        }
                    }
                }

            def send_to_token(t: str) -> dict:
                if credentials.expiry and credentials.expired:
                    credentials.refresh(Request())
                    headers["Authorization"] = f"Bearer {credentials.token}"

                resp = requests.post(
                    self.FCM_SEND_URL,
                    headers=headers,
                    data=json.dumps(build_message(t))
                )
                success = resp.status_code == 200
                return {
                    "token": t,
                    "success": success,
                    "status_code": resp.status_code,
                    "response": resp.text
                }

            tokens = token if isinstance(token, list) else [token]
            send_results = [send_to_token(t) for t in tokens]

            any_success = any(r["success"] for r in send_results)

            return {
                "status": "success" if any_success else "error",
                "results": send_results if isinstance(token, list) else send_results[0]
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}
    

    def send_fcm_with_wave_notification_new(
        self, token, full_name, login_id, waved_to_user_id, profile_pic, title, notification_body, wave_type
    ):
        """Send a silent, data-only FCM notification for wave or waveback actions."""
        if not token:
            return {"status": "error", "message": "No device token provided."}

        try:
            # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE]
            )
            auth_request = Request()
            credentials.refresh(auth_request)

            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": self.CONTENT_TYPE,
            }

            url = self.FCM_SEND_URL

            # Silent data payload
            payload_data = {
                "type": wave_type,
                "user_id": str(login_id),
                "waved_to_user_id": str(waved_to_user_id),
                "full_name": full_name,
                "profile_pic": profile_pic,
            }

            # Helper: send one payload
            def send_payload(t):
                payload = {
                    "message": {
                        "token": t,
                        "notification": {
                        "title": title,
                        "body": notification_body
                    },
                        "data": payload_data,  # data-only, silent
                        "android": {
                            "priority": "high"  # ensures timely delivery on Android
                        },
                        "apns": {
                            "headers": {
                                "apns-priority": "10"  # immediate delivery on iOS
                            },
                            "payload": {
                                "aps": {
                                    "mutable-content": 1 
                                }
                            }
                        }
                    }
                }

                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                return {"status_code": resp.status_code, "response": resp.text}

            # 🔹 If token is a list — send to each device
            if isinstance(token, list):
                responses = [send_payload(t) for t in token]
                result = {"status": "success", "messages": responses}

            # 🔹 If single token — send to that one device
            else:
                resp = send_payload(token)
                result = {"status": "success", "message": resp}

            # Optional: log the silent wave in DB
            results = self.make_db_entry_of_wavesnotification(
                login_id, waved_to_user_id, wave_type, notification_body, profile_pic
            )
            if results.get("status") == 'success':
                print("Database entry success for silent wave notification")
            else:
                print("Database entry failure for silent wave notification")

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def upload_selfie_pic(self, user_id):
            try:
                # === Load AWS credentials from JSON file ===
                aws_config = self._load_vibr_config()

                s3 = boto3.client(
                    's3',
                    region_name=aws_config["region"],
                    aws_access_key_id=aws_config["aws_access_key_id"],
                    aws_secret_access_key=aws_config["aws_secret_access_key"]
                )

                
                # Set the bucket name and region
                bucket_name = 'sparoptui'
                region = 'ap-south-1'
                folder_name = 'netwirkUserPics/'
                file = request.files['image']
                original_filename = file.filename

                # sanitize filename to avoid path traversal or bad chars
                safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_filename)

                # Add current date and time to filename
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Netwirkuser_{user_id}_{now}_{safe_filename}"

                # build safe storage path
                save_dir = os.path.join(os.getcwd(), "profilepics")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, filename)

                # only safe path is used
                file.save(save_path)
                save_path = self._compress_image_for_s3(save_path)
                filename = os.path.basename(save_path)

                # Upload to S3
                s3.upload_file(save_path, bucket_name, folder_name + filename,
                            ExtraArgs={'ContentType': 'image/jpeg'})
                
                # Get the URL of the file
                url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{folder_name}{filename}"
                aesthetic_score = 0.2

                # === OpenAI Aesthetic Scoring ===
                # openai.api_key = aws_config["openai_api_key"]

            
                # with open(save_path, "rb") as img_file:
                #     img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                # try:
                #     intent_response = openai.ChatCompletion.create(
                #         model="gpt-4-turbo",
                #         messages=[
                #             {
                #                 "role": "system",
                #                 "content": (
                #                     "Analyze the provided image of a human face (STRICTLY HUMAN FACE, no animals or statues, if not human face just say- 'NOT HUMAN')."
                #                     "Rate the face on a scale of 0 to 1 based on three factors: Facial Symmetry (0–0.33): Measure the balance between left and right facial features using landmark points (e.g., eyes, nose). "
                #                     "Feature Proportionality (0–0.33): Assess the harmony of facial features (e.g., eye size, nose width, skin clarity) relative to standard proportions. Face Alignment (0–0.33): Evaluate the positioning of facial features (e.g., eye level, nose centering). Provide a total score (sum of the three) and briefly explain each factor's score."
                #                     "Focus on technical metrics, not subjective opinions, for research purposes and people's upliftment."
                #                     "JUST RETURN 'total_score' in JSON format in response AND NO OTHER DATA, nothing else. NO Yapping."
                #                 )
                #             },
                #             {
                #                 "role": "user",
                #                 "content": [
                #                     {
                #                         "type": "text",
                #                         "text": "Rate this profile picture between 1 (poor) and 10 (excellent)."
                #                     },
                #                     {
                #                         "type": "image_url",
                #                         "image_url": {
                #                             "url": f"data:image/jpeg;base64,{img_base64}"
                #                         }
                #                     }
                #                 ]
                #             }
                #         ],
                #         max_tokens=10
                #     )

                #     gpt_reply = intent_response['choices'][0]['message']['content']
                #     match = re.search(r'\d+(\.\d+)?', gpt_reply)
                #     aesthetic_score = float(match.group()) if match else 0.2

                # except Exception as e:
                #     print("Image scoring error:", str(e))
                #     aesthetic_score = 0.2


                # Inserting profile pic url into db table
                cursor = self.db.cur()
                cursor.execute("UPDATE icebreaker.profile SET selfiePic = %s, aesthetic_score = %s where userId = %s", (url,aesthetic_score, user_id))
                # cursor.execute("UPDATE icebreaker.userPictures SET picture_urls = %s WHERE loginUserId = %s AND IsProfilePicture = 1", (url,user_id))
                self.db.conn.commit()           
                cursor.close()

                message = 'Image uploaded successfully !'
                return jsonify({'image_url':url, 'status' : 'Success','message': message, 'userId':user_id})


            except Exception as e:
                message = 'Failed to upload image !'
                return jsonify({'status' : 'Failure','message': str(e)})

    def user_live_location(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            lat = data['latitude']
            lon = data['longitude']

            cursor = self.db.cur()

            cursor.execute("""
                SELECT notify_connection
                FROM icebreaker.profile
                WHERE userId = %s
            """, (user_id,))

            row = cursor.fetchone()

            if row and row[0] == 0:  # user has disabled notifications
                return {
                    "status": "success",
                    "message": "User doesn't want to send notification"
                }

            # --- Fetch user's name ---
            cursor.execute("""
                SELECT fullName
                FROM icebreaker.registration 
                WHERE userId = %s
            """, (user_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(self.USER_NOT_FOUND_MESSAGE)
            full_name = row[0]

            # --- Insert or update user location ---
            cursor.execute(
                "SELECT user_id FROM icebreaker.user_live_location WHERE user_id = %s",
                (user_id,)
            )

            if cursor.fetchone():
                cursor.execute(
                    "UPDATE icebreaker.user_live_location SET latitude = %s, longitude = %s WHERE user_id = %s",
                    (lat, lon, user_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO icebreaker.user_live_location (user_id, latitude, longitude) VALUES (%s, %s, %s)",
                    (user_id, lat, lon)
                )

            self.db.conn.commit()

            # --- Distance function ---
            def haversine_distance(lat1, lon1, lat2, lon2):
                R = 3958.8  # miles
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                return R * c

            # --- Find users within 1 mile ---
            cursor.execute("""
                SELECT user_id, latitude, longitude FROM icebreaker.user_live_location WHERE user_id != %s
            """, (user_id,))

            all_nearby_users = [
                uid for uid, u_lat, u_lon in cursor.fetchall()
                if haversine_distance(lat, lon, u_lat, u_lon) <= 10.0
            ]

            # --- Find nearby users with "Connected" status ---
            connected_nearby_users = []
            for near_uid in all_nearby_users:
                status = self._get_interaction_status(cursor, user_id, near_uid)
                if status == "Connected":
                    connected_nearby_users.append(near_uid)

            # --- Determine which users are allowed for notification ---
            sent_user_ids = []
            final_tokens = []
            blocked_notifications = []

            for near_uid in connected_nearby_users:

                # Check 4-hour cooldown
                if not self._can_send_notification(cursor, user_id, near_uid):
                    blocked_notifications.append(near_uid)
                    continue

                # Fetch token
                cursor.execute("""
                    SELECT deviceToken 
                    FROM icebreaker.registration 
                    WHERE userId = %s AND deviceToken IS NOT NULL AND deviceToken != ''
                """, (near_uid,))

                token_row = cursor.fetchone()
                if not token_row:
                    continue

                token = token_row[0]
                final_tokens.append(token)
                sent_user_ids.append(near_uid)

                # Log notification
                cursor.execute("""
                    INSERT INTO icebreaker.user_notification_log 
                    (sender_user_id, receiver_user_id)
                    VALUES (%s, %s)
                """, (user_id, near_uid))

            self.db.conn.commit()

            # --- Send notifications ---
            if final_tokens:
                self.send_fcm_with_nearby_user(final_tokens, full_name, user_id)

            # --- Response ---
            return {
                "status": "success",
                "message": "Location stored successfully",
                "notifications": {
                    "sent_to": sent_user_ids,
                    "blocked_already_sent": blocked_notifications,
                    "connected_users": connected_nearby_users,
                    "within_1_mile": all_nearby_users
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

        finally:
            cursor.close()

    def send_fcm_with_nearby_user(self, tokens, full_name, user_id):
        # print("Sending to:", tokens)

        title = f"{full_name} is nearby"
        body = "Say hello"
        data_type = "nearby_user"

        try:
            # SERVICE_ACCOUNT_FILE = r"D:\icebreakerNew\vibrr\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE

            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=[self.CLOUD_PLATFORM_SCOPE]
            )

            auth_request = Request()
            credentials.refresh(auth_request)

            headers = {
                'Authorization': f'Bearer {credentials.token}',
                'Content-Type': 'application/json; UTF-8',
            }

            url = self.FCM_SEND_URL

            for token in tokens:  # ✔ iterate tokens properly
                payload = {
                    "message": {
                        "token": token,  # ✔ correct field
                        "notification": {
                            "title": title,
                            "body": body     # ✔ add body
                        },
                        "data": {"type": data_type,
                                "user_id": str(user_id)
                                },
                        "android": {
                            "priority": "high",
                            "collapse_key": "type_a"
                        },
                        "apns": {
                            "headers": {"apns-priority": "10"},
                            "payload": {
                                "aps": {
                                    "alert": {
                                        "title": title,
                                        "body": body
                                    },
                                    "mutable-content": 1
                                }
                            }
                        }
                    }
                }

                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                print("FCM Response:", resp.text)

            return {"status": "success"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _can_send_notification(self, cursor, sender_id, receiver_id):
        cursor.execute("""
            SELECT created_at 
            FROM icebreaker.user_notification_log
            WHERE sender_user_id = %s AND receiver_user_id = %s
            ORDER BY created_at DESC LIMIT 1
        """, (sender_id, receiver_id))
        
        row = cursor.fetchone()
        if not row:
            return True  # no previous notification

        last_sent_time = row[0]

        four_hours_ago = datetime.utcnow() - timedelta(minutes=5)

        if last_sent_time >= four_hours_ago:
            return False

        return True


        # last_sent_time = row[0]
        # four_hours_ago = datetime.utcnow() - timedelta(hours=4)

        # return last_sent_time < four_hours_ago

    def get_vibepersonality_qns1(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns1')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        
    def get_vibepersonality_qns2(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns2')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        
    def get_vibepersonality_qns3(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns3')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        
    def get_vibepersonality_qns4(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns4')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})

    def get_vibepersonality_qns5(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns5')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        
    def get_vibepersonality_qns6(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns6')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        
    def get_vibepersonality_qns7(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns7')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})

    def get_vibepersonality_qns8(self):
        try:
            cursor = self.db.cur()
            vibe_list = []

            cursor.execute('SELECT id,answer FROM icebreaker.vibepersonalityqns8')
            vibes = cursor.fetchall()

            for vibe in vibes:
                vibe_dict= {
                    'id': vibe[0],
                    'answer': vibe[1]
                }
                
                vibe_list.append(vibe_dict)

            return jsonify({'Status': 'Success', 'vibe List': vibe_list })

        except Exception as e:
            return jsonify({'Status': 'Failure', 'Message': str(e)})
        
    def get_user_profile_details_new(self):
        try:
            request_data = request.get_json()
            login_user_id = int(request_data['loginUserId'])
            requested_user_id =int (request_data['requestedUserId'])
     

            cursor = self.db.cur()

            login_web_token = self._get_login_token_google(cursor, login_user_id)
            print(login_web_token)
            block_flag = self._get_block_status_new(cursor, login_user_id, requested_user_id)

            user_dict = self._get_user_details_profile_new(cursor, requested_user_id, block_flag, login_user_id)
            if not user_dict:
                return self._user_not_found_response_new(cursor)
        
            # user_dict["status"] = self._get_interaction_status_new(cursor, login_user_id, requested_user_id)

            status = self._get_interaction_status_new(cursor, login_user_id, requested_user_id)
            user_dict["status"] = status
            user_dict["loginWebToken"] = login_web_token
            user_dict["pictures"] = self._get_user_pictures_new(cursor, requested_user_id)
            user_dict["wave_status"]= self.get_wave_status_sender(cursor,login_user_id,requested_user_id )

            incognito, cognito_time = self._get_incognito_status_new(cursor, requested_user_id)
            is_incognito = bool(incognito)

            if is_incognito and status in ("Connected", "Request_Received"):
                user_dict["incognito"] = False
            else:
                user_dict["incognito"] = is_incognito
                if is_incognito:
                    user_dict["cognito_time"] = cognito_time

            cursor.close()
            self.db.conn.close()
            return jsonify(user_dict)

        except Exception as e:
            self.db.conn.rollback()
            error = {'error': str(e)}
            print(error)
            return jsonify(error)

    # ----------------- HELPER METHODS -------------------

    def _get_block_status_new(self, cursor, login_user_id, requested_user_id):
        cursor.execute(
            'SELECT 1 FROM icebreaker.blockedUsers WHERE blockedId = %s AND userId = %s',
            (requested_user_id, login_user_id)
        )
        return 1 if cursor.fetchone() else 0
    
    def _get_incognito_status_new(self, cursor, user_id):
        cursor.execute("""
            SELECT incognito FROM icebreaker.location
            WHERE userId = %s
        """, (user_id,))
        row = cursor.fetchone()

        if row and row[0] == 1:
            cursor.execute(
                "SELECT cognito_time FROM icebreaker.location WHERE userId = %s",
                (user_id,)
            )
            row2 = cursor.fetchone()
            cognito_time = row2[0] if row2 else None
            return 1, cognito_time

        return False, None
        
    def _get_user_details_profile_new(self, cursor, requested_user_id, block_flag,login_user_id):
        cursor.execute("""
                SELECT r.userId, 
            p.profilePic,
            p.selfiePic, 
            r.fullName, 
            r.email, 
            r.firebaseId, 
            r.signupType, 
            r.deviceToken, 
            r.authenticationId, 
            r.Gender, 
            r.Preference, 
            r.DOB, 
            p.vibes, 
            p.vibing, 
            r.countryCode, 
            r.phoneNo, 
            r.city, 
            r.state, 
            r.country, 
            p.zodiac_sign,
            p.answer1,
            p.answer2,
            p.answer3,
            p.answer4,
            p.height,
            p.answer5,
            p.answer6,
            p.age_preference,
            p.prompt1question,
            p.prompt1answer,
            p.prompt2question,
            p.prompt2answer
                              
        FROM icebreaker.registration r
        LEFT JOIN icebreaker.profile p ON r.userId = p.userId
        WHERE r.userId = %s
        """, (requested_user_id,))
        row = cursor.fetchone()

        if not row:
            return None
           
        return {
            "userId": row[0],
            "profilePic": row[1],
            "selfiePic": row[2],
            "fullName": row[3],
            "email": row[4],
            "firebaseId": row[5],
            "signupType": row[6],
            "deviceToken": row[7],
            "authenticationId": row[8],
            "Gender": row[9],
            "Preference": row[10],
            "DOB": row[11].strftime('%m-%d-%Y') if row[11] else None,
            "vibes": row[12],
            "vibing": row[13],
            "countryCode": row[14],
            "phoneNo": row[15],
            "city": row[16],
            "state": row[17],
            "country": row[18],
            "zodiac_sign": row[19],
            "what_makes_my_friends_laugh_about_me":row[20],
            "weekend_usually_looks_like":row[21],
            "Describe_your_communication_style":row[22],
            "What_brings_you_to_Vyb":row[23],
            "height":row[24],
            "What_kind_of_partner_do_you_vibe_with_most":row[25],
            "What_does_everyday_vyb_look_like":row[26],
            "age_preference":row[27],
            "prompt1question":row[28],
            "prompt1answer":row[29],
            "prompt2question":row[30],
            "prompt2answer":row[31],
            "block_flag": block_flag,
            # "wave_status": wave_status
        }

    def _get_interaction_status_new(self, cursor, login_user_id, requested_user_id):
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

    def _get_user_pictures_new(self, cursor, user_id):
        cursor.execute(
            "SELECT picture_urls, IsProfilePicture FROM icebreaker.userPictures WHERE loginUserId = %s",
            (user_id,)
        )
        pics_data = cursor.fetchall()
        return [
            {"url": pic_url, "isProfilePicture": bool(is_profile)}
            for pic_url, is_profile in pics_data
        ]

    def _user_not_found_response_new(self, cursor):
        cursor.close()
        self.db.conn.close()
        return jsonify({"error": self.USER_NOT_FOUND_MESSAGE})

    def edit_profile_details_new(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            full_name = data['fullName']
            gender = data['Gender']
            preference = data.get('Preference')
            dob = data['DOB']
            vibe_I_bring = data['vibe_I_bring']
            height = data["height"]
            country_code = data.get('countryCode')
            phone_no = data.get('phoneNo')
            city = data.get('city')
            state = data.get('state')
            country = data.get('country')
            notify_connection = bool(data.get('notify_connection', False))
            background_notify = bool(data.get('background_notify', False))
            what_makes_my_friends_laugh_about_me = data.get('what_makes_my_friends_laugh_about_me')
            weekend_usually_looks_like = data.get('weekend_usually_looks_like', None)
            Describe_your_communication_style  = data.get('Describe_your_communication_style', None)
            What_brings_you_to_VYB = data.get('What_brings_you_to_VYB', None)
            What_kind_of_partner_do_you_vibe_with_most = data.get('What_kind_of_partner_do_you_vibe_with_most', None)
            # How_active_are_you_in_your_daily_life = data.get('How_active_are_you_in_your_daily_life', None)
            # How_do_you_usually_unwind_after_a_long_day = data.get('How_do_you_usually_unwind_after_a_long_day', None)
            # Do_you_prefer_planned_or_spontaneous_days = data.get('Do_you_prefer_planned_or_spontaneous_days',None)
            What_does_everyday_vyb_look_like=data.get('What_does_everyday_vyb_look_like', None)
            age_preference = data.get('age_preference')
            prompt1question = data.get('prompt1question')
            prompt1answer = data.get('prompt1answer')
            prompt2question = data.get('prompt2question')
            prompt2answer = data.get('prompt2answer')
            cursor = self.db.cur()

            # Fetch loginWebToken of user from Profile table
            cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
            login_token = cursor.fetchall()
            login_web_token = login_token[0][0]

            # === Check if profile exists ===
            cursor.execute("SELECT COUNT(*) FROM icebreaker.profile WHERE userId = %s", (user_id,))
            profile_exists = cursor.fetchone()[0] 

            if profile_exists:
                # --- Update existing record ---
                cursor.execute("""
                            UPDATE icebreaker.profile p
                            JOIN icebreaker.registration r
                            ON p.userId = r.userId
                            SET r.fullName = %s, r.Gender = %s, r.Preference = %s,
                                p.vibes = %s, r.countryCode = %s, r.phoneNo = %s, 
                                r.city = %s, r.state = %s, r.country = %s, p.notify_connection = %s,p.background_notify = %s, p.answer1 = %s,
                                p.answer2 = %s, p.answer3 = %s, p.answer4 = %s, height = %s,  p.answer5 = %s,  p.answer6 = %s, p.age_preference = %s, p.prompt1question = %s, p.prompt1answer = %s, p.prompt2question = %s, p.prompt2answer = %s
                            WHERE r.userId = %s""",
                            (full_name, gender, preference, vibe_I_bring,
                            country_code, phone_no, city, state, country, notify_connection, background_notify,what_makes_my_friends_laugh_about_me, weekend_usually_looks_like, Describe_your_communication_style, What_brings_you_to_VYB, 
                            height, What_kind_of_partner_do_you_vibe_with_most, What_does_everyday_vyb_look_like, age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer, user_id))
            else:
                # --- Insert new profile record ---
                cursor.execute("""
                            INSERT INTO icebreaker.profile (userId, vibes, zodiac_sign, notify_connection, background_notify, answer1, answer2, answer3, answer4, height, answer5, answer6, age_preference, prompt1question, prompt1answer, prompt2question, prompt2answer )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s)""",
                            (user_id, vibe_I_bring, None, notify_connection, background_notify,what_makes_my_friends_laugh_about_me, weekend_usually_looks_like, Describe_your_communication_style, What_brings_you_to_VYB, height, What_kind_of_partner_do_you_vibe_with_most, What_does_everyday_vyb_look_like, age_preference, 
                             prompt1question, prompt1answer, prompt2question, prompt2answer))

                # Also update registration table with user details
                cursor.execute("""
                            UPDATE icebreaker.registration
                            SET fullName = %s, Gender = %s, Preference = %s,
                                countryCode = %s, phoneNo = %s, city = %s, state = %s, country = %s
                            WHERE userId = %s""",
                            (full_name, gender, preference, country_code,
                            phone_no, city, state, country, user_id))

            # self.store_embedding(user_id, prompt1answer, prompt2answer)
            
            self.db.conn.commit()

            # Fetch updated user details
            cursor.execute("""
                            SELECT r.Gender, r.Preference, r.DOB, p.vibes, 
                                r.fullName, r.countryCode, r.phoneNo, r.city, r.state, 
                                r.country, p.zodiac_sign, p.notify_connection, p.background_notify,
                                p.answer1 AS what_makes_my_friends_laugh_about_me, 
                                p.answer2 AS weekend_usually_looks_like, 
                                p.answer3 AS Describe_your_communication_style, 
                                p.answer4 AS What_brings_you_to_VYB,
                                p.height,
                                p.answer5 AS What_kind_of_partner_do_you_vibe_with_most, 
                                p.answer6 AS What_does_everyday_vyb_look_like,
                                p.age_preference, 
                                p.prompt1question,
                                p.prompt1answer,
                                p.prompt2question,
                                p.prompt2answer

                            FROM icebreaker.registration r
                            JOIN icebreaker.profile p
                            ON r.userId = p.userId
                            WHERE r.userId = %s""", (user_id,))
            row = cursor.fetchone()

            cursor.close()
            self.db.conn.close()

            if not row:
                return jsonify({
                    "error": self.USER_NOT_FOUND_MESSAGE,
                    "loginWebToken": login_web_token
                })
            
            keys = ["Gender", "Preference", "DOB", "vibes", "fullName",
                    "countryCode", "phoneNo", "city", "state", "country", "zodiac_sign","notify_connection","background_notify","what_makes_my_friends_laugh_about_me","weekend_usually_looks_like","what_brings_you_to_VYB","what_are_you_open_to","height",
                    "What_kind_of_partner_do_you_vibe_with_most","What_does_everyday_vyb_look_like", "age_preference", "prompt1question", "prompt1answer", "prompt2question", "prompt2answer"]

            def format_value(index):
                val = row[index]
                if val is None:
                    return ''
                if keys[index] == 'DOB':
                    return val.strftime('%m-%d-%Y') if hasattr(val, 'strftime') else ''
                return val

            user_dict = {key: format_value(i) for i, key in enumerate(keys)}
            user_dict["loginWebToken"] = login_web_token

            return jsonify(user_dict)
        
        except Exception as e:
            self.db.conn.rollback()
            return jsonify({'error': str(e)})
        
    def get_liked_user_list_dummy(self):
        try:
            data = request.get_json()
            user_id = data['userId']

            # placeId as STRING (as per your requirement)
            # place_id = data.get('placeId') or "ChIJRVR9ciHP5zsRtrFD4nhXuBQ"

            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page

            cursor = self.db.cur()
            place_id = self._resolve_place_id(user_id, data, cursor)

            # ---------------------------------------------------
            # 1️⃣ TOTAL UNIQUE LIKES COUNT (DEDUPED)
            # ---------------------------------------------------
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT userId FROM icebreaker.wave
                    WHERE wavedToUserId = %s 

                    UNION
                    SELECT userId FROM icebreaker.waveback
                    WHERE wavedToUserId = %s 

                    UNION
                    SELECT userId FROM icebreaker.waveNear
                    WHERE wavedToUserId = %s 

                    UNION
                    SELECT userId FROM icebreaker.wavebackNear
                    WHERE wavedToUserId = %s 
                ) AS liked_users
            """, (
                user_id, 
                user_id, 
                user_id,
                user_id
            ))

            total_likes = cursor.fetchone()[0]

            # ---------------------------------------------------
            # 2️⃣ PAGINATED UNIQUE USER LIST
            # ---------------------------------------------------
            cursor.execute("""
                SELECT
                    r.userId, r.fullName, p.profilePic, r.email,
                    r.Gender, r.Preference,
                    p.vibes, p.vibing, p.zodiac_sign, p.height, p.selfiePic, r.DOB, r.firebaseId, 
                    r.signupType, r.deviceToken, r.authenticationId,r.countryCode, r.phoneNo, r.city, r.state, r.country
                FROM icebreaker.registration r
                JOIN icebreaker.profile p
                    ON r.userId = p.userId
                JOIN (
                    SELECT userId, MAX(wavedTime) AS latest_waved_time
                    FROM (
                        SELECT userId, wavedTime FROM icebreaker.wave WHERE wavedToUserId = %s
                        UNION ALL
                        SELECT userId, wavedTime FROM icebreaker.waveback WHERE wavedToUserId = %s
                        UNION ALL
                        SELECT userId, wavedTime FROM icebreaker.waveNear WHERE wavedToUserId = %s
                        UNION ALL
                        SELECT userId, wavedTime FROM icebreaker.wavebackNear WHERE wavedToUserId = %s
                    ) all_waves
                    GROUP BY userId
                ) liked_users
                    ON liked_users.userId = r.userId
                ORDER BY liked_users.latest_waved_time  DESC
                LIMIT %s OFFSET %s
            """, (
                user_id, 
                user_id, 
                user_id, 
                user_id, 
                per_page, offset
            ))

            rows = cursor.fetchall()
            liked_users = []

            # ---------------------------------------------------
            # 3️⃣ DETERMINE WAVE STATUS (PER USER)
            # ---------------------------------------------------
            for row in rows:
                waved_user_id = row[0]

                cursor.execute("""
                    SELECT
                        -- THEY waved YOU (normal or near)
                        EXISTS (
                            SELECT 1 FROM icebreaker.wave
                            WHERE userId = %s AND wavedToUserId = %s
                        )
                        OR EXISTS (
                            SELECT 1 FROM icebreaker.waveNear
                            WHERE userId = %s AND wavedToUserId = %s
                        ) AS they_waved_you,

                        -- YOU waved THEM (normal or near)
                        EXISTS (
                            SELECT 1 FROM icebreaker.wave
                            WHERE userId = %s AND wavedToUserId = %s
                        )
                        OR EXISTS (
                            SELECT 1 FROM icebreaker.waveNear
                            WHERE userId = %s AND wavedToUserId = %s
                        ) AS you_waved_them,

                        -- MUTUAL (waveback or near)
                        EXISTS (
                            SELECT 1 FROM icebreaker.waveback
                            WHERE userId = %s AND wavedToUserId = %s
                        )
                        OR EXISTS (
                            SELECT 1 FROM icebreaker.wavebackNear
                            WHERE userId = %s AND wavedToUserId = %s
                        ) AS mutual
                """, (
                    waved_user_id, user_id,      # THEY → YOU
                    waved_user_id, user_id,

                    user_id, waved_user_id,      # YOU → THEM
                    user_id, waved_user_id,

                    user_id, waved_user_id,      # MUTUAL
                    user_id, waved_user_id
                ))


                they_waved_you, you_waved_them, mutual = cursor.fetchone()

                if mutual:
                    wave_status = "waved"
                elif they_waved_you:
                    wave_status = "wave_back"
                elif you_waved_them:
                    wave_status = "already_waved"
                else:
                    wave_status = "no_wave"


                user_dict = {
                    "userId": row[0],
                    "fullName": row[1],
                    "profilePic": row[2],
                    "email": row[3],
                    "Gender": row[4],
                    "Preference": row[5],
                    "vibes": row[6],
                    "vibing": row[7],
                    "zodiac_sign": row[8],
                    "height": row[9],
                    "selfiePic": row[10],
                    "DOB":row[11],
                    "firebaseId":row[12],
                    "signupType":row[13],
                    "deviceToken":row[14],
                    "authenticationId":row[15],
                    "countryCode":row[16],
                    "phoneNo":row[17],
                    "city":row[18],
                    "state":row[19],
                    "country":row[20],
                    "wave_status": wave_status
                }

                # Existing industry status logic (unchanged)
                accepted_status, pending_status, received_status = self._fetch_request_status_industry_v1(
                    cursor, user_id, waved_user_id, place_id
                )
                user_dict["status"] = self._determine_status_industry_v1(
                    accepted_status, pending_status, received_status
                )

                liked_users.append(user_dict)

            return {
                "status": True,
                "page": page,
                "per_page": per_page,
                "TotalLikes": total_likes,
                "data": liked_users
            }

        except Exception as e:
            return {"status": False, "error": str(e)}   

        

    def get_current_liked_user_list(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            # place_id_static = 'ChIJRVR9ciHP5zsRtrFD4nhXuBQ'
            # place_id = place_id_static if place_id_real else 'None'
            cursor = self.db.cur()
            place_id = self._resolve_place_id(user_id, data, cursor)

            SESSION_GAP = timedelta(minutes=3)

            # 1️⃣ Get all sessions for this user at this place
            # cursor.execute("""
            #     SELECT checkintime, checkouttime
            #     FROM icebreaker.sessiontracker
            #     WHERE userid = %s AND placeid = %s
            #     ORDER BY checkintime
            # """, (user_id, place_id))
            # sessions = cursor.fetchall()
            # print("session -",sessions)

            # if not sessions:
            #     # User not checked in
            #     return {"status": True, "data": []}

            # # 2️⃣ Determine effective session start (merge sessions within SESSION_GAP)
            # effective_checkin = None
            # last_checkout = None

            # for checkin, checkout in sessions:
            #     checkout_time = checkout or checkin  # if checkout is NULL, treat as current session
            #     print("checkout_time -",checkout_time)
            #     if effective_checkin is None:
            #         # first session
            #         effective_checkin = checkin
            #         print("effective_checkin -",effective_checkin)
            #         last_checkout = checkout_time
            #         print("last_checkout -",last_checkout)
            #         continue

            #     gap = checkin - last_checkout
               
            #     print("gap -",gap)
            #     if gap <= SESSION_GAP:
            #         # same continuous session → extend last_checkout
            #         last_checkout = checkout_time
            #         print("last_checkout -",last_checkout)
            #     else:
            #         # new session → reset effective_checkin
            #         effective_checkin = checkin
            #         last_checkout = checkout_time
            #         print("effective_checkin -",effective_checkin)
            #         print("last_checkout -",last_checkout)

            # 3️⃣ Fetch users who liked this user after effective_checkin
            # cursor.execute("""
            #     SELECT DISTINCT r.userId, r.fullName, p.profilePic, r.email,
            #                     r.Gender, r.Preference, p.vibes, p.vibing,
            #                     p.zodiac_sign, p.height, p.selfiePic, r.DOB, r.firebaseId, 
            #         r.signupType, r.deviceToken, r.authenticationId,r.countryCode, r.phoneNo, r.city, r.state, r.country, p.zodiac_sign
            #     FROM icebreaker.registration r
            #     JOIN icebreaker.profile p ON r.userId = p.userId
            #     JOIN icebreaker.session_likes s ON r.userId = s.user_id
            #     WHERE s.wavedToUserId = %s
            #     AND s.place_id = %s
            #     AND s.wavedTime >= %s
            #     ORDER BY wavedTime DESC
                
            # """, (user_id, place_id, effective_checkin))

            cursor.execute("""
                SELECT
                    r.userId, r.fullName, p.profilePic, r.email,
                    r.Gender, r.Preference, p.vibes, p.vibing,
                    p.zodiac_sign, p.height, p.selfiePic, r.DOB, r.firebaseId,
                    r.signupType, r.deviceToken, r.authenticationId, r.countryCode,
                    r.phoneNo, r.city, r.state, r.country, p.zodiac_sign
                FROM icebreaker.registration r
                JOIN icebreaker.profile p ON r.userId = p.userId
                JOIN (
                    SELECT user_id, MAX(wavedTime) AS wavedTime
                    FROM icebreaker.session_likes
                    WHERE wavedToUserId = %s AND place_id = %s
                    GROUP BY user_id
                ) s ON r.userId = s.user_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b1
                    WHERE b1.blockedId = s.user_id AND b1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b2
                    WHERE b2.userId = s.user_id AND b2.blockedId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r1
                    WHERE r1.reportedUserId = s.user_id AND r1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r2
                    WHERE r2.userId = s.user_id AND r2.reportedUserId = %s
                )
                ORDER BY s.wavedTime DESC
            """, (user_id, place_id, user_id, user_id, user_id, user_id))

            rows = cursor.fetchall()
            liked_users = []

            # 4️⃣ Wave / waveback logic (unchanged)
            for row in rows:
                waved_user_id = row[0]
                user_dict = {
                    "userId": waved_user_id,
                    "fullName": row[1],
                    "profilePic": row[2],
                    "email": row[3],
                    "Gender": row[4],
                    "Preference": row[5],
                    "vibes": row[6],
                    "vibing": row[7],
                    "zodiac_sign": row[8],
                    "height": row[9],
                    "selfiePic": row[10],
                    "DOB": row[11],
                    "firebaseId":row[12],
                    "signupType":row[13],
                    "deviceToken":row[14],
                    "authenticationId":row[15],
                    "countryCode":row[16],
                    "phoneNo":row[17],
                    "city":row[18],
                    "state":row[19],
                    "country":row[20],
                    "zodiac_sign":row[21]
                }

                user_dict["wave_status"] = self.get_wave_status_sender_here(cursor, user_id, waved_user_id, place_id )

                accepted, pending, received = self._fetch_request_status_industry_v1(
                    cursor, user_id, waved_user_id, place_id
                )
                user_dict["status"] = self._determine_status_industry_v1(
                    accepted, pending, received
                )

                liked_users.append(user_dict)

            return {"status": True, "data": liked_users}

        except Exception as e:
            return {"status": False, "error": str(e)}

    def delete_received_wave(self):
        try:
            data = request.get_json()
            user_id = data['userId']         
            waved_by_id = data['wavedByUserId'] 
       

            cursor = self.db.cur()

      
            # cursor.execute("""
            #     DELETE FROM icebreaker.user_likes
            #     WHERE sender_id = %s AND receiver_id = %s 
            # """, (waved_by_id, user_id))

            # cursor.execute("""
            #     DELETE FROM icebreaker.user_likes_near
            #     WHERE sender_id = %s AND receiver_id = %s 
            # """, (waved_by_id, user_id))

           
            cursor.execute("""
                DELETE FROM icebreaker.session_likes
                WHERE user_id = %s
                AND wavedTouserId = %s
                AND like_type = 'wave'
            """, (waved_by_id, user_id))

            cursor.execute("""
                DELETE FROM icebreaker.session_likes_near
                WHERE user_id = %s
                AND wavedTouserId = %s
                AND like_type = 'wave'
            """, (waved_by_id, user_id))

            self.db.conn.commit()
            cursor.close()

            return {"status": True, "message": "Wave removed successfully"}

        except Exception as e:
            self.db.conn.rollback()
            return {"status": False, "error": str(e)}
        
    def get_total_likes(self):
        data = request.get_json()
        user_id = data['userId']         
       

        cursor = self.db.cur()

        try:
            cursor.execute("""
                            SELECT COUNT(DISTINCT liker_id) AS likes_count
            FROM (
                SELECT sender_id AS liker_id
                FROM icebreaker.user_likes ul
                JOIN icebreaker.registration r ON r.userId = ul.sender_id
                WHERE ul.receiver_id = %s
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b1
                    WHERE b1.blockedId = ul.sender_id AND b1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b2
                    WHERE b2.userId = ul.sender_id AND b2.blockedId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r1
                    WHERE r1.reportedUserId = ul.sender_id AND r1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r2
                    WHERE r2.userId = ul.sender_id AND r2.reportedUserId = %s
                )

                UNION

                SELECT sender_id AS liker_id
                FROM icebreaker.user_likes_near uln
                JOIN icebreaker.registration r ON r.userId = uln.sender_id
                WHERE uln.receiver_id = %s
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b1
                    WHERE b1.blockedId = uln.sender_id AND b1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b2
                    WHERE b2.userId = uln.sender_id AND b2.blockedId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r1
                    WHERE r1.reportedUserId = uln.sender_id AND r1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r2
                    WHERE r2.userId = uln.sender_id AND r2.reportedUserId = %s
                )

            ) AS combined_likes
            """, (
                user_id, user_id, user_id, user_id, user_id,   # user_likes params
                user_id, user_id, user_id, user_id, user_id    # user_likes_near params
            ))

            result = cursor.fetchone()
            total_likes = result[0] if result else 0

          
            return jsonify({"totalLikes": total_likes}), 200

        except Exception as e:
            print("DB error:", e)
            return jsonify({"error": "Database query failed"}), 500

        finally:
            cursor.close()  

    def update_devicetoken(self):
        data = request.get_json()

        device_token = data.get('deviceToken')
        user_id = data.get('user_id')

        if not device_token or not user_id:
            return {"message": "Missing deviceToken or user_id"}, 400

        try:
            cursor = self.db.cur()

            # Check if user exists
            cursor.execute(
                """
                SELECT userId
                FROM icebreaker.registration
                WHERE userId = %s
                """,
                (user_id,)
            )

            result = cursor.fetchone()

            if result:
                # Update device token
                cursor.execute(
                    """
                    UPDATE icebreaker.registration
                    SET deviceToken = %s
                    WHERE userId = %s
                    """,
                    (device_token, user_id)
                )
                self.db.conn.commit()
                cursor.close()
                return {"message": "Device token updated successfully"}, 200
            else:
                return {"message": "No user found"}, 404

        except Exception as e:
            self.db.rollback()
            return {"error": str(e)}, 500

    def handle_send_wave_near_dummy(self):
        data = request.get_json()
        login_id = data['loginUserId']          
        waved_to_id = data['wavedToUserId']     
        status = data.get('status', 'wave')     
        place_id = None 

        cursor = self.db.cur()
        wave_time = datetime.now()

        cursor.execute('''
            SELECT r.deviceToken, r.fullName, p.profilePic
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s
        ''', (login_id,))
        sender_info = cursor.fetchone()
        full_name = sender_info[1].title()
        new_wave_inserted = False



        # Decide target table based on wave type
        if status == 'wave':
            # --- Check all tables first ---
            # cursor.execute('''SELECT waveId FROM icebreaker.wave
            #                 WHERE userId = %s AND wavedToUserId = %s''',
            #             (login_id, waved_to_id))
            # existing_wave = cursor.fetchone()

            # cursor.execute('''SELECT waveId FROM icebreaker.waveback
            #                 WHERE userId = %s AND wavedToUserId = %s ''',
            #             (login_id, waved_to_id))
            # existing_waveback = cursor.fetchone()

            cursor.execute('''SELECT waveId FROM icebreaker.waveNear
                            WHERE userId = %s AND wavedToUserId = %s''',
                        (login_id, waved_to_id))
            existing_wave_near = cursor.fetchone()

            cursor.execute('''SELECT waveId FROM icebreaker.wavebackNear
                            WHERE userId = %s AND wavedToUserId = %s''',
                        (login_id, waved_to_id))
            existing_waveback_near = cursor.fetchone()

            actions_taken = []

            # --- Handle wave ---
            if existing_wave_near:
                cursor.execute('''DELETE FROM icebreaker.wave
                                WHERE userId = %s AND wavedToUserId = %s''',
                            (login_id, waved_to_id))
                cursor.execute('''DELETE FROM icebreaker.waveNear
                                WHERE userId = %s AND wavedToUserId = %s''',
                            (login_id, waved_to_id))
                
                cursor.execute('''DELETE FROM icebreaker.session_likes
                                WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                            (login_id, waved_to_id))
                cursor.execute('''DELETE FROM icebreaker.session_likes_near
                                WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                            (login_id, waved_to_id))
                cursor.execute('''DELETE FROM icebreaker.notifications
                            WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'wave_received' ''',
                        (waved_to_id, login_id))
                actions_taken.append("Wave removed")

     
             # --- Handle wavebackNear ---
            if existing_waveback_near:
                cursor.execute('''DELETE FROM icebreaker.wavebackNear
                                WHERE ((userId = %s AND wavedToUserId = %s) OR (userId = %s AND wavedToUserId = %s))
                                ''',
                            (login_id, waved_to_id, waved_to_id, login_id))

                waveback_near_deleted = cursor.rowcount

                if waveback_near_deleted > 0:

                    
                    cursor.execute('''DELETE FROM icebreaker.session_likes_near
                                    WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                                (login_id, waved_to_id))
                    cursor.execute('''DELETE FROM icebreaker.notifications
                                    WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'waveback_received' ''',
                                (waved_to_id, login_id))
                    # Insert waveNear as wavebackNear conversion
                    cursor.execute('''INSERT INTO icebreaker.waveNear (userId, wavedToUserId, wavedTime, status)
                                    VALUES (%s, %s, %s, 'wave')''',
                                (waved_to_id, login_id, wave_time))
                    actions_taken.append("WavebackNear converted to waveNear")

                cursor.execute('''SELECT placeID, wavedTime FROM icebreaker.waveback
                            WHERE userId = %s AND wavedToUserId = %s 
                            ORDER BY wavedTime DESC
                            LIMIT 1''',
                        (login_id, waved_to_id))
                row = cursor.fetchone()

                if row:
                    original_place_id, original_waved_time = row


                cursor.execute('''DELETE FROM icebreaker.waveback
                                WHERE ((userId = %s AND wavedToUserId = %s) OR (userId = %s AND wavedToUserId = %s))
                                 ''',
                            (login_id, waved_to_id, waved_to_id, login_id))
                
                waveback_deleted = cursor.rowcount

                if waveback_deleted > 0:
              
                    cursor.execute('''DELETE FROM icebreaker.session_likes
                                    WHERE user_id = %s AND wavedTouserId = %s AND like_type = 'wave' ''',
                                (login_id, waved_to_id))
                    cursor.execute('''DELETE FROM icebreaker.notifications
                                    WHERE userId = %s AND requestedUserId = %s AND requestStatus = 'waveback_received' ''',
                                (waved_to_id, login_id))
                    # Insert wave as a waveback conversion
                    cursor.execute('''INSERT INTO icebreaker.wave (userId, wavedToUserId, wavedTime,placeId, status)
                                    VALUES (%s, %s, %s, %s, 'wave')''',
                                (waved_to_id, login_id, original_waved_time,original_place_id ))
                    actions_taken.append("Waveback converted to wave")

           

            # --- Insert new wave if none existed ---
            if not any([existing_wave_near, existing_waveback_near]):
    
                cursor.execute('''INSERT INTO icebreaker.session_likes_near (user_id, wavedToUserId, like_type, wavedTime)
                                VALUES (%s, %s, 'wave', %s)''',
                            (login_id, waved_to_id, wave_time))
                cursor.execute('''INSERT INTO icebreaker.waveNear (userId, wavedToUserId, wavedTime, status)
                                VALUES (%s, %s, %s, 'wave')''',
                            (login_id, waved_to_id, wave_time))
                actions_taken.append("Wave inserted")
                new_wave_inserted = True

            self.db.conn.commit()

            if new_wave_inserted:
                notification_title = "Like Received"
                notification_body = f"{full_name} liked you"
                wave_type = "wave_received"

                cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (waved_to_id,))
                tokens = cursor.fetchone()

                # --- Send notification if enabled ---
                if tokens and tokens[2] == 1:
                    receiver_token = tokens[0]
                    if isinstance(receiver_token, list):
                        receiver_token = [t for t in receiver_token if t != sender_info[0]]

                    self.send_fcm_with_wave_notification_new(
                        receiver_token,
                        sender_info[1],     # full name
                        login_id,           # sender ID
                        waved_to_id,        # receiver ID
                        sender_info[2],     # profile pic
                        notification_title,
                        notification_body,
                        wave_type
                    )

                # Close connection
                self.db.conn.close()
            

            return jsonify({"message": ", ".join(actions_taken), "status": "Success"})


        elif status == 'waveback':
            # --- Check if already waved back ---
            cursor.execute('''
                SELECT waveId FROM icebreaker.wavebackNear
                WHERE userId = %s AND wavedToUserId = %s
            ''', (login_id, waved_to_id))
            existing_waveback = cursor.fetchone()

            if existing_waveback:
                message = "User has already waved back to this user"
                return jsonify({"message": message, "status": 'Success'})

            cursor.execute('''
                DELETE FROM icebreaker.waveNear
                WHERE userId = %s AND wavedToUserId = %s
            ''', (waved_to_id,login_id))

            # --- Insert into waveback table ---
            cursor.execute('''
                INSERT INTO icebreaker.wavebackNear (userId, wavedToUserId, wavedTime, status)
                VALUES (%s, %s, %s, %s)
            ''', (login_id, waved_to_id, wave_time, 'waveback'))
            cursor.execute('''
                INSERT INTO icebreaker.wavebackNear (userId, wavedToUserId, wavedTime, status)
                VALUES (%s, %s, %s, %s)
            ''', (waved_to_id,login_id, wave_time, 'waveback'))
            notification_title = "Like Back "
            notification_body = f" {full_name} liked you back"
            wave_type = "waveback_received"

            
            cursor.execute('''
                INSERT INTO icebreaker.session_likes_near (user_id, wavedTouserId, like_type, wavedTime)
                VALUES (%s, %s, 'wave', %s)
            ''', (login_id, waved_to_id, wave_time))
            # print(notification_body)

        else:
            return jsonify({"message": "Invalid status. Must be 'wave' or 'waveback'.", "status": "Error"})

        # Commit DB insert
        self.db.conn.commit()

        

        # --- Fetch receiver’s notification preferences ---
        cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (waved_to_id,))
        tokens = cursor.fetchone()

        # --- Send notification if enabled ---
        if tokens and tokens[2] == 1:
            receiver_token = tokens[0]
            if isinstance(receiver_token, list):
                receiver_token = [t for t in receiver_token if t != sender_info[0]]

            self.send_fcm_with_wave_notification_new(
                receiver_token,
                sender_info[1],     # full name
                login_id,           # sender ID
                waved_to_id,        # receiver ID
                sender_info[2],     # profile pic
                notification_title,
                notification_body,
                wave_type
            )

        # Close connection
        self.db.conn.close()

        message = f"User {login_id} performed '{status}' action to user {waved_to_id}."
        return jsonify({"message": message, "status": 'Success'})
        
    # def handle_send_wave_near(self):
    #     data = request.get_json()
    #     login_id = data['loginUserId']          # user who is waving
    #     waved_to_id = data['wavedToUserId']     # user who is being waved to
    #     status = data.get('status', 'wave')     # either 'wave' or 'waveback'
    #     place_id = None  # static place_id for now

    #     cursor = self.db.cur()
    #     wave_time = datetime.now()

    #     # --- Fetch sender info ---
    #     cursor.execute('''
    #         SELECT r.deviceToken, r.fullName, p.profilePic
    #         FROM icebreaker.registration r
    #         JOIN icebreaker.profile p ON r.userId = p.userId
    #         WHERE r.userId = %s
    #     ''', (login_id,))
    #     sender_info = cursor.fetchone()
    #     full_name = sender_info[1].title()


    #     # Decide target table based on wave type
    #     if status == 'wave':
    #         # --- Check if already waved ---
    #         cursor.execute('''
    #             SELECT waveId FROM icebreaker.waveNear
    #             WHERE userId = %s AND wavedToUserId = %s
    #         ''', (login_id, waved_to_id))
    #         existing_wave = cursor.fetchone()

    #         if existing_wave:
    #             cursor.execute('''
    #                 DELETE FROM icebreaker.waveNear
    #                 WHERE userId = %s AND wavedToUserId = %s 
    #             ''', (login_id, waved_to_id))

               
    #             cursor.execute('''
    #                 DELETE FROM icebreaker.session_likes_near
    #                 WHERE user_id = %s
    #                 AND wavedTouserId = %s
    #                 AND like_type = 'wave'
    #             ''', ( login_id, waved_to_id))

    #             cursor.execute('''
    #                 DELETE FROM icebreaker.notifications
    #                 WHERE userId = %s
    #                 AND requestedUserId = %s
    #                 AND requestStatus = %s
    #             ''', (waved_to_id, login_id, 'wave_received'))
                    
    #             self.db.conn.commit()
    #             return jsonify({"message": "Wave removed", "status": "Success"})
            
    #         cursor.execute('''
    #             SELECT waveId FROM icebreaker.wavebackNear
    #             WHERE userId = %s AND wavedToUserId = %s 
    #         ''', (login_id, waved_to_id))
    #         existing_wave = cursor.fetchone()

    #         if existing_wave:
    #             # Delete both waveback entries
    #             cursor.execute('''
    #                 DELETE FROM icebreaker.wavebackNear
    #                 WHERE (userId = %s AND wavedToUserId = %s)
    #                 OR (userId = %s AND wavedToUserId = %s)
                    
    #             ''', (login_id, waved_to_id, waved_to_id, login_id))

                
    #             cursor.execute('''
    #                 DELETE FROM icebreaker.session_likes_near
    #                 WHERE user_id = %s
    #                 AND wavedTouserId = %s
    #                 AND like_type = 'wave'
    #             ''', (login_id, waved_to_id))

    #             cursor.execute('''
    #                 DELETE FROM icebreaker.notifications
    #                 WHERE userId = %s
    #                 AND requestedUserId = %s
    #                 AND requestStatus = %s
    #             ''', (waved_to_id, login_id, 'waveback_received'))
                    
    #              # --- Insert into waves table ---
    #             cursor.execute('''
    #                 INSERT INTO icebreaker.waveNear (userId, wavedToUserId, wavedTime, status)
    #                 VALUES (%s, %s, %s, %s )
    #             ''', (waved_to_id, login_id, wave_time,'wave'))
    #             self.db.conn.commit()
    #             return jsonify({"message": "Waveback withdrawn, wave inserted", "status": "Success"})
            
    #         cursor.execute('''
    #             INSERT INTO icebreaker.session_likes_near (user_id, wavedToUserId, like_type, wavedTime)
    #             VALUES (%s, %s, %s, %s )
    #         ''', (login_id, waved_to_id,'wave', wave_time))
    #         # --- Insert into waves table ---
    #         cursor.execute('''
    #             INSERT INTO icebreaker.waveNear (userId, wavedToUserId, wavedTime, status)
    #             VALUES (%s, %s, %s, %s )
    #         ''', (login_id, waved_to_id, wave_time,'wave'))
    #         notification_title = "Like Received"
    #         notification_body = f"{full_name} liked you"
    #         wave_type = "wave_received"
    #         # print(notification_body)

    #     elif status == 'waveback':
    #         # --- Check if already waved back ---
    #         cursor.execute('''
    #             SELECT waveId FROM icebreaker.wavebackNear
    #             WHERE userId = %s AND wavedToUserId = %s
    #         ''', (login_id, waved_to_id))
    #         existing_waveback = cursor.fetchone()

    #         if existing_waveback:
    #             message = "User has already waved back to this user"
    #             return jsonify({"message": message, "status": 'Success'})

    #         cursor.execute('''
    #             DELETE FROM icebreaker.waveNear
    #             WHERE userId = %s AND wavedToUserId = %s
    #         ''', (waved_to_id,login_id))

    #         # --- Insert into waveback table ---
    #         cursor.execute('''
    #             INSERT INTO icebreaker.wavebackNear (userId, wavedToUserId, wavedTime, status)
    #             VALUES (%s, %s, %s, %s)
    #         ''', (login_id, waved_to_id, wave_time, 'waveback'))
    #         cursor.execute('''
    #             INSERT INTO icebreaker.wavebackNear (userId, wavedToUserId, wavedTime, status)
    #             VALUES (%s, %s, %s, %s)
    #         ''', (waved_to_id,login_id, wave_time, 'waveback'))
    #         notification_title = "Like Back "
    #         notification_body = f" {full_name} liked you back"
    #         wave_type = "waveback_received"

            
    #         cursor.execute('''
    #             INSERT INTO icebreaker.session_likes_near (user_id, wavedTouserId, like_type, wavedTime)
    #             VALUES (%s, %s, 'wave', %s)
    #         ''', (login_id, waved_to_id, wave_time))
    #         # print(notification_body)

    #     else:
    #         return jsonify({"message": "Invalid status. Must be 'wave' or 'waveback'.", "status": "Error"})

    #     # Commit DB insert
    #     self.db.conn.commit()

        

    #     # --- Fetch receiver’s notification preferences ---
    #     cursor.execute(self.GET_USER_NOTIFICATION_PREFS_QUERY, (waved_to_id,))
    #     tokens = cursor.fetchone()

    #     # --- Send notification if enabled ---
    #     if tokens and tokens[2] == 1:
    #         receiver_token = tokens[0]
    #         if isinstance(receiver_token, list):
    #             receiver_token = [t for t in receiver_token if t != sender_info[0]]

    #         self.send_fcm_with_wave_notification_new(
    #             receiver_token,
    #             sender_info[1],     # full name
    #             login_id,           # sender ID
    #             waved_to_id,        # receiver ID
    #             sender_info[2],     # profile pic
    #             notification_title,
    #             notification_body,
    #             wave_type
    #         )

    #     # Close connection
    #     self.db.conn.close()

    #     message = f"User {login_id} performed '{status}' action to user {waved_to_id}."
    #     return jsonify({"message": message, "status": 'Success'})
    
    def get_current_liked_user_list_near(self):
        data = request.get_json()
        user_id = data['userId']

        cursor = self.db.cur()

        try:
            cursor.execute("""
                SELECT
                    r.userId,
                    r.fullName,
                    p.profilePic,
                    r.email,
                    r.Gender,
                    r.Preference,
                    p.vibes,
                    p.vibing,
                    p.zodiac_sign,
                    p.height,
                    p.selfiePic,
                    r.DOB,
                    r.firebaseId,
                    r.signupType,
                    r.deviceToken,
                    r.authenticationId,
                    r.countryCode,
                    r.phoneNo,
                    r.city,
                    r.state,
                    r.country
                FROM icebreaker.registration r
                JOIN icebreaker.profile p ON p.userId = r.userId
                JOIN (
                    SELECT user_id, MAX(wavedTime) AS wavedTime
                    FROM icebreaker.session_likes_near
                    WHERE wavedTouserId = %s
                    GROUP BY user_id
                ) s ON r.userId = s.user_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b1
                    WHERE b1.blockedId = s.user_id AND b1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b2
                    WHERE b2.userId = s.user_id AND b2.blockedId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r1
                    WHERE r1.reportedUserId = s.user_id AND r1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r2
                    WHERE r2.userId = s.user_id AND r2.reportedUserId = %s
                )
                ORDER BY s.wavedTime DESC
            """, (user_id, user_id, user_id, user_id, user_id))

            rows = cursor.fetchall()
            liked_users = []

            for row in rows:
                liked_user_id = row[0]

                status = self.get_wave_status_sender_near(cursor, user_id, liked_user_id)

                connection_status = self.get_connection_status_radius(
                    cursor, user_id, liked_user_id
                )

                liked_users.append({
                    "userId": row[0],
                    "fullName": row[1],
                    "profilePic": row[2],
                    "email": row[3],
                    "Gender": row[4],
                    "Preference": row[5],
                    "vibes": row[6],
                    "vibing": row[7],
                    "zodiac_sign": row[8],
                    "height": row[9],
                    "selfiePic": row[10],
                    "DOB": row[11],
                    "firebaseId": row[12],
                    "signupType": row[13],
                    "deviceToken": row[14],
                    "authenticationId": row[15],
                    "countryCode": row[16],
                    "phoneNo": row[17],
                    "city": row[18],
                    "state": row[19],
                    "country": row[20],
                    "wave_status": status,
                    "status": connection_status
                })

            return jsonify({
                "status": True,
                "totalLikes": len(liked_users),
                "data": liked_users
            }), 200

        except Exception as e:
            logging.error("get_current_liked_user_list_near DB error: %s", e)
            return jsonify({"status": False, "error": "Database query failed"}), 500
        
    def get_liked_user_list(self):
        try:
            data = request.get_json()
            user_id = data['userId']

            # place_id = data.get('placeId') or "ChIJRVR9ciHP5zsRtrFD4nhXuBQ"

            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page
            search_text = data.get('searchText', '')

            cursor = self.db.cur()
            place_id = self._resolve_place_id(user_id, data, cursor)

            cursor.execute("""
                SELECT
                    r.userId, r.fullName, p.profilePic, r.email,
                    r.Gender, r.Preference,
                    p.vibes, p.vibing, p.zodiac_sign, p.height, p.selfiePic, r.DOB, r.firebaseId,
                    r.signupType, r.deviceToken, r.authenticationId, r.countryCode, r.phoneNo, r.city, r.state, r.country
                FROM icebreaker.registration r
                JOIN icebreaker.profile p
                    ON r.userId = p.userId
                JOIN (
                    SELECT sender_id, MAX(created_at) AS latest_waved_time
                    FROM (
                        SELECT sender_id, created_at FROM icebreaker.user_likes ul
                        WHERE ul.receiver_id = %s
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b1
                            WHERE b1.blockedId = ul.sender_id AND b1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b2
                            WHERE b2.userId = ul.sender_id AND b2.blockedId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r1
                            WHERE r1.reportedUserId = ul.sender_id AND r1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r2
                            WHERE r2.userId = ul.sender_id AND r2.reportedUserId = %s
                        )

                        UNION ALL

                        SELECT sender_id, created_at FROM icebreaker.user_likes_near uln
                        WHERE uln.receiver_id = %s
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b1
                            WHERE b1.blockedId = uln.sender_id AND b1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b2
                            WHERE b2.userId = uln.sender_id AND b2.blockedId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r1
                            WHERE r1.reportedUserId = uln.sender_id AND r1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r2
                            WHERE r2.userId = uln.sender_id AND r2.reportedUserId = %s
                        )

                    ) all_waves
                    GROUP BY sender_id
                ) liked_users
                    ON liked_users.sender_id = r.userId
                ORDER BY liked_users.latest_waved_time DESC
                LIMIT %s OFFSET %s
            """, (
                user_id, user_id, user_id, user_id, user_id,   # user_likes params
                user_id, user_id, user_id, user_id, user_id,   # user_likes_near params
                per_page, offset
            ))

            rows = cursor.fetchall()
            liked_users = []

            # ---------------------------------------------------
            # 3️⃣ DETERMINE WAVE STATUS (PER USER)
            # ---------------------------------------------------
            for row in rows:
                waved_user_id = row[0]

                status = self.get_wave_status_sender(cursor, user_id, waved_user_id)

                cursor.execute("""
                    SELECT lastping
                    FROM icebreaker.user_pings
                    WHERE userId = %s
                """, (waved_user_id,))
                ping_result = cursor.fetchone()

                online_status = "offline"
                last_seen = ""

                if ping_result and ping_result[0]:
                    last_ping = ping_result[0]

                    if isinstance(last_ping, str):
                        last_ping_dt = datetime.strptime(last_ping, "%Y-%m-%d %H:%M:%S")
                    else:
                        last_ping_dt = last_ping

                    now = datetime.utcnow()
                    time_diff = (now - last_ping_dt).total_seconds()

                    if time_diff <= 30:
                        online_status = "online"

                    last_seen = last_ping_dt.strftime("%Y-%m-%d %H:%M:%S")

                user_dict = {
                    "userId": row[0],
                    "fullName": row[1],
                    "profilePic": row[2],
                    "email": row[3],
                    "Gender": row[4],
                    "Preference": row[5],
                    "vibes": row[6],
                    "vibing": row[7],
                    "zodiac_sign": row[8],
                    "height": row[9],
                    "selfiePic": row[10],
                    "DOB": row[11],
                    "firebaseId": row[12],
                    "signupType": row[13],
                    "deviceToken": row[14],
                    "authenticationId": row[15],
                    "countryCode": row[16],
                    "phoneNo": row[17],
                    "city": row[18],
                    "state": row[19],
                    "country": row[20],
                    "wave_status": status,
                    "online_status": online_status,
                    "last_seen": last_seen
                }

                accepted_status, pending_status, received_status = self._fetch_request_status_industry_v1(
                    cursor, user_id, waved_user_id, place_id
                )
                user_dict["status"] = self._determine_status_industry_v1(
                    accepted_status, pending_status, received_status
                )

                liked_users.append(user_dict)

            if search_text:
                liked_users = self._filter_liked_users_search(liked_users, search_text)
                total_likes = len(liked_users)

            total_likes = len(liked_users)

            return {
                "status": True,
                "page": page,
                "per_page": per_page,
                "TotalLikes": total_likes,
                "data": liked_users
            }

        except Exception as e:
            return {"status": False, "error": str(e)}
        
    def _filter_liked_users_search(self, results, search_text):
        search_text = search_text.lower()
        def matches(user):
            return (
                search_text in user.get("fullName", "").lower()
            )
        return [user for user in results if matches(user)]

    def delete_received_profile_likes(self):
        try:
            data = request.get_json()
            user_id = data['userId']         
            likedByUserId = data['likedByUserId'] 
            cursor = self.db.cur()

            # like_exists = False

            # tables = [
            #     ("icebreaker.user_likes", "sender_id", "receiver_id"),
            #     ("icebreaker.session_likes", "user_id", "wavedTouserId"),
            #     ("icebreaker.user_likes_near", "sender_id", "receiver_id"),
            #     ("icebreaker.session_likes_near", "user_id", "wavedTouserId"),
            # ]

            # for table, col1, col2 in tables:
            #     cursor.execute(f"""
            #         SELECT 1 FROM {table}
            #         WHERE ({col1} = %s AND {col2} = %s)
            #         OR ({col1} = %s AND {col2} = %s)
            #         LIMIT 1
            #     """, (likedByUserId, user_id, user_id, likedByUserId))

            #     if cursor.fetchone():
            #         like_exists = True
            #         break

            # if not like_exists:
            #     cursor.close()
            #     return {"status": False, "message": "No likes to be removed"}
   
            cursor.execute("""
                DELETE FROM icebreaker.user_likes
                WHERE sender_id = %s AND receiver_id = %s 
            """, (likedByUserId, user_id))

            cursor.execute("""
                DELETE FROM icebreaker.user_likes
                WHERE sender_id = %s AND receiver_id = %s 
            """, (user_id, likedByUserId))

            cursor.execute("""
                DELETE FROM icebreaker.session_likes
                WHERE user_id = %s AND wavedTouserId = %s 
            """, (likedByUserId, user_id))

            cursor.execute("""
                DELETE FROM icebreaker.session_likes
                WHERE user_id = %s AND wavedTouserId = %s 
            """, (user_id, likedByUserId))

            cursor.execute("""
                DELETE FROM icebreaker.user_likes_near
                WHERE sender_id = %s AND receiver_id = %s 
            """, (likedByUserId, user_id))

            cursor.execute("""
                DELETE FROM icebreaker.user_likes_near
                WHERE sender_id = %s AND receiver_id = %s 
            """, (user_id, likedByUserId))

            cursor.execute("""
                DELETE FROM icebreaker.session_likes_near
                WHERE user_id = %s AND wavedTouserId = %s 
            """, (likedByUserId, user_id))

            cursor.execute("""
                DELETE FROM icebreaker.session_likes_near
                WHERE user_id = %s AND wavedTouserId = %s 
            """, (user_id, likedByUserId))

            self.db.conn.commit()

            cursor.execute('''
                SELECT deviceToken
                FROM icebreaker.registration
                WHERE userId = %s
            ''', (likedByUserId,))
            receiver_info = cursor.fetchone()

            if receiver_info and receiver_info[0]:
                self.send_fcm_wave_deleted_silent(
                    token=receiver_info[0],
                    login_id=user_id,
                    waved_to_user_id=likedByUserId
                )

            cursor.close()

            return {"status": True, "message": "Like removed successfully"}

        except Exception as e:
            self.db.conn.rollback()
            return {"status": False, "error": str(e)} 


    def get_list_of_Requests_pending(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page
            cursor = self.db.cur()

            login_web_token = self._fetch_login_token_users_pending_user(cursor, user_id)
            logged_in_user_place_id = self._fetch_user_place_pending(cursor, user_id)

            total_accepted_users = self._fetch_total_pending_users(cursor, user_id)
            rows = self._fetch_paginated_pending_users(cursor, user_id, per_page, offset)

            results = []

            for row in rows:
                user_dict = self._build_user_dict_new_pending(cursor, row)
                if user_dict:  # prevents null in response
                    results.append(user_dict)

            return jsonify({
                'List': results,
                'Status': 'Success',
                'Page': page,
                'PerPage': per_page,
                'TotalAcceptedUsers': total_accepted_users,
                "loginWebToken": login_web_token
            })

        except Exception as e:
            self.db.conn.rollback()
            return {'error': str(e)}


    # ---------------- Helper methods ---------------- #

    def _fetch_login_token_users_pending_user(self, cursor, user_id):
        return self._get_login_web_token_confirm_api(cursor, user_id)


    def _fetch_user_place_pending(self, cursor, user_id):
        cursor.execute(self.SELECT_PLACE_ID_BY_USER, (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None


    def _fetch_total_pending_users(self, cursor, user_id):
        cursor.execute("""
            SELECT COUNT(netwirkRequestsId)
            FROM netwirkRequestsPending
            WHERE userId = %s
            AND requestedUserId NOT IN (
                SELECT reportedUserId 
                FROM icebreaker.netwirkReports 
                WHERE userId = %s
            )
        """, (user_id, user_id))
        return cursor.fetchone()[0]


    def _fetch_paginated_pending_users(self, cursor, user_id, per_page, offset):
        cursor.execute("""
            SELECT requestTime, userId, requestedUserID, status, netwirkRequestsId
            FROM netwirkRequestsPending
            WHERE userId = %s
            AND requestedUserId NOT IN (
                SELECT reportedUserId 
                FROM icebreaker.netwirkReports 
                WHERE userId = %s
            )
            ORDER BY requestTime DESC
            LIMIT %s OFFSET %s
        """, (user_id, user_id, per_page, offset))
        return cursor.fetchall()


    def _build_user_dict_new_pending(self, cursor, row):
        request_time, user_id, requested_user_id, status, _ = row

   
        cursor.execute("""
            SELECT 
                p.profilePic,
                r.fullName,
                r.email,
                r.firebaseId,
                r.signupType,
                r.authenticationId,
                r.Gender,
                r.Preference,
                r.DOB,
                p.vibes,
                p.vibing,
                l.placeId
            FROM icebreaker.registration r
            LEFT JOIN icebreaker.profile p ON p.userId = r.userId
            LEFT JOIN icebreaker.location l ON r.userId = l.userId
            WHERE r.userId = %s
        """, (requested_user_id,))

        res = cursor.fetchone()

        if not res:
            return None 

  
        cursor.execute(self.CHECK_BLOCKED_USER_REVERSE_QUERY, (requested_user_id, user_id))
        block_flag = 1 if cursor.fetchone() else 0

        (
            profilePic,
            fullName,
            email,
            firebaseId,
            signupType,
            authenticationId,
            Gender,
            Preference,
            DOB,
            vibes,
            vibing,
            placeId
        ) = res
        
        return {
            "RequestTime": request_time,
            "UserId": user_id,
            "RequestedUserID": requested_user_id,
            "Status": status,
            "profilePic": profilePic or '',
            "fullName": fullName or '',
            "email": email or '',
            "firebaseId": firebaseId or '',
            "signupType": signupType or '',
            "authenticationId": authenticationId or '',
            "Gender": Gender or '',
            "Preference": Preference or '',
            "DOB": DOB.strftime('%m-%d-%Y') if DOB else '',
            "vibes": vibes or '',
            "vibing": vibing or '',
            "placeId": placeId,
            "block_flag": block_flag
        }
    
    def count_of_vibe_notifications(self):
        try:
            data = request.get_json()
            user_id = data.get("loginUserId")

            cursor = self.db.cur()

            # 1️⃣ Count of requests received
            cursor.execute("""
                SELECT COUNT(nrp.netwirkRequestsId)
                FROM icebreaker.netwirkRequestsPending nrp
                JOIN icebreaker.registration r ON nrp.userId = r.userId
                WHERE nrp.requestedUserId = %s
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b1
                    WHERE b1.blockedId = nrp.userId AND b1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b2
                    WHERE b2.userId = nrp.userId AND b2.blockedId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r1
                    WHERE r1.reportedUserId = nrp.userId AND r1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r2
                    WHERE r2.userId = nrp.userId AND r2.reportedUserId = %s
                )
            """, (user_id, user_id, user_id, user_id, user_id))
            count_of_request_received = cursor.fetchone()[0]

            # 2️⃣ Count of requests sent
            cursor.execute("""
                SELECT COUNT(nrp.netwirkRequestsId)
                FROM icebreaker.netwirkRequestsPending nrp
                JOIN icebreaker.registration r ON nrp.requestedUserId = r.userId
                WHERE nrp.userId = %s
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b1
                    WHERE b1.blockedId = nrp.requestedUserId AND b1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b2
                    WHERE b2.userId = nrp.requestedUserId AND b2.blockedId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r1
                    WHERE r1.reportedUserId = nrp.requestedUserId AND r1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r2
                    WHERE r2.userId = nrp.requestedUserId AND r2.reportedUserId = %s
                )
            """, (user_id, user_id, user_id, user_id, user_id))
            count_of_request_sent = cursor.fetchone()[0]

            # 3️⃣ Count of likes (deduplicated across both tables)
            cursor.execute("""
                SELECT COUNT(DISTINCT liker_id) AS likes_count
                FROM (
                    SELECT ul.sender_id AS liker_id
                    FROM icebreaker.user_likes ul
                    JOIN icebreaker.registration r ON ul.sender_id = r.userId
                    WHERE ul.receiver_id = %s
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.blockedUsers b1
                        WHERE b1.blockedId = ul.sender_id AND b1.userId = %s
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.blockedUsers b2
                        WHERE b2.userId = ul.sender_id AND b2.blockedId = %s
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.netwirkReports r1
                        WHERE r1.reportedUserId = ul.sender_id AND r1.userId = %s
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.netwirkReports r2
                        WHERE r2.userId = ul.sender_id AND r2.reportedUserId = %s
                    )

                    UNION

                    SELECT uln.sender_id AS liker_id
                    FROM icebreaker.user_likes_near uln
                    JOIN icebreaker.registration r ON uln.sender_id = r.userId
                    WHERE uln.receiver_id = %s
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.blockedUsers b1
                        WHERE b1.blockedId = uln.sender_id AND b1.userId = %s
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.blockedUsers b2
                        WHERE b2.userId = uln.sender_id AND b2.blockedId = %s
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.netwirkReports r1
                        WHERE r1.reportedUserId = uln.sender_id AND r1.userId = %s
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM icebreaker.netwirkReports r2
                        WHERE r2.userId = uln.sender_id AND r2.reportedUserId = %s
                    )
                ) AS combined_likes
            """, (
                user_id, user_id, user_id, user_id, user_id,   # user_likes params
                user_id, user_id, user_id, user_id, user_id    # user_likes_near params
            ))
            count_of_likes = cursor.fetchone()[0]

            # 4️⃣ Count of connections
            cursor.execute("""
                SELECT COUNT(*)
                FROM icebreaker.netwirkRequestsAccepted nra
                JOIN icebreaker.registration r ON nra.userId = r.userId
                WHERE nra.userId = %s
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b1
                    WHERE b1.blockedId = nra.requestedUserId AND b1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.blockedUsers b2
                    WHERE b2.userId = nra.requestedUserId AND b2.blockedId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r1
                    WHERE r1.reportedUserId = nra.requestedUserId AND r1.userId = %s
                )
                AND NOT EXISTS (
                    SELECT 1 FROM icebreaker.netwirkReports r2
                    WHERE r2.userId = nra.requestedUserId AND r2.reportedUserId = %s
                )
            """, (user_id, user_id, user_id, user_id, user_id))
            count_of_connection = cursor.fetchone()[0]

            return {
                "count_of_request_received": count_of_request_received,
                "count_of_request_sent": count_of_request_sent,
                "count_of_likes": count_of_likes,
                "count_of_connection": count_of_connection
            }

        except Exception as e:
            return {"error": str(e)}, 500

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Haversine distance in meters between two lat/lon points."""
        R = 6_371_000

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )

        return R * 2 * asin(sqrt(a))


    # def search_within_100m(self):
    #     try:
    #         data = request.get_json(force=True, silent=True) or {}

    #         user_id = data.get("user_id")
    #         query = data.get("query", "").strip()

    #         # ✅ Fix: use explicit None checks so 0.0 is valid
    #         lat = data.get("latitude")
    #         lng = data.get("longitude")

    #         GOOGLE_API_KEY = self._google_api_key()

    #         if lat is None or lng is None or not query:
    #             return jsonify({"error": "latitude, longitude and query are required"}), 400

    #         # ✅ Fix: cast to float early to catch bad input immediately
    #         try:
    #             lat, lng = float(lat), float(lng)
    #         except (TypeError, ValueError):
    #             return jsonify({"error": "latitude and longitude must be numbers"}), 400

    #         # ✅ Fix: load from environment, never hardcode
    #         api_key = GOOGLE_API_KEY
    #         if not api_key:
    #             return jsonify({"error": "Server misconfiguration: missing API key"}), 500

    #         url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    #         params = {
    #             "query": query,
    #             "location": f"{lat},{lng}",
    #             "radius": 500,   # tighter candidate radius — 100m filter is strict anyway
    #             "key": api_key,
    #         }

    #         # ✅ Fix: paginate through all results (up to 3 pages / 60 places)
    #         candidates = []
    #         while True:
    #             resp = requests.get(url, params=params, timeout=5)
    #             resp.raise_for_status()
    #             body = resp.json()

    #             # print("Google API status:", body.get("status"))
    #             # print("Raw results count:", len(body.get("results", [])))
    #             # for place in body.get("results", []):
    #             #     loc = place["geometry"]["location"]
    #             #     dist = self.calculate_distance(lat, lng, loc["lat"], loc["lng"])
    #             #     print(f"  {place['name']} — {round(dist, 1)}m away")

    #             candidates.extend(body.get("results", []))

    #             next_token = body.get("next_page_token")
    #             if not next_token or len(candidates) >= 60:
    #                 break

    #             import time
    #             time.sleep(2)  # Google requires a short delay before next_page_token is valid
    #             params = {"pagetoken": next_token, "key": api_key}

    #         # ✅ Filter to strict 100 m radius
    #         RADIUS_M = 100
    #         final_results = []

    #         for place in candidates:
    #             place_name = place.get("name", "")
                
    #             # ✅ Skip places that don't match the partial query
    #             if not self.is_name_match(query, place_name):
    #                 continue

    #             loc = place.get("geometry", {}).get("location", {})
    #             place_lat = loc.get("lat")
    #             place_lng = loc.get("lng")

    #             if place_lat is None or place_lng is None:
    #                 continue

    #             distance = self.calculate_distance(lat, lng, place_lat, place_lng)


    #             if distance <= RADIUS_M:
    #                 final_results.append({
    #                     "place_id": place.get("place_id"),
    #                     "name": place.get("name"),
    #                     "address": place.get("formatted_address"),
    #                     "latitude": place_lat,
    #                     "longitude": place_lng,
    #                     "rating": place.get("rating", 0),
    #                     "distance_meters": round(distance, 2),
    #                 })

    #         final_results.sort(key=lambda x: x["distance_meters"])

    #         return jsonify({"count": len(final_results), "results": final_results})

    #     except requests.exceptions.RequestException as e:
    #         return jsonify({"error": f"Upstream API error: {e}"}), 502
    #     except Exception as e:
    #         return jsonify({"error": str(e)}), 500

    def is_name_match(self, query, place_name):
        query = query.lower()
        place_name = place_name.lower()
        

        # ✅ Direct partial match first
        if query in place_name:
            return True

        # ✅ Fuzzy match — 0.6 threshold catches typos
        ratio = SequenceMatcher(None, query, place_name).ratio()
        return ratio >= 0.6

    def search_within_100m(self):
        try:
            data = request.get_json(force=True, silent=True) or {}

            lat = data.get("latitude")
            lng = data.get("longitude")
            query = data.get("query", "").strip()
            user_id = data.get("user_id")
            radius = 300

            GOOGLE_API_KEY = self._google_api_key()

            if lat is None or lng is None or not query:
                return jsonify({"error": "latitude, longitude and query are required"}), 400

            try:
                lat, lng = float(lat), float(lng)
            except (TypeError, ValueError):
                return jsonify({"error": "latitude and longitude must be numbers"}), 400

            if len(query) < 2:
                return jsonify({"count": 0, "results": [], "message": "Type at least 2 characters"})

            api_key = GOOGLE_API_KEY
            if not api_key:
                return jsonify({"error": "Server misconfiguration: missing API key"}), 500

            cursor = self.db.cur()

            restricted_place_ids = self._get_restricted_places(cursor, user_id)

            # ─── Step 1: Get autocomplete suggestions ───────────────────────
            autocomplete_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
            params = {
                "input": query,
                "location": f"{lat},{lng}",
                "radius": radius,
                "strictbounds": True,
                "types": "establishment",
                "key": api_key,
            }

            resp = requests.get(autocomplete_url, params=params, timeout=5)
            resp.raise_for_status()
            body = resp.json()

            if body.get("status") not in ("OK", "ZERO_RESULTS"):
                return jsonify({"error": body.get("status"), "message": body.get("error_message")}), 400

            predictions = body.get("predictions", [])

            # ─── Step 2: Fetch coordinates for each suggestion ───────────────
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            suggestions = []

            predictions = predictions[:5]

            for prediction in predictions:
                place_id = prediction.get("place_id")

                # Get lat/lng for this place
                detail_resp = requests.get(details_url, params={
                    "place_id": place_id,
                    "fields": "name,geometry,formatted_address,rating",
                    "key": api_key,
                }, timeout=5)

                detail_body = detail_resp.json()

                if detail_body.get("status") != "OK":
                    continue

                result = detail_body.get("result", {})
                place_lat = result.get("geometry", {}).get("location", {}).get("lat")
                place_lng = result.get("geometry", {}).get("location", {}).get("lng")

                if place_lat is None or place_lng is None:
                    continue

                distance = self.calculate_distance(lat, lng, place_lat, place_lng)

                if distance <= radius:

                    cursor.execute(
                        "SELECT count(userId) FROM icebreaker.location WHERE placeId = %s",
                        (place_id,)
                    )
                    user_count = cursor.fetchone()[0]

                    incognito_allowed = place_id not in restricted_place_ids
                    suggestions.append({
                        "place_id": place_id,
                        "name": result.get("name"),
                        "address": result.get("formatted_address"),
                        "latitude": place_lat,
                        "longitude": place_lng,
                        "rating": result.get("rating", 0),
                        "distance_meters": round(distance, 2),
                        "matched_text": prediction.get("structured_formatting", {}).get("main_text"),
                        "user_count": user_count,                       
                        "incognito_checkin_allowed": incognito_allowed
                    })

            suggestions.sort(key=lambda x: x["distance_meters"])

            return jsonify({"count": len(suggestions), "results": suggestions})

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Upstream API error: {e}"}), 502
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def hide_user_from_list(self):
        cursor = None
        try:
            data = request.get_json()
            user_id = int(data['user_id'])
            target_user_id = int(data['target_user_id'])
            cursor = self.db.cur()

            # Insert into hidden list (ignore if already hidden)
            cursor.execute("""
                INSERT IGNORE INTO icebreaker.hidden_users (userId, hiddenUserId, hiddenAt)
                VALUES (%s, %s, NOW())
            """, (user_id, target_user_id))

            self.on_hide(cursor, user_id, target_user_id, recsys_action="hide_1_month")

            self.db.conn.commit()

            # Fetch device token(s) for the target user
            cursor.execute("""
                SELECT deviceToken FROM icebreaker.registration
                WHERE userId = %s AND deviceToken IS NOT NULL
            """, (target_user_id,))
            rows = cursor.fetchall()
            tokens = [row[0] for row in rows if row[0]]

            # Send silent FCM notification if token(s) exist
            if tokens:
                self.send_fcm_user_removed_silent(
                    token=tokens,
                    login_id=user_id,
                    waved_to_user_id=target_user_id
                )

            return jsonify({
                "Status": "Success",
                "Message": "User hidden from your list"
            })

        except KeyError as e:
            return jsonify({"Status": "Failure", "Message": f"Missing field: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"Status": "Failure", "Message": str(e)}), 500
        finally:
            if cursor:
                cursor.close()

    def send_fcm_user_removed_silent(self, token, login_id, waved_to_user_id):
        """Send a silent, data-only FCM notification when a wave is deleted."""
        if not token:
            return {"status": "error", "message": "No device token provided."}

        try:
            # SERVICE_ACCOUNT_FILE = f"D:\\icebreakerNew\\vibrr\\icebreakerfinal-firebase-adminsdk-6clsb-cf90b713b7.json"
            SERVICE_ACCOUNT_FILE = self.FIREBASE_SERVICE_ACCOUNT_FILE
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE,
                scopes=[self.CLOUD_PLATFORM_SCOPE]
            )
            credentials.refresh(Request())

            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": self.CONTENT_TYPE,
            }

            payload_data = {
                "type": "refresh_all_users",
                "user_id": str(login_id),
                "waved_to_user_id": str(waved_to_user_id),
            }

            def build_message(t: str) -> dict:
                return {
                    "message": {
                        "token": t,
                        "data": payload_data,
                        "android": {
                            "priority": "high"
                        },
                        "apns": {
                            "headers": {
                                "apns-priority": "10"
                            },
                            "payload": {
                                "aps": {
                                    "content-available": 1
                                }
                            }
                        }
                    }
                }

            def send_to_token(t: str) -> dict:
                if credentials.expiry and credentials.expired:
                    credentials.refresh(Request())
                    headers["Authorization"] = f"Bearer {credentials.token}"

                resp = requests.post(
                    self.FCM_SEND_URL,
                    headers=headers,
                    data=json.dumps(build_message(t))
                )
                success = resp.status_code == 200
                return {
                    "token": t,
                    "success": success,
                    "status_code": resp.status_code,
                    "response": resp.text
                }

            tokens = token if isinstance(token, list) else [token]
            send_results = [send_to_token(t) for t in tokens]

            any_success = any(r["success"] for r in send_results)

            return {
                "status": "success" if any_success else "error",
                "results": send_results if isinstance(token, list) else send_results[0]
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def haversine_km(self,lat1, lng1, lat2, lng2):
        R = 6371
        to_rad = math.radians
        d_lat = to_rad(lat2 - lat1)
        d_lng = to_rad(lng2 - lng1)
        a = (math.sin(d_lat / 2) ** 2
            + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(d_lng / 2) ** 2)
        return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)
    
 
    def resolve_metro(self,lat, lng):
        matched_metro = None
        nearest_metro = None
        nearest_dist = float("inf")
    
        for metro in SUPPORTED_METROS:
            dist = self.haversine_km(lat, lng, metro["lat"], metro["lng"])
    
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_metro = {**metro, "distanceKm": dist}
    
            if matched_metro is None and dist <= metro["radiusKm"]:
                matched_metro = {**metro, "distanceKm": dist}
    
        return matched_metro, nearest_metro
    
    
    
    def validate_location(self):
        try:
            data = request.get_json()

            user_id     = data.get("userId")          # parsed but available for future use
            lat         = float(data["latitude"])      # fixed: was "lat"
            lng         = float(data["longitude"])     # fixed: was "lng"
            device_city = data.get("deviceCity")       # optional

            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return jsonify({
                    "Status": "Failure",
                    "Message": "lat/lng values are out of valid range"
                }), 400

             # ── Waitlist lookup ──────────────────────────────
            on_waitlist = False
            if user_id:
                try:
                    cur = self.db.cur()
                    cur.execute("""
                        SELECT waitlist
                        FROM icebreaker.registration
                        WHERE userId = %s
                        LIMIT 1
                    """, (user_id,))
                    row = cur.fetchone()
                    cur.close()
                    if row and row[0] == 1:
                        on_waitlist = True
                except Exception:
                    on_waitlist = False 

            matched_metro, nearest_metro = self.resolve_metro(lat, lng)

            supported_metros = [
                {
                    "name":        m["name"],
                    "displayName": m["displayName"],
                    "lat":         m["lat"],
                    "lng":         m["lng"],
                    "radiusKm":    m["radiusKm"],
                }
                for m in SUPPORTED_METROS
            ]

            if matched_metro:
                return jsonify({
                    "allowed": True,
                    "reason": None,
                    "waitlist": on_waitlist, 
                    "matchedMetro": {
                        "name":        matched_metro["name"],
                        "displayName": matched_metro["displayName"],
                        "lat":         matched_metro["lat"],
                        "lng":         matched_metro["lng"],
                        "radiusKm":    matched_metro["radiusKm"],
                        "distanceKm":  matched_metro["distanceKm"],
                    },
                    "supportedMetros": supported_metros,
                })

            return jsonify({
                "allowed": False,
                "reason": "outside_radius",
                "waitlist": on_waitlist, 
                "nearestMetro": {
                    "name":        nearest_metro["name"],
                    "displayName": nearest_metro["displayName"],
                    "lat":         nearest_metro["lat"],
                    "lng":         nearest_metro["lng"],
                    "radiusKm":    nearest_metro["radiusKm"],
                    "distanceKm":  nearest_metro["distanceKm"],
                },
                "supportedMetros": supported_metros,
            })

        except KeyError as e:
            return jsonify({"Status": "Failure", "Message": f"Missing field: {str(e)}"}), 400
        except (ValueError, TypeError) as e:
            return jsonify({"Status": "Failure", "Message": f"Invalid value: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"Status": "Failure", "Message": str(e)}), 500

    def add_to_waitlist(self):
        try:
            data = request.get_json()

            email   = data.get("email")
            user_id = data.get("userId")

            if not email or not user_id:
                return jsonify({
                    "Status": "Failure",
                    "Message": "Both 'email' and 'userId' are required"
                }), 400

            cursor = self.db.cur()               

            cursor.execute("""
                UPDATE icebreaker.registration
                SET waitlist = 1
                WHERE userId = %s
            """, (user_id,))

            # if cursor.rowcount == 0:
            #     cursor.execute("""
            #         INSERT INTO icebreaker.registration (user_id, email, waitlist)
            #         VALUES (%s, %s, 1)
            #     """, (user_id, email))

            self.db.conn.commit()                         
            cursor.close()

            return jsonify({
                "Status":  "Success",
                "Message": "User added to waitlist",
                "userId":  user_id,
                "email":   email,
                "waitlist": True
            }), 200

        except Exception as e:
            return jsonify({"Status": "Failure", "Message": str(e)}), 500

    def get_user_list_offline(self):
   
        try:
            data = request.get_json()
            user_id = int(data['userId'])
            search_text = data.get('searchText', '')
            page = int(data.get('page', 1))
            # per_page = int(data.get('perPage', 1))
            per_page = 1
            offset = (page - 1) * per_page

            cursor = self.db.cur()

            ctx = self._fetch_offline_viewer_context(cursor, user_id)
            if ctx is None:
                return jsonify({'message': 'User not found.', 'Status': 'Failure'}), 404

            login_web_token, my_gender, my_preference, age_ranges, my_age, gender_filter, viewer_profile = ctx

            row = self._fetch_users_offline(
                cursor, user_id, gender_filter, age_ranges,
                my_age, my_gender,
            )



            return self._handle_recommend_mode_offline(
                cursor, user_id, row, page, per_page, search_text, login_web_token,
                viewer_profile=viewer_profile,
            )

        except Exception as e:
            payload = locals().get("data")
            logger.exception(
                "offline_feed event=error stage=handler user_id=%s",
                payload.get("userId") if isinstance(payload, dict) else None,
            )
            return jsonify({'message': 'An error occurred.', 'Status': 'Failure'}), 500

        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                self.db.conn.close()
            except Exception:
                pass


    # ──────────────────────────────────────────────
    # Data-fetch helpers (offline variants)
    # ──────────────────────────────────────────────

    def _fetch_offline_viewer_context(self, cursor, user_id):
        """Single query for viewer metadata used by offline list + SQL filters."""
        cursor.execute("""
            SELECT r.loginWebToken, r.Gender, r.Preference, r.DOB, p.age_preference,
                   r.fullName, p.vibes, p.answer3, p.answer4,
                   p.prompt1answer, p.prompt2answer,
                   live.latitude, live.longitude
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            LEFT JOIN icebreaker.user_live_location_offline live
                ON live.user_id = r.userId
            WHERE r.userId = %s
        """, (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        login_web_token, gender, preference, dob, age_pref_raw = row[:5]
        gender_filter = self._translate_preference_common(preference)
        age_ranges = self._translate_age_preference_industry_v1(age_pref_raw)
        my_age = self._calculate_age(dob) if dob else None
        viewer_profile = {
            "full_name": row[5] or "",
            "vibes": row[6],
            "answer3": row[7],
            "answer4": row[8],
            "prompt1": row[9],
            "prompt2": row[10],
            "lat": row[11],
            "lng": row[12],
        }
        return login_web_token, gender, preference, age_ranges, my_age, gender_filter, viewer_profile

    def _fetch_login_token_offline(self, cursor, user_id):
        cursor.execute(self.LOGIN_WEBTOKEN_QUERY, (user_id,))
        result = cursor.fetchall()
        return result[0][0] if result else None


    def _fetch_gender_preference_offline(self, cursor, user_id):
        cursor.execute(
            'SELECT Gender, Preference FROM icebreaker.registration WHERE userId = %s',
            (user_id,)
        )
        row = cursor.fetchone()
        return row if row else (None, None)


    def _fetch_age_preference_offline(self, cursor, user_id):
        cursor.execute(
            'SELECT age_preference FROM icebreaker.profile WHERE userId = %s',
            (user_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None


    def _fetch_user_dob_offline(self, cursor, user_id):
        cursor.execute(
            'SELECT DOB FROM icebreaker.registration WHERE userId = %s',
            (user_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None


    def _fetch_users_offline(
        self, cursor, user_id, gender_filter,
        age_ranges=None, my_age=None, my_gender=None,
        per_page=None, offset=0
    ):
        
        age_clause, age_params = self._build_age_filter_sql(age_ranges)
        reverse_clause, reverse_params = self._build_reverse_age_filter_sql(my_age)
        reverse_gender_clause, reverse_gender_params = self._build_reverse_gender_filter_sql(my_gender)

        query = f"""
            SELECT
                p.userId,
                r.fullName,
                p.vibes,
                p.answer3,
                p.answer4,
                p.prompt1answer,
                p.prompt2answer,
                live.latitude,
                live.longitude
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            LEFT JOIN icebreaker.blockedUsers b1
                ON p.userId = b1.blockedId AND b1.userId = %s
            LEFT JOIN icebreaker.blockedUsers b2
                ON p.userId = b2.userId  AND b2.blockedId = %s
            LEFT JOIN icebreaker.netwirkReports r1
                ON p.userId = r1.reportedUserId AND r1.userId = %s
            LEFT JOIN icebreaker.netwirkReports r2
                ON p.userId = r2.userId  AND r2.reportedUserId = %s
            LEFT JOIN icebreaker.hidden_users h
                ON p.userId = h.hiddenUserId AND h.userId = %s
            LEFT JOIN icebreaker.hidden_users h2
                ON p.userId = h2.userId  AND h2.hiddenUserId = %s

            -- 1-day hides (only active ones)
            LEFT JOIN icebreaker.hidden_users_1day hd1
                ON p.userId = hd1.hiddenUserId AND hd1.userId = %s AND hd1.expiresAt > NOW()
            LEFT JOIN icebreaker.hidden_users_1day hd2
                ON p.userId = hd2.userId  AND hd2.hiddenUserId = %s AND hd2.expiresAt > NOW()

            -- 1-month hides (only active ones)
            LEFT JOIN icebreaker.hidden_users_1month hm1
                ON p.userId = hm1.hiddenUserId AND hm1.userId = %s AND hm1.expiresAt > NOW()
            LEFT JOIN icebreaker.hidden_users_1month hm2
                ON p.userId = hm2.userId  AND hm2.hiddenUserId = %s AND hm2.expiresAt > NOW()
            LEFT JOIN icebreaker.user_likes_near lk
                ON lk.sender_id = %s AND lk.receiver_id = p.userId
            LEFT JOIN icebreaker.user_likes lk2
                ON lk2.sender_id = %s AND lk2.receiver_id = p.userId
            LEFT JOIN icebreaker.location loc ON p.userId = loc.userId
            LEFT JOIN icebreaker.radius rad   ON p.userId = rad.userId
            LEFT JOIN icebreaker.user_live_location_offline live
                ON live.user_id = p.userId
            WHERE p.userId != %s
            AND b1.blockedId   IS NULL
            AND b2.blockedId   IS NULL
            AND r1.reportedUserId IS NULL
            AND r2.reportedUserId IS NULL
            AND h.hiddenUserId IS NULL
            AND h2.userId      IS NULL
            -- 1-day hide filters
            AND hd1.hiddenUserId  IS NULL
            AND hd2.userId        IS NULL

            -- 1-month hide filters
            AND hm1.hiddenUserId  IS NULL
            AND hm2.userId        IS NULL
            AND (%s IS NULL OR r.Gender = %s)
            AND loc.userId IS NULL
            AND rad.userId IS NULL

            AND p.profilePic IS NOT NULL AND p.profilePic != ''
            AND p.answer3 IS NOT NULL AND p.answer3 != ''
            AND p.answer4 IS NOT NULL AND p.answer4 != ''
            AND p.prompt1question IS NOT NULL AND p.prompt1question != ''
            AND p.prompt1answer IS NOT NULL AND p.prompt1answer != ''
            AND p.prompt2question IS NOT NULL AND p.prompt2question != ''
            AND p.prompt2answer IS NOT NULL AND p.prompt2answer != ''
            AND lk.receiver_id IS NULL
            AND lk2.receiver_id IS NULL
            
            {"AND " + age_clause if age_clause else ""}
            {"AND " + reverse_clause if reverse_clause else ""}
            {"AND " + reverse_gender_clause if reverse_gender_clause else ""}
            {f"LIMIT %s OFFSET %s" if per_page is not None else ""}
        """

        params = (
            user_id, user_id,
            user_id, user_id,
            user_id, user_id,
            user_id, user_id,
            user_id, user_id,
            user_id,
            user_id,
            user_id,
            gender_filter, gender_filter,
            *age_params,
            *reverse_params,
            *reverse_gender_params,
        )
        if per_page is not None:
            params = (*params, per_page, offset)

        cursor.execute(query, params)
        return cursor.fetchall()


    # ──────────────────────────────────────────────
    # Recommend-mode handler (offline)
    # ──────────────────────────────────────────────

    # def _handle_recommend_mode_offline(self, cursor, user_id, rows, page, per_page, 
    #                                 search_text, login_web_token):
    #     try:
    #         recommendations = self.best_matches_cosine_vibr_offline(user_id, rows)
            
    #         output_list = recommendations.get('Matches', []) if recommendations.get('Status') == 'Success' else []

    #         if search_text:
    #             needle = search_text.lower()
    #             output_list = [u for u in output_list if needle in u.get('fullName', '').lower()]

    #         # Batch enrich everything at once
    #         if output_list:
    #             self._batch_enrich_users(cursor, user_id, output_list, [u['userId'] for u in output_list])

    #         # Final sorting
    #         output_list.sort(
    #             key=lambda u: (
    #                 -max(u.get('wavedTime', 0), u.get('wavedbackTime', 0)),
    #                 -u.get('similarity_score', 0)
    #             )
    #         )

    #         start = (page - 1) * per_page
    #         paginated_list = output_list[start:start + per_page]

    #         return jsonify({
    #             'List': paginated_list,
    #             'Status': 'Success',
    #             'Page': page,
    #             'PerPage': per_page,
    #             'TotalUsers': len(output_list),
    #             'loginWebToken': login_web_token,
    #         })

    #     except Exception as e:
    #         return jsonify({'message': 'An error occurred.', 'Status': 'Failure'}), 500
        
    def _batch_enrich_users(self, cursor, login_user_id, output_list, user_ids):
        """One-time batch fetch for all extra data"""
        fmt = ','.join(['%s'] * len(user_ids))
        
        # 1. Wave times (already good)
        cursor.execute(f"""
            SELECT sender_id, MAX(created_at)
            FROM icebreaker.user_likes_near
            WHERE sender_id IN ({fmt}) AND receiver_id = %s
            GROUP BY sender_id
        """, (*user_ids, login_user_id))
        waved = {row[0]: int(row[1].timestamp()) if row[1] else 0 for row in cursor.fetchall()}

        cursor.execute(f"""
            SELECT receiver_id, MAX(created_at)
            FROM icebreaker.user_likes_near
            WHERE sender_id = %s AND receiver_id IN ({fmt})
            GROUP BY receiver_id
        """, (login_user_id, *user_ids))
        wavedback = {row[0]: int(row[1].timestamp()) if row[1] else 0 for row in cursor.fetchall()}

        # 2. Pictures (batch)
        cursor.execute("""
            SELECT loginUserId, picture_urls, IsProfilePicture 
            FROM icebreaker.userPictures 
            WHERE loginUserId IN ({})
        """.format(fmt), user_ids)
        pics_map = {}
        for uid, url, is_profile in cursor.fetchall():
            pics_map.setdefault(uid, []).append({"url": url, "isProfilePicture": bool(is_profile)})

        # 3. Incognito + cognito_time
        cursor.execute(f"""
            SELECT userId, incognito, cognito_time 
            FROM icebreaker.location 
            WHERE userId IN ({fmt})
        """, user_ids)
        incognito_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        # Enrich
        for user in output_list:
            uid = user['userId']
            user['wavedTime'] = waved.get(uid, 0)
            user['wavedbackTime'] = wavedback.get(uid, 0)
            user['pictures'] = pics_map.get(uid, [])
            
            incog, cog_time = incognito_map.get(uid, (0, None))
            user['incognito'] = bool(incog)
            user['cognito_time'] = cog_time

    def best_matches_cosine_vibr_offline(self, user_id, row):
        try:
            cursor = self.db.cur()
            cursor.execute('''
                SELECT r.gender, r.Preference, p.vibes, p.vibing, r.state, r.city, 
                    p.zodiac_sign, p.aesthetic_score, r.DOB
                FROM icebreaker.registration r 
                JOIN icebreaker.profile p ON r.userId = p.userId 
                WHERE r.userId = %s''', (user_id,))
            user_row = cursor.fetchone()

            if not user_row:
                return {"Status": "Failure", "Message": "User not found"}

            my_gender, my_preference, my_vibes, _, my_state, my_city, my_zodiac, my_aesthetic, my_dob = user_row
            cursor.execute("""
                SELECT COUNT(*)
                FROM (
                    SELECT id
                    FROM icebreaker.user_likes
                    WHERE sender_id = %s

                    UNION ALL

                    SELECT id
                    FROM icebreaker.user_likes_near
                    WHERE sender_id = %s
                ) t
            """, (user_id, user_id))

            total_likes = cursor.fetchone()[0]

            my_vibes_set = set(my_vibes.lower().split(',')) if my_vibes else set()
            compatible_signs = self.find_compatible_signs(my_zodiac)

            prioritized_users = self._filter_by_gender_preference(row, my_gender, my_preference)

            if total_likes > 5:
                cursor.execute("""
                    SELECT receiver_id FROM (
                        SELECT receiver_id, created_at FROM icebreaker.user_likes      WHERE sender_id = %s
                        UNION ALL
                        SELECT receiver_id, created_at FROM icebreaker.user_likes_near WHERE sender_id = %s
                    ) t
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (user_id, user_id))
                past_sent = cursor.fetchall()


                return self._behavior_based_recommendation_offline(
                    cursor=cursor,
                    past_sent=past_sent,
                    prioritized_users=prioritized_users,
                    my_aesthetic_score=my_aesthetic,
                    max_users=50,
                    page=1,
                    per_page=1,
                    user_id=user_id
                )
                            
            return self._fallback_cosine_recommendation_offline(
                prioritized_users, my_gender, my_vibes_set, my_state, my_city,
                compatible_signs, my_aesthetic, my_dob, user_id,max_users=50,page=1,per_page=1
            )

        except Exception as e:
            return {"Status": "Failure", "Message": str(e)}

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass

    def _fallback_cosine_recommendation_offline(
        self, prioritized_users, my_gender, my_vibes_set,
        my_state, my_city, compatible_signs, my_aesthetic_score,
        my_dob, user_id, max_users=20, page=1, per_page=1
    ):
        final_row = prioritized_users[:max_users]
        if not final_row:
            return {"Status": "Success", "Matches": [], "Page": page, "PerPage": per_page, "TotalMatches": 0}

        user_map, encoded_data = self._prepare_encoding_data_offline(
            final_row, my_gender, my_vibes_set, my_state, my_city,
            compatible_signs, my_aesthetic_score, user_id
        )

        cos_sim = cosine_similarity(encoded_data)
        similarities = cos_sim[0][1:]  # Skip self
        sorted_uids = list(user_map.keys())
        sorted_scores = sorted(zip(sorted_uids, similarities), key=lambda x: x[1], reverse=True)
        # top_matches = sorted_scores[:max_users]
        top_matches = sorted_scores

        results = [
            {**user_map[uid], "similarity_score": round(score * 100, 2)}
            for uid, score in top_matches
        ]
        return {
            "Status": "Success",
            "Matches": results,
            "Page": page,
            "PerPage": per_page,
            "TotalMatches": len(sorted_scores)
        }
    
    def _safe_answer(self,val):
        if val is None:
            return ''
        if isinstance(val, (bytes, bytearray)):
            return val.decode('utf-8', errors='replace').strip().lower()
        return str(val).strip().lower()

    def _get_embedding(self, text: str):
        
        if not text or not text.strip():
            return np.zeros(384)  # all-MiniLM-L6-v2 output dim
        # return model.encode(text.strip(), normalize_embeddings=True)
        return np.zeros(384)

    def _cosine_similarity(self, vec1, vec2):
        """Returns cosine similarity between two vectors (already normalized → dot product)."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2))

    def _prepare_encoding_data_offline(
        self, final_row, my_gender, my_vibes_set, my_state, my_city,
        compatible_signs, my_aesthetic_score, user_id, my_dob=None
    ):
        if not final_row:
            return {}, np.zeros((1, 1))

        user_ids = [user[0] for user in final_row]

        # ====================== BATCH QUERIES ======================
        cursor = self.db.cur()
        fmt = ','.join(['%s'] * len(user_ids))

        # 1. Pictures
        cursor.execute(f"""
            SELECT loginUserId, picture_urls, IsProfilePicture 
            FROM icebreaker.userPictures 
            WHERE loginUserId IN ({fmt})
        """, user_ids)
        pics_map = {}
        for uid, url, is_prof in cursor.fetchall():
            pics_map.setdefault(uid, []).append({
                "url": url, 
                "isProfilePicture": bool(is_prof)
            })

        # Incognito + Cognito Time
        cursor.execute(f"""
            SELECT userId, incognito, cognito_time 
            FROM icebreaker.location 
            WHERE userId IN ({fmt})
        """, user_ids)
        incognito_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        # 3.Like Status (Batch)
        cursor.execute(f"""
            SELECT sender_id, receiver_id, MAX(created_at) as ts
            FROM icebreaker.user_likes_near
            WHERE (sender_id IN ({fmt}) AND receiver_id = %s) 
            OR (sender_id = %s AND receiver_id IN ({fmt}))
            GROUP BY sender_id, receiver_id
        """, (*user_ids, user_id, user_id, *user_ids))

        waved_time_map = {}
        wavedback_time_map = {}
        for sender, receiver, ts in cursor.fetchall():
            timestamp = int(ts.timestamp()) if ts else 0
            if sender == user_id:
                wavedback_time_map[receiver] = timestamp
            else:
                waved_time_map[sender] = timestamp

        # 4. Answers + Prompts (Batch)
        cursor.execute(f"""
            SELECT userId, answer3, answer4, prompt1answer, prompt2answer, prompt_embedding, prompt_embedding2
            FROM icebreaker.profile 
            WHERE userId IN ({fmt})
        """, user_ids)
        profile_data = {row[0]: row[1:] for row in cursor.fetchall()}

        cursor.execute(f"""
            SELECT user_id, latitude, longitude
            FROM icebreaker.user_live_location_offline
            WHERE user_id IN ({fmt})
        """, user_ids)
        coords_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        # 6. NEW: Fetch current user's coordinates
        cursor.execute("""
            SELECT latitude, longitude FROM icebreaker.user_live_location_offline WHERE user_id = %s
        """, (user_id,))
        my_coords_row = cursor.fetchone()
        my_coords = (my_coords_row[0], my_coords_row[1]) if my_coords_row else None

        cursor.execute("""
            SELECT prompt_embedding, prompt_embedding2
            FROM icebreaker.profile 
            WHERE userId = %s
        """, (user_id,))
        my_emb_row = cursor.fetchone()
        my_emb1 = self._load_embedding(my_emb_row[0]) if my_emb_row else None
        my_emb2 = self._load_embedding(my_emb_row[1]) if my_emb_row else None

        cursor.close()

        # ====================== PREPARE LISTS FOR ENCODING ======================
        user_map = {}
        genders, vibes_list, states = [], [], []
        zodiac_scores, aesthetics_score_list, ages = [], [], []
        answers_list = [[], []]     
        prompt_similarities = []
        proximity_scores = []

        

        for user in final_row:
            uid = user[0]
            dob = user[10]
            age = self.calculate_age(dob)

            # Get batched profile data
            ans3, ans4, p1, p2, raw_emb1, raw_emb2  = profile_data.get(uid, ('', '', '', '',None, None))
            user_ans3 = self._safe_answer(ans3)
            user_ans4 = self._safe_answer(ans4)
            user_p1 = p1 or ''
            user_p2 = p2 or ''

            user_emb1 = self._load_embedding(raw_emb1)
            user_emb2 = self._load_embedding(raw_emb2)

            sim1 = self._cosine_similarity_fast(my_emb1, user_emb1)
            sim2 = self._cosine_similarity_fast(my_emb2, user_emb2)

            # Average only available ones
            valid_sims = [s for s in [sim1, sim2] if s is not None]
            avg_prompt_sim = sum(valid_sims) / len(valid_sims) if valid_sims else 0.0

            # Incognito info from batch
            incog, cog_time = incognito_map.get(uid, (0, None))

            # Build user dict
            user_map[uid] = {
                "userId": uid,
                "selfiePic": user[2],
                "fullName": user[3],
                "email": user[4],
                "gender": user[8],
                "vibes": user[11],
                "profilePic": user[1],
                "firebaseId": user[5],
                "signupType": user[6],
                "authenticationId": user[7],
                "state": user[13],
                "city": user[14],
                "user_zodiac_sign": user[15],
                "user_aesthetic_score": user[16],
                "incognito": bool(incog),
                "cognito_time": cog_time,
                "wavedTime": waved_time_map.get(uid, 0),
                "wavedbackTime": wavedback_time_map.get(uid, 0),
                "wave_status": self.get_wave_status_sender(cursor, user_id, uid),
                "countryCode": user[18],
                "phoneNo": user[19],
                "country": user[20],
                "deviceToken": user[21],
                "height": user[28],
                "age_preference": user[29],
                "prompt1question": user[30],
                "prompt1answer": user_p1,
                "prompt2question": user[32],
                "prompt2answer": user_p2,
                "pictures": pics_map.get(uid, []),
                "DOB": str(dob) if dob else None,
                "what_makes_my_friends_laugh_about_me": user[22],
                "weekend_usually_looks_like": user[23],
                "Describe_your_communication_style": user[24],
                "What_brings_you_to_Vyb": user[25],
                "What_kind_of_partner_do_you_vibe_with_most": user[26],
                "What_does_everyday_vyb_look_like": user[27],
            }

            user_coords = coords_map.get(uid)
            prox = self._proximity_score(my_coords, user_coords, max_km=200)
            proximity_scores.append(prox)

            # Collect features for ML encoding
            genders.append(user[8] or '')
            vibes_list.append(user[11].lower().split(',') if user[11] else [])
            states.append(user[13] or '')
            zodiac_scores.append(1 if user[15] and str(user[15]).lower() in compatible_signs else 0)
            aesthetics_score_list.append(user[16] if user[16] is not None else 0)
            ages.append(age)
            answers_list[0].append(user_ans3)
            answers_list[1].append(user_ans4)
            prompt_similarities.append(avg_prompt_sim)

        # ====================== ADD CURRENT USER ======================
        self._add_current_user_to_encoding(
            user_map, genders, vibes_list, states, zodiac_scores,
            aesthetics_score_list, ages, answers_list, prompt_similarities,proximity_scores,
            my_gender, my_vibes_set, my_state, my_aesthetic_score,
            my_dob,my_p1=None, my_p2=None
        )

        # ====================== FINAL ENCODING ======================
        encoded_features = self._encode_features_offline(
            vibes_list, states, zodiac_scores, aesthetics_score_list,
            ages, answers_list, prompt_similarities,proximity_scores
        )

        return user_map, encoded_features


    def store_embedding(self, user_id, prompt1answer, prompt2answer):
        emb1 = self._get_embedding(prompt1answer or '')
        emb2 = self._get_embedding(prompt2answer or '')
        
        cursor = self.db.cur()
        cursor.execute("""
            UPDATE icebreaker.profile 
            SET prompt_embedding = %s,
                prompt_embedding2 = %s
            WHERE userId = %s
        """, (
            json.dumps(emb1.tolist()) if emb1 is not None else None,
            json.dumps(emb2.tolist()) if emb2 is not None else None,
            user_id
        ))
        cursor.close()
    
    def _cosine_similarity_fast(self, vec_a, vec_b):
        """Fast numpy cosine similarity — no model call."""
        if vec_a is None or vec_b is None:
            return None
        a = np.array(vec_a)
        b = np.array(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _load_embedding(self, raw):
        """Deserialize from DB (JSON string → list)."""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _derive_wave_status(self, user_id, uid, waved_time, wavedback_time):
   
        both = waved_time and wavedback_time

        if both:
            return "waved"          # mutual — matches `if mutual`
        elif wavedback_time:
            return "already_waved"  # current user waved, no reply yet
        elif waved_time:
            return "wave_back"      # they waved, current user hasn't replied
        else:
            return "no_wave"        # no interaction         # no wave interaction

    def _batch_get_wave_status(self, cursor, login_user_id, candidate_ids):
        """Batch wave status for offline/recsys (user_likes + user_likes_near)."""
        if not candidate_ids:
            return {}

        fmt = ",".join(["%s"] * len(candidate_ids))
        in_params = (*candidate_ids, login_user_id, *candidate_ids)

        cursor.execute(
            f"""
            SELECT sender_id FROM (
                SELECT sender_id FROM icebreaker.user_likes
                WHERE receiver_id = %s AND sender_id IN ({fmt})
                UNION
                SELECT sender_id FROM icebreaker.user_likes_near
                WHERE receiver_id = %s AND sender_id IN ({fmt})
            ) t
            """,
            (login_user_id, *in_params),
        )
        received_from = {row[0] for row in cursor.fetchall()}

        cursor.execute(
            f"""
            SELECT receiver_id FROM (
                SELECT receiver_id FROM icebreaker.user_likes
                WHERE sender_id = %s AND receiver_id IN ({fmt})
                UNION
                SELECT receiver_id FROM icebreaker.user_likes_near
                WHERE sender_id = %s AND receiver_id IN ({fmt})
            ) t
            """,
            (login_user_id, *in_params),
        )
        sent_to = {row[0] for row in cursor.fetchall()}

        return {
            uid: self._derive_wave_status(
                login_user_id,
                uid,
                1 if uid in received_from else 0,
                1 if uid in sent_to else 0,
            )
            for uid in candidate_ids
        }
        
    def _get_my_prompt_answers(self, user_id):
        cursor = self.db.cur()
        cursor.execute("""
            SELECT prompt1answer, prompt2answer 
            FROM icebreaker.profile WHERE userId = %s
        """, (user_id,))
        row = cursor.fetchone()
        cursor.close()
        return (row[0] or '', row[1] or '') if row else ('', '')


    def _add_current_user_to_encoding(self, user_map, genders, vibes_list, states,
                                    zodiac_scores, aesthetics_score_list, ages, answers_list,
                                    prompt_similarities,proximity_scores, my_gender, my_vibes_set, my_state,
                                     my_aesthetic, my_dob, my_p1=None, my_p2=None):
        
        my_age = self.calculate_age(my_dob)
        
        genders.insert(0, my_gender or '')
        vibes_list.insert(0, list(my_vibes_set))
        states.insert(0, my_state or '')
        zodiac_scores.insert(0, 1)
        aesthetics_score_list.insert(0, my_aesthetic if my_aesthetic is not None else 0)
        ages.insert(0, my_age)
        answers_list[0].insert(0, self._safe_answer(my_p1))
        answers_list[1].insert(0, self._safe_answer(my_p2))
        prompt_similarities.insert(0, 1.0)
        proximity_scores.insert(0, 1.0)


    def _encode_features_offline(self, vibes_list, states, zodiac_scores,
                            aesthetics_score_list, ages, answers_list, prompt_similarities,proximity_scores):
        
        vibes_mlb = MultiLabelBinarizer()
        vibes_encoded = vibes_mlb.fit_transform(vibes_list)

        state_encoded = OneHotEncoder(sparse_output=False, handle_unknown='ignore')\
                        .fit_transform(np.array(states).reshape(-1, 1))
        
        proximity_array = np.array(proximity_scores).reshape(-1, 1)

        # Answer encoding
        answer_encoded = self._encode_answers(answers_list)

        weights = {
            'vibes':      0.20,  
            'state':      0.05,  
            'proximity':  0.35,   
            'age':        0.10,   
            'answers':    0.30,   
            'prompt_sim': 0.10,   
        }
        encoded_features = np.concatenate([
            vibes_encoded * weights['vibes'],
            state_encoded * weights['state'],
            proximity_array * weights['proximity'],
            np.array(zodiac_scores).reshape(-1, 1) * 0,
            np.array(aesthetics_score_list).reshape(-1, 1).astype(float) * 0,
            MinMaxScaler().fit_transform(np.array(ages).reshape(-1, 1)) * weights['age'],
            answer_encoded * weights['answers'],
            np.array(prompt_similarities).reshape(-1, 1) * weights['prompt_sim'],
        ], axis=1)

        return encoded_features

    def _haversine_km(self, coord1, coord2):
        """Returns distance in km between two (lat, lng) tuples."""
        if not coord1 or not coord2:
            return None
        R = 6371
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    def _proximity_score(self, coord1, coord2, max_km=200):
        """
        Returns [0, 1]. Same spot → 1.0, max_km away → 0.0.
        Missing coords → 0.5 (neutral, not penalized).
        """
        dist = self._haversine_km(coord1, coord2)
        if dist is None:
            return 0.5
        return max(0.0, 1.0 - dist / max_km)


    def _encode_answers(self, answers_list):
        """Encode answer3 and answer4"""
        answer_encoded_parts = []
        for i in range(2):
            col = np.array(answers_list[i]).reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            answer_encoded_parts.append(enc.fit_transform(col))
        return np.concatenate(answer_encoded_parts, axis=1)

    def _age_from_dob(self, dob):
        """Return integer age from a date/datetime object, or None."""
        if not dob:
            return None
        today = date.today()
        return today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )
    
    
    def _age_compat(self, age_a, age_b):
        """
        Return 0.0–1.0 compatibility score based on age gap.
        Returns 0.5 (neutral) if either age is unknown.
        """
        if age_a is None or age_b is None:
            return 0.5
        diff = abs(age_a - age_b)
        if diff <= 1: return 1.00
        if diff <= 2: return 0.90
        if diff <= 3: return 0.78
        if diff <= 5: return 0.60
        if diff <= 8: return 0.35
        return 0.15

            
        
    def _behavior_based_recommendation_offline(
        self, cursor, past_sent, prioritized_users, my_aesthetic_score,
        max_users, page, per_page, user_id
    ):
        if not past_sent or not prioritized_users:
            return {
                "Status": "Success", "Matches": [], "Page": page,
                "PerPage": per_page, "TotalMatches": 0
            }
    
        # ── 1. Collect IDs ─────────────────────────────────────────────────────
        requested_ids = [r[0] for r in past_sent]
        candidate_ids = [u[0] for u in prioritized_users]
    
        fmt_req  = ','.join(['%s'] * len(requested_ids))
        fmt_cand = ','.join(['%s'] * len(candidate_ids))
    
        # ── 2. Fetch past-liked users → build behaviour taste profile ──────────
        # answer3 = Describe_your_communication_style
        # answer4 = What_brings_you_to_Vyb
        cursor.execute(f"""
            SELECT p.userId, p.vibes, p.answer3, p.answer4,
                p.prompt_embedding, p.prompt_embedding2, r.DOB
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            WHERE p.userId IN ({fmt_req})
        """, requested_ids)
    
        behavior_vibes       = set()
        behavior_comm        = set()   # comm styles seen in liked users
        behavior_intent      = set()   # intents seen in liked users
        behavior_embeddings1 = []
        behavior_embeddings2 = []
        behavior_ages = []

    
        for row in cursor.fetchall():
            uid, vibes, ans3, ans4, emb1_raw, emb2_raw, dob = row
        
            if vibes:
                behavior_vibes.update(v.strip() for v in vibes.lower().split(','))
            if self._safe_answer(ans3):
                behavior_comm.add(self._safe_answer(ans3))
            if self._safe_answer(ans4):
                behavior_intent.add(self._safe_answer(ans4))
            e1 = self._load_embedding(emb1_raw)
            e2 = self._load_embedding(emb2_raw)
            if e1 is not None:
                behavior_embeddings1.append(e1)
            if e2 is not None:
                behavior_embeddings2.append(e2)

            age = self._age_from_dob(dob)
            if age is not None:
                behavior_ages.append(age)
    
        # Average embedding of past liked users → "taste vector"
        avg_behavior_emb1 = (
            np.mean(behavior_embeddings1, axis=0).tolist()
            if behavior_embeddings1 else None
        )
        avg_behavior_emb2 = (
            np.mean(behavior_embeddings2, axis=0).tolist()
            if behavior_embeddings2 else None
        )

        if behavior_ages:
            pref_age_min = min(behavior_ages)
            pref_age_max = max(behavior_ages)
        else:
            pref_age_min = None
            pref_age_max = None
    
        # ── 3. Fetch MY own profile ────────────────────────────────────────────
        cursor.execute("""
            SELECT p.answer3, p.answer4,
                p.prompt_embedding, p.prompt_embedding2,
                p.vibes, r.DOB
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            WHERE p.userId = %s
        """, (user_id,))
        my_row = cursor.fetchone()
    
        my_ans3      = self._safe_answer(my_row[0]) if my_row else ''  # comm style
        my_ans4      = self._safe_answer(my_row[1]) if my_row else ''  # Vyb intent
        my_emb1      = self._load_embedding(my_row[2]) if my_row else None
        my_emb2      = self._load_embedding(my_row[3]) if my_row else None
        my_vibes_set = (
            set(v.strip() for v in my_row[4].lower().split(','))
            if my_row and my_row[4] else set()
        )
        my_dob       = my_row[5] if my_row else None
        my_age       = self._age_from_dob(my_dob)
    
        # Map my comm style and intent strings → matrix indices
        my_comm_idx   = COMM_STYLE_MAP.get(my_ans3)  # None if blank/unrecognised
        my_intent_idx = INTENT_MAP.get(my_ans4)       # None if blank/unrecognised
    
        # ── 4. Batch fetch candidate profiles ─────────────────────────────────
        cursor.execute(f"""
            SELECT userId, answer3, answer4,
                prompt1answer, prompt2answer,
                prompt_embedding, prompt_embedding2
            FROM icebreaker.profile
            WHERE userId IN ({fmt_cand})
        """, candidate_ids)
    
        candidate_profile_map = {}
        for row in cursor.fetchall():
            uid, ans3, ans4, p1, p2, emb1_raw, emb2_raw = row
            candidate_profile_map[uid] = {
                'ans3': self._safe_answer(ans3),  # comm style
                'ans4': self._safe_answer(ans4),  # Vyb intent
                'p1':   p1 or '',
                'p2':   p2 or '',
                'emb1': self._load_embedding(emb1_raw),
                'emb2': self._load_embedding(emb2_raw),
            }
    
        # ── 5. Batch fetch wave times ──────────────────────────────────────────
        cursor.execute(f"""
            SELECT sender_id, receiver_id, MAX(created_at) AS ts
            FROM icebreaker.user_likes_near
            WHERE (sender_id = %s AND receiver_id IN ({fmt_cand}))
            OR (receiver_id = %s AND sender_id IN ({fmt_cand}))
            GROUP BY sender_id, receiver_id
        """, (user_id, *candidate_ids, user_id, *candidate_ids))
    
        waved_time_map     = {}  # they liked me
        wavedback_time_map = {}  # I liked them
        for sender, receiver, ts in cursor.fetchall():
            timestamp = int(ts.timestamp()) if ts else 0
            if sender == user_id:
                wavedback_time_map[receiver] = timestamp
            else:
                waved_time_map[sender] = timestamp
    
        # ── 6. Batch fetch pictures ────────────────────────────────────────────
        cursor.execute(f"""
            SELECT loginUserId, picture_urls, IsProfilePicture
            FROM icebreaker.userPictures
            WHERE loginUserId IN ({fmt_cand})
        """, candidate_ids)
        pics_map = {}
        for uid, url, is_profile in cursor.fetchall():
            pics_map.setdefault(uid, []).append({
                "url": url,
                "isProfilePicture": bool(is_profile)
            })
    
        # ── 7. Batch fetch incognito ───────────────────────────────────────────
        cursor.execute(f"""
            SELECT userId, incognito, cognito_time
            FROM icebreaker.location
            WHERE userId IN ({fmt_cand})
        """, candidate_ids)
        incognito_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    
        # ── 8. Batch fetch live coordinates ───────────────────────────────────
        cursor.execute(f"""
            SELECT user_id, latitude, longitude
            FROM icebreaker.user_live_location_offline
            WHERE user_id IN ({fmt_cand})
        """, candidate_ids)
        coords_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    
        cursor.execute("""
            SELECT latitude, longitude
            FROM icebreaker.user_live_location_offline
            WHERE user_id = %s
        """, (user_id,))
        my_coords_row = cursor.fetchone()
        my_coords = (my_coords_row[0], my_coords_row[1]) if my_coords_row else None
    
        # ── 9. Score every candidate ───────────────────────────────────────────
        scored_users = []
    
        for user in prioritized_users:
            uid = user[0]
            dob = user[10]
    
            prof      = candidate_profile_map.get(uid, {})
            cand_ans3 = prof.get('ans3', '')   # comm style string
            cand_ans4 = prof.get('ans4', '')   # Vyb intent string
            cand_emb1 = prof.get('emb1')
            cand_emb2 = prof.get('emb2')
    
            # ── Signal 1: Vibe score ───────────────────────────────────────────
            # 60% behaviour taste (overlap with liked users' vibes)
            # 40% direct overlap with my own vibes
            vibe_score = 0.0
            if user[11]:
                candidate_vibes  = set(v.strip() for v in user[11].lower().split(','))
                behavior_overlap = (
                    len(behavior_vibes & candidate_vibes) / max(1, len(behavior_vibes))
                )
                my_overlap = (
                    len(my_vibes_set & candidate_vibes) / max(1, len(my_vibes_set))
                    if my_vibes_set else 0.0
                )
                vibe_score = 0.6 * behavior_overlap + 0.4 * my_overlap
    
            # ── Signal 2: Communication style compatibility ────────────────────
            # Matrix lookup — partial credit for adjacent styles
            # e.g. Text-first ↔ Voice notes = 0.80, Text-first ↔ In-person = 0.20
            # Also checks if candidate's comm style matches styles seen in liked users
            cand_comm_idx = COMM_STYLE_MAP.get(cand_ans3)
            if my_comm_idx is not None and cand_comm_idx is not None:
                direct_comm_score    = COMM_COMPAT[my_comm_idx][cand_comm_idx]
                behavior_comm_match  = 1.0 if cand_ans3 in behavior_comm else 0.0
                comm_score = 0.6 * direct_comm_score + 0.4 * behavior_comm_match
            else:
                comm_score = 0.5   # neutral — one or both sides didn't answer
    
            # ── Signal 3: What brings you to Vyb (dating intent) ──────────────
            # Overlap scoring — Long-term ↔ Long-term open to short = 0.85
            # Long-term ↔ Short-term = 0.10 (penalised but not zeroed)
            # Also checks if candidate's intent matches intents seen in liked users
            cand_intent_idx = INTENT_MAP.get(cand_ans4)
            if my_intent_idx is not None and cand_intent_idx is not None:
                direct_intent_score    = INTENT_COMPAT[my_intent_idx][cand_intent_idx]
                behavior_intent_match  = 1.0 if cand_ans4 in behavior_intent else 0.0
                intent_score = 0.6 * direct_intent_score + 0.4 * behavior_intent_match
            else:
                intent_score = 0.5   # neutral — one or both sides didn't answer
    
            # ── Signal 4: Prompt semantic similarity ───────────────────────────
            # 60% behaviour taste (avg embedding of liked users' prompts)
            # 40% my own prompt embeddings
            sim_behavior1 = self._cosine_similarity_fast(avg_behavior_emb1, cand_emb1)
            sim_behavior2 = self._cosine_similarity_fast(avg_behavior_emb2, cand_emb2)
            sim_mine1     = self._cosine_similarity_fast(my_emb1, cand_emb1)
            sim_mine2     = self._cosine_similarity_fast(my_emb2, cand_emb2)
    
            valid_behavior = [s for s in [sim_behavior1, sim_behavior2] if s is not None]
            valid_mine     = [s for s in [sim_mine1, sim_mine2]         if s is not None]
    
            avg_behavior_prompt = (
                sum(valid_behavior) / len(valid_behavior) if valid_behavior else 0.0
            )
            avg_mine_prompt = (
                sum(valid_mine) / len(valid_mine) if valid_mine else 0.0
            )
            prompt_score = 0.6 * avg_behavior_prompt + 0.4 * avg_mine_prompt
    
            # ── Signal 5: Age compatibility ────────────────────────────────────
            cand_age  = self._age_from_dob(dob)
            # Behaviour-learned age preference
            if pref_age_min is not None and cand_age is not None:
                behavior_age_score = 1.0 if pref_age_min <= cand_age <= pref_age_max else 0.0
            else:
                behavior_age_score = 0.5  # neutral fallback

            # 50% your own age compat + 50% what you've been liking
            age_score = 0.5 * self._age_compat(my_age, cand_age) + 0.5 * behavior_age_score
    
            # ── Signal 6: Proximity ────────────────────────────────────────────
            user_coords     = coords_map.get(uid)
            proximity_score = self._proximity_score(my_coords, user_coords, max_km=200)
    
            # ── Final weighted score (sums to 1.0) ─────────────────────────────
            final_score = (
                0.25 * prompt_score    +   # semantic prompt similarity
                0.20 * vibe_score      +   # behaviour + direct vibe overlap
                0.20 * comm_score      +   # communication style compatibility
                0.20 * intent_score    +   # dating intent overlap
                0.10 * proximity_score +   # distance
                0.05 * age_score           # age gap
            )
    
            incog, cog_time = incognito_map.get(uid, (0, None))
    
            scored_users.append((uid, final_score, {
                "userId":           uid,
                "fullName":         user[3],
                "email":            user[4],
                "gender":           user[8],
                "vibes":            user[11],
                "profilePic":       user[1],
                "selfiePic":        user[2],
                "firebaseId":       user[5],
                "signupType":       user[6],
                "authenticationId": user[7],
                "state":            user[13],
                "city":             user[14],
                "user_zodiac_sign": user[15],
                "user_aesthetic_score": user[16],
                "incognito":        bool(incog),
                "cognito_time":     cog_time,
                "wave_status": self.get_wave_status_sender(cursor, user_id, uid),
                "wavedTime":        waved_time_map.get(uid, 0),
                "wavedbackTime":    wavedback_time_map.get(uid, 0),
                "countryCode":      user[18],
                "phoneNo":          user[19],
                "country":          user[20],
                "deviceToken":      user[21],
                "what_makes_my_friends_laugh_about_me": user[22],
                "weekend_usually_looks_like":           user[23],
                "Describe_your_communication_style":    user[24],
                "What_brings_you_to_Vyb":               user[25],
                "What_kind_of_partner_do_you_vibe_with_most": user[26],
                "What_does_everyday_vyb_look_like":     user[27],
                "height":           user[28],
                "age_preference":   user[29],
                "prompt1question":  user[30],
                "prompt1answer":    prof.get('p1', ''),
                "prompt2question":  user[32],
                "prompt2answer":    prof.get('p2', ''),
                "pictures":         pics_map.get(uid, []),
                "DOB":              str(dob) if dob else None,
            }))
    
        # ── 10. Sort and return ────────────────────────────────────────────────
        sorted_scores = sorted(scored_users, key=lambda x: x[1], reverse=True)
        results = [
            {**profile, "similarity_score": round(score * 100, 2)}
            for uid, score, profile in sorted_scores
        ]
        return {
            "Status":       "Success",
            "Matches":      results,
            "Page":         page,
            "PerPage":      per_page,
            "TotalMatches": len(sorted_scores)
        }

    def hide_user_from_offline_list(self):
        cursor = None
        try:
            data = request.get_json()
            user_id = int(data['user_id'])
            target_user_id = int(data['target_user_id'])
            hide_duration = data.get('hide_duration')

            VALID_DURATIONS = ('1_day', '1_month')
            if hide_duration not in VALID_DURATIONS:
                return jsonify({
                    "Status": "Failure",
                    "Message": f"Invalid hide_duration. Must be one of: {', '.join(VALID_DURATIONS)}"
                }), 400

            cursor = self.db.cur()

            if hide_duration == '1_day':
                LATER_LIMIT = 3

                # Count hides within the last 10 hours (rolling window)
                cursor.execute("""
                    SELECT COUNT(*) FROM icebreaker.hidden_users_1day
                    WHERE userId = %s
                    AND hiddenAt >= NOW() - INTERVAL 10 HOUR
                """, (user_id,))
                used_count = cursor.fetchone()[0]

                if used_count >= LATER_LIMIT:
                    # Fetch when the oldest of the 3 will expire, so client knows when slot reopens
                    cursor.execute("""
                        SELECT MIN(hiddenAt) FROM (
                            SELECT hiddenAt FROM icebreaker.hidden_users_1day
                            WHERE userId = %s
                            AND hiddenAt >= NOW() - INTERVAL 10 HOUR
                            ORDER BY hiddenAt ASC
                            LIMIT 3
                        ) AS recent
                    """, (user_id,))
                    oldest_hidden_at = cursor.fetchone()[0]

                    return jsonify({
                        "Status": "Failure",
                        "Message": "'Later' limit reached. You can use 'later' 3 times per 10 hours.",
                        "later_remaining": 0,
                        # "retry_after": str(oldest_hidden_at)  # oldest slot reopens after 10hrs from this
                    }), 429

                cursor.execute("""
                    INSERT INTO icebreaker.hidden_users_1day
                        (userId, hiddenUserId, hiddenAt, expiresAt)
                    VALUES (%s, %s, NOW(), DATE_ADD(NOW(), INTERVAL 1 DAY))
                    ON DUPLICATE KEY UPDATE
                        hiddenAt = NOW(),
                        expiresAt = DATE_ADD(NOW(), INTERVAL 1 DAY)
                """, (user_id, target_user_id))

                self.on_hide(cursor, user_id, target_user_id, recsys_action="hide_1_day")


                self.db.conn.commit()

                # Re-count after insert so remaining is always accurate
                cursor.execute("""
                    SELECT COUNT(*) FROM icebreaker.hidden_users_1day
                    WHERE userId = %s
                    AND hiddenAt >= NOW() - INTERVAL 10 HOUR
                """, (user_id,))
                used_count_after = cursor.fetchone()[0]
                later_remaining = LATER_LIMIT - used_count_after

                self._invalidate_offline_feed_cache(user_id)
                return jsonify({
                    "Status": "Success",
                    "Message": "User hidden from your list",
                    "later_remaining": later_remaining
                })

            elif hide_duration == '1_month':
                cursor.execute("""
                    INSERT INTO icebreaker.hidden_users_1month
                        (userId, hiddenUserId, hiddenAt, expiresAt)
                    VALUES (%s, %s, NOW(), DATE_ADD(NOW(), INTERVAL 1 MONTH))
                    ON DUPLICATE KEY UPDATE
                        hiddenAt = NOW(),
                        expiresAt = DATE_ADD(NOW(), INTERVAL 1 MONTH)
                """, (user_id, target_user_id))

                self.on_hide(cursor, user_id, target_user_id, recsys_action="hide_1_month")

                self.db.conn.commit()

                self._invalidate_offline_feed_cache(user_id)
                return jsonify({
                    "Status": "Success",
                    "Message": "User hidden from your list for 1 month"
                })

        except KeyError as e:
            return jsonify({"Status": "Failure", "Message": f"Missing field: {str(e)}"}), 400
        except ValueError:
            return jsonify({"Status": "Failure", "Message": "user_id and target_user_id must be integers"}), 400
        except Exception as e:
            return jsonify({"Status": "Failure", "Message": str(e)}), 500
        finally:
            if cursor:
                cursor.close()

    def get_liked_by_me_user_list(self):
        try:
            data = request.get_json()
            user_id = data['userId']

            # place_id = data.get('placeId') or "ChIJRVR9ciHP5zsRtrFD4nhXuBQ"

            page = int(data.get('page', 1))
            per_page = int(data.get('perPage', 10))
            offset = (page - 1) * per_page
            search_text = data.get('searchText', '')

            cursor = self.db.cur()
            place_id = self._resolve_place_id(user_id, data, cursor)

            cursor.execute("""
                SELECT
                    r.userId, r.fullName, p.profilePic, r.email,
                    r.Gender, r.Preference,
                    p.vibes, p.vibing, p.zodiac_sign, p.height, p.selfiePic, r.DOB, r.firebaseId,
                    r.signupType, r.deviceToken, r.authenticationId, r.countryCode, r.phoneNo, r.city, r.state, r.country
                FROM icebreaker.registration r
                JOIN icebreaker.profile p
                    ON r.userId = p.userId
                JOIN (
                    SELECT receiver_id, MAX(created_at) AS latest_waved_time
                    FROM (
                        SELECT receiver_id, created_at FROM icebreaker.user_likes ul
                        WHERE ul.sender_id = %s
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b1
                            WHERE b1.blockedId = ul.receiver_id AND b1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b2
                            WHERE b2.userId = ul.receiver_id AND b2.blockedId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r1
                            WHERE r1.reportedUserId = ul.receiver_id AND r1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r2
                            WHERE r2.userId = ul.receiver_id AND r2.reportedUserId = %s
                        )

                        UNION ALL

                        SELECT receiver_id, created_at FROM icebreaker.user_likes_near uln
                        WHERE uln.sender_id = %s
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b1
                            WHERE b1.blockedId = uln.receiver_id AND b1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.blockedUsers b2
                            WHERE b2.userId = uln.receiver_id AND b2.blockedId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r1
                            WHERE r1.reportedUserId = uln.receiver_id AND r1.userId = %s
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM icebreaker.netwirkReports r2
                            WHERE r2.userId = uln.receiver_id AND r2.reportedUserId = %s
                        )

                    ) all_waves
                    GROUP BY receiver_id
                ) liked_users
                    ON liked_users.receiver_id = r.userId
                ORDER BY liked_users.latest_waved_time DESC
                LIMIT %s OFFSET %s
            """, (
                user_id, user_id, user_id, user_id, user_id,   # user_likes params
                user_id, user_id, user_id, user_id, user_id,   # user_likes_near params
                per_page, offset
            ))

            rows = cursor.fetchall()
            liked_users = []

            for row in rows:
                waved_user_id = row[0]

                status = self.get_wave_status_sender(cursor, user_id, waved_user_id)

                cursor.execute("""
                    SELECT lastping
                    FROM icebreaker.user_pings
                    WHERE userId = %s
                """, (waved_user_id,))
                ping_result = cursor.fetchone()

                online_status = "offline"
                last_seen = ""

                if ping_result and ping_result[0]:
                    last_ping = ping_result[0]

                    if isinstance(last_ping, str):
                        last_ping_dt = datetime.strptime(last_ping, "%Y-%m-%d %H:%M:%S")
                    else:
                        last_ping_dt = last_ping

                    now = datetime.utcnow()
                    time_diff = (now - last_ping_dt).total_seconds()

                    if time_diff <= 30:
                        online_status = "online"

                    last_seen = last_ping_dt.strftime("%Y-%m-%d %H:%M:%S")

                user_dict = {
                    "userId": row[0],
                    "fullName": row[1],
                    "profilePic": row[2],
                    "email": row[3],
                    "Gender": row[4],
                    "Preference": row[5],
                    "vibes": row[6],
                    "vibing": row[7],
                    "zodiac_sign": row[8],
                    "height": row[9],
                    "selfiePic": row[10],
                    "DOB": row[11],
                    "firebaseId": row[12],
                    "signupType": row[13],
                    "deviceToken": row[14],
                    "authenticationId": row[15],
                    "countryCode": row[16],
                    "phoneNo": row[17],
                    "city": row[18],
                    "state": row[19],
                    "country": row[20],
                    "wave_status": status,
                    "online_status": online_status,
                    "last_seen": last_seen
                }

                accepted_status, pending_status, received_status = self._fetch_request_status_industry_v1(
                    cursor, user_id, waved_user_id, place_id
                )
                user_dict["status"] = self._determine_status_industry_v1(
                    accepted_status, pending_status, received_status
                )

                liked_users.append(user_dict)

            if search_text:
                liked_users = self._filter_liked_users_search(liked_users, search_text)
                total_likes = len(liked_users)

            total_likes = len(liked_users)

            return {
                "status": True,
                "page": page,
                "per_page": per_page,
                "TotalLikes": total_likes,
                "data": liked_users
            }

        except Exception as e:
            return {"status": False, "error": str(e)}

    def user_live_location_offline(self):
        try:
            data = request.get_json()
            user_id = data['userId']
            lat = data['latitude']
            lon = data['longitude']
            city = data['city']
            state = data['state']

            cursor = self.db.cur()

            

            # --- Insert or update user location ---
            cursor.execute(
                "SELECT user_id FROM icebreaker.user_live_location_offline WHERE user_id = %s",
                (user_id,)
            )

            if cursor.fetchone():
                cursor.execute(
                    "UPDATE icebreaker.user_live_location_offline SET latitude = %s, longitude = %s, city = %s, state = %s WHERE user_id = %s",
                    (lat, lon, city, state, user_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO icebreaker.user_live_location_offline (user_id, latitude, longitude, city, state) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, lat, lon, city, state)
                )

            self.db.conn.commit()

            return {
            "status": "success",
            "message": "Location updated successfully"
        }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

        finally:
            cursor.close()

    def get_later_remaining(self):
        cursor = None
        try:
            data = request.get_json()
            user_id = int(data['userId'])

            LATER_LIMIT = 3

            cursor = self.db.cur()

            # Rolling 10-hour window
            cursor.execute("""
                SELECT COUNT(*) FROM icebreaker.hidden_users_1day
                WHERE userId = %s
                AND hiddenAt >= NOW() - INTERVAL 10 HOUR
            """, (user_id,))
            used_count = cursor.fetchone()[0]
            remaining = max(0, LATER_LIMIT - used_count)

            # Tell the client when the next slot reopens
            next_available_at = None
            if remaining == 0:
                cursor.execute("""
                    SELECT MIN(hiddenAt) FROM (
                        SELECT hiddenAt FROM icebreaker.hidden_users_1day
                        WHERE userId = %s
                        AND hiddenAt >= NOW() - INTERVAL 10 HOUR
                        ORDER BY hiddenAt ASC
                        LIMIT 3
                    ) AS recent
                """, (user_id,))
                oldest = cursor.fetchone()[0]
                if oldest:
                    next_available_at = str(oldest)  # client adds 10hrs to this

            return jsonify({
                "Status": "Success",
                "user_id": user_id,
                "later_remaining": remaining,
                # "next_available_at": next_available_at  # null if slots are available
            })

        except KeyError as e:
            return jsonify({"Status": "Failure", "Message": f"Missing field: {str(e)}"}), 400
        except ValueError:
            return jsonify({"Status": "Failure", "Message": "user_id must be an integer"}), 400
        except Exception as e:
            return jsonify({"Status": "Failure", "Message": str(e)}), 500
        finally:
            if cursor:
                cursor.close()

    def _attach_offline_wave_times(self, cursor, viewer_id, matches):
        """Incoming and outgoing wave times for sort. One query per direction, not per candidate."""
        if not matches:
            return
        cursor.execute(
            """
            SELECT sender_id, MAX(created_at)
            FROM (
                SELECT sender_id, created_at
                FROM icebreaker.user_likes_near
                WHERE receiver_id = %s
                UNION ALL
                SELECT sender_id, created_at
                FROM icebreaker.user_likes
                WHERE receiver_id = %s
            ) t
            GROUP BY sender_id
            """,
            (viewer_id, viewer_id),
        )
        waved = {
            row[0]: int(row[1].timestamp()) if row[1] else 0
            for row in cursor.fetchall()
        }
        cursor.execute(
            """
            SELECT receiver_id, MAX(created_at)
            FROM (
                SELECT receiver_id, created_at
                FROM icebreaker.user_likes_near
                WHERE sender_id = %s
                UNION ALL
                SELECT receiver_id, created_at
                FROM icebreaker.user_likes
                WHERE sender_id = %s
            ) t
            GROUP BY receiver_id
            """,
            (viewer_id, viewer_id),
        )
        wavedback = {
            row[0]: int(row[1].timestamp()) if row[1] else 0
            for row in cursor.fetchall()
        }
        for match in matches:
            uid = match["userId"]
            match["wavedTime"] = waved.get(uid, 0)
            match["wavedbackTime"] = wavedback.get(uid, 0)

    def _offline_feed_card(self, row, similarity_score, ranker):
        return {
            "userId": row[0],
            "profilePic": row[1],
            "selfiePic": row[2],
            "fullName": row[3],
            "email": row[4],
            "firebaseId": row[5],
            "signupType": row[6],
            "authenticationId": row[7],
            "gender": row[8],
            "preference": row[9],
            "DOB": str(row[10]) if row[10] else None,
            "vibes": row[11],
            "state": row[13],
            "city": row[14],
            "user_zodiac_sign": row[15],
            "user_aesthetic_score": row[16],
            "countryCode": row[18],
            "phoneNo": row[19],
            "country": row[20],
            "deviceToken": row[21],
            "what_makes_my_friends_laugh_about_me": row[22],
            "weekend_usually_looks_like": row[23],
            "Describe_your_communication_style": row[24],
            "What_brings_you_to_Vyb": row[25],
            "What_kind_of_partner_do_you_vibe_with_most": row[26],
            "What_does_everyday_vyb_look_like": row[27],
            "height": row[28],
            "age_preference": row[29],
            "prompt1question": row[30],
            "prompt1answer": row[31] or "",
            "prompt2question": row[32],
            "prompt2answer": row[33] or "",
            "similarity_score": similarity_score,
            "ranker": ranker,
        }

    def _hydrate_offline_feed_page(self, cursor, viewer_id, page_matches):
        """Load the full profile, pictures, and wave status for the page only."""
        if not page_matches:
            return []
        ids = [match["userId"] for match in page_matches]
        meta = {match["userId"]: match for match in page_matches}
        fmt = ",".join(["%s"] * len(ids))
        cursor.execute(
            f"""
            SELECT
                p.userId,
                p.profilePic,
                p.selfiePic,
                r.fullName,
                r.email,
                r.firebaseId,
                r.signupType,
                r.authenticationId,
                r.Gender,
                r.Preference,
                r.DOB,
                p.vibes,
                p.vibing,
                r.state,
                r.city,
                p.zodiac_sign,
                p.aesthetic_score,
                0 AS incognito,
                r.countryCode,
                r.phoneNo,
                r.country,
                r.deviceToken,
                p.answer1,
                p.answer2,
                p.answer3,
                p.answer4,
                p.answer5,
                p.answer6,
                p.height,
                p.age_preference,
                p.prompt1question,
                p.prompt1answer,
                p.prompt2question,
                p.prompt2answer
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            WHERE p.userId IN ({fmt})
            """,
            tuple(ids),
        )
        rows = {row[0]: row for row in cursor.fetchall()}
        cards = []
        for uid in ids:
            row = rows.get(uid)
            if not row:
                continue
            match = meta[uid]
            cards.append(self._offline_feed_card(
                row, match.get("similarity_score", 0), match.get("ranker"),
            ))
        if not cards:
            return []
        card_ids = [card["userId"] for card in cards]
        fmt = ",".join(["%s"] * len(card_ids))
        cursor.execute(
            f"""
            SELECT loginUserId, picture_urls, IsProfilePicture
            FROM icebreaker.userPictures
            WHERE loginUserId IN ({fmt})
            """,
            tuple(card_ids),
        )
        pics_map = {}
        for uid, url, is_profile in cursor.fetchall():
            pics_map.setdefault(uid, []).append({
                "url": url,
                "isProfilePicture": bool(is_profile),
            })
        for card in cards:
            match = meta[card["userId"]]
            waved_time = match.get("wavedTime", 0)
            wavedback_time = match.get("wavedbackTime", 0)
            card["wavedTime"] = waved_time
            card["wavedbackTime"] = wavedback_time
            card["wave_status"] = self._derive_wave_status(
                viewer_id, card["userId"], waved_time, wavedback_time,
            )
            card["pictures"] = pics_map.get(card["userId"], [])
            card["incognito"] = False
            card["cognito_time"] = None
        return cards

    def _handle_recommend_mode_offline(self, cursor, user_id, rows, page, per_page,
                                        search_text, login_web_token,
                                        session_hidden_ids=None,
                                        viewer_profile=None):  # ← add this param
        try:
          
            result = self.recommend_recsys(
                cursor=cursor,
                user_id=user_id,
                candidates=rows,
                session_hidden_ids=session_hidden_ids or [],
                viewer_profile=viewer_profile,
            )
            output_list = result.get('Matches', [])
            # ──────────────────────────────────────────────────────────────

            if search_text:
                needle = search_text.lower()
                output_list = [u for u in output_list if needle in u.get('fullName', '').lower()]

            self._attach_offline_wave_times(cursor, user_id, output_list)

            if result.get("ranker") == "sql_fallback":
                # Someone who waved you comes first, newest wave first.
                # Everyone else stays in nearest-user order.
                output_list.sort(
                    key=lambda u: (
                        -u.get("wavedTime", 0),
                        -u.get("similarity_score", 0),
                    )
                )
            else:
                output_list.sort(
                    key=lambda u: (
                        -max(u.get('wavedTime', 0), u.get('wavedbackTime', 0)),
                        -u.get('similarity_score', 0)
                    )
                )

            start = (page - 1) * per_page
            paginated_list = output_list[start:start + per_page]
            paginated_list = self._hydrate_offline_feed_page(cursor, user_id, paginated_list)

            return jsonify({
                'List': paginated_list,
                'Status': 'Success',
                'Page': page,
                'PerPage': per_page,
                'TotalUsers': result['TotalMatches'],  # ← use result, not len(output_list)
                'loginWebToken': login_web_token,
            })

        except Exception as e:
            logger.exception(
                "offline_feed event=error stage=recommend user_id=%s",
                user_id,
            )
            return jsonify({'message': 'An error occurred.', 'Status': 'Failure'}), 500

    def recommend_recsys(
        self,
        cursor,
        user_id: int,
        candidates: list,
        session_hidden_ids: list = None,
        max_users: int = None,
        viewer_profile: dict = None,
    ) -> dict:
        """Rank SQL candidates via Amazon Personalize behavioural recommendations."""
        from Model.recsys.flask_adapter import OfflineRecsysRunner
        runner = OfflineRecsysRunner(self)
        ranking_k = max_users if max_users is not None else len(candidates)
        return runner.run(
            cursor=cursor,
            user_id=user_id,
            sql_candidates=candidates,
            session_hidden_ids=session_hidden_ids or [],
            ranking_k=ranking_k,
            viewer_profile=viewer_profile,
        )

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
 
    def _bp_rebuild(self, cursor, user_id: int) -> dict:
        """
        Full rebuild from DB history.
        Reads user_likes + user_likes_near for positive signals.
        Reads hidden_users_1day + hidden_users_1month for negative signals.
        """
        profile = self._bp_empty()
 
        # ── Positive: waves sent (sender_id / receiver_id / created_at) ──────
        cursor.execute("""
            SELECT receiver_id, created_at
            FROM (
                SELECT receiver_id, created_at
                FROM icebreaker.user_likes
                WHERE sender_id = %s
                UNION ALL
                SELECT receiver_id, created_at
                FROM icebreaker.user_likes_near
                WHERE sender_id = %s
            ) t
            ORDER BY created_at DESC
            LIMIT 100
        """, (user_id, user_id))
        positive_interactions = cursor.fetchall()
 
        # ── Mutual matches ────────────────────────────────────────────────────
        cursor.execute("""
            SELECT DISTINCT a.receiver_id
            FROM (
                SELECT receiver_id FROM icebreaker.user_likes      WHERE sender_id = %s
                UNION
                SELECT receiver_id FROM icebreaker.user_likes_near WHERE sender_id = %s
            ) a
            JOIN (
                SELECT sender_id FROM icebreaker.user_likes      WHERE receiver_id = %s
                UNION
                SELECT sender_id FROM icebreaker.user_likes_near WHERE receiver_id = %s
            ) b ON a.receiver_id = b.sender_id
        """, (user_id, user_id, user_id, user_id))
        mutual_ids = {r[0] for r in cursor.fetchall()}
 
        # ── Negative: hidden users (userId / hiddenUserId / hiddenAt) ─────────
        cursor.execute("""
            SELECT hiddenUserId, hiddenAt
            FROM (
                SELECT hiddenUserId, hiddenAt
                FROM icebreaker.hidden_users_1day
                WHERE userId = %s
                UNION ALL
                SELECT hiddenUserId, hiddenAt
                FROM icebreaker.hidden_users_1month
                WHERE userId = %s
            ) t
            ORDER BY hiddenAt DESC
            LIMIT 300
        """, (user_id, user_id))
        hidden_interactions = cursor.fetchall()
 
        if positive_interactions:
            profile = self._bp_build_positive(cursor, positive_interactions, mutual_ids, profile)
 
        if hidden_interactions:
            profile = self._bp_build_negative(cursor, [r[0] for r in hidden_interactions], profile)
 
        profile["interaction_counts"] = {
            "waves":   len(positive_interactions),
            "hides":   len(hidden_interactions),
            "matches": len(mutual_ids),
        }
        profile["dynamic_weights"] = self._bp_learn_weights(profile)
 
        self._bp_save(cursor, user_id, profile)
        return profile
 
    # =========================================================================
    # BEHAVIOUR PROFILE — Incremental Updates
    # =========================================================================
 
    def on_wave_sent(self, cursor, user_id: int, target_id: int):
        """
        Call this after inserting into user_likes or user_likes_near.
        target_id = receiver_id of the wave just sent.
        """
        profile = self._bp_load(cursor, user_id)
 
        cursor.execute("""
            SELECT p.vibes, p.answer3, p.answer4,
                   p.prompt_embedding, p.prompt_embedding2, r.DOB
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            WHERE p.userId = %s
        """, (target_id,))
        row = cursor.fetchone()
        if not row:
            return
 
        vibes, ans3, ans4, emb1_raw, emb2_raw, dob = row
 
        if vibes:
            for v in vibes.lower().split(','):
                v = v.strip()
                profile["taste_vibes"][v] = profile["taste_vibes"].get(v, 0) + 1.0
 
        profile = self._bp_update_embedding(
            profile, self._load_embedding(emb1_raw), self._load_embedding(emb2_raw), weight=1.0
        )
 
        age = self._age_from_dob(dob)
        if age:
            mn = profile["pref_age_min"]
            mx = profile["pref_age_max"]
            profile["pref_age_min"] = min(mn, age) if mn is not None else age
            profile["pref_age_max"] = max(mx, age) if mx is not None else age
 
        profile["interaction_counts"]["waves"] = \
            profile["interaction_counts"].get("waves", 0) + 1
        profile["dynamic_weights"] = self._bp_learn_weights(profile)

        self._log_recsys_interaction(cursor, user_id, target_id, "like", "on_wave_sent")
        self._bp_save(cursor, user_id, profile)

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

    # =========================================================================
    # RECOMMENDATION — Main Entry Point
    # =========================================================================
 
    def _get_user_tag_counts(self, cursor, user_id, months=3):
        """Aggregate check-in counts per lifestyle tag from sessiontracker."""
        cursor.execute("""
            SELECT primary_type, COUNT(*) AS cnt
            FROM icebreaker.sessiontracker
            WHERE userid = %s
              AND checkintime >= DATE_SUB(NOW(), INTERVAL %s MONTH)
            GROUP BY primary_type
        """, (user_id, months))

        tag_counts = {tag: 0 for tag in ALL_TAGS}
        for primary_type, cnt in cursor.fetchall():
            tag = primary_type_to_tag(primary_type)
            if tag:
                tag_counts[tag] += cnt
        return tag_counts

    def _normalize_tag_weights(self, tag_counts):
        """sqrt-normalize raw counts into comparable weights (per doc example)."""
        weights = {tag: math.sqrt(count) for tag, count in tag_counts.items()}
        total = sum(weights.values())
        return weights, total

    def _lifestyle_score(self, my_tag_weights, my_total, cand_tag_counts):
        if my_total <= 0:
            return 0.5
        cand_weights, _ = self._normalize_tag_weights(cand_tag_counts)
        weighted_shared = 0.0
        for tag in ALL_TAGS:
            mine, theirs = my_tag_weights.get(tag, 0.0), cand_weights.get(tag, 0.0)
            if mine > 0 and theirs > 0:
                weighted_shared += math.sqrt(mine * theirs)
        return min(1.0, weighted_shared / my_total)

    # ---------------------------------------------------------------------
    # 2. INTEREST — direct overlap + taste signal − skip penalty (on vibes)
    # ---------------------------------------------------------------------

    def _interest_score(self, my_vibes_set, cand_vibes_str, taste_vibes, skip_vibes):
        if not cand_vibes_str:
            return 0.0

        cand_set = {v.strip() for v in cand_vibes_str.lower().split(',') if v.strip()}
        if not cand_set:
            return 0.0

        shared = my_vibes_set & cand_set
        direct_score = len(shared) / max(1, max(len(my_vibes_set), len(cand_set)))

        taste_total = sum(taste_vibes.values()) if taste_vibes else 0
        taste = (sum(taste_vibes.get(t, 0) for t in cand_set) / taste_total) if taste_total > 0 else 0.0

        skip_total = sum(skip_vibes.values()) if skip_vibes else 0
        skip_p = (sum(skip_vibes.get(t, 0) for t in cand_set) / skip_total) if skip_total > 0 else 0.0

        return (0.5 * direct_score + 0.5 * taste) * (1 - 0.4 * skip_p)

    # ---------------------------------------------------------------------
    # 3. INTENT — IC matrix fading into learned taste ratio with like volume
    # ---------------------------------------------------------------------

    def _intent_score(self, my_intent_idx, cand_intent_idx, intent_like_counts):
        """
        intent_like_counts: dict of {intent_label: count} from people the
        user has previously liked (keyed by the INTENT_MAP label strings).
        """
        if my_intent_idx is None or cand_intent_idx is None:
            return 0.5

        taste_tot = sum(intent_like_counts.values()) if intent_like_counts else 0
        confidence = min(1.0, taste_tot / 10.0)
        ic_weight = 1.0 - confidence
        taste_weight = confidence

        ic_base = INTENT_COMPAT[my_intent_idx][cand_intent_idx]

        # candidate's intent label, by reverse lookup of INTENT_MAP
        cand_label = None
        for label, idx in INTENT_MAP.items():
            if idx == cand_intent_idx:
                cand_label = label
                break

        taste_ratio = (
            intent_like_counts.get(cand_label, 0) / taste_tot
            if taste_tot > 0 and cand_label else 0.0
        )

        return (ic_weight * ic_base) + (taste_weight * taste_ratio)

    # ---------------------------------------------------------------------
    # 4. PROMPT — AI semantic similarity + taste boost − skip penalty
    # ---------------------------------------------------------------------

    @staticmethod
    def _tier_boost(diff):
        if diff <= 0.10: return 0.10
        if diff <= 0.20: return 0.06
        if diff <= 0.30: return 0.03
        return 0.00

    @staticmethod
    def _tier_penalty(diff):
        if diff <= 0.10: return 0.15
        if diff <= 0.20: return 0.08
        if diff <= 0.30: return 0.03
        return 0.00

    def _prompt_score(self, raw_ai_score, avg_liked_score, avg_skipped_score):
        if avg_liked_score is None and avg_skipped_score is None:
            return max(0.0, min(1.0, raw_ai_score))

        boost = 0.0
        if avg_liked_score is not None:
            diff_liked = abs(raw_ai_score - avg_liked_score)
            boost = self._tier_boost(diff_liked)

        penalty = 0.0
        if avg_skipped_score is not None:
            diff_skipped = abs(raw_ai_score - avg_skipped_score)
            penalty = self._tier_penalty(diff_skipped)

        return max(0.0, min(1.0, raw_ai_score + boost - penalty))

    def _get_avg_prompt_scores(self, my_emb1, my_emb2, liked_embeddings, skipped_embeddings):
        """Average AI-similarity score (to current user) across liked/skipped sets."""
        def avg_sim(emb_list):
            if not emb_list:
                return None
            sims = []
            for e1, e2 in emb_list:
                s1 = self._cos_sim(my_emb1, e1)
                s2 = self._cos_sim(my_emb2, e2)
                pair = [s for s in [s1, s2] if s is not None]
                if pair:
                    sims.append(sum(pair) / len(pair))
            return sum(sims) / len(sims) if sims else None

        return avg_sim(liked_embeddings), avg_sim(skipped_embeddings)

    # ---------------------------------------------------------------------
    # 5. AGE — base gap score + taste boost (nearest liked age) − skip penalty
    # ---------------------------------------------------------------------

    @staticmethod
    def _age_base_score(diff):
        if diff <= 1: return 1.00
        if diff <= 3: return 0.85
        if diff <= 5: return 0.65
        if diff <= 8: return 0.45
        return 0.25

    @staticmethod
    def _age_taste_boost(min_diff):
        if min_diff <= 2: return 0.15
        if min_diff <= 5: return 0.10
        if min_diff <= 8: return 0.05
        return 0.00

    @staticmethod
    def _age_skip_penalty(min_diff_skip):
        if min_diff_skip is None:
            return 0.00
        if min_diff_skip <= 2: return 0.15
        if min_diff_skip <= 5: return 0.08
        if min_diff_skip <= 8: return 0.03
        return 0.00

    def _age_score_v2(self, my_age, cand_age, liked_ages, skipped_ages):
        if my_age is None or cand_age is None:
            return 0.5

        diff = abs(cand_age - my_age)
        base = self._age_base_score(diff)

        taste_boost = 0.0
        if liked_ages:
            min_diff = min(abs(cand_age - a) for a in liked_ages)
            taste_boost = self._age_taste_boost(min_diff)

        skip_penalty = 0.0
        if skipped_ages:
            min_diff_skip = min(abs(cand_age - a) for a in skipped_ages)
            skip_penalty = self._age_skip_penalty(min_diff_skip)

        return max(0.0, min(1.0, base + taste_boost - skip_penalty))

    # ---------------------------------------------------------------------
    # 6. CHECK-IN OVERLAP — shared physical places between user and candidate
    # ---------------------------------------------------------------------

    def _checkin_overlap_score(self, my_places, cand_places):
        if not my_places or not cand_places:
            return 0.0
        return len(my_places & cand_places) / max(1, len(my_places))
    # ---------------------------------------------------------------------
    # MASTER SCORER — wires all 7 factors together per the weight table
    # ---------------------------------------------------------------------

    def _score_candidate_v2(
        self, cursor, uid, user, my_data, profile, cand_prof,
        cand_coords, my_coords, my_tag_weights, my_tag_total,
        liked_embeddings, skipped_embeddings, liked_ages, skipped_ages,
        intent_like_counts, cand_tag_counts, my_places, cand_places,
    ):
        cand_age = self._age_from_dob(user[10])

        proximity_score = self._proximity_score(my_coords, cand_coords)

        lifestyle_score = self._lifestyle_score(
            my_tag_weights, my_tag_total, cand_tag_counts
        )

        age_score = self._age_score_v2(
            my_data.get("age"), cand_age, liked_ages, skipped_ages
        )

        interest_score = self._interest_score(
            my_data.get("vibes_set", set()),
            user[11],
            profile.get("taste_vibes", {}),
            profile.get("skip_vibes", {}),
        )

        cand_ans4 = (cand_prof.get("ans4") or "").strip().lower()
        cand_intent_idx = INTENT_MAP.get(cand_ans4)
        intent_score = self._intent_score(
            my_data.get("intent_idx"), cand_intent_idx, intent_like_counts
        )

        cand_emb1, cand_emb2 = cand_prof.get("emb1"), cand_prof.get("emb2")
        sim1 = self._cos_sim(my_data.get("emb1"), cand_emb1)
        sim2 = self._cos_sim(my_data.get("emb2"), cand_emb2)
        pair = [s for s in [sim1, sim2] if s is not None]
        raw_ai_score = sum(pair) / len(pair) if pair else 0.0

        avg_liked, avg_skipped = self._get_avg_prompt_scores(
            my_data.get("emb1"), my_data.get("emb2"), liked_embeddings, skipped_embeddings
        )
        prompt_score = self._prompt_score(raw_ai_score, avg_liked, avg_skipped)

        checkin_score = self._checkin_overlap_score(
            my_places, cand_places
        )

        w = MATCHING_WEIGHTS_V2
        final_score = (
            w["proximity"]       * proximity_score +
            w["lifestyle"]       * lifestyle_score +
            w["age"]             * age_score +
            w["interest"]        * interest_score +
            w["intent"]          * intent_score +
            w["prompt"]          * prompt_score +
            w["checkin_overlap"] * checkin_score
        )

        breakdown = {
            "proximity":       round(proximity_score, 3),
            "lifestyle":       round(lifestyle_score, 3),
            "age":             round(age_score, 3),
            "interest":        round(interest_score, 3),
            "intent":          round(intent_score, 3),
            "prompt":          round(prompt_score, 3),
            "checkin_overlap": round(checkin_score, 3),
            "weights":         w,
        }

        return final_score, breakdown

    # ---------------------------------------------------------------------
    # Behaviour-history fetchers needed by the V2 scorer
    # ---------------------------------------------------------------------

    def _fetch_liked_skipped_signals(self, cursor, user_id):
        """Returns embeddings/ages/intents for liked + skipped (hidden) users."""
        cursor.execute("""
            SELECT receiver_id FROM (
                SELECT receiver_id, created_at FROM icebreaker.user_likes WHERE sender_id = %s
                UNION ALL
                SELECT receiver_id, created_at FROM icebreaker.user_likes_near WHERE sender_id = %s
            ) t ORDER BY created_at DESC LIMIT 50
        """, (user_id, user_id))
        liked_ids = [r[0] for r in cursor.fetchall()]

        cursor.execute("""
            SELECT hiddenUserId FROM (
                SELECT hiddenUserId, hiddenAt FROM icebreaker.hidden_users_1day WHERE userId = %s
                UNION ALL
                SELECT hiddenUserId, hiddenAt FROM icebreaker.hidden_users_1month WHERE userId = %s
            ) t ORDER BY hiddenAt DESC LIMIT 50
        """, (user_id, user_id))
        skipped_ids = [r[0] for r in cursor.fetchall()]

        def fetch_profiles(ids):
            if not ids:
                return [], [], []
            fmt = ','.join(['%s'] * len(ids))
            cursor.execute(f"""
                SELECT p.userId, p.prompt_embedding, p.prompt_embedding2,
                       p.answer4, r.DOB
                FROM icebreaker.profile p
                JOIN icebreaker.registration r ON p.userId = r.userId
                WHERE p.userId IN ({fmt})
            """, ids)
            embeddings, ages, intents = [], [], {}
            for uid, e1_raw, e2_raw, ans4, dob in cursor.fetchall():
                embeddings.append((self._load_embedding(e1_raw), self._load_embedding(e2_raw)))
                age = self._age_from_dob(dob)
                if age:
                    ages.append(age)
                if ans4:
                    label = ans4.strip()
                    intents[label] = intents.get(label, 0) + 1
            return embeddings, ages, intents

        liked_embeddings, liked_ages, liked_intents = fetch_profiles(liked_ids)
        skipped_embeddings, skipped_ages, _ = fetch_profiles(skipped_ids)

        return {
            "liked_embeddings": liked_embeddings,
            "liked_ages": liked_ages,
            "liked_intents": liked_intents,
            "skipped_embeddings": skipped_embeddings,
            "skipped_ages": skipped_ages,
        }

    # ---------------------------------------------------------------------
    # UPDATED recommend() — now uses the V2 7-factor scorer
    # ---------------------------------------------------------------------

    def recommend(
        self,
        cursor,
        user_id: int,
        candidates: list,
        session_hidden_ids: list = None,
        max_users: int = 50,
        page: int = 1,
        per_page: int = 10,
    ) -> dict:
        session_hidden = set(session_hidden_ids or [])

        if not candidates:
            return {"Status": "Success", "Matches": [], "Page": page,
                    "PerPage": per_page, "TotalMatches": 0}

        profile = self._bp_load(cursor, user_id)
        if profile["interaction_counts"].get("waves", 0) == 0:
            profile = self._bp_rebuild(cursor, user_id)

        my_data = self._fetch_my_data(cursor, user_id)
        my_data["user_id"] = user_id
        candidate_ids = [c[0] for c in candidates]
        all_ids = [user_id] + candidate_ids

        tag_counts_map = self._batch_get_user_tag_counts(cursor, all_ids)
        places_map     = self._batch_get_places(cursor, all_ids)

        my_tag_weights, my_tag_total = self._normalize_tag_weights(tag_counts_map.get(user_id, {}))
        my_places = places_map.get(user_id, set())

        cand_profiles = self._batch_fetch_candidates(cursor, candidate_ids)
        coords_map = self._batch_fetch_coords(cursor, candidate_ids)
        my_coords = self._fetch_my_coords(cursor, user_id)


        my_tag_counts = self._get_user_tag_counts(cursor, user_id)
        # my_tag_weights, my_tag_total = self._normalize_tag_weights(my_tag_counts)

        signals = self._fetch_liked_skipped_signals(cursor, user_id)

        scored = []
        for user in candidates[:max_users]:
            uid = user[0]
            if uid in session_hidden:
                continue
            score, breakdown = self._score_candidate_v2(
                cursor, uid, user, my_data, profile,
                cand_profiles.get(uid, {}),
                coords_map.get(uid), my_coords,
                my_tag_weights, my_tag_total,
                signals["liked_embeddings"], signals["skipped_embeddings"],
                signals["liked_ages"], signals["skipped_ages"],
                signals["liked_intents"],
                tag_counts_map.get(uid, {}),   # this candidate's tag counts
                my_places,                      # requesting user's place set
                places_map.get(uid, set()),     # this candidate's place set
            )
            scored.append((uid, score, breakdown, user))

            scored.sort(key=lambda x: x[1], reverse=True)

        wave_status_map = self._batch_get_wave_status(cursor, user_id, [uid for uid, *_ in scored])


        results = []
        for uid, score, breakdown, user in scored:
            prof = cand_profiles.get(uid, {})
            results.append({
                "userId":           uid,
                "fullName":         user[3],
                "email":            user[4],
                "gender":           user[8],
                "vibes":            user[11],
                "profilePic":       user[1],
                "selfiePic":        user[2],
                "firebaseId":       user[5],
                "signupType":       user[6],
                "authenticationId": user[7],
                "state":            user[13],
                "city":             user[14],
                "user_zodiac_sign": user[15],
                "user_aesthetic_score": user[16],
                "countryCode":      user[18],
                "phoneNo":          user[19],
                "country":          user[20],
                "deviceToken":      user[21],
                "what_makes_my_friends_laugh_about_me": user[22],
                "weekend_usually_looks_like":           user[23],
                "Describe_your_communication_style":    user[24],
                "What_brings_you_to_Vyb":               user[25],
                "What_kind_of_partner_do_you_vibe_with_most": user[26],
                "What_does_everyday_vyb_look_like":     user[27],
                "height":           user[28],
                "age_preference":   user[29],
                "prompt1question":  user[30],
                "prompt1answer":    prof.get("p1", ""),
                "prompt2question":  user[32],
                "prompt2answer":    prof.get("p2", ""),
                "DOB":              str(user[10]) if user[10] else None,
                "similarity_score": round(score * 100, 2),
                # "score_breakdown":  breakdown,
                "preference":user[9],
                "wave_status": wave_status_map.get(uid, "no_wave"),
            })

        start = (page - 1) * per_page
        return {
            "Status":       "Success",
            "Matches":      results,
            "Page":         page,
            "PerPage":      per_page,
            "TotalMatches": len(results),
        }
 
 
 
    # =========================================================================
    # RECOMMENDATION — Data Fetchers
    # =========================================================================
 
    def _fetch_my_data(self, cursor, user_id: int) -> dict:
        cursor.execute("""
            SELECT r.Gender, r.Preference, r.DOB, p.vibes,
                   p.answer3, p.answer4, p.prompt_embedding, p.prompt_embedding2
            FROM icebreaker.registration r
            JOIN icebreaker.profile p ON r.userId = p.userId
            WHERE r.userId = %s
        """, (user_id,))
        row = cursor.fetchone()
        if not row:
            return {}
        gender, pref, dob, vibes, ans3, ans4, emb1_raw, emb2_raw = row
        vibes_set = {v.strip() for v in vibes.lower().split(',')} if vibes else set()
        return {
            "gender":     gender,
            "preference": pref,
            "dob":        dob,
            "age":        self._age_from_dob(dob),
            "vibes_set":  vibes_set,
            "ans3":       (ans3 or "").strip().lower(),
            "ans4":       (ans4 or "").strip().lower(),
            "comm_idx":   COMM_STYLE_MAP.get((ans3 or "").strip().lower()),
            "intent_idx": INTENT_MAP.get((ans4 or "").strip().lower()),
            "emb1":       self._load_embedding(emb1_raw),
            "emb2":       self._load_embedding(emb2_raw),
        }
 
    def _batch_fetch_candidates(self, cursor, candidate_ids: list) -> dict:
        if not candidate_ids:
            return {}
        fmt = ','.join(['%s'] * len(candidate_ids))
        cursor.execute(f"""
            SELECT userId, answer3, answer4, prompt1answer, prompt2answer,
                   prompt_embedding, prompt_embedding2, vibes
            FROM icebreaker.profile
            WHERE userId IN ({fmt})
        """, candidate_ids)
        result = {}
        for row in cursor.fetchall():
            uid, ans3, ans4, p1, p2, emb1_raw, emb2_raw, vibes = row
            result[uid] = {
                "ans3":  (ans3 or "").strip().lower(),
                "ans4":  (ans4 or "").strip().lower(),
                "p1":    p1 or "",
                "p2":    p2 or "",
                "emb1":  self._load_embedding(emb1_raw),
                "emb2":  self._load_embedding(emb2_raw),
                "vibes": vibes or "",
            }
        return result
 
    def _batch_fetch_coords(self, cursor, candidate_ids: list) -> dict:
        if not candidate_ids:
            return {}
        fmt = ','.join(['%s'] * len(candidate_ids))
        cursor.execute(f"""
            SELECT user_id, latitude, longitude
            FROM icebreaker.user_live_location_offline
            WHERE user_id IN ({fmt})
        """, candidate_ids)
        return {r[0]: (r[1], r[2]) for r in cursor.fetchall()}
 
    def _fetch_my_coords(self, cursor, user_id: int):
        cursor.execute("""
            SELECT latitude, longitude
            FROM icebreaker.user_live_location_offline
            WHERE user_id = %s
        """, (user_id,))
        row = cursor.fetchone()
        return (row[0], row[1]) if row else None
    
    def _batch_get_user_tag_counts(self, cursor, user_ids, months=3):
        if not user_ids:
            return {}
        fmt = ','.join(['%s'] * len(user_ids))
        cursor.execute(f"""
            SELECT userid, primary_type, COUNT(*) AS cnt
            FROM icebreaker.sessiontracker
            WHERE userid IN ({fmt})
            AND checkintime >= DATE_SUB(NOW(), INTERVAL %s MONTH)
            GROUP BY userid, primary_type
        """, (*user_ids, months))

        result = {uid: {tag: 0 for tag in ALL_TAGS} for uid in user_ids}
        for userid, primary_type, cnt in cursor.fetchall():
            tag = primary_type_to_tag(primary_type)
            if tag:
                result[userid][tag] += cnt
        return result

    def _batch_get_places(self, cursor, user_ids, months=3):
        if not user_ids:
            return {}
        fmt = ','.join(['%s'] * len(user_ids))
        cursor.execute(f"""
            SELECT userid, placeid FROM icebreaker.sessiontracker
            WHERE userid IN ({fmt})
            AND checkintime >= DATE_SUB(NOW(), INTERVAL %s MONTH)
        """, (*user_ids, months))

        places = {uid: set() for uid in user_ids}
        for userid, placeid in cursor.fetchall():
            places[userid].add(placeid)
        return places
 
    # =========================================================================
    # BEHAVIOUR PROFILE — Private Builders
    # =========================================================================
 
    def _bp_build_positive(self, cursor, interactions, mutual_ids, profile):
        today        = datetime.utcnow()
        receiver_ids = [r[0] for r in interactions]
        fmt          = ','.join(['%s'] * len(receiver_ids))
 
        cursor.execute(f"""
            SELECT p.userId, p.vibes, p.answer3, p.answer4,
                   p.prompt_embedding, p.prompt_embedding2, r.DOB
            FROM icebreaker.profile p
            JOIN icebreaker.registration r ON p.userId = r.userId
            WHERE p.userId IN ({fmt})
        """, receiver_ids)
        rows = {r[0]: r[1:] for r in cursor.fetchall()}
 
        taste_vibes       = {}
        weighted_emb1_sum = None
        weighted_emb2_sum = None
        total_weight      = 0.0
        ages              = []
 
        for receiver_id, created_at in interactions:
            if receiver_id not in rows:
                continue
            vibes, ans3, ans4, emb1_raw, emb2_raw, dob = rows[receiver_id]
 
            days_ago = (today - created_at).days if created_at else 30
            recency  = math.exp(-0.693 * days_ago / RECENCY_HALF_LIFE_DAYS)
            weight   = recency * (2.0 if receiver_id in mutual_ids else 1.0)
 
            if vibes:
                for v in vibes.lower().split(','):
                    v = v.strip()
                    taste_vibes[v] = taste_vibes.get(v, 0) + weight
 
            emb1 = self._load_embedding(emb1_raw)
            emb2 = self._load_embedding(emb2_raw)
            if emb1 is not None:
                arr = np.array(emb1) * weight
                weighted_emb1_sum = arr if weighted_emb1_sum is None else weighted_emb1_sum + arr
            if emb2 is not None:
                arr = np.array(emb2) * weight
                weighted_emb2_sum = arr if weighted_emb2_sum is None else weighted_emb2_sum + arr
 
            total_weight += weight
            age = self._age_from_dob(dob)
            if age:
                ages.append(age)
 
        profile["taste_vibes"] = taste_vibes
        if weighted_emb1_sum is not None and total_weight > 0:
            profile["taste_emb1"] = (weighted_emb1_sum / total_weight).tolist()
        if weighted_emb2_sum is not None and total_weight > 0:
            profile["taste_emb2"] = (weighted_emb2_sum / total_weight).tolist()
        if ages:
            profile["pref_age_min"] = min(ages)
            profile["pref_age_max"] = max(ages)
 
        return profile
 
    def _bp_build_negative(self, cursor, hidden_ids: list, profile: dict) -> dict:
        fmt = ','.join(['%s'] * len(hidden_ids))
        cursor.execute(f"""
            SELECT p.vibes, p.answer3, p.answer4
            FROM icebreaker.profile p
            WHERE p.userId IN ({fmt})
        """, hidden_ids)
 
        skip_vibes  = {}
        skip_comm   = {}
        skip_intent = {}
 
        for vibes, ans3, ans4 in cursor.fetchall():
            if vibes:
                for v in vibes.lower().split(','):
                    v = v.strip()
                    skip_vibes[v] = skip_vibes.get(v, 0) + 1
            if ans3:
                k = ans3.strip().lower()
                skip_comm[k] = skip_comm.get(k, 0) + 1
            if ans4:
                k = ans4.strip().lower()
                skip_intent[k] = skip_intent.get(k, 0) + 1
 
        profile["skip_vibes"]  = skip_vibes
        profile["skip_comm"]   = skip_comm
        profile["skip_intent"] = skip_intent
        return profile
 
    def _bp_learn_weights(self, profile: dict) -> dict:
        counts     = profile.get("interaction_counts", {})
        total      = counts.get("waves", 0) + counts.get("matches", 0)
 
        if total < MIN_INTERACTIONS_FOR_DYNAMIC_WEIGHTS:
            return DEFAULT_WEIGHTS.copy()
 
        taste_vibes     = profile.get("taste_vibes", {})
        has_emb         = profile.get("taste_emb1") is not None
        match_rate      = counts.get("matches", 0) / max(1, total)
        vibe_signal     = min(1.0, sum(taste_vibes.values()) / max(1, total * 3))
        semantic_signal = 1.0 if has_emb else 0.3
 
        vibe_w    = 0.12 + 0.14 * vibe_signal
        prompt_w  = 0.20 + 0.12 * semantic_signal
        mutual_w  = 0.03 + 0.07 * match_rate
        remainder = 1.0 - vibe_w - prompt_w - mutual_w
 
        return {
            "prompt_sim":    round(prompt_w, 3),
            "vibe":          round(vibe_w, 3),
            "comm":          round(remainder * 0.38, 3),
            "intent":        round(remainder * 0.38, 3),
            "proximity":     round(remainder * 0.16, 3),
            "age":           round(remainder * 0.08, 3),
            "mutual_signal": round(mutual_w, 3),
        }
 
    def _bp_update_embedding(self, profile, emb1, emb2, weight):
        alpha = 0.15
        if emb1 is not None:
            existing = profile.get("taste_emb1")
            if existing is None:
                profile["taste_emb1"] = emb1
            else:
                profile["taste_emb1"] = (
                    (1 - alpha) * np.array(existing) + alpha * np.array(emb1) * weight
                ).tolist()
        if emb2 is not None:
            existing = profile.get("taste_emb2")
            if existing is None:
                profile["taste_emb2"] = emb2
            else:
                profile["taste_emb2"] = (
                    (1 - alpha) * np.array(existing) + alpha * np.array(emb2) * weight
                ).tolist()
        return profile
 
    @staticmethod
    def _build_session_hide_profile(session_hidden: set, cand_profiles: dict) -> dict:
        vibes  = set()
        comm   = set()
        intent = set()
        for uid in session_hidden:
            prof = cand_profiles.get(uid, {})
            raw_vibes = prof.get("vibes", "")
            if raw_vibes:
                for v in raw_vibes.lower().split(','):
                    v = v.strip()
                    if v:
                        vibes.add(v)
            if prof.get("ans3"):
                comm.add(prof["ans3"])
            if prof.get("ans4"):
                intent.add(prof["ans4"])
        return {"vibes": vibes, "comm": comm, "intent": intent}
 
    # =========================================================================
    # SHARED UTILITIES
    # =========================================================================
 
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
    def _cos_sim(a, b):
        if a is None or b is None:
            return None
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
 
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
    def _age_compat(a, b):
        if a is None or b is None:
            return 0.5
        diff = abs(a - b)
        if diff <= 1: return 1.00
        if diff <= 2: return 0.90
        if diff <= 3: return 0.78
        if diff <= 5: return 0.60
        if diff <= 8: return 0.35
        return 0.15
 
    @staticmethod
    def _proximity_score(coord1, coord2, max_km=200):
        if not coord1 or not coord2:
            return 0.5
        R = 6371
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return max(0.0, 1.0 - R * 2 * atan2(sqrt(a), sqrt(1 - a)) / max_km)
 
    @staticmethod
    def _load_emb(raw):
        if not raw:
            return None
        try:
            return json.loads(raw)
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
