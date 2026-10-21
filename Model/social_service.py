"""4. Social Graph microservice model."""

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


from Model.common_base import ProfileBase, profile


class SocialService(ProfileBase):
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

    def _fetch_login_token_users_accepted_user(self, cursor, user_id):    
        return self._get_login_web_token_confirm_api(cursor, user_id)

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

    def is_name_match(self, query, place_name):
        query = query.lower()
        place_name = place_name.lower()
        

        # ✅ Direct partial match first
        if query in place_name:
            return True

        # ✅ Fuzzy match — 0.6 threshold catches typos
        ratio = SequenceMatcher(None, query, place_name).ratio()
        return ratio >= 0.6

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
