"""2. Profile & Persona microservice model."""

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


class ProfileService(ProfileBase):
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

    def _user_not_found_response(self, cursor):
        cursor.close()
        self.db.conn.close()
        return jsonify({"error": self.USER_NOT_FOUND_MESSAGE})

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
