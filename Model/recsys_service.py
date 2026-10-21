"""6. RecSys & AI microservice model."""

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


class RecsysService(ProfileBase):
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

    def _encode_answers(self, answers_list):
        """Encode answer3 and answer4"""
        answer_encoded_parts = []
        for i in range(2):
            col = np.array(answers_list[i]).reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            answer_encoded_parts.append(enc.fit_transform(col))
        return np.concatenate(answer_encoded_parts, axis=1)

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

    def _checkin_overlap_score(self, my_places, cand_places):
        if not my_places or not cand_places:
            return 0.0
        return len(my_places & cand_places) / max(1, len(my_places))

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

    @staticmethod
    def _cos_sim(a, b):
        if a is None or b is None:
            return None
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

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

