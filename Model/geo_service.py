"""3. Geo & Check-in microservice model."""

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


class GeoService(ProfileBase):
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

    def _haversine_meters(self, lat1, lng1, lat2, lng2):
        radius = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lng / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(a))

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

    def _translate_preference_industry_v1(self, my_preference):
        return self._translate_preference_common(my_preference)

    def _fetch_age_preference_industry_v1(self, cursor, user_id):
        cursor.execute(
            'SELECT age_preference FROM icebreaker.profile WHERE userId = %s', 
            (user_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

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

    def _fetch_user_dob_industry_v1(self, cursor, user_id):
        cursor.execute('SELECT DOB FROM icebreaker.registration WHERE userId = %s', (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

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

    def delete_selfie_pic(self, cursor, user_id):
        cursor.execute('UPDATE icebreaker.profile SET selfiePic = NULL WHERE userId = %s',(user_id,))
        self.db.conn.commit()

    def cleanup_session_likes(self, cursor, user_id):
        cursor.execute('DELETE FROM icebreaker.session_likes WHERE wavedToUserId = %s', (user_id,))
        self.db.conn.commit()

    def cleanup_hidden_users(self, cursor, user_id):
        cursor.execute('DELETE FROM icebreaker.hidden_users WHERE userId = %s', (user_id,))
        self.db.conn.commit()

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

    def filter_by_search_text_radius(self, output_list, search_text):
        search_text = search_text.lower()
        return [user for user in output_list if search_text in user["fullName"].lower()]

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

    def cleanup_pending_requests(self, cursor, user_id):
        cursor.execute('DELETE FROM icebreaker.netwirkRequestsPending WHERE userId = %s', (user_id,))
        self.db.conn.commit()

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

    def _get_gender_filter(self, cursor, user_id):
        cursor.execute('SELECT Gender, Preference FROM icebreaker.registration WHERE userId = %s', (user_id,))
        _, preference = cursor.fetchone() or (None, None)

        if preference == "Women":
            return "Female"
        elif preference == "Men":
            return "Male"
        return None

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
