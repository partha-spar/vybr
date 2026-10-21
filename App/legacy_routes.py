"""Legacy microservice routes for remaining endpoints."""
import logging
from flask import Blueprint, jsonify, request
from flasgger import swag_from
from App.matchDemoRoute import matchDemoRoute
from Model.legacy_service import LegacyProfileService as profile

legacyProfileRoute = Blueprint('legacyProfileRoute', __name__, url_prefix='/netwirkProfileRoute')
BLUEPRINTS = (legacyProfileRoute, matchDemoRoute)

