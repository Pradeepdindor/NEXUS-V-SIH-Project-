"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Master System Configuration (config/config.py)
"""

import os
import sys
from pathlib import Path

# ==========================================
# BASE PATHS & DIRECTORY STRUCTURE
# ==========================================
# Resolves to project root regardless of where this file is imported from
CONFIG_DIR = Path(__file__).resolve().parent
if CONFIG_DIR.name == "config":
    BASE_DIR = CONFIG_DIR.parent
else:
    BASE_DIR = CONFIG_DIR

# Ensure Project Root is on sys.path for global modular imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Component Root Directories
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Data & Model Weights Storage Paths
DATA_DIR = DATASET_DIR / "data"
# Fallback support if data is at root
if not DATA_DIR.exists() and (BASE_DIR / "data").exists():
    DATA_DIR = BASE_DIR / "data"

ALERTS_DIR = DATA_DIR / "alerts"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
TICKETS_DIR = DATA_DIR / "tickets"
RECORDINGS_DIR = DATA_DIR / "recordings"

MODELS_DIR = MODEL_DIR / "weights"
if not MODELS_DIR.exists() and (BASE_DIR / "models").exists():
    MODELS_DIR = BASE_DIR / "models"

TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

for directory in [
    BACKEND_DIR,
    FRONTEND_DIR,
    MODEL_DIR,
    DATASET_DIR,
    SCRIPTS_DIR,
    DATA_DIR,
    ALERTS_DIR,
    SNAPSHOTS_DIR,
    TICKETS_DIR,
    RECORDINGS_DIR,
    MODELS_DIR,
    TEMPLATES_DIR,
    STATIC_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# BACKEND & DASHBOARD NETWORKING
# ==========================================
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
WS_LIVE_FEED_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/live_feed"
DASHBOARD_PORT = 8501
DASHBOARD_URL = f"http://localhost:{DASHBOARD_PORT}"

# ==========================================
# EDGE DEVICE & FLEET METADATA
# ==========================================
DEFAULT_BUS_ID = "BUS-KA-01-F-1204"
DEFAULT_ROUTE_ID = "ROUTE-335E"
DEFAULT_FLEET_OPERATOR = "BMTC Urban Smart Fleet"
DEFAULT_DEVICE_ID = "EDGE-AI-NODE-04"

FLEET_BUS_OPTIONS = [
    {"bus_id": "BUS-KA-01-F-1204", "route_id": "ROUTE-335E", "name": "Majestic - Whitefield ITPL Corridor"},
    {"bus_id": "BUS-KA-01-F-2210", "route_id": "ROUTE-500D", "name": "Silk Board - Hebbal Outer Ring Road"},
    {"bus_id": "BUS-DL-02-C-8810", "route_id": "ROUTE-CBD-EXPRESS", "name": "Connaught Place - Airport Express"},
    {"bus_id": "BUS-MH-12-Q-4091", "route_id": "ROUTE-METRO-FEEDER", "name": "Kothrud - Hinjewadi Tech Park"},
]

# Camera Angle Options
CAMERA_ANGLE_FRONT = "Front (Road Defect & Traffic)"
CAMERA_ANGLE_REAR = "Rear (Following Traffic)"
CAMERA_ANGLE_SIDE = "Side (Pavement & Signage)"
CAMERA_ANGLE_CABIN = "Cabin (Driver Safety & Occupancy)"

CAMERA_ANGLES = [
    CAMERA_ANGLE_FRONT,
    CAMERA_ANGLE_REAR,
    CAMERA_ANGLE_SIDE,
    CAMERA_ANGLE_CABIN,
]

# ==========================================
# VIDEO INGESTION & PROCESSING CONFIG
# ==========================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
CAMERA_INDEX = 0

# ==========================================
# AI DETECTION & THRESHOLDS
# ==========================================
DEFAULT_MODEL_WEIGHTS = str(MODELS_DIR / "yolov8n.pt") if (MODELS_DIR / "yolov8n.pt").exists() else "yolov8n.pt"
DEFECT_CONFIDENCE_THRESHOLD = 0.75  # Trigger alert & ticket if >= 0.75
VEHICLE_CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

# Interactive Judge/Webcam Demo Configuration:
DEMO_WEBCAM_FACE_AS_POTHOLE = False

# COCO Class IDs
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}

COCO_TRAFFIC_CLASSES = {
    0: "pedestrian",
    9: "traffic light",
    11: "stop sign",
}

# Defect & Infrastructure Hazard Categories
CATEGORY_ROAD_DEFECT = "Road Defect"
CATEGORY_MISSING_INFRASTRUCTURE = "Missing Infrastructure"
CATEGORY_TRAFFIC_SAFETY = "Traffic & Safety"

# ==========================================
# MUNICIPAL DEPARTMENT ROUTING & SLA RULES
# (pothole->PWD, crash->Police, medical->Ambulance, other->Municipal Corp)
# ==========================================
DEPARTMENT_PWD = "Public Works Department (PWD) - Road Maintenance Division"
DEPARTMENT_POLICE = "City Traffic Police & Road Safety Department"
DEPARTMENT_AMBULANCE = "Emergency Medical & Ambulance Services (EMS 108)"
DEPARTMENT_MUNICIPAL = "City Municipal Corporation (BBMP / Urban Local Body)"
DEPARTMENT_DRAINAGE = "Urban Drainage & Sewage Board (BWSSB/BBMP)"
DEPARTMENT_TRAFFIC_ENG = "Municipal Traffic Engineering & Road Safety Cell"
DEPARTMENT_TRAFFIC_POLICE = "City Traffic Police & Signals Management"
DEPARTMENT_EMERGENCY = "Emergency First Responders & Disaster Control"

DEFECT_TYPES = {
    "pothole": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Pothole",
        "severity": "High",
        "color_bgr": (0, 0, 255),  # Red
        "department": DEPARTMENT_PWD,
        "sla_hours": 48,
        "estimated_cost_inr": 4500,
        "action_required": "Cold-mix asphalt patching and compaction required.",
    },
    "damaged_pavement": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Damaged Pavement",
        "severity": "Medium",
        "color_bgr": (0, 140, 255),  # Orange-Red
        "department": DEPARTMENT_PWD,
        "sla_hours": 72,
        "estimated_cost_inr": 12000,
        "action_required": "Milling and surface re-layering required.",
    },
    "crash": {
        "category": CATEGORY_TRAFFIC_SAFETY,
        "name": "Vehicle Crash / Collision",
        "severity": "Critical",
        "color_bgr": (0, 0, 255),  # Red
        "department": DEPARTMENT_POLICE,
        "sla_hours": 1,
        "estimated_cost_inr": 0,
        "action_required": "Emergency police dispatch, traffic diversion, and accident scene clearance.",
    },
    "accident_hazard": {
        "category": CATEGORY_TRAFFIC_SAFETY,
        "name": "Accident / Road Hazard",
        "severity": "Critical",
        "color_bgr": (0, 0, 220),  # Crimson Red
        "department": DEPARTMENT_POLICE,
        "sla_hours": 2,
        "estimated_cost_inr": 0,
        "action_required": "Immediate traffic redirection, police response, and roadway clearing.",
    },
    "medical_emergency": {
        "category": CATEGORY_TRAFFIC_SAFETY,
        "name": "Medical / Pedestrian Incident",
        "severity": "Critical",
        "color_bgr": (0, 30, 255),  # Deep Crimson
        "department": DEPARTMENT_AMBULANCE,
        "sla_hours": 1,
        "estimated_cost_inr": 0,
        "action_required": "Immediate ambulance dispatch and trauma medical intervention.",
    },
    "waterlogging": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Waterlogging",
        "severity": "High",
        "color_bgr": (255, 128, 0),  # Blue-Orange
        "department": DEPARTMENT_MUNICIPAL,
        "sla_hours": 24,
        "estimated_cost_inr": 8500,
        "action_required": "Clear blocked storm-water drains and suction pump deployment.",
    },
    "missing_zebra_crossing": {
        "category": CATEGORY_MISSING_INFRASTRUCTURE,
        "name": "Missing Zebra Crossing",
        "severity": "Medium",
        "color_bgr": (0, 215, 255),  # Yellow
        "department": DEPARTMENT_MUNICIPAL,
        "sla_hours": 120,
        "estimated_cost_inr": 3500,
        "action_required": "Thermoplastic reflective paint marking renewal.",
    },
    "damaged_road_divider": {
        "category": CATEGORY_MISSING_INFRASTRUCTURE,
        "name": "Damaged Road Divider",
        "severity": "High",
        "color_bgr": (255, 0, 255),  # Magenta
        "department": DEPARTMENT_MUNICIPAL,
        "sla_hours": 48,
        "estimated_cost_inr": 16000,
        "action_required": "Reinstall reinforced concrete median blocks & retroreflective studs.",
    },
    "damaged_traffic_sign": {
        "category": CATEGORY_MISSING_INFRASTRUCTURE,
        "name": "Damaged/Missing Traffic Sign",
        "severity": "Medium",
        "color_bgr": (180, 105, 255),  # Pinkish
        "department": DEPARTMENT_POLICE,
        "sla_hours": 72,
        "estimated_cost_inr": 2800,
        "action_required": "Fabricate and mount high-intensity microprismatic signboard.",
    },
    "crack": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Road Surface Crack",
        "severity": "Medium",
        "color_bgr": (0, 140, 255),  # Orange-Red
        "department": DEPARTMENT_PWD,
        "sla_hours": 72,
        "estimated_cost_inr": 7500,
        "action_required": "Bitumen crack injection and road surface sealing.",
    },
    "manhole": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Manhole / Utility Cover",
        "severity": "Low",
        "color_bgr": (255, 191, 0),  # Amber
        "department": DEPARTMENT_MUNICIPAL,
        "sla_hours": 96,
        "estimated_cost_inr": 5000,
        "action_required": "Inspect utility cover alignment and flush-level road seating.",
    },
    # RDD2022 Benchmark Specific Defect Classes & Aliases
    "longitudinal_crack": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Longitudinal Crack (D00)",
        "severity": "Medium",
        "color_bgr": (0, 165, 255),  # Amber
        "department": DEPARTMENT_PWD,
        "sla_hours": 72,
        "estimated_cost_inr": 8000,
        "action_required": "Asphalt crack sealing with polymer-modified bitumen emulsion.",
    },
    "transverse_crack": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Transverse Crack (D10)",
        "severity": "Medium",
        "color_bgr": (0, 140, 255),  # Orange-Red
        "department": DEPARTMENT_PWD,
        "sla_hours": 72,
        "estimated_cost_inr": 8500,
        "action_required": "Joint saw cutting and elastomeric sealant injection.",
    },
    "alligator_crack": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Alligator Fatigue Crack (D20)",
        "severity": "High",
        "color_bgr": (0, 69, 255),  # Deep Orange-Red
        "department": DEPARTMENT_PWD,
        "sla_hours": 48,
        "estimated_cost_inr": 18000,
        "action_required": "Full-depth asphalt reclamation (FDR) and sub-base structural stabilization.",
    },
    "crosswalk_blur": {
        "category": CATEGORY_MISSING_INFRASTRUCTURE,
        "name": "Crosswalk / Lane Marking Blur (D44)",
        "severity": "Medium",
        "color_bgr": (0, 215, 255),  # Yellow
        "department": DEPARTMENT_MUNICIPAL,
        "sla_hours": 120,
        "estimated_cost_inr": 3500,
        "action_required": "Thermoplastic reflective paint marking renewal.",
    },
    "d00": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Longitudinal Crack (D00)",
        "severity": "Medium",
        "color_bgr": (0, 165, 255),
        "department": DEPARTMENT_PWD,
        "sla_hours": 72,
        "estimated_cost_inr": 8000,
        "action_required": "Asphalt crack sealing with polymer-modified bitumen emulsion.",
    },
    "d10": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Transverse Crack (D10)",
        "severity": "Medium",
        "color_bgr": (0, 140, 255),
        "department": DEPARTMENT_PWD,
        "sla_hours": 72,
        "estimated_cost_inr": 8500,
        "action_required": "Joint saw cutting and elastomeric sealant injection.",
    },
    "d20": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Alligator Fatigue Crack (D20)",
        "severity": "High",
        "color_bgr": (0, 69, 255),
        "department": DEPARTMENT_PWD,
        "sla_hours": 48,
        "estimated_cost_inr": 18000,
        "action_required": "Full-depth asphalt reclamation (FDR) and sub-base structural stabilization.",
    },
    "d40": {
        "category": CATEGORY_ROAD_DEFECT,
        "name": "Pothole (D40)",
        "severity": "High",
        "color_bgr": (0, 0, 255),
        "department": DEPARTMENT_PWD,
        "sla_hours": 48,
        "estimated_cost_inr": 4500,
        "action_required": "Cold-mix asphalt patching and compaction required.",
    },
    "d44": {
        "category": CATEGORY_MISSING_INFRASTRUCTURE,
        "name": "Crosswalk / Lane Marking Blur (D44)",
        "severity": "Medium",
        "color_bgr": (0, 215, 255),
        "department": DEPARTMENT_MUNICIPAL,
        "sla_hours": 120,
        "estimated_cost_inr": 3500,
        "action_required": "Thermoplastic reflective paint marking renewal.",
    },
}

# ==========================================
# EDGE PRIVACY & ANONYMIZATION CONFIG
# ==========================================
ANONYMIZE_FACES_AND_PLATES = True
ANONYMIZATION_BLUR_KERNEL = (31, 31)

# ==========================================
# STORE-AND-FORWARD BUFFERING CONFIG
# ==========================================
OFFLINE_BUFFER_FILE = ALERTS_DIR / "edge_offline_queue.json"
STORE_FORWARD_RETRY_INTERVAL = 3.0
MAX_OFFLINE_BUFFER_EVENTS = 500

# ==========================================
# MULTI-BUS SPATIAL DEDUPLICATION CONFIG
# ==========================================
SPATIAL_DEDUP_RADIUS_METERS = 25.0
SPATIAL_DEDUP_WINDOW_HOURS = 24.0

# ==========================================
# SPATIAL DATABASE & POSTGIS CONFIG
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "")  # e.g., postgresql://postgres:postgres@localhost:5432/sih_db
SPATIAL_SQLITE_PATH = DATA_DIR / "urban_intelligence_spatial.db"

# ==========================================
# FLASK REST API SERVER CONFIG
# ==========================================
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = 5000
FLASK_URL = f"http://127.0.0.1:{FLASK_PORT}"

# ==========================================
# TRAFFIC CONGESTION THRESHOLDS
# ==========================================
CONGESTION_LEVEL_LOW = "Low"
CONGESTION_LEVEL_MEDIUM = "Medium"
CONGESTION_LEVEL_HIGH = "High"

CONGESTION_THRESHOLDS = {
    "low_max": 3,
    "medium_max": 8,
}

# ==========================================
# ALERT DEBOUNCE & GEO-FILTERING
# ==========================================
ALERT_DEBOUNCE_TIME_SECONDS = 4.0
ALERT_DEBOUNCE_DISTANCE_METERS = 15.0

# ==========================================
# SIMULATED URBAN ROUTE WAYPOINTS (Bangalore Metro Corridor)
# ==========================================
SIMULATED_ROUTE_WAYPOINTS = [
    {"name": "Majestic Central Bus Station", "lat": 12.9767, "lon": 77.5713, "speed_limit": 30, "ward": "Ward 94 - Gandhinagar"},
    {"name": "Vidhana Soudha Junction", "lat": 12.9796, "lon": 77.5907, "speed_limit": 40, "ward": "Ward 110 - Sampangiram Nagar"},
    {"name": "Cubbon Park East Gate", "lat": 12.9750, "lon": 77.5980, "speed_limit": 45, "ward": "Ward 111 - Shantala Nagar"},
    {"name": "MG Road Metro Station", "lat": 12.9756, "lon": 77.6095, "speed_limit": 35, "ward": "Ward 112 - Domlur Core"},
    {"name": "Trinity Circle", "lat": 12.9729, "lon": 77.6201, "speed_limit": 40, "ward": "Ward 113 - Ulsoor East"},
    {"name": "Indiranagar 100ft Road", "lat": 12.9644, "lon": 77.6412, "speed_limit": 50, "ward": "Ward 114 - Indiranagar"},
    {"name": "Domlur Flyover", "lat": 12.9610, "lon": 77.6480, "speed_limit": 55, "ward": "Ward 115 - Domlur Central"},
    {"name": "HAL Main Gate", "lat": 12.9555, "lon": 77.6650, "speed_limit": 50, "ward": "Ward 116 - HAL Airport"},
    {"name": "Marathahalli Bridge", "lat": 12.9560, "lon": 77.7010, "speed_limit": 40, "ward": "Ward 117 - Marathahalli"},
    {"name": "Kundalahalli Gate", "lat": 12.9660, "lon": 77.7180, "speed_limit": 45, "ward": "Ward 118 - Brookfield"},
    {"name": "ITPB / Whitefield Tech Corridor", "lat": 12.9850, "lon": 77.7310, "speed_limit": 40, "ward": "Ward 119 - Whitefield"},
]
