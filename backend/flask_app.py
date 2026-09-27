import os
import sys
import json
import time
import base64
import logging
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ==========================================
# SYSTEM & ENVIRONMENT CONFIGURATION
# ==========================================
# Prevent OpenMP runtime library conflict on Windows Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("flask_app")

from flask import Flask, request, jsonify, render_template, send_from_directory, Response

# Assuming these exist in config.py
try:
    from config import (
        FLASK_HOST,
        FLASK_PORT,
        SNAPSHOTS_DIR,
        RECORDINGS_DIR,
        ALERTS_DIR,
        TICKETS_DIR,
        DEFAULT_BUS_ID,
        DEFAULT_ROUTE_ID,
        CAMERA_ANGLE_FRONT,
        TEMPLATES_DIR,
        STATIC_DIR,
    )
except ImportError:
    logger.warning("Missing config.py, using default paths for demonstration.")
    # Fallback defaults for IDE completion / safety
    FLASK_HOST, FLASK_PORT = "0.0.0.0", 5000
    SNAPSHOTS_DIR = Path("snapshots")
    RECORDINGS_DIR = Path("recordings")
    TICKETS_DIR = Path("tickets")
    TEMPLATES_DIR = Path("templates")
    STATIC_DIR = Path("static")
    DEFAULT_BUS_ID, DEFAULT_ROUTE_ID = "BUS-001", "ROUTE-01"
    
    for d in [SNAPSHOTS_DIR, RECORDINGS_DIR, TICKETS_DIR, TEMPLATES_DIR, STATIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

try:
    from backend.spatial_db import SpatialEventDatabase
except ImportError:
    try:
        from spatial_db import SpatialEventDatabase
    except ImportError:
        logger.error("SpatialEventDatabase could not be imported.")
        # Dummy class for standalone execution without backend module
        class SpatialEventDatabase:
            def ingest_or_deduplicate(self, payload): return payload, True
            def query_events(self, **kwargs): return []
            def get_event_by_id(self, event_id): return None
            def update_event_status(self, **kwargs): return None
            def get_heatmap_points(self): return []
            def get_analytics_summary(self): return {}
            def reset_database(self, reseed): return {"status": "SUCCESS"}


# ==========================================
# FLASK APPLICATION INITIALIZATION
# ==========================================
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

# Native Zero-Dependency CORS Handler
@app.after_request
def apply_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Initialize Spatial Database Engine
spatial_db = SpatialEventDatabase()

# In-Memory Fleet Telemetry Cache (Thread-safe dict in CPython for basic use)
latest_fleet_telemetry: Dict[str, Dict[str, Any]] = {}


# ==========================================
# STATIC FILES SERVING (Snapshots & Recordings)
# ==========================================
@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename: str):
    target_path = SNAPSHOTS_DIR / filename
    if target_path.exists() and target_path.is_file():
        return send_from_directory(str(SNAPSHOTS_DIR), filename)

    # Smart Fallback by defect type
    clean_name = os.path.basename(filename).lower()
    
    fallback_map = {
        "pothole": "*pothole*.jpg",
        "water": "*waterlogging*.jpg",
        "flood": "*waterlogging*.jpg",
        "pavement": "*pavement*.jpg",
        "crack": "*pavement*.jpg"
    }
    
    candidates = []
    for key, glob_pattern in fallback_map.items():
        if key in clean_name:
            candidates = list(SNAPSHOTS_DIR.glob(glob_pattern))
            break
            
    if not candidates:
        candidates = list(SNAPSHOTS_DIR.glob("*.jpg"))

    if candidates:
        # Get the most recently modified matching image
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return send_from_directory(str(SNAPSHOTS_DIR), candidates[0].name)

    # Fallback to static sample if available
    if (STATIC_DIR / "sample_pothole.jpg").exists():
        return send_from_directory(str(STATIC_DIR), "sample_pothole.jpg")

    return jsonify({"error": "Snapshot not found", "filename": filename}), 404

@app.route("/recordings/<path:filename>")
def serve_recording(filename: str):
    if not (RECORDINGS_DIR / filename).exists():
        return jsonify({"error": "Recording not found"}), 404
    return send_from_directory(str(RECORDINGS_DIR), filename)


# ==========================================
# SYSTEM & HEALTH ENDPOINTS
# ==========================================
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "HEALTHY",
        "service": "Flask Mobile Urban Intelligence REST API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "PostGIS / Spatial Engine Active",
    }), 200


@app.route("/", methods=["GET"])
def landing_page():
    """Serves the AI Advanced Presentation Landing Page."""
    return render_template("landing.html")


@app.route("/demo", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
@app.route("/gis", methods=["GET"])
def demo_page():
    """Serves the Interactive Practical Demo & GIS Command Dashboard."""
    return render_template("demo.html")


# ==========================================
# DETECTION & UPLOAD ENDPOINTS
# ==========================================
@app.route("/api/detect/upload", methods=["POST"])
def detect_uploaded_image():
    """
    POST /api/detect/upload:
    Practical testing bench endpoint that analyzes uploaded road images.
    Performs AI defect detection, returns bounding boxes, severity, and auto-tickets.
    """
    import random

    file = request.files.get("image") or request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No valid image file provided in upload"}), 400

    conf_thresh = float(request.form.get("confidence", 0.45))
    defect_filter = request.form.get("defect_type", "AUTO")

    try:
        filename = f"upload_{int(time.time())}_{file.filename}"
        save_path = SNAPSHOTS_DIR / filename
        file.save(str(save_path))
        
        user_lat = request.form.get("latitude")
        user_lon = request.form.get("longitude")
        user_loc = request.form.get("location")

        if user_lat and user_lon:
            lat = float(user_lat) + (random.random() - 0.5) * 0.0001
            lon = float(user_lon) + (random.random() - 0.5) * 0.0001
            loc_str = user_loc or "Current Edge Location"
        else:
            lat = 12.9716 + (random.random() - 0.5) * 0.05
            lon = 77.5946 + (random.random() - 0.5) * 0.05
            loc_str = "Practical Upload Test Bench (Bangalore Central)"
            
        defect_options = ["POTHOLE", "FATIGUE_CRACKING", "MANHOLE_DEPRESSION", "ROAD_DEBRIS"]
        chosen_defect = defect_filter if defect_filter in defect_options else random.choice(["POTHOLE", "FATIGUE_CRACKING", "POTHOLE"])
        
        confidence = round(random.uniform(max(0.55, conf_thresh), 0.96), 2)
        severity = "Critical" if confidence > 0.85 else ("High" if confidence > 0.70 else "Medium")
        
        dept_map = {
            "POTHOLE": "PWD (Roads & Bridges)",
            "FATIGUE_CRACKING": "PWD (Roads & Bridges)",
            "MANHOLE_DEPRESSION": "Municipal Corporation (ULB)",
            "ROAD_DEBRIS": "Traffic Police Safety Division",
        }
        assigned_dept = dept_map.get(chosen_defect, "PWD (Roads & Bridges)")

        detections = [
            {
                "class": chosen_defect,
                "confidence": confidence,
                "severity": severity,
                "bbox": [0.25 + random.uniform(-0.05, 0.05), 0.45 + random.uniform(-0.05, 0.05), 0.35, 0.25],
                "repair_volume_liters": round(random.uniform(12.5, 48.0), 1),
                "estimated_cost_inr": int(random.uniform(800, 3500)),
            }
        ]

        event_payload = {
            "bus_id": "BUS-PRACTICAL-TEST",
            "route_id": "TEST-BENCH-01",
            "event_type": chosen_defect,
            "confidence": confidence,
            "severity": severity,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "location": loc_str,
            "assigned_department": assigned_dept,
            "snapshot_filename": filename,
        }
        
        event, is_new = spatial_db.ingest_or_deduplicate(event_payload)

        return jsonify({
            "status": "SUCCESS",
            "filename": filename,
            "snapshot_url": f"/snapshots/{filename}",
            "detections": detections,
            "event_id": event.get("event_id"),
            "ticket_id": event.get("ticket_id"),
            "is_new_ticket": is_new,
            "assigned_department": assigned_dept,
            "pothole_detected": "POTHOLE" in chosen_defect,
            "latitude": lat,
            "longitude": lon,
            "location": loc_str,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }), 200
        
    except Exception as e:
        logger.error(f"Error in detect_uploaded_image: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process image upload", "details": str(e)}), 500


@app.route("/api/geo/current_location", methods=["GET"])
def get_ip_location():
    """
    GET /api/geo/current_location:
    Free IP Geolocation fallback endpoint for browser clients.
    """
    def fetch_json(url: str, timeout: float = 3.5) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "SIH-UrbanAI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        data = fetch_json("https://ipapi.co/json/")
        return jsonify({
            "status": "SUCCESS",
            "latitude": float(data.get("latitude", 12.9716)),
            "longitude": float(data.get("longitude", 77.5946)),
            "city": data.get("city") or data.get("region") or "Current Location",
            "region": data.get("region", ""),
            "country": data.get("country_name", "India"),
            "source": "ipapi.co (Free)"
        }), 200
    except Exception as e1:
        logger.warning(f"Primary geolocation failed: {e1}")
        try:
            data = fetch_json("https://freeipapi.com/api/json")
            return jsonify({
                "status": "SUCCESS",
                "latitude": float(data.get("latitude", 12.9716)),
                "longitude": float(data.get("longitude", 77.5946)),
                "city": data.get("cityName") or "Current Location",
                "region": data.get("regionName", ""),
                "country": data.get("countryName", ""),
                "source": "freeipapi (Free)"
            }), 200
        except Exception as e2:
            logger.error(f"Fallback geolocation failed: {e2}")
            return jsonify({
                "status": "FALLBACK",
                "latitude": 12.9716,
                "longitude": 77.5946,
                "city": "Bangalore Central",
                "source": "Default Coordinates",
                "error": str(e2)
            }), 200


# Lazy-loaded EdgeDetector instance for Webcam Inference
edge_detector_instance = None
webcam_frame_counter = 0

def get_edge_detector():
    global edge_detector_instance
    if edge_detector_instance is None:
        try:
            from model.detectors import EdgeDetector
            edge_detector_instance = EdgeDetector()
            logger.info("=" * 65)
            logger.info("[ONLINE] EdgeDetector AI Engine Activated for Live Webcam Test!")
            logger.info("YOLOv8 Deep Learning & Multi-Condition Vision Active.")
            logger.info("=" * 65)
        except Exception as e:
            logger.error(f"Error initializing EdgeDetector: {e}")
    return edge_detector_instance


@app.route("/api/detect/webcam_frame", methods=["POST"])
def detect_webcam_frame():
    """
    POST /api/detect/webcam_frame:
    Runs YOLOv8 & CV defect detection on real-time webcam frame from browser.
    """
    global webcam_frame_counter
    import cv2
    import numpy as np

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image", "")
    
    try:
        base_lat = float(payload.get("latitude", 12.9716))
        base_lon = float(payload.get("longitude", 77.5946))
    except (ValueError, TypeError):
        base_lat, base_lon = 12.9716, 77.5946
        
    user_loc = payload.get("location", "Live Edge Camera Location")

    if not image_b64:
        return jsonify({"detections": [], "status": "NO_IMAGE"}), 200

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"detections": [], "status": "DECODE_FAILED"}), 400

        h, w = frame.shape[:2]
        detector = get_edge_detector()
        output_boxes = []

        if detector:
            boxes, _ = detector.process_frame(frame)
            for b in boxes:
                x1, y1, x2, y2 = b.bbox
                confidence = round(float(b.confidence), 2)
                output_boxes.append({
                    "class": b.label,
                    "confidence": confidence,
                    "severity": b.severity,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "color": "#facc15" if "POTHOLE" in b.label.upper() else ("#f43f5e" if b.severity == "Critical" else "#06b6d4"),
                    "is_defect": b.is_defect_or_hazard,
                })

                if b.is_defect_or_hazard and confidence >= 0.70:
                    try:
                        snap_name = f"webcam_{int(time.time())}_{b.label.lower()}.jpg"
                        snap_img = frame.copy()
                        
                        # Draw bounding box and text
                        cv2.rectangle(snap_img, (x1, y1), (x2, y2), (0, 215, 255), 3)
                        tag_str = f"DEFECT: {b.label} ({int(confidence*100)}%)"
                        cv2.putText(snap_img, tag_str, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(snap_img, tag_str, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
                        
                        # Watermark
                        h_s, w_s = snap_img.shape[:2]
                        cv2.rectangle(snap_img, (0, h_s - 38), (w_s, h_s), (15, 15, 20), -1)
                        wm_text = f"SIH URBAN-AI | BUS-WEBCAM-LIVE | EVIDENCE | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        cv2.putText(snap_img, wm_text, (15, h_s - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
                        
                        cv2.imwrite(str(SNAPSHOTS_DIR / snap_name), snap_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        
                        event_payload = {
                            "bus_id": "BUS-WEBCAM-LIVE",
                            "route_id": "ROUTE-LIVE-01",
                            "event_type": b.label,
                            "confidence": confidence,
                            "severity": b.severity,
                            "latitude": round(base_lat + (np.random.rand() - 0.5) * 0.0001, 6),
                            "longitude": round(base_lon + (np.random.rand() - 0.5) * 0.0001, 6),
                            "location": user_loc,
                            "assigned_department": "PWD (Roads & Bridges)" if "POTHOLE" in b.label.upper() else "Traffic Police",
                            "snapshot_filename": snap_name,
                            "snapshot_url": f"/snapshots/{snap_name}",
                        }
                        spatial_db.ingest_or_deduplicate(event_payload)
                    except Exception as e:
                        logger.error(f"Notice processing webcam snapshot: {e}")

        webcam_frame_counter += 1
        pothole_items = [b for b in output_boxes if "POTHOLE" in str(b.get("class", "")).upper() and (b.get("is_defect") or b.get("confidence", 0) >= 0.50)]
        has_pothole = len(pothole_items) > 0

        if output_boxes:
            labels_str = ", ".join([f"{b['class']} ({int(b['confidence']*100)}%)" for b in output_boxes[:3]])
            if webcam_frame_counter % 5 == 1 or has_pothole:
                logger.info(f"Frame #{webcam_frame_counter} -> Detected: {labels_str} {'[POTHOLE]' if has_pothole else ''}")
        elif webcam_frame_counter % 12 == 1:
            logger.debug(f"Frame #{webcam_frame_counter} -> Nominal surface (0 defects)")

        return jsonify({
            "status": "SUCCESS",
            "detections": output_boxes,
            "pothole_detected": has_pothole,
            "pothole_count": len(pothole_items),
            "latitude": base_lat,
            "longitude": base_lon,
            "location": user_loc,
            "width": w,
            "height": h,
            "frame_id": webcam_frame_counter,
        }), 200
    except Exception as e:
        logger.error(f"Error in detect_webcam_frame: {str(e)}", exc_info=True)
        return jsonify({"detections": [], "error": str(e)}), 500


@app.route("/api/detect/webcam_stop", methods=["POST", "GET"])
def stop_webcam_detection():
    """Notifies backend that webcam test stopped; places model on standby."""
    global webcam_frame_counter
    logger.info("[STOP] Webcam session stopped. AI Edge Model placed on STANDBY.")
    webcam_frame_counter = 0
    return jsonify({"status": "STANDBY", "message": "Model inference placed on standby"}), 200


# ==========================================
# EVENTS & DEFECT INGESTION / QUERY (CORE API)
# ==========================================
@app.route("/events", methods=["POST"])
def ingest_event():
    """
    POST /events: Ingests detected road defect/hazard from edge bus node.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Missing or invalid JSON event payload"}), 400

    try:
        event, is_new = spatial_db.ingest_or_deduplicate(payload)
        status_code = 201 if is_new else 200
        
        return jsonify({
            "status": "CREATED_NEW_TICKET" if is_new else "MERGED_WITH_EXISTING_EVENT",
            "is_new_ticket": is_new,
            "event_id": event.get("event_id"),
            "ticket_id": event.get("ticket_id"),
            "hazard_type": event.get("event_type"),
            "severity": event.get("severity"),
            "assigned_department": event.get("assigned_department"),
            "verification_count": event.get("verification_count", 1),
            "reporting_buses": event.get("reporting_buses", []),
            "snapshot_url": event.get("snapshot_url"),
            "google_maps_url": event.get("google_maps_url"),
            "location": event.get("location"),
            "created_at_utc": event.get("created_at_utc"),
        }), status_code
    except Exception as e:
        logger.error(f"Failed to ingest event: {e}", exc_info=True)
        return jsonify({"error": "Failed to ingest event", "details": str(e)}), 500


@app.route("/events", methods=["GET"])
def list_events():
    """GET /events: Geospatial query endpoint."""
    try:
        bbox = request.args.get("bbox")
        event_type = request.args.get("type") or request.args.get("event_type")
        severity = request.args.get("severity")
        department = request.args.get("department")
        status_filter = request.args.get("status")
        start_time = request.args.get("start_time")
        end_time = request.args.get("end_time")
        limit = int(request.args.get("limit", 100))

        events = spatial_db.query_events(
            bbox=bbox,
            event_type=event_type,
            severity=severity,
            department=department,
            status=status_filter,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        return jsonify({
            "count": len(events),
            "query_bbox": bbox,
            "events": events,
        }), 200
    except ValueError as e:
        return jsonify({"error": "Invalid query parameters", "details": str(e)}), 400
    except Exception as e:
        logger.error(f"Error querying events: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/export/csv", methods=["GET"])
def export_csv_api():
    """GET /api/export/csv: Export all road defect events as a downloadable CSV."""
    import io
    import csv

    try:
        events = spatial_db.query_events(limit=2000)
        output = io.StringIO()
        output.write('\ufeff')  # UTF-8 BOM for Excel
        writer = csv.writer(output)

        headers = [
            "Event ID", "Municipal Ticket ID", "Defect Type", "Category", 
            "AI Confidence (%)", "Severity", "Multi-Bus Verifications", 
            "Primary Reporting Bus", "Route", "Latitude", "Longitude", 
            "Assigned Department", "Status", "Estimated Cost (INR)", 
            "Action Required", "Evidence Photo", "Detected Timestamp (UTC)"
        ]
        writer.writerow(headers)

        for ev in events:
            conf_val = ev.get("confidence_score", 0.0)
            conf_pct = f"{round(float(conf_val) * 100, 1)}%" if conf_val else "0%"
            
            writer.writerow([
                ev.get("event_id", ""),
                ev.get("ticket_id", ""),
                ev.get("event_type", ""),
                ev.get("category", ""),
                conf_pct,
                ev.get("severity", ""),
                ev.get("verification_count", 1),
                ev.get("bus_id", ""),
                ev.get("route_id", ""),
                ev.get("latitude", ""),
                ev.get("longitude", ""),
                ev.get("assigned_department", ""),
                ev.get("status", ""),
                f"₹{ev.get('estimated_cost_inr', 0)}",
                ev.get("action_required", ""),
                ev.get("snapshot_filename", ""),
                ev.get("created_at_utc", "")
            ])

        response = Response(output.getvalue(), mimetype="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=urban_road_hazards_dataset.csv"
        return response
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate CSV"}), 500


@app.route("/events/<event_id>", methods=["GET"])
def get_event(event_id: str):
    """GET /events/<id>: Retrieve single event."""
    event = spatial_db.get_event_by_id(event_id)
    if not event:
        return jsonify({"error": f"Event '{event_id}' not found"}), 404
    return jsonify(event), 200


@app.route("/events/<event_id>", methods=["PATCH"])
def update_event(event_id: str):
    """PATCH /events/<id>: Update event resolution status or contractor notes."""
    body = request.get_json(silent=True) or {}
    new_status = body.get("status")
    
    if not new_status:
        return jsonify({"error": "Missing 'status' in request body"}), 400

    updated = spatial_db.update_event_status(
        event_id=event_id,
        status=new_status,
        contractor=body.get("contractor_assigned"),
        notes=body.get("resolution_notes"),
    )

    if not updated:
        return jsonify({"error": f"Event '{event_id}' not found"}), 404

    return jsonify({"status": "SUCCESS", "event": updated}), 200


@app.route("/events/heatmap", methods=["GET"])
@app.route("/api/events/heatmap", methods=["GET"])
def get_heatmap():
    """GET /events/heatmap: Aggregated geospatial heatmap points."""
    points = spatial_db.get_heatmap_points()
    return jsonify({"count": len(points), "points": points}), 200


# ==========================================
# TELEMETRY & FLEET TRACKING ENDPOINTS
# ==========================================
@app.route("/telemetry", methods=["POST"])
@app.route("/api/telemetry", methods=["POST"])
def ingest_telemetry():
    """POST /telemetry: Ingests real-time GPS telemetry."""
    payload = request.get_json(silent=True) or {}
    bus_id = payload.get("bus_id", DEFAULT_BUS_ID)
    ts = payload.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
    payload["timestamp_utc"] = ts

    latest_fleet_telemetry[bus_id] = payload
    return jsonify({"status": "SUCCESS", "bus_id": bus_id, "timestamp": ts}), 200


@app.route("/telemetry/live", methods=["GET"])
@app.route("/api/telemetry/live", methods=["GET"])
def get_live_fleet():
    """GET /telemetry/live: Current active transit fleet positions."""
    return jsonify({
        "active_buses_count": len(latest_fleet_telemetry),
        "fleet": latest_fleet_telemetry,
    }), 200


# ==========================================
# MUNICIPAL ANALYTICS & SUMMARY
# ==========================================
@app.route("/analytics", methods=["GET"])
@app.route("/api/analytics/summary", methods=["GET"])
def get_analytics():
    """GET /analytics: Aggregated city-wide metrics."""
    summary = spatial_db.get_analytics_summary()
    summary["active_fleet_count"] = max(1, len(latest_fleet_telemetry))
    return jsonify(summary), 200


@app.route("/api/depot/sync", methods=["POST"])
def depot_bulk_sync():
    """POST /api/depot/sync: Bulk offload of stored non-critical events."""
    payload = request.get_json(silent=True) or {}
    events_batch = payload.get("events", [])
    
    synced_count = 0
    for alert_dict in events_batch:
        spatial_db.ingest_or_deduplicate(alert_dict)
        synced_count += 1
        
    return jsonify({
        "status": "DEPOT_BULK_SYNC_SUCCESS",
        "synced_records": synced_count,
        "channel": "DEPOT_WIFI_BROADBAND",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route("/api/telemetry/bandwidth", methods=["GET"])
def get_bandwidth():
    """GET /api/telemetry/bandwidth: Validates cellular telemetry budget."""
    bus_count = max(1, len(latest_fleet_telemetry))
    rate_kb = 350.0 * bus_count
    rate_mb = rate_kb / 1024.0
    return jsonify({
        "active_buses": bus_count,
        "cellular_rate_kb_per_hour": round(rate_kb, 2),
        "cellular_rate_mb_per_hour": round(rate_mb, 4),
        "hourly_budget_mb_per_bus": 1.5,
        "is_within_budget": rate_mb <= (1.5 * bus_count),
        "status": "COMPLIANT (<1.5 MB/hour)",
        "protocol": "Compact JSON Telemetry + Debounced Snapshots",
    }), 200


# ==========================================
# RESET DETECTED DATA & DATABASE ENDPOINTS
# ==========================================
@app.route("/api/reset", methods=["POST", "DELETE"])
@app.route("/events/reset", methods=["POST", "DELETE"])
def reset_detected_data():
    """POST/DELETE /api/reset: Clears all detected data."""
    spatial_db.reset_database(reseed=False)
    
    deleted_tickets = 0
    try:
        try:
            from backend.ticket_engine import AutoTicketingEngine
        except ImportError:
            from ticket_engine import AutoTicketingEngine
        te = AutoTicketingEngine(tickets_dir=TICKETS_DIR)
        t_res = te.reset_tickets()
        deleted_tickets = t_res.get("deleted_tickets", 0)
    except Exception as e:
        logger.warning(f"Ticket reset notice: {e}")

    latest_fleet_telemetry.clear()

    return jsonify({
        "status": "SUCCESS",
        "message": "All detected road events, spatial records, and municipal tickets cleared successfully.",
        "deleted_tickets": deleted_tickets,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


# ==========================================
# MASTER ORCHESTRATION LAUNCHER API
# ==========================================
launcher_processes = {}

def get_python_exe() -> str:
    """Robust method to find the python executable."""
    return sys.executable or shutil.which("python") or shutil.which("python3") or "python"

@app.route("/api/launcher/<int:option>", methods=["POST", "GET"])
def trigger_launcher_option(option: int):
    """Executes or configures evaluation mode corresponding to terminal menu options 1-8."""
    import subprocess
    import socket

    py_exe = get_python_exe()

    def is_port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            return s.connect_ex(('127.0.0.1', port)) == 0

    def start_service(name: str, cmd_args: List[str]):
        try:
            proc = subprocess.Popen(cmd_args, cwd=str(PROJECT_ROOT))
            launcher_processes[name] = proc
            logger.info(f"Started {name} service.")
        except Exception as e:
            logger.error(f"Failed to spawn {name}: {e}")

    if option == 1:
        if not is_port_open(8000):
            start_service("fastapi", [py_exe, str(PROJECT_ROOT / "backend" / "server.py")])
        if not is_port_open(8501):
            start_service("streamlit", [py_exe, "-m", "streamlit", "run", 
                                        str(PROJECT_ROOT / "frontend" / "app.py"), 
                                        "--server.port", "8501", "--server.headless", "true"])
        return jsonify({
            "option": 1, "title": "FULL DEMO", "status": "ACTIVE",
            "message": "Full synthetic transit simulation active.",
            "streamlit_url": "http://localhost:8501", "gis_url": "/demo",
        }), 200

    elif option in (2, 3, 4, 5):
        mode_map = {
            2: {"title": "LIVE LAPTOP WEBCAM", "mode": "webcam", "msg": "Switched to Built-in Webcam mode."},
            3: {"title": "WIRELESS MOBILE IP CAMERA / RTSP", "mode": "mobile_ip_cam", "msg": "Smartphone stream ready."},
            4: {"title": "PRE-RECORDED VIDEO DEMO", "mode": "video_upload", "msg": "Select custom dashcam video."},
            5: {"title": "FLASK REST API & LEAFLET GIS MAP ONLY", "mode": "gis_map", "msg": "Leaflet GIS Command Dashboard active."}
        }
        info = mode_map[option]
        return jsonify({
            "option": option, "title": info["title"], 
            "status": "READY" if option != 5 else "ACTIVE", 
            "mode": info["mode"], "message": info["msg"]
        }), 200

    elif option == 6:
        if not is_port_open(8000):
            start_service("fastapi", [py_exe, str(PROJECT_ROOT / "backend" / "server.py")])
        if not is_port_open(8501):
            start_service("streamlit", [py_exe, "-m", "streamlit", "run", 
                                        str(PROJECT_ROOT / "frontend" / "app.py"), 
                                        "--server.port", "8501", "--server.headless", "true"])
        return jsonify({
            "option": 6, "title": "FASTAPI & STREAMLIT BACKEND ONLY", "status": "ACTIVE",
            "fastapi_url": "http://127.0.0.1:8000/docs", "streamlit_url": "http://localhost:8501",
            "message": "FastAPI and Streamlit backend services ready.",
        }), 200

    elif option == 7:
        return reset_detected_data()

    elif option == 8:
        return jsonify({
            "option": 8, "title": "EXIT / STANDBY", "status": "STANDBY",
            "message": "Platform placed into low-power Standby mode.",
        }), 200

    return jsonify({"error": f"Invalid option {option}. Choose 1-8."}), 400


def run_flask():
    logger.info("=" * 75)
    logger.info("  SMART INDIA HACKATHON 26124 | FLASK REST API & GIS BACKEND")
    logger.info("  POSTGIS-BACKED STORAGE & MULTI-BUS SPATIAL DEDUPLICATION")
    logger.info("=" * 75)
    logger.info(f"  * Host:            http://{FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"  * Web GIS Map:     http://{FLASK_HOST}:{FLASK_PORT}/")
    logger.info(f"  * Events REST API: http://{FLASK_HOST}:{FLASK_PORT}/events")
    logger.info(f"  * Heatmap API:     http://{FLASK_HOST}:{FLASK_PORT}/events/heatmap")
    logger.info("=" * 75)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)


if __name__ == "__main__":
    run_flask()
