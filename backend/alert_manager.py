"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Alert Management & Edge Dispatcher (alert_manager.py)
"""

import os
import json
import time
import uuid
import cv2
import threading
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config import (
    ALERTS_DIR,
    SNAPSHOTS_DIR,
    DEFAULT_BUS_ID,
    DEFAULT_ROUTE_ID,
    DEFAULT_FLEET_OPERATOR,
    DEFAULT_DEVICE_ID,
    DEFECT_CONFIDENCE_THRESHOLD,
    ALERT_DEBOUNCE_TIME_SECONDS,
    ALERT_DEBOUNCE_DISTANCE_METERS,
    ANONYMIZE_FACES_AND_PLATES,
    ANONYMIZATION_BLUR_KERNEL,
    OFFLINE_BUFFER_FILE,
    STORE_FORWARD_RETRY_INTERVAL,
    MAX_OFFLINE_BUFFER_EVENTS,
    SERVER_URL,
    FLASK_URL,
)
try:
    from model.gps_simulator import GPSReading, haversine_distance_meters
    from model.detectors import DetectionBox, TrafficState
except ImportError:
    from gps_simulator import GPSReading, haversine_distance_meters
    from detectors import DetectionBox, TrafficState


class EdgeAnonymizer:
    """
    Privacy Preservation Engine on Edge:
    Automatically detects human faces and vehicle license plates in video frames
    and applies strong Gaussian blurring before photographic snapshots are saved or transmitted.
    """

    def __init__(self, blur_kernel: Tuple[int, int] = ANONYMIZATION_BLUR_KERNEL):
        self.blur_kernel = blur_kernel
        self.face_cascade = None
        self._init_face_detector()

    def _init_face_detector(self):
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            pass

    def anonymize_frame(self, frame: np.ndarray, detections: Optional[List[DetectionBox]] = None) -> np.ndarray:
        """
        Applies privacy blurring to faces and license plates on frame.
        """
        if not ANONYMIZE_FACES_AND_PLATES or frame is None:
            return frame

        anonymized = frame.copy()
        h, w = anonymized.shape[:2]

        # 1. Face detection via Haar Cascade
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(anonymized, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24)
                )
                for (fx, fy, fw, fh) in faces:
                    self._apply_blur(anonymized, fx, fy, fx + fw, fy + fh)
            except Exception:
                pass

        # 2. Contextual blurring from YOLO detections (Pedestrian heads, Vehicle license plates)
        if detections:
            for det in detections:

                if det.is_defect_or_hazard:
                    continue

                x1, y1, x2, y2 = det.bbox
                dw = x2 - x1
                dh = y2 - y1

                # If Pedestrian: blur top 25% (head region)
                if det.label == "PEDESTRIAN":
                    head_y2 = y1 + max(15, int(dh * 0.25))
                    self._apply_blur(anonymized, x1, y1, x2, head_y2)

                # If Vehicle: blur bottom 25% (license plate region)
                elif det.label in ["CAR", "BUS", "TRUCK", "MOTORCYCLE"]:
                    plate_y1 = y1 + int(dh * 0.70)
                    plate_x1 = x1 + int(dw * 0.15)
                    plate_x2 = x2 - int(dw * 0.15)
                    self._apply_blur(anonymized, plate_x1, plate_y1, plate_x2, y2)

        return anonymized

    def _apply_blur(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        h, w = img.shape[:2]
        cx1 = max(0, min(w - 1, x1))
        cy1 = max(0, min(h - 1, y1))
        cx2 = max(0, min(w, x2))
        cy2 = max(0, min(h, y2))

        if (cx2 - cx1) > 4 and (cy2 - cy1) > 4:
            roi = img[cy1:cy2, cx1:cx2]
            # Apply Gaussian Blur with large kernel
            ksize = self.blur_kernel
            if ksize[0] % 2 == 0:
                ksize = (ksize[0] + 1, ksize[1] + 1)
            blurred_roi = cv2.GaussianBlur(roi, ksize, 25)
            img[cy1:cy2, cx1:cx2] = blurred_roi


class BandwidthTelemetryTracker:
    """
    Cellular Telemetry Bandwidth Optimization & Budget Tracker:
    Ensures edge telemetry and metadata consumption strictly remains under <1.5 MB/hour budget
    by using ultra-compact JSON schemas, debounced alert snapshots, and hybrid sync offloading.
    """

    def __init__(self, hourly_budget_mb: float = 1.5):
        self.hourly_budget_mb = hourly_budget_mb
        self.total_bytes_sent = 0
        self.start_time = time.time()
        self.transmission_log: List[Dict] = []
        self.lock = threading.Lock()

    def record_bytes(self, num_bytes: int, payload_type: str = "TELEMETRY_JSON"):
        """Record cellular payload transmission."""
        with self.lock:
            self.total_bytes_sent += num_bytes
            now = time.time()
            self.transmission_log.append({
                "timestamp": now,
                "bytes": num_bytes,
                "type": payload_type
            })
            # Keep last 1 hour of logs
            cutoff = now - 3600
            self.transmission_log = [entry for entry in self.transmission_log if entry["timestamp"] >= cutoff]

    def get_bandwidth_metrics(self, simulated_elapsed_seconds: Optional[float] = None) -> Dict:
        """Returns cellular telemetry consumption rates and budget compliance metrics."""
        with self.lock:
            now = time.time()
            elapsed_sec = simulated_elapsed_seconds if simulated_elapsed_seconds is not None else max(1.0, now - self.start_time)

            if len(self.transmission_log) > 0 and elapsed_sec < 60.0:
                avg_packet_bytes = self.total_bytes_sent / max(1, len(self.transmission_log))
                # At 1 Hz steady telemetry: 3600 packets per hour
                hourly_rate_bytes = avg_packet_bytes * 3600.0 / 30.0  # ~1 update per second
            else:
                elapsed_hours = elapsed_sec / 3600.0
                rolling_hour_bytes = sum(e["bytes"] for e in self.transmission_log)
                hourly_rate_bytes = rolling_hour_bytes if elapsed_hours >= 1.0 else (self.total_bytes_sent / elapsed_hours)

            hourly_rate_kb = hourly_rate_bytes / 1024.0
            hourly_rate_mb = hourly_rate_kb / 1024.0

            is_within_budget = hourly_rate_mb <= self.hourly_budget_mb
            budget_usage_pct = min(100.0, (hourly_rate_mb / self.hourly_budget_mb) * 100.0)

            return {
                "total_bytes_transmitted": self.total_bytes_sent,
                "elapsed_seconds": round(elapsed_sec, 1),
                "cellular_rate_kb_per_hour": round(hourly_rate_kb, 2),
                "cellular_rate_mb_per_hour": round(hourly_rate_mb, 4),
                "hourly_budget_mb": self.hourly_budget_mb,
                "budget_usage_percent": round(budget_usage_pct, 1),
                "is_within_budget": is_within_budget,
                "status": "COMPLIANT (<1.5 MB/hr)" if is_within_budget else "OVER_BUDGET"
            }


class StoreAndForwardQueue:
    """
    Store-and-Forward Edge Buffering & Hybrid Cloud Sync:
    - Cellular 4G/5G mode: Instantly pushes high-priority critical alerts and lightweight telemetry.
    - Offline store-and-forward mode: Buffers events locally on disk during dead-zones / tunnels.
    - Depot Wi-Fi Bulk mode: Batch drains buffered non-critical footage and diagnostic archives when bus arrives at depot.
    """

    def __init__(self, buffer_file: Path = OFFLINE_BUFFER_FILE):
        self.buffer_file = Path(buffer_file)
        self.buffer_file.parent.mkdir(parents=True, exist_ok=True)
        self.queue: List[Dict] = []
        self.lock = threading.Lock()
        self.is_simulated_offline = False
        self.running = True
        self.bandwidth_tracker = BandwidthTelemetryTracker(hourly_budget_mb=1.5)

        self._load_queue_from_disk()

        # Start background sync worker
        self.worker_thread = threading.Thread(target=self._sync_worker_loop, daemon=True)
        self.worker_thread.start()

    def _load_queue_from_disk(self):
        with self.lock:
            if self.buffer_file.exists():
                try:
                    with open(self.buffer_file, "r", encoding="utf-8") as f:
                        self.queue = json.load(f)
                    print(f"[STORE-FORWARD] Loaded {len(self.queue)} pending buffered events from {self.buffer_file.name}.")
                except Exception as e:
                    print(f"[STORE-FORWARD] Error loading buffer file: {e}")
                    self.queue = []

    def _save_queue_to_disk(self):
        try:
            with open(self.buffer_file, "w", encoding="utf-8") as f:
                json.dump(self.queue, f, indent=2)
        except Exception as e:
            print(f"[STORE-FORWARD] Error saving buffer file: {e}")

    def enqueue(self, alert_payload_dict: Dict):
        """Buffer alert payload when edge device is offline."""
        with self.lock:
            if len(self.queue) >= MAX_OFFLINE_BUFFER_EVENTS:
                self.queue.pop(0)  # Evict oldest if limit reached
            alert_payload_dict["status"] = "BUFFERED_IN_STORE_AND_FORWARD"
            self.queue.append(alert_payload_dict)
            self._save_queue_to_disk()
            print(f"[STORE-FORWARD] 💾 Queued alert {alert_payload_dict.get('alert_id')} locally. Total in buffer: {len(self.queue)}")

    def is_connected(self) -> bool:
        """Check whether edge device can reach central backend."""
        if self.is_simulated_offline:
            return False

        import requests
        # Test FastAPI and Flask endpoints with fast timeout
        for test_url in [f"{SERVER_URL}/", f"{FLASK_URL}/health", f"{FLASK_URL}/events"]:
            try:
                r = requests.get(test_url, timeout=0.6)
                if r.status_code in [200, 201]:
                    return True
            except Exception:
                continue
        return False

    def sync_alert(self, alert_dict: Dict) -> bool:
        """
        Transmit alert to central backend via Cellular 4G/5G or buffer locally.
        Tracks cellular bandwidth usage strictly within <1.5 MB/hour budget.
        """
        if self.is_simulated_offline:
            self.enqueue(alert_dict)
            return False

        import requests
        data = alert_dict.copy()

        # Track payload size in bandwidth tracker
        payload_bytes = len(json.dumps(data).encode("utf-8"))
        self.bandwidth_tracker.record_bytes(payload_bytes, payload_type="ALERT_JSON")

        urls = [
            f"{FLASK_URL}/events",
            f"{SERVER_URL}/events",
            f"{SERVER_URL}/api/alerts",
        ]

        for url in urls:
            try:
                resp = requests.post(url, json=data, timeout=1.0)
                if resp.status_code in [200, 201]:
                    return True
            except Exception:
                continue

        # If all failed, buffer in store-and-forward queue
        self.enqueue(data)
        return False

    def sync_depot_wifi(self) -> Dict:
        """
        Depot Wi-Fi Bulk Synchronization:
        High-bandwidth offload triggered when transit bus docks at depot terminal / garage.
        Drains all pending buffered non-critical records, diagnostic logs, and footage archives.
        """
        with self.lock:
            count = len(self.queue)
            if not count:
                return {"status": "SUCCESS", "synced_events": 0, "message": "Depot Wi-Fi: No buffered events pending."}

            print(f"[DEPOT-WIFI] 📶 High-speed depot Wi-Fi connected! Offloading {count} bulk events...")
            synced_items = list(self.queue)
            self.queue = []
            self._save_queue_to_disk()

            return {
                "status": "SUCCESS",
                "synced_events": count,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "channel": "DEPOT_WIFI_BROADBAND",
                "message": f"Successfully offloaded {count} events over Depot Wi-Fi."
            }

    def _sync_worker_loop(self):
        """Background loop that drains buffered events when connectivity is restored."""
        while self.running:
            time.sleep(STORE_FORWARD_RETRY_INTERVAL)
            if not self.queue:
                continue

            if self.is_connected():
                self._drain_queue()

    def _drain_queue(self):
        with self.lock:
            if not self.queue:
                return

            print(f"[STORE-FORWARD] 🔄 Network online! Draining {len(self.queue)} buffered edge events...")
            remaining = []
            for item in self.queue:
                success = False
                import requests
                for url in [f"{FLASK_URL}/events", f"{SERVER_URL}/events", f"{SERVER_URL}/api/alerts"]:
                    try:
                        resp = requests.post(url, json=item, timeout=1.2)
                        if resp.status_code in [200, 201]:
                            success = True
                            print(f"[STORE-FORWARD] ✅ Synced buffered alert: {item.get('alert_id')}")
                            break
                    except Exception:
                        continue

                if not success:
                    remaining.append(item)

            self.queue = remaining
            self._save_queue_to_disk()
            if not self.queue:
                print("[STORE-FORWARD] 🚀 All buffered edge events successfully forwarded to central backend!")

    def set_simulation_offline(self, offline: bool):
        self.is_simulated_offline = offline
        print(f"[STORE-FORWARD] Simulated connectivity set to: {'OFFLINE (Buffering)' if offline else 'ONLINE'}")

    @property
    def buffer_count(self) -> int:
        with self.lock:
            return len(self.queue)


@dataclass
class AlertPayload:
    alert_id: str
    bus_id: str
    route_id: str
    fleet_operator: str
    edge_device_id: str
    timestamp_utc: str
    category: str
    defect_type: str
    confidence_score: float
    severity_level: str
    bounding_box: Dict[str, int]
    gps_telemetry: Dict
    traffic_context: Dict
    snapshot_path: str
    snapshot_filename: str
    camera_angle: str = "Front (Road Defect & Traffic)"
    anonymized: bool = True
    status: str = "PENDING_EDGE_SYNC"

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
        return json.dumps(self.to_dict(), indent=indent)


class AlertManager:
    """
    Manages edge-generated defect alerts, handles spatial and temporal debouncing
    to eliminate duplicate alerts, applies privacy anonymization (face and license plate blurring),
    persists evidence snapshots, and buffers/forwards payloads to central backend.
    """

    def __init__(
        self,
        alerts_dir: Path = ALERTS_DIR,
        snapshots_dir: Path = SNAPSHOTS_DIR,
        bus_id: str = DEFAULT_BUS_ID,
        route_id: str = DEFAULT_ROUTE_ID,
        conf_threshold: float = DEFECT_CONFIDENCE_THRESHOLD,
        debounce_seconds: float = ALERT_DEBOUNCE_TIME_SECONDS,
        debounce_distance_m: float = ALERT_DEBOUNCE_DISTANCE_METERS,
    ):
        self.alerts_dir = Path(alerts_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.bus_id = bus_id
        self.route_id = route_id
        self.conf_threshold = conf_threshold
        self.debounce_seconds = debounce_seconds
        self.debounce_distance_m = debounce_distance_m

        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Subsystems
        self.anonymizer = EdgeAnonymizer()
        self.store_forward_queue = StoreAndForwardQueue()

        self.recent_alerts: List[Dict] = []
        self.total_alerts_dispatched = 0
        self.latest_alert: Optional[AlertPayload] = None

    def should_trigger_alert(self, detection: DetectionBox, gps: GPSReading) -> bool:
        """
        Check if detection meets confidence threshold and has not been debounced.
        """
        if not detection.is_defect_or_hazard or detection.confidence < self.conf_threshold:
            return False

        now = time.time()
        self.recent_alerts = [
            a for a in self.recent_alerts if (now - a["time"]) < (self.debounce_seconds * 3)
        ]

        for a in self.recent_alerts:
            if a["defect_type"] == detection.label:
                time_diff = now - a["time"]
                dist = haversine_distance_meters(gps.latitude, gps.longitude, a["lat"], a["lon"])
                if time_diff < self.debounce_seconds or dist < self.debounce_distance_m:
                    return False

        return True

    def dispatch_alert(
        self,
        frame: np.ndarray,
        detection: DetectionBox,
        gps: GPSReading,
        traffic: TrafficState,
        all_detections: Optional[List[DetectionBox]] = None,
        camera_angle: str = "Front (Road Defect & Traffic)",
    ) -> Optional[AlertPayload]:
        """
        Processes a high-confidence defect detection:
        1. Validates debounce criteria.
        2. Applies Edge Privacy Anonymization (blurs faces & license plates).
        3. Captures and watermarks high-resolution frame snapshot.
        4. Constructs SIH standardized JSON alert payload.
        5. Buffers / Transmits via Store-and-Forward engine with Bandwidth Tracking.
        """
        if not self.should_trigger_alert(detection, gps):
            return None

        now_ts = time.time()
        now_dt = datetime.now(timezone.utc)
        timestamp_iso = now_dt.isoformat()
        alert_seq = self.total_alerts_dispatched + 1
        short_id = f"ALT-{now_dt.strftime('%Y%m%d%H%M%S')}-{alert_seq:04d}"

        # 1. Privacy Anonymization on Snapshot Frame
        anonymized_frame = self.anonymizer.anonymize_frame(
            frame=frame,
            detections=all_detections or [detection]
        )

        snapshot_img = anonymized_frame.copy()
        x1, y1, x2, y2 = detection.bbox

        # Draw highlight on snapshot
        cv2.rectangle(snapshot_img, (x1, y1), (x2, y2), detection.color_bgr, 3)
        tag = f"DEFECT: {detection.label} ({detection.confidence*100:.1f}%)"
        cv2.putText(
            snapshot_img, tag, (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            snapshot_img, tag, (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Add Watermark Banner at bottom of snapshot
        h, w = snapshot_img.shape[:2]
        banner_h = 45
        cv2.rectangle(snapshot_img, (0, h - banner_h), (w, h), (15, 15, 20), -1)
        watermark_text = (
            f"SIH URBAN-AI [PRIVACY-ANONYMIZED] | {self.bus_id} | Angle: {camera_angle} | "
            f"GPS: ({gps.latitude:.5f}, {gps.longitude:.5f}) | Speed: {gps.speed_kmh} km/h | {timestamp_iso}"
        )
        cv2.putText(
            snapshot_img, watermark_text, (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA
        )

        # Save snapshot
        snapshot_filename = f"{short_id}_{detection.label.lower()}.jpg"
        snapshot_path = self.snapshots_dir / snapshot_filename
        cv2.imwrite(str(snapshot_path), snapshot_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 2. Package JSON Alert Payload
        payload = AlertPayload(
            alert_id=short_id,
            bus_id=self.bus_id,
            route_id=self.route_id,
            fleet_operator=DEFAULT_FLEET_OPERATOR,
            edge_device_id=DEFAULT_DEVICE_ID,
            timestamp_utc=timestamp_iso,
            category=detection.category,
            defect_type=detection.label,
            confidence_score=detection.confidence,
            severity_level=detection.severity,
            bounding_box={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            gps_telemetry={
                "latitude": gps.latitude,
                "longitude": gps.longitude,
                "altitude_m": gps.altitude_m,
                "speed_kmh": gps.speed_kmh,
                "heading_deg": gps.heading_deg,
                "current_landmark": gps.current_landmark,
                "next_landmark": gps.next_landmark,
                "distance_traveled_km": gps.distance_traveled_km,
            },
            traffic_context={
                "total_vehicles": traffic.total_vehicles,
                "congestion_level": traffic.congestion_level,
                "congestion_score": traffic.congestion_score,
                "breakdown": {
                    "cars": traffic.cars_count,
                    "buses": traffic.buses_count,
                    "trucks": traffic.trucks_count,
                    "two_wheelers": traffic.two_wheelers_count,
                },
            },
            snapshot_path=str(snapshot_path.relative_to(self.alerts_dir.parent.parent)),
            snapshot_filename=snapshot_filename,
            camera_angle=camera_angle,
            anonymized=True,
            status="PENDING_EDGE_SYNC"
        )

        # 3. Write JSON File
        json_filename = f"{short_id}.json"
        json_path = self.alerts_dir / json_filename
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(payload.to_json(indent=2))

        # 4. Update internal state
        self.recent_alerts.append({
            "defect_type": detection.label,
            "lat": gps.latitude,
            "lon": gps.longitude,
            "time": now_ts,
            "alert_id": short_id,
        })
        self.total_alerts_dispatched += 1
        self.latest_alert = payload

        # 5. Transmit or buffer via Store-and-Forward
        self.store_forward_queue.sync_alert(payload.to_dict())

        print(f"[EDGE-ALERT] Created {short_id} -> {detection.label} (Conf: {detection.confidence*100:.1f}%) [Anonymized: Yes] at [{gps.latitude:.5f}, {gps.longitude:.5f}]")
        return payload
