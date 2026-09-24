"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: PostGIS & Spatial Database Engine with Multi-Bus Deduplication (spatial_db.py)
"""

import os
import json
import math
import sqlite3
import random
import string
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from config import (
    DATA_DIR,
    TICKETS_DIR,
    ALERTS_DIR,
    SNAPSHOTS_DIR,
    DATABASE_URL,
    SPATIAL_SQLITE_PATH,
    SPATIAL_DEDUP_RADIUS_METERS,
    SPATIAL_DEDUP_WINDOW_HOURS,
    DEFECT_TYPES,
    SIMULATED_ROUTE_WAYPOINTS,
)
try:
    from model.gps_simulator import haversine_distance_meters
except ImportError:
    from gps_simulator import haversine_distance_meters


class SpatialEventDatabase:
    """
    Geospatial Event Storage & Analysis Engine:
    - PostGIS / Spatial SQLite backing with real geometry columns & bounding-box queries.
    - Multi-bus spatial deduplication: merges nearby reports of identical defects from different fleet buses.
    - Spatial aggregation for Leaflet heatmap generation.
    - Automated municipal ticket synchronization.
    """

    def __init__(self, db_path: Path = SPATIAL_SQLITE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pg_conn_str = DATABASE_URL
        self.is_postgres = False

        self._init_database()
        self._seed_existing_data()

    def _get_connection(self):
        """Returns a SQLite connection with dict row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Create SQLite events schema with spatial indexing support."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            ticket_id TEXT UNIQUE,
            tracking_code TEXT,
            bus_id TEXT NOT NULL,
            route_id TEXT NOT NULL,
            category TEXT NOT NULL,
            event_type TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            severity TEXT NOT NULL,
            assigned_department TEXT NOT NULL,
            ward_zone TEXT,
            status TEXT DEFAULT 'DISPATCHED_TO_CONTRACTOR',
            action_required TEXT,
            estimated_cost_inr INTEGER DEFAULT 0,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            altitude_m REAL DEFAULT 920.0,
            speed_kmh REAL DEFAULT 30.0,
            landmark TEXT,
            snapshot_filename TEXT,
            snapshot_url TEXT,
            google_maps_url TEXT,
            anonymized INTEGER DEFAULT 1,
            verification_count INTEGER DEFAULT 1,
            reporting_buses TEXT,
            traffic_context TEXT,
            audit_trail TEXT,
            created_at_utc TEXT,
            updated_at_utc TEXT,
            geojson_geom TEXT
        );
        """)

        # Create Spatial and Query Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_lat_lon ON events(latitude, longitude);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at_utc);")

        conn.commit()
        conn.close()

    def _find_nearby_duplicate(
        self,
        lat: float,
        lon: float,
        defect_type: str,
        max_dist_m: float = SPATIAL_DEDUP_RADIUS_METERS,
        time_window_hours: float = SPATIAL_DEDUP_WINDOW_HOURS,
    ) -> Optional[Dict]:
        """
        Spatial Deduplication Query:
        Checks if the same defect type was already reported within `max_dist_m` meters and `time_window_hours`.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Bounding box rough filter (~0.001 deg ~ 111m)
        deg_delta = (max_dist_m * 2) / 111000.0
        min_lat, max_lat = lat - deg_delta, lat + deg_delta
        min_lon, max_lon = lon - deg_delta, lon + deg_delta

        cursor.execute("""
            SELECT * FROM events
            WHERE event_type = ?
              AND latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
            ORDER BY created_at_utc DESC
        """, (defect_type, min_lat, max_lat, min_lon, max_lon))

        candidates = [dict(row) for row in cursor.fetchall()]
        conn.close()

        now_dt = datetime.now(timezone.utc)
        for cand in candidates:
            # Check exact haversine distance
            dist = haversine_distance_meters(lat, lon, cand["latitude"], cand["longitude"])
            if dist <= max_dist_m:
                # Check time window
                try:
                    c_time = datetime.fromisoformat(cand["created_at_utc"].replace("Z", "+00:00"))
                    if (now_dt - c_time).total_seconds() <= (time_window_hours * 3600):
                        return cand
                except Exception:
                    return cand

        return None

    def ingest_or_deduplicate(self, alert_data: Dict) -> Tuple[Dict, bool]:
        """
        Main Event Ingestion with Multi-Bus Spatial Deduplication:
        - If an event is already reported nearby, merge into existing record, increment verification count, and update audit trail.
        - If unique, create new record and official municipal ticket.
        Returns: (event_dict, is_new_ticket)
        """
        gps = alert_data.get("gps_telemetry", {})
        lat = float(gps.get("latitude", alert_data.get("latitude", 12.9767)))
        lon = float(gps.get("longitude", alert_data.get("longitude", 77.5713)))
        defect_type = alert_data.get("defect_type", "pothole").upper()
        bus_id = alert_data.get("bus_id", "BUS-KA-01-F-1204")
        route_id = alert_data.get("route_id", "ROUTE-335E")
        conf = float(alert_data.get("confidence_score", 0.85))

        # Check for nearby duplicate
        existing = self._find_nearby_duplicate(lat, lon, defect_type)

        conn = self._get_connection()
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()

        if existing:
            # -------------------------------------------------------------
            # MERGE MULTI-BUS REPORT INTO EXISTING EVENT
            # -------------------------------------------------------------
            event_id = existing["event_id"]
            new_count = existing["verification_count"] + 1

            # Update reporting buses list
            try:
                buses_list = json.loads(existing.get("reporting_buses") or "[]")
            except Exception:
                buses_list = [existing.get("bus_id")]
            if bus_id not in buses_list:
                buses_list.append(bus_id)

            # Update confidence score (higher confidence on multi-bus validation)
            merged_conf = min(0.99, max(existing["confidence_score"], conf) + 0.03)

            # Update audit trail
            try:
                audit_list = json.loads(existing.get("audit_trail") or "[]")
            except Exception:
                audit_list = []

            audit_list.append({
                "timestamp": now_iso,
                "action": "MULTI_BUS_SPATIAL_VERIFICATION",
                "actor": f"FLEET_NODE_{bus_id}",
                "details": f"Secondary independent confirmation by {bus_id} on {route_id}. Total multi-bus confirmations: {new_count}."
            })

            cursor.execute("""
                UPDATE events
                SET verification_count = ?,
                    reporting_buses = ?,
                    confidence_score = ?,
                    audit_trail = ?,
                    updated_at_utc = ?
                WHERE event_id = ?
            """, (new_count, json.dumps(buses_list), merged_conf, json.dumps(audit_list), now_iso, event_id))

            conn.commit()

            cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            updated_event = dict(cursor.fetchone())
            conn.close()

            print(f"[SPATIAL-DEDUP] Merged duplicate {defect_type} report from {bus_id} into existing Event: {event_id} (Verifications: {new_count})")
            return self._format_event(updated_event), False

        else:
            # -------------------------------------------------------------
            # CREATE NEW UNIQUE GEOMETRY EVENT & TICKET
            # -------------------------------------------------------------
            defect_key = defect_type.lower()
            defect_info = DEFECT_TYPES.get(defect_key, DEFECT_TYPES.get("pothole"))

            dept_prefix = "PWD"
            if "Police" in defect_info["department"]:
                dept_prefix = "BLR-POLICE"
            elif "Ambulance" in defect_info["department"] or "Emergency" in defect_info["department"]:
                dept_prefix = "EMS-108"
            elif "Municipal" in defect_info["department"] or "Drainage" in defect_info["department"]:
                dept_prefix = "BBMP"

            year_str = datetime.now(timezone.utc).strftime("%Y")
            rand_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
            ticket_id = f"{dept_prefix}-{year_str}-{rand_suffix}"
            event_id = alert_data.get("alert_id") or f"EVT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
            tracking_code = f"TRK-{random.randint(100000, 999999)}"

            ward_zone = self._resolve_ward(lat, lon, gps.get("current_landmark", "Central Zone"))
            maps_url = f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"

            geojson_geom = json.dumps({
                "type": "Point",
                "coordinates": [round(lon, 6), round(lat, 6)]
            })

            audit_trail = [
                {
                    "timestamp": now_iso,
                    "action": "AUTO_EVENT_CAPTURED",
                    "actor": f"EDGE_NODE_{bus_id}",
                    "details": f"Detected {defect_info['name']} with {conf*100:.1f}% AI confidence.",
                },
                {
                    "timestamp": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
                    "action": "MUNICIPAL_DISPATCHED",
                    "actor": "CENTRAL_COMMAND_ENGINE",
                    "details": f"Work order dispatched to {defect_info['department']} (SLA: {defect_info.get('sla_hours', 48)}h).",
                }
            ]

            snapshot_filename = alert_data.get("snapshot_filename")
            if not snapshot_filename or not (SNAPSHOTS_DIR / snapshot_filename).exists():
                candidates = list(SNAPSHOTS_DIR.glob(f"*{defect_key}*.jpg"))
                if not candidates:
                    candidates = list(SNAPSHOTS_DIR.glob("*.jpg"))
                if candidates:
                    snapshot_filename = candidates[0].name
                else:
                    snapshot_filename = f"{event_id}_{defect_key}.jpg"
            snapshot_url = f"/snapshots/{snapshot_filename}"


            cursor.execute("""
                INSERT INTO events (
                    event_id, ticket_id, tracking_code, bus_id, route_id, category,
                    event_type, confidence_score, severity, assigned_department, ward_zone,
                    status, action_required, estimated_cost_inr, latitude, longitude,
                    altitude_m, speed_kmh, landmark, snapshot_filename, snapshot_url,
                    google_maps_url, anonymized, verification_count, reporting_buses,
                    traffic_context, audit_trail, created_at_utc, updated_at_utc, geojson_geom
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                event_id, ticket_id, tracking_code, bus_id, route_id,
                alert_data.get("category", defect_info["category"]),
                defect_info["name"].upper(), conf,
                alert_data.get("severity_level", defect_info["severity"]),
                defect_info["department"], ward_zone,
                "DISPATCHED_TO_CONTRACTOR", defect_info["action_required"],
                defect_info.get("estimated_cost_inr", 0), lat, lon,
                float(gps.get("altitude_m", 920.0)), float(gps.get("speed_kmh", 30.0)),
                gps.get("current_landmark", "Urban Corridor"),
                snapshot_filename, snapshot_url, maps_url,
                1 if alert_data.get("anonymized", True) else 0,
                1, json.dumps([bus_id]),
                json.dumps(alert_data.get("traffic_context", {})),
                json.dumps(audit_trail),
                now_iso, now_iso, geojson_geom
            ))

            conn.commit()

            cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            new_event = dict(cursor.fetchone())
            conn.close()

            # Sync to tickets folder as JSON for backward compatibility
            self._save_ticket_file(new_event)

            print(f"[SPATIAL-DB] Created New Event: {event_id} ({defect_info['name']}) -> {defect_info['department']}")
            return self._format_event(new_event), True

    def query_events(
        self,
        bbox: Optional[Union[str, List[float]]] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        department: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Geospatial and attribute query endpoint:
        - `bbox`: min_lon, min_lat, max_lon, max_lat (string or 4-element list/tuple)
        - `start_time` / `end_time`: ISO UTC timestamps
        - `event_type`, `severity`, `department`, `status`
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        # 1. Bounding Box Spatial Filter (min_lon, min_lat, max_lon, max_lat)
        if bbox:
            try:
                if isinstance(bbox, str):
                    parts = [float(v.strip()) for v in bbox.split(",")]
                else:
                    parts = [float(v) for v in bbox]

                if len(parts) == 4:
                    min_lon, min_lat, max_lon, max_lat = parts
                    query += " AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
                    params.extend([min_lat, max_lat, min_lon, max_lon])
            except Exception as e:
                print(f"[SPATIAL-QUERY] Invalid bbox format '{bbox}': {e}")

        # 2. Attribute Filters
        if event_type and event_type != "ALL":
            query += " AND (UPPER(event_type) LIKE ? OR category LIKE ?)"
            params.extend([f"%{event_type.upper()}%", f"%{event_type}%"])

        if severity and severity != "ALL":
            query += " AND severity = ?"
            params.append(severity)

        if department and department != "ALL":
            query += " AND assigned_department LIKE ?"
            params.append(f"%{department}%")

        if status and status != "ALL":
            query += " AND status = ?"
            params.append(status)

        if start_time:
            query += " AND created_at_utc >= ?"
            params.append(start_time)

        if end_time:
            query += " AND created_at_utc <= ?"
            params.append(end_time)

        query += " ORDER BY created_at_utc DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = [self._format_event(dict(r)) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_heatmap_points(self) -> List[Dict]:
        """
        Returns aggregated geospatial damage and traffic hotspot points:
        `[{"lat": float, "lon": float, "intensity": float, "type": str, "severity": str}]`
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT latitude, longitude, severity, verification_count, event_type FROM events")
        rows = cursor.fetchall()
        conn.close()

        heatmap_points = []
        for r in rows:
            sev = r["severity"]
            base_weight = 1.0 if sev == "Critical" else (0.75 if sev == "High" else (0.50 if sev == "Medium" else 0.25))
            # Scale slightly with multi-bus confirmation
            weight = min(1.0, base_weight * (1.0 + 0.15 * (r["verification_count"] - 1)))
            heatmap_points.append({
                "lat": round(r["latitude"], 6),
                "lon": round(r["longitude"], 6),
                "intensity": round(weight, 2),
                "event_type": r["event_type"],
                "severity": sev,
                "verification_count": r["verification_count"]
            })
        return heatmap_points

    def get_event_by_id(self, event_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE event_id = ? OR ticket_id = ?", (event_id, event_id))
        row = cursor.fetchone()
        conn.close()
        if row:
            return self._format_event(dict(row))
        return None

    def update_event_status(
        self,
        event_id: str,
        status: str,
        contractor: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Optional[Dict]:
        event = self.get_event_by_id(event_id)
        if not event:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()

        audit_list = event.get("audit_trail", [])
        audit_list.append({
            "timestamp": now_iso,
            "action": f"STATUS_UPDATED_{status}",
            "actor": "MUNICIPAL_OFFICER",
            "details": notes or f"Ticket status changed to {status}. Assigned: {contractor or 'Default'}.",
        })

        cursor.execute("""
            UPDATE events
            SET status = ?,
                audit_trail = ?,
                updated_at_utc = ?
            WHERE event_id = ? OR ticket_id = ?
        """, (status, json.dumps(audit_list), now_iso, event_id, event_id))

        conn.commit()
        cursor.execute("SELECT * FROM events WHERE event_id = ? OR ticket_id = ?", (event_id, event_id))
        updated = dict(cursor.fetchone())
        conn.close()

        formatted = self._format_event(updated)
        self._save_ticket_file(formatted)
        return formatted

    def get_analytics_summary(self) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM events")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as open_cnt FROM events WHERE status IN ('PENDING_DISPATCH', 'DISPATCHED_TO_CONTRACTOR', 'IN_PROGRESS')")
        open_cnt = cursor.fetchone()["open_cnt"]

        cursor.execute("SELECT COUNT(*) as res_cnt FROM events WHERE status IN ('RESOLVED', 'CLOSED')")
        res_cnt = cursor.fetchone()["res_cnt"]

        cursor.execute("SELECT COUNT(*) as crit FROM events WHERE severity = 'Critical'")
        crit = cursor.fetchone()["crit"]

        cursor.execute("SELECT COUNT(*) as high FROM events WHERE severity = 'High'")
        high = cursor.fetchone()["high"]

        cursor.execute("SELECT SUM(estimated_cost_inr) as cost FROM events")
        cost = cursor.fetchone()["cost"] or 0

        cursor.execute("SELECT assigned_department, COUNT(*) as cnt FROM events GROUP BY assigned_department")
        dept_dist = {r["assigned_department"]: r["cnt"] for r in cursor.fetchall()}

        cursor.execute("SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type")
        type_dist = {r["event_type"]: r["cnt"] for r in cursor.fetchall()}

        conn.close()

        health_deductions = (crit * 8) + (high * 4) + ((total - crit - high) * 1.5)
        road_health = max(30.0, min(100.0, 100.0 - health_deductions))

        return {
            "total_events": total,
            "open_tickets": open_cnt,
            "resolved_tickets": res_cnt,
            "critical_severity_count": crit,
            "high_severity_count": high,
            "total_estimated_repair_cost_inr": cost,
            "city_road_health_index": round(road_health, 1),
            "department_distribution": dept_dist,
            "hazard_type_distribution": type_dist,
        }

    def _format_event(self, d: Dict) -> Dict:
        """Parses JSON strings and constructs GeoJSON geometry."""
        out = dict(d)
        for json_key in ["reporting_buses", "traffic_context", "audit_trail"]:
            if isinstance(out.get(json_key), str):
                try:
                    out[json_key] = json.loads(out[json_key])
                except Exception:
                    pass

        out["geometry"] = {
            "type": "Point",
            "coordinates": [round(out["longitude"], 6), round(out["latitude"], 6)]
        }
        out["location"] = {
            "latitude": out["latitude"],
            "longitude": out["longitude"],
            "altitude_m": out.get("altitude_m", 920.0),
            "speed_kmh": out.get("speed_kmh", 30.0),
            "current_landmark": out.get("landmark", "Bangalore Corridor"),
        }
        return out

    def _resolve_ward(self, lat: float, lon: float, landmark: str) -> str:
        for wp in SIMULATED_ROUTE_WAYPOINTS:
            if wp["name"].lower() in landmark.lower() or landmark.lower() in wp["name"].lower():
                return wp.get("ward", "Ward 112 - Domlur Core")
        return "Ward 112 - Central Urban Zone"

    def _save_ticket_file(self, event_dict: Dict):
        """Saves ticket as JSON in data/tickets/ for filesystem interoperability."""
        tid = event_dict.get("ticket_id") or event_dict.get("event_id")
        tfile = TICKETS_DIR / f"{tid}.json"
        try:
            with open(tfile, "w", encoding="utf-8") as f:
                json.dump(event_dict, f, indent=2)
        except Exception:
            pass

    def _seed_existing_data(self):
        """Seed SQLite spatial database from data/tickets/ or data/alerts/ if empty."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM events")
        count = cursor.fetchone()["count"]
        conn.close()

        if count == 0:
            # Check tickets
            ticket_files = list(TICKETS_DIR.glob("*.json"))
            if ticket_files:
                for tf in ticket_files:
                    try:
                        with open(tf, "r", encoding="utf-8") as f:
                            tdata = json.load(f)
                            loc = tdata.get("location", {})
                            alert_mock = {
                                "alert_id": tdata.get("ticket_id"),
                                "bus_id": tdata.get("reporting_bus_id", "BUS-KA-01-F-1204"),
                                "route_id": tdata.get("reporting_route_id", "ROUTE-335E"),
                                "defect_type": tdata.get("hazard_type", "pothole"),
                                "category": tdata.get("hazard_category", "Road Defect"),
                                "severity_level": tdata.get("severity", "High"),
                                "confidence_score": tdata.get("detection_confidence", 0.88),
                                "gps_telemetry": {
                                    "latitude": loc.get("latitude", 12.9767),
                                    "longitude": loc.get("longitude", 77.5713),
                                    "current_landmark": loc.get("current_landmark", "Central Station"),
                                },
                                "snapshot_filename": tdata.get("snapshot_filename"),
                            }
                            self.ingest_or_deduplicate(alert_mock)
                    except Exception as e:
                        print(f"[SPATIAL-DB] Seed error: {e}")

    def reset_database(self, reseed: bool = False):
        """
        Clears all events, deduplicated records, and spatial geometry from database.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events;")
        conn.commit()
        conn.close()
        print("[SPATIAL-DB] 🧹 All spatial events cleared from database.")
        if reseed:
            self._seed_existing_data()
        return {"status": "SUCCESS", "message": "Spatial events cleared successfully."}
