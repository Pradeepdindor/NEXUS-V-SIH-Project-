"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Municipal Repair Auto-Ticketing Engine (ticket_engine.py)
"""

import os
import json
import random
import string
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from pathlib import Path

from config import (
    TICKETS_DIR,
    SNAPSHOTS_DIR,
    DEFECT_TYPES,
    DEPARTMENT_PWD,
    DEPARTMENT_DRAINAGE,
    DEPARTMENT_TRAFFIC_ENG,
    DEPARTMENT_TRAFFIC_POLICE,
    DEPARTMENT_EMERGENCY,
    SIMULATED_ROUTE_WAYPOINTS,
)


@dataclass
class MunicipalTicket:
    ticket_id: str  # e.g., PWD-2026-X89B
    tracking_code: str
    created_at_utc: str
    sla_deadline_utc: str
    sla_hours: int
    hazard_type: str
    hazard_category: str
    severity: str  # Critical / High / Medium / Low
    priority_level: str  # P1 - Immediate, P2 - Urgent, P3 - Scheduled
    assigned_department: str
    ward_zone: str
    status: str  # PENDING_DISPATCH, DISPATCHED_TO_CONTRACTOR, IN_PROGRESS, RESOLVED, CLOSED
    action_required: str
    estimated_repair_cost_inr: int
    location: Dict
    google_maps_url: str
    snapshot_filename: str
    snapshot_url: str
    detection_confidence: float
    reporting_bus_id: str
    reporting_route_id: str
    contractor_assigned: Optional[str] = "Govt Urban Works Empranelled Contractor A"
    resolution_notes: Optional[str] = None
    resolved_at_utc: Optional[str] = None
    audit_trail: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class AutoTicketingEngine:
    """
    Automated Municipal Repair Ticket Generator:
    Converts raw AI-detected road hazards and infrastructure alerts into official,
    actionable government repair work orders (PWD/BBMP/Traffic Police) with SLA tracking.
    """

    def __init__(self, tickets_dir: Path = TICKETS_DIR):
        self.tickets_dir = Path(tickets_dir)
        self.tickets_dir.mkdir(parents=True, exist_ok=True)
        self._seed_existing_tickets()

    def generate_ticket(self, alert_data: Dict) -> MunicipalTicket:
        """
        Creates a new municipal repair ticket from an alert payload.
        """
        defect_type_key = alert_data.get("defect_type", "pothole").lower()
        defect_config = DEFECT_TYPES.get(defect_type_key, DEFECT_TYPES.get("pothole"))

        # 1. Department Prefix & Unique Tracking ID (e.g. PWD-2026-X89B)
        dept_prefix = "PWD"
        if "Police" in defect_config["department"]:
            dept_prefix = "POLICE"
        elif "Ambulance" in defect_config["department"] or "EMS" in defect_config["department"] or "Emergency" in defect_config["department"]:
            dept_prefix = "EMS-108"
        elif "Municipal" in defect_config["department"] or "BBMP" in defect_config["department"] or "Drainage" in defect_config["department"]:
            dept_prefix = "BBMP"

        year_str = datetime.now(timezone.utc).strftime("%Y")
        random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        ticket_id = f"{dept_prefix}-{year_str}-{random_suffix}"

        # 2. SLA Deadlines & Priority
        severity = alert_data.get("severity_level", defect_config["severity"])
        sla_hours = defect_config.get("sla_hours", 48)

        if severity == "Critical":
            priority = "P1 - IMMEDIATE INTERVENTION"
            sla_hours = 2
        elif severity == "High":
            priority = "P2 - HIGH PRIORITY (URGENT)"
            sla_hours = min(sla_hours, 48)
        elif severity == "Medium":
            priority = "P3 - SCHEDULED MUNICIPAL WORK"
        else:
            priority = "P4 - ROUTINE REPAIR"

        created_dt = datetime.now(timezone.utc)
        sla_deadline_dt = created_dt + timedelta(hours=sla_hours)

        # 3. Location & Ward Resolution
        gps = alert_data.get("gps_telemetry", {})
        lat = gps.get("latitude", 12.9767)
        lon = gps.get("longitude", 77.5713)

        # Find closest simulated ward
        ward_zone = self._resolve_ward(lat, lon, gps.get("current_landmark", "Central Zone"))
        maps_url = f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"

        # 4. Snapshot Linking
        snapshot_filename = alert_data.get("snapshot_filename", "")
        snapshot_url = f"/snapshots/{snapshot_filename}" if snapshot_filename else ""

        # 5. Build Audit Trail
        audit_trail = [
            {
                "timestamp": created_dt.isoformat(),
                "action": "AUTO_TICKET_GENERATED",
                "actor": "AI-EDGE-SENSING-ENGINE",
                "details": f"Detected {defect_config['name']} with {alert_data.get('confidence_score', 0.85)*100:.1f}% AI confidence.",
            },
            {
                "timestamp": (created_dt + timedelta(seconds=2)).isoformat(),
                "action": "MUNICIPAL_DISPATCHED",
                "actor": "CENTRAL-COMMAND-DISPATCHER",
                "details": f"Work order routed to {defect_config['department']}. Target SLA: {sla_hours}h.",
            }
        ]

        ticket = MunicipalTicket(
            ticket_id=ticket_id,
            tracking_code=f"TRK-{random.randint(100000, 999999)}",
            created_at_utc=created_dt.isoformat(),
            sla_deadline_utc=sla_deadline_dt.isoformat(),
            sla_hours=sla_hours,
            hazard_type=defect_config["name"],
            hazard_category=alert_data.get("category", defect_config["category"]),
            severity=severity,
            priority_level=priority,
            assigned_department=defect_config["department"],
            ward_zone=ward_zone,
            status="DISPATCHED_TO_CONTRACTOR",
            action_required=defect_config["action_required"],
            estimated_repair_cost_inr=defect_config["estimated_cost_inr"],
            location={
                "latitude": lat,
                "longitude": lon,
                "altitude_m": gps.get("altitude_m", 920.0),
                "speed_kmh": gps.get("speed_kmh", 30.0),
                "current_landmark": gps.get("current_landmark", "Urban Corridor"),
                "next_landmark": gps.get("next_landmark", "Next Checkpoint"),
            },
            google_maps_url=maps_url,
            snapshot_filename=snapshot_filename,
            snapshot_url=snapshot_url,
            detection_confidence=alert_data.get("confidence_score", 0.85),
            reporting_bus_id=alert_data.get("bus_id", "BUS-KA-01-F-1204"),
            reporting_route_id=alert_data.get("route_id", "ROUTE-335E"),
            contractor_assigned="Govt Urban Works Empanelled Contractor A",
            audit_trail=audit_trail,
        )

        # Save to disk
        ticket_file = self.tickets_dir / f"{ticket_id}.json"
        with open(ticket_file, "w", encoding="utf-8") as f:
            f.write(ticket.to_json(indent=2))

        print(f"[TICKET-ENGINE] Generated Official Municipal Ticket: {ticket_id} ({ticket.hazard_type}) -> {ticket.assigned_department}")
        return ticket

    def _resolve_ward(self, lat: float, lon: float, landmark: str) -> str:
        for wp in SIMULATED_ROUTE_WAYPOINTS:
            if wp["name"].lower() in landmark.lower() or landmark.lower() in wp["name"].lower():
                return wp.get("ward", "Ward 112 - Domlur Core")
        return "Ward 112 - Central Urban Zone"

    def get_all_tickets(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        department: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch and filter all municipal tickets."""
        tickets = []
        for file in sorted(self.tickets_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if status and data.get("status") != status:
                        continue
                    if severity and data.get("severity") != severity:
                        continue
                    if department and department.lower() not in data.get("assigned_department", "").lower():
                        continue
                    tickets.append(data)
            except Exception as e:
                print(f"[TICKET-ENGINE] Error reading {file}: {e}")
        return tickets

    def get_ticket_by_id(self, ticket_id: str) -> Optional[Dict]:
        ticket_file = self.tickets_dir / f"{ticket_id}.json"
        if ticket_file.exists():
            with open(ticket_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def update_ticket_status(
        self,
        ticket_id: str,
        status: str,
        contractor: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> Optional[Dict]:
        ticket = self.get_ticket_by_id(ticket_id)
        if not ticket:
            return None

        ticket["status"] = status
        if contractor:
            ticket["contractor_assigned"] = contractor
        if resolution_notes:
            ticket["resolution_notes"] = resolution_notes
        if status in ["RESOLVED", "CLOSED"]:
            ticket["resolved_at_utc"] = datetime.now(timezone.utc).isoformat()

        ticket.setdefault("audit_trail", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": f"STATUS_UPDATED_{status}",
            "actor": "MUNICIPAL_OFFICER",
            "details": resolution_notes or f"Ticket status changed to {status}",
        })

        ticket_file = self.tickets_dir / f"{ticket_id}.json"
        with open(ticket_file, "w", encoding="utf-8") as f:
            json.dump(ticket, f, indent=2)

        return ticket

    def _seed_existing_tickets(self):
        """Seed tickets from existing alert files in data/alerts/ if ticket dir is empty."""
        existing_tickets = list(self.tickets_dir.glob("*.json"))
        if not existing_tickets:
            alerts_dir = self.tickets_dir.parent / "alerts"
            if alerts_dir.exists():
                for alert_file in sorted(alerts_dir.glob("*.json")):
                    try:
                        with open(alert_file, "r", encoding="utf-8") as f:
                            alert_data = json.load(f)
                            self.generate_ticket(alert_data)
                    except Exception as e:
                        print(f"[TICKET-ENGINE] Error seeding from {alert_file}: {e}")

    def reset_tickets(self):
        """
        Clears all generated municipal work-order tickets from disk storage.
        """
        deleted_count = 0
        for tf in self.tickets_dir.glob("*.json"):
            try:
                tf.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"[TICKET-ENGINE] Error deleting ticket {tf}: {e}")
        print(f"[TICKET-ENGINE] 🧹 Cleared {deleted_count} municipal tickets from storage.")
        return {"status": "SUCCESS", "deleted_tickets": deleted_count}
