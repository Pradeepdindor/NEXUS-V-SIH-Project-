"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: GPS & Route Telemetry Simulator (gps_simulator.py)
"""

import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from config import SIMULATED_ROUTE_WAYPOINTS, DEFAULT_ROUTE_ID


@dataclass
class GPSReading:
    latitude: float
    longitude: float
    altitude_m: float
    speed_kmh: float
    heading_deg: float
    current_landmark: str
    next_landmark: str
    distance_traveled_km: float
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the compass bearing (heading in degrees 0-360) between two coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


class GPSSimulator:
    """
    Simulates high-precision GPS telemetry for a moving public transit bus along
    pre-configured urban waypoints, modeling acceleration, traffic stops, and turns.
    """

    def __init__(self, waypoints: Optional[List[Dict]] = None, route_id: str = DEFAULT_ROUTE_ID, speed_factor: float = 1.0):
        self.waypoints = waypoints or SIMULATED_ROUTE_WAYPOINTS
        if len(self.waypoints) < 2:
            raise ValueError("GPSSimulator requires at least 2 waypoints.")

        self.route_id = route_id
        self.speed_factor = speed_factor

        self.current_idx = 0
        self.next_idx = 1
        self.segment_progress = 0.0  # 0.0 to 1.0 along current waypoint segment

        self.lat = self.waypoints[0]["lat"]
        self.lon = self.waypoints[0]["lon"]
        self.speed_kmh = 24.0
        self.target_speed_kmh = self.waypoints[0].get("speed_limit", 40)
        self.heading_deg = calculate_bearing(
            self.waypoints[0]["lat"], self.waypoints[0]["lon"],
            self.waypoints[1]["lat"], self.waypoints[1]["lon"]
        )
        self.altitude_m = 920.0  # Bangalore elevation ~920m
        self.total_distance_traveled_m = 0.0
        self.last_update_time = time.time()
        self.stop_dwell_time_remaining = 0.0  # simulated seconds waiting at bus stop

    def update(self, dt: Optional[float] = None) -> GPSReading:
        """
        Update GPS position by advancing the bus along the route segment according to simulated speed.
        """
        now = time.time()
        if dt is None:
            dt = max(0.001, min(now - self.last_update_time, 0.5))
        self.last_update_time = now

        # If bus is stopped at a stop/traffic light
        if self.stop_dwell_time_remaining > 0:
            self.stop_dwell_time_remaining -= dt
            self.speed_kmh = max(0.0, self.speed_kmh - 15.0 * dt)
            return self._build_reading()

        # Current and next waypoints
        wp1 = self.waypoints[self.current_idx]
        wp2 = self.waypoints[self.next_idx]

        segment_distance_m = haversine_distance_meters(wp1["lat"], wp1["lon"], wp2["lat"], wp2["lon"])
        if segment_distance_m <= 0:
            segment_distance_m = 1.0

        # Adjust speed gradually towards target speed limit
        self.target_speed_kmh = wp1.get("speed_limit", 40)
        speed_delta = (self.target_speed_kmh - self.speed_kmh) * 0.8 * dt
        self.speed_kmh = max(10.0, min(65.0, self.speed_kmh + speed_delta))

        # Effective speed with speed_factor
        effective_speed_mps = (self.speed_kmh * self.speed_factor) * (1000.0 / 3600.0)
        distance_step_m = effective_speed_mps * dt
        self.total_distance_traveled_m += distance_step_m

        # Advance progress
        self.segment_progress += distance_step_m / segment_distance_m

        if self.segment_progress >= 1.0:
            # Reached next waypoint
            self.segment_progress = 0.0
            self.current_idx = self.next_idx
            self.next_idx = (self.next_idx + 1) % len(self.waypoints)

            # Reached a major bus stop - simulate a brief 2-second slowdown
            self.stop_dwell_time_remaining = 1.5

            wp1 = self.waypoints[self.current_idx]
            wp2 = self.waypoints[self.next_idx]

        # Interpolate Lat / Lon
        self.lat = wp1["lat"] + (wp2["lat"] - wp1["lat"]) * self.segment_progress
        self.lon = wp1["lon"] + (wp2["lon"] - wp1["lon"]) * self.segment_progress

        # Re-compute heading
        self.heading_deg = calculate_bearing(wp1["lat"], wp1["lon"], wp2["lat"], wp2["lon"])

        return self._build_reading()

    def _build_reading(self) -> GPSReading:
        wp1 = self.waypoints[self.current_idx]
        wp2 = self.waypoints[self.next_idx]
        return GPSReading(
            latitude=round(self.lat, 6),
            longitude=round(self.lon, 6),
            altitude_m=round(self.altitude_m, 1),
            speed_kmh=round(self.speed_kmh, 1),
            heading_deg=round(self.heading_deg, 1),
            current_landmark=wp1["name"],
            next_landmark=wp2["name"],
            distance_traveled_km=round(self.total_distance_traveled_m / 1000.0, 3),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def get_current_reading(self) -> GPSReading:
        return self._build_reading()
