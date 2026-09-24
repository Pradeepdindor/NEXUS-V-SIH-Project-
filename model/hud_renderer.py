"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Real-Time Glassmorphic HUD Renderer (hud_renderer.py)
"""

import cv2
import numpy as np
import time
from typing import List, Optional, Tuple
try:
    from model.gps_simulator import GPSReading
    from model.detectors import DetectionBox, TrafficState
except ImportError:
    from gps_simulator import GPSReading
    from detectors import DetectionBox, TrafficState
from config import CONGESTION_LEVEL_LOW, CONGESTION_LEVEL_MEDIUM, CONGESTION_LEVEL_HIGH


class HUDRenderer:
    """
    Renders high-definition graphical Heads-Up Display (HUD) overlays on video streams,
    providing telemetry, detection bounding boxes, congestion stats, and live hazard alerts.
    """

    def __init__(self):
        # Color Palettes (BGR)
        self.COLOR_BG_DARK = (15, 18, 24)
        self.COLOR_ACCENT_CYAN = (230, 216, 0)
        self.COLOR_ACCENT_GREEN = (80, 220, 50)
        self.COLOR_ACCENT_YELLOW = (0, 215, 255)
        self.COLOR_ACCENT_RED = (40, 40, 245)
        self.COLOR_TEXT_WHITE = (245, 245, 245)
        self.COLOR_TEXT_DIM = (160, 160, 160)

        # Pulse animation timer for warning banners
        self.last_flash_time = time.time()
        self.flash_state = True

    def render(
        self,
        frame: np.ndarray,
        detections: List[DetectionBox],
        gps: GPSReading,
        traffic: TrafficState,
        bus_id: str,
        route_id: str,
        fps: float,
        total_alerts: int,
        latest_alert_text: Optional[str] = None,
    ) -> np.ndarray:
        """Render complete HUD overlay on frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        # Update flash state for pulsing alert banner
        now = time.time()
        if now - self.last_flash_time > 0.4:
            self.flash_state = not self.flash_state
            self.last_flash_time = now

        # 1. Render Detection Bounding Boxes
        self._render_bounding_boxes(output, detections)

        # 2. Render Top Telemetry Status Bar
        self._render_top_bar(output, gps, bus_id, route_id, fps, w)

        # 3. Render Left Telemetry & Congestion Dashboard
        self._render_left_dashboard(output, traffic, total_alerts, gps)

        # 4. Render Active Hazard Alert Banner (if defect on screen or recent alert)
        self._render_hazard_banner(output, detections, latest_alert_text, w, h)

        # 5. Render Bottom Mini Route Strip
        self._render_bottom_strip(output, gps, w, h)

        return output

    def _render_bounding_boxes(self, frame: np.ndarray, detections: List[DetectionBox]):
        """Render stylized bounding boxes and tags for detected objects and hazards."""
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = d.color_bgr
            is_defect = d.is_defect_or_hazard

            # Main bounding box
            thickness = 3 if is_defect else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Corner accents
            corner_len = min(18, max(5, int((x2 - x1) * 0.2)))
            # Top-Left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), thickness + 1)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), thickness + 1)
            # Top-Right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), thickness + 1)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), thickness + 1)
            # Bottom-Left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), thickness + 1)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), thickness + 1)
            # Bottom-Right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), thickness + 1)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), thickness + 1)

            # Label Header Tag
            label_text = f"{d.label} {int(d.confidence * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50 if is_defect else 0.45
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            tag_x1 = x1
            tag_x2 = min(frame.shape[1], x1 + tw + 12)

            # Tag background
            cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
            # Tag text
            text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255)
            cv2.putText(
                frame, label_text, (tag_x1 + 6, tag_y2 - 5),
                font, font_scale, text_color, 1, cv2.LINE_AA
            )

    def _render_top_bar(self, frame: np.ndarray, gps: GPSReading, bus_id: str, route_id: str, fps: float, w: int):
        """Render top glassmorphic telemetry header."""
        bar_h = 44
        # Glassmorphism dark background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border separator
        cv2.line(frame, (0, bar_h), (w, bar_h), (50, 60, 75), 1)

        # Title / Brand
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "SIH-26124 | FLEET EDGE-AI", (14, 28), font, 0.55, self.COLOR_ACCENT_CYAN, 2, cv2.LINE_AA)

        # Bus & Route ID
        bus_str = f"BUS: {bus_id} [{route_id}]"
        cv2.putText(frame, bus_str, (270, 28), font, 0.48, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # GPS Coordinates
        gps_str = f"GPS: {gps.latitude:.4f} N, {gps.longitude:.4f} E"
        cv2.putText(frame, gps_str, (540, 28), font, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        # Speed
        speed_color = self.COLOR_ACCENT_GREEN if gps.speed_kmh < 45 else self.COLOR_ACCENT_YELLOW
        speed_str = f"SPD: {int(gps.speed_kmh)} km/h"
        cv2.putText(frame, speed_str, (820, 28), font, 0.50, speed_color, 2, cv2.LINE_AA)

        # FPS & System Time
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_str, (w - 110, 28), font, 0.48, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    def _render_left_dashboard(self, frame: np.ndarray, traffic: TrafficState, total_alerts: int, gps: GPSReading):
        """Render left telemetry panel with vehicle counts, congestion level, and alerts tally."""
        x1, y1 = 12, 54
        card_w, card_h = 240, 215

        # Glassmorphic Card Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + card_w, y1 + card_h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Card Border
        cv2.rectangle(frame, (x1, y1), (x1 + card_w, y1 + card_h), (60, 70, 85), 1)

        # Header Title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "URBAN EDGE SENSING", (x1 + 10, y1 + 22), font, 0.44, self.COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 10, y1 + 28), (x1 + card_w - 10, y1 + 28), (50, 60, 75), 1)

        # Traffic Congestion Gauge
        cv2.putText(frame, "Congestion Level:", (x1 + 10, y1 + 48), font, 0.42, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        if traffic.congestion_level == CONGESTION_LEVEL_LOW:
            cg_color = self.COLOR_ACCENT_GREEN
        elif traffic.congestion_level == CONGESTION_LEVEL_MEDIUM:
            cg_color = self.COLOR_ACCENT_YELLOW
        else:
            cg_color = self.COLOR_ACCENT_RED

        cv2.putText(frame, traffic.congestion_level.upper(), (x1 + 140, y1 + 48), font, 0.48, cg_color, 2, cv2.LINE_AA)

        # Congestion Bar Meter
        bar_x, bar_y, bar_w, bar_h = x1 + 10, y1 + 56, card_w - 20, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 40, 50), -1)
        filled_w = int(bar_w * max(0.05, min(1.0, traffic.congestion_score)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), cg_color, -1)

        # Vehicle Breakdown Counters
        row_y = y1 + 86
        cv2.putText(frame, f"Cars: {traffic.cars_count}", (x1 + 10, row_y), font, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Buses: {traffic.buses_count}", (x1 + 125, row_y), font, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        row_y += 22
        cv2.putText(frame, f"Trucks: {traffic.trucks_count}", (x1 + 10, row_y), font, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"2-Wheelers: {traffic.two_wheelers_count}", (x1 + 125, row_y), font, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        row_y += 22
        cv2.putText(frame, f"Pedestrians: {traffic.pedestrians_count}", (x1 + 10, row_y), font, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Signals: {traffic.traffic_lights_count}", (x1 + 125, row_y), font, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # Total Alerts Logged
        cv2.line(frame, (x1 + 10, row_y + 12), (x1 + card_w - 10, row_y + 12), (50, 60, 75), 1)
        row_y += 32
        cv2.putText(frame, f"ALERTS LOGGED: {total_alerts}", (x1 + 10, row_y), font, 0.46, (0, 165, 255), 2, cv2.LINE_AA)

    def _render_hazard_banner(
        self,
        frame: np.ndarray,
        detections: List[DetectionBox],
        latest_alert_text: Optional[str],
        w: int,
        h: int,
    ):
        """Render pulsing alert banner when a defect or road hazard is actively detected."""
        active_defects = [d for d in detections if d.is_defect_or_hazard]

        if not active_defects and not latest_alert_text:
            return

        banner_w = min(800, w - 40)
        banner_h = 42
        bx1 = (w - banner_w) // 2
        by1 = h - 75
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        # Determine banner text and color
        if active_defects:
            top_defect = max(active_defects, key=lambda d: d.confidence)
            alert_msg = f"[!] HAZARD DETECTED: {top_defect.label} ({int(top_defect.confidence*100)}%) - CAPTURING TELEMETRY"
            bg_color = (0, 0, 180) if self.flash_state else (20, 20, 120)
        else:
            alert_msg = f"[SYNC] {latest_alert_text}"
            bg_color = (15, 60, 30)

        # Draw Banner
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(alert_msg, font, 0.50, 2)
        tx = bx1 + (banner_w - tw) // 2
        ty = by1 + (banner_h + th) // 2
        cv2.putText(frame, alert_msg, (tx, ty), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    def _render_bottom_strip(self, frame: np.ndarray, gps: GPSReading, w: int, h: int):
        """Render bottom status bar with route landmark progression."""
        strip_h = 24
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - strip_h), (w, h), (10, 12, 16), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        route_text = (
            f"TRANSIT WAYPOINT: {gps.current_landmark}  -->  NEXT: {gps.next_landmark}  |  "
            f"Distance: {gps.distance_traveled_km:.2f} km"
        )
        cv2.putText(frame, route_text, (15, h - 7), font, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
