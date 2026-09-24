"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Synthetic Urban Driving Video Generator (generate_demo_video.py)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR

OUTPUT_VIDEO_PATH = DATA_DIR / "sample_driving_feed.mp4"


def generate_synthetic_urban_video(output_path: Path = OUTPUT_VIDEO_PATH, duration_sec: int = 20, fps: int = 30):
    """
    Generates a 720p driving simulation video complete with:
    - Moving road perspective and lane markings
    - Surrounding traffic vehicles (cars, buses, bikes)
    - Dynamic road hazards: Potholes, waterlogged puddles, cracked pavement
    - Traffic signals and urban buildings
    """
    print(f"[DEMO-GEN] Generating realistic {duration_sec}s urban driving video at {output_path}...")
    width, height = 1280, 720
    total_frames = duration_sec * fps

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    horizon_y = int(height * 0.45)
    lane_offset = 0.0

    # Moving vehicles state (rel_x, rel_y, speed, color, type)
    vehicles = [
        {"x": 420, "y": horizon_y + 40, "speed": 1.8, "scale": 0.35, "color": (180, 50, 40), "name": "CAR"},
        {"x": 780, "y": horizon_y + 70, "speed": 2.2, "scale": 0.45, "color": (40, 160, 220), "name": "CAR"},
        {"x": 280, "y": horizon_y + 110, "speed": 1.4, "scale": 0.55, "color": (30, 180, 80), "name": "BUS"},
    ]

    # Road anomalies (rel_frame_start, rel_frame_end, type, start_x, start_y)
    anomalies = [
        {"start_frame": 45, "end_frame": 120, "type": "pothole", "x": 680, "y": horizon_y + 50},
        {"start_frame": 160, "end_frame": 240, "type": "waterlogging", "x": 520, "y": horizon_y + 60},
        {"start_frame": 280, "end_frame": 360, "type": "pothole", "x": 460, "y": horizon_y + 40},
        {"start_frame": 400, "end_frame": 490, "type": "damaged_pavement", "x": 720, "y": horizon_y + 50},
    ]

    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Sky & Urban Horizon
        cv2.rectangle(frame, (0, 0), (width, horizon_y), (180, 150, 120), -1)  # Dusky urban sky

        # City Skyline Silhouette
        for b_x in range(0, width, 60):
            b_h = 40 + int(35 * np.sin(b_x * 0.05 + 2))
            cv2.rectangle(frame, (b_x, horizon_y - b_h), (b_x + 50, horizon_y), (100, 90, 85), -1)

        # 2. Road Asphalt Surface
        cv2.rectangle(frame, (0, horizon_y), (width, height), (75, 75, 80), -1)

        # Road perspective polygon
        road_pts = np.array([
            [width * 0.35, horizon_y],
            [width * 0.65, horizon_y],
            [width * 0.95, height],
            [width * 0.05, height]
        ], np.int32)
        cv2.fillPoly(frame, [road_pts], (70, 70, 75))

        # Road Curbs / Sidewalks
        cv2.line(frame, (int(width * 0.35), horizon_y), (int(width * 0.05), height), (130, 130, 140), 4)
        cv2.line(frame, (int(width * 0.65), horizon_y), (int(width * 0.95), height), (130, 130, 140), 4)

        # 3. Animated Lane Markings (Dashed Center and Lane Dividers)
        lane_offset = (lane_offset + 12) % 60
        for y_dash in range(horizon_y, height, 50):
            curr_y = y_dash + int(lane_offset)
            if curr_y >= height or curr_y < horizon_y:
                continue
            progress = (curr_y - horizon_y) / (height - horizon_y)
            dash_w = int(4 + progress * 10)
            dash_h = int(12 + progress * 24)

            # Center divider line
            cv2.rectangle(frame, (int(width * 0.50 - dash_w // 2), curr_y),
                          (int(width * 0.50 + dash_w // 2), curr_y + dash_h), (220, 220, 230), -1)

            # Left lane dash
            lx = int(width * (0.35 + 0.15 * progress) - dash_w // 2)
            cv2.rectangle(frame, (lx, curr_y), (lx + dash_w, curr_y + dash_h), (200, 200, 210), -1)

            # Right lane dash
            rx = int(width * (0.65 - 0.15 * progress) - dash_w // 2)
            cv2.rectangle(frame, (rx, curr_y), (rx + dash_w, curr_y + dash_h), (200, 200, 210), -1)

        # 4. Traffic Signal Post on Right Side
        signal_x = int(width * 0.88)
        signal_y = horizon_y - 80
        cv2.rectangle(frame, (signal_x, signal_y), (signal_x + 24, horizon_y), (60, 60, 60), -1)
        cv2.rectangle(frame, (signal_x - 10, signal_y - 65), (signal_x + 34, signal_y), (30, 30, 35), -1)
        # Green light active
        cv2.circle(frame, (signal_x + 12, signal_y - 50), 6, (40, 40, 40), -1)
        cv2.circle(frame, (signal_x + 12, signal_y - 32), 6, (40, 40, 40), -1)
        cv2.circle(frame, (signal_x + 12, signal_y - 14), 7, (0, 255, 120), -1)

        # 5. Render Moving Surrounding Vehicles
        for v in vehicles:
            v["y"] += v["speed"] * (1.0 + (v["y"] - horizon_y) / 100.0)
            if v["y"] > height + 50:
                v["y"] = horizon_y + 20
                v["x"] = int(width * (0.38 + 0.24 * np.random.rand()))

            prog = (v["y"] - horizon_y) / (height - horizon_y)
            car_w = int(60 + prog * 160)
            car_h = int(35 + prog * 95)
            cx = int(v["x"])
            cy = int(v["y"])

            # Car Body
            cv2.rectangle(frame, (cx - car_w // 2, cy - car_h), (cx + car_w // 2, cy), v["color"], -1)
            # Windshield / Cabin
            cv2.rectangle(frame, (cx - int(car_w * 0.35), cy - car_h + 4),
                          (cx + int(car_w * 0.35), cy - int(car_h * 0.45)), (50, 60, 70), -1)
            # Taillights
            cv2.circle(frame, (cx - int(car_w * 0.4), cy - 8), int(3 + prog * 5), (0, 0, 255), -1)
            cv2.circle(frame, (cx + int(car_w * 0.4), cy - 8), int(3 + prog * 5), (0, 0, 255), -1)

        # 6. Render Road Hazards & Defects (Potholes, Waterlogging)
        for anom in anomalies:
            if anom["start_frame"] <= frame_idx <= anom["end_frame"]:
                prog = (frame_idx - anom["start_frame"]) / (anom["end_frame"] - anom["start_frame"])
                curr_py = anom["y"] + int(prog * (height - horizon_y - 30))
                curr_px = anom["x"] + int((prog * 40))

                scale = 0.5 + prog * 1.5
                pw = int(45 * scale)
                ph = int(22 * scale)

                if anom["type"] == "pothole":
                    # Dark elliptical cavity with rough edge
                    cv2.ellipse(frame, (curr_px, curr_py), (pw, ph), 0, 0, 360, (20, 20, 25), -1)
                    cv2.ellipse(frame, (curr_px, curr_py), (int(pw * 0.8), int(ph * 0.7)), 0, 0, 360, (10, 10, 15), -1)
                    cv2.ellipse(frame, (curr_px, curr_py), (pw + 2, ph + 2), 0, 0, 360, (65, 65, 70), 2)
                elif anom["type"] == "waterlogging":
                    # Specular puddle reflection (Sky blue / high contrast)
                    cv2.ellipse(frame, (curr_px, curr_py), (int(pw * 1.4), int(ph * 1.1)), -15, 0, 360, (190, 160, 110), -1)
                    cv2.ellipse(frame, (curr_px, curr_py), (int(pw * 1.1), int(ph * 0.8)), -15, 0, 360, (220, 190, 140), -1)
                elif anom["type"] == "damaged_pavement":
                    # Cracked asphalt fissures
                    cv2.line(frame, (curr_px - pw, curr_py - ph), (curr_px + pw, curr_py + ph), (18, 18, 22), 3)
                    cv2.line(frame, (curr_px - int(pw*0.5), curr_py + ph), (curr_px + int(pw*0.5), curr_py - ph), (18, 18, 22), 2)

        out.write(frame)

    out.release()
    print(f"[DEMO-GEN] Synthetic driving video successfully created: {output_path} ({total_frames} frames)")
    return output_path


if __name__ == "__main__":
    generate_synthetic_urban_video()
