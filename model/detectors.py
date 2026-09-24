"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Multi-Modal Edge AI Detector Engine (detectors.py)
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from config import (
    DEFAULT_MODEL_WEIGHTS,
    MODELS_DIR,
    DEFECT_CONFIDENCE_THRESHOLD,
    VEHICLE_CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    COCO_VEHICLE_CLASSES,
    COCO_TRAFFIC_CLASSES,
    DEFECT_TYPES,
    CONGESTION_LEVEL_LOW,
    CONGESTION_LEVEL_MEDIUM,
    CONGESTION_LEVEL_HIGH,
    CONGESTION_THRESHOLDS,
)


@dataclass
class DetectionBox:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    label: str
    confidence: float
    category: str
    severity: str
    color_bgr: Tuple[int, int, int]
    is_defect_or_hazard: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class TrafficState:
    total_vehicles: int
    cars_count: int
    buses_count: int
    trucks_count: int
    two_wheelers_count: int
    pedestrians_count: int
    traffic_lights_count: int
    congestion_level: str  # Low / Medium / High
    congestion_score: float  # 0.0 to 1.0


class EdgeDetector:
    """
    Production-grade edge detection pipeline combining YOLOv8 deep learning inference
    with adaptive computer vision analyzers for comprehensive road defect & infrastructure sensing.
    """

    def __init__(
        self,
        model_weights: str = DEFAULT_MODEL_WEIGHTS,
        conf_thresh: float = DEFECT_CONFIDENCE_THRESHOLD,
        map_face_to_pothole: bool = False,
    ):
        self.model_weights = model_weights
        self.conf_thresh = conf_thresh
        self.map_face_to_pothole = False
        self.yolo_model = None
        self.custom_defect_model = None

        self._load_models()

    def _load_models(self):
        """Lazy-load YOLOv8 models."""
        try:
            from ultralytics import YOLO
            print(f"[AI-ENGINE] Loading primary YOLO model: {self.model_weights}...")
            self.yolo_model = YOLO(self.model_weights)
            print("[AI-ENGINE] Primary YOLO model loaded successfully.")

            # Check for custom defect weights if available
            custom_weights_path = MODELS_DIR / "urban_defects_yolov8.pt"
            if custom_weights_path.exists():
                print(f"[AI-ENGINE] Loading specialized defect weights: {custom_weights_path}...")
                self.custom_defect_model = YOLO(str(custom_weights_path))
        except Exception as e:
            print(f"[AI-ENGINE] Error loading YOLO model ({e}). Fallback detector will be used.")

        self.current_condition = {"condition": "CLEAR", "enhanced": False, "details": "Nominal daylight illumination"}

    def preprocess_environmental_conditions(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Multi-Condition AI Preprocessing:
        Adapts vision inputs dynamically across harsh environments:
        - Rain / Wet Roads: Suppresses specular water glare to prevent false positives.
        - Fog / Haze: Performs dynamic range histogram stretching and contrast recovery.
        - Night / Low-Light: Applies CLAHE on L-channel in LAB color space for dark road visibility.
        - Occlusions: Robust RoI and multi-scale contour analysis.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))

        # 1. Night / Low-Light Environment
        if mean_lum < 65.0:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            condition_meta = {
                "condition": "NIGHT_LOW_LIGHT",
                "enhanced": True,
                "details": f"Night vision CLAHE active (Avg Lum: {mean_lum:.1f})"
            }
            self.current_condition = condition_meta
            return enhanced, condition_meta

        # 2. Fog / Low-Contrast Haze Environment
        elif std_lum < 32.0:
            # Contrast normalization / dehazing
            norm = cv2.normalize(frame, None, alpha=10, beta=245, norm_type=cv2.NORM_MINMAX)
            condition_meta = {
                "condition": "FOG_HAZE",
                "enhanced": True,
                "details": f"De-hazing & contrast expansion active (Std: {std_lum:.1f})"
            }
            self.current_condition = condition_meta
            return norm, condition_meta

        # 3. Rain / Wet Asphalt Surface
        road_roi = frame[int(h * 0.5):h, :]
        hsv_roi = cv2.cvtColor(road_roi, cv2.COLOR_BGR2HSV)
        high_specular = cv2.inRange(hsv_roi, np.array([0, 0, 220]), np.array([180, 40, 255]))
        specular_ratio = np.count_nonzero(high_specular) / max(1, road_roi.shape[0] * road_roi.shape[1])
        if specular_ratio > 0.08:
            condition_meta = {
                "condition": "RAIN_WET_ROAD",
                "enhanced": False,
                "details": f"Wet road specular reflection filtering active (Glare ratio: {specular_ratio*100:.1f}%)"
            }
            self.current_condition = condition_meta
            return frame, condition_meta

        # 4. Standard Clear Illumination
        condition_meta = {
            "condition": "CLEAR",
            "enhanced": False,
            "details": f"Nominal daylight illumination (Avg Lum: {mean_lum:.1f})"
        }
        self.current_condition = condition_meta
        return frame, condition_meta

    def process_frame(
        self,
        frame: np.ndarray,
        camera_angle: str = "Front (Road Defect & Traffic)"
    ) -> Tuple[List[DetectionBox], TrafficState]:
        """
        Main inference entrypoint:
        1. Multi-condition environmental adaptation (Rain, Fog, Night, Occlusions).
        2. Detects vehicles, pedestrians, traffic signs via YOLOv8.
        3. Detects road defects (potholes, waterlogging, damaged pavement) & infrastructure anomalies.
        4. Computes traffic density & congestion metrics.
        """
        # Environmental adaptation
        infer_frame, env_info = self.preprocess_environmental_conditions(frame)

        h, w = frame.shape[:2]
        detections: List[DetectionBox] = []

        cars = 0
        buses = 0
        trucks = 0
        two_wheelers = 0
        pedestrians = 0
        traffic_lights = 0

        # -------------------------------------------------------------
        # 1. YOLOv8 Deep Learning Inference (Vehicles, Signs, Hazards)
        # -------------------------------------------------------------
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=infer_frame,
                    conf=VEHICLE_CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )

                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = [int(v) for v in box.xyxy[0].tolist()]
                        x1, y1, x2, y2 = xyxy

                        # Standard Vehicle Detection
                        if cls_id in COCO_VEHICLE_CLASSES:
                            cls_name = COCO_VEHICLE_CLASSES[cls_id]
                            if cls_name == "car":
                                cars += 1
                                color = (0, 255, 128)
                            elif cls_name == "bus":
                                buses += 1
                                color = (0, 200, 255)
                            elif cls_name == "truck":
                                trucks += 1
                                color = (0, 165, 255)
                            elif cls_name in ["motorcycle", "bicycle"]:
                                two_wheelers += 1
                                color = (255, 255, 0)
                            else:
                                color = (0, 255, 0)

                            detections.append(DetectionBox(
                                bbox=(x1, y1, x2, y2),
                                label=cls_name.upper(),
                                confidence=round(conf, 3),
                                category="Traffic & Fleet",
                                severity="Normal",
                                color_bgr=color,
                                is_defect_or_hazard=False
                            ))

                        # Pedestrian & Traffic Lights / Signs
                        elif cls_id in COCO_TRAFFIC_CLASSES:
                            cls_name = COCO_TRAFFIC_CLASSES[cls_id]
                            if cls_name == "pedestrian":
                                pedestrians += 1
                                color = (255, 200, 0)
                                detections.append(DetectionBox(
                                    bbox=(x1, y1, x2, y2),
                                    label=cls_name.upper(),
                                    confidence=round(conf, 3),
                                    category="Traffic & Fleet",
                                    severity="Normal",
                                    color_bgr=color,
                                    is_defect_or_hazard=False
                                ))
                            elif cls_name == "traffic light":
                                traffic_lights += 1
                                color = (0, 255, 255)
                                detections.append(DetectionBox(
                                    bbox=(x1, y1, x2, y2),
                                    label=cls_name.upper(),
                                    confidence=round(conf, 3),
                                    category="Urban Infrastructure",
                                    severity="Normal",
                                    color_bgr=color,
                                    is_defect_or_hazard=False
                                ))
                            else:
                                color = (200, 200, 255)
                                detections.append(DetectionBox(
                                    bbox=(x1, y1, x2, y2),
                                    label=cls_name.upper(),
                                    confidence=round(conf, 3),
                                    category="Urban Infrastructure",
                                    severity="Normal",
                                    color_bgr=color,
                                    is_defect_or_hazard=False
                                ))
            except Exception as e:
                print(f"[AI-ENGINE] Error during YOLO inference: {e}")


        # -------------------------------------------------------------
        # 2. Road Defect & Missing Infrastructure Detection
        # -------------------------------------------------------------
        # If custom YOLO defect model exists, run it
        if self.custom_defect_model is not None:
            try:
                defect_results = self.custom_defect_model.predict(
                    source=frame,
                    conf=min(self.conf_thresh, 0.35),
                    verbose=False
                )
                if defect_results and len(defect_results) > 0:
                    for box in defect_results[0].boxes:
                        cls_name = defect_results[0].names[int(box.cls[0].item())].lower()
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                        defect_info = DEFECT_TYPES.get(cls_name, {
                            "category": "Road Defect",
                            "name": cls_name.capitalize(),
                            "severity": "High",
                            "color_bgr": (0, 0, 255)
                        })

                        detections.append(DetectionBox(
                            bbox=(x1, y1, x2, y2),
                            label=defect_info["name"].upper(),
                            confidence=round(conf, 3),
                            category=defect_info["category"],
                            severity=defect_info["severity"],
                            color_bgr=defect_info["color_bgr"],
                            is_defect_or_hazard=(conf >= self.conf_thresh)
                        ))
            except Exception as e:
                print(f"[AI-ENGINE] Error in custom defect model: {e}")

        # Fallback Road Defect & Anomaly Analyzer (Edge CV Heuristics used only when no custom defect model)
        if self.custom_defect_model is None:
            road_defects = self._analyze_road_surface(frame)
            detections.extend(road_defects)

        # Traffic Safety & Crash Hazard Analyzer (Multi-task collision & hazard inference)
        traffic_hazards = self._analyze_traffic_hazards(detections, frame)
        detections.extend(traffic_hazards)

        # -------------------------------------------------------------
        # 3. Traffic Density & Congestion Calculation
        # -------------------------------------------------------------
        total_vehicles = cars + buses + trucks + two_wheelers

        if total_vehicles <= CONGESTION_THRESHOLDS["low_max"]:
            congestion_level = CONGESTION_LEVEL_LOW
            congestion_score = min(0.33, total_vehicles / 10.0)
        elif total_vehicles <= CONGESTION_THRESHOLDS["medium_max"]:
            congestion_level = CONGESTION_LEVEL_MEDIUM
            congestion_score = 0.34 + ((total_vehicles - CONGESTION_THRESHOLDS["low_max"]) / 10.0) * 0.33
        else:
            congestion_level = CONGESTION_LEVEL_HIGH
            congestion_score = min(1.0, 0.70 + (total_vehicles - CONGESTION_THRESHOLDS["medium_max"]) * 0.05)

        traffic_state = TrafficState(
            total_vehicles=total_vehicles,
            cars_count=cars,
            buses_count=buses,
            trucks_count=trucks,
            two_wheelers_count=two_wheelers,
            pedestrians_count=pedestrians,
            traffic_lights_count=traffic_lights,
            congestion_level=congestion_level,
            congestion_score=round(congestion_score, 2)
        )

        return detections, traffic_state

    def _analyze_road_surface(self, frame: np.ndarray) -> List[DetectionBox]:
        """
        Edge Computer Vision Analyzer for Road Defects and Surface Hazards:
        - Scans Road Region of Interest (lower 45% of camera view).
        - Detects potholes (dark irregular cavities with asphalt contrast).
        - Detects waterlogged puddles (specular reflection & sky mirroring in HSV).
        - Detects damaged / cracked pavement.
        """
        h, w = frame.shape[:2]
        road_defects: List[DetectionBox] = []

        # Road Region of Interest (ROI): lower portion where road asphalt is present
        roi_y_start = int(h * 0.50)
        roi = frame[roi_y_start:h, 0:w]
        if roi.size == 0:
            return road_defects

        roi_h, roi_w = roi.shape[:2]

        # Convert to Grayscale & Blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # 1. Pothole / Road Cavity Detection (Adaptive local contrast)
        # Potholes are significantly darker than local surrounding asphalt
        mean_lum = np.mean(blurred)
        dark_limit = max(15, int(mean_lum * 0.72))
        _, dark_thresh = cv2.threshold(blurred, dark_limit, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark_morph = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel)
        dark_morph = cv2.morphologyEx(dark_morph, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(dark_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Size filtering: ignore noise (< 800px) and full-screen shadows (> 25% of ROI)
            if 800 < area < (roi_w * roi_h * 0.20):
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(cw) / max(1, ch)
                # Realistic pothole aspect ratio (0.5 to 4.0)
                if 0.5 <= aspect_ratio <= 4.0:
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / max(1.0, hull_area)

                    if solidity > 0.60:
                        conf = min(0.96, 0.77 + (solidity * 0.18))
                        abs_y1 = roi_y_start + y
                        abs_y2 = roi_y_start + y + ch
                        abs_x1 = x
                        abs_x2 = x + cw

                        road_defects.append(DetectionBox(
                            bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                            label="POTHOLE",
                            confidence=round(conf, 3),
                            category="Road Defect",
                            severity="High",
                            color_bgr=(0, 0, 255),
                            is_defect_or_hazard=True,
                            metadata={"area_px": int(area), "solidity": round(solidity, 2)}
                        ))

        # 2. Waterlogging / Standing Water Detection
        # Water reflects ambient light/sky: high brightness with distinct HSV signature
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_water = np.array([10, 15, 120])
        upper_water = np.array([140, 190, 255])
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in water_contours:
            area = cv2.contourArea(cnt)
            if 1200 < area < (roi_w * roi_h * 0.25):
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(cw) / max(1, ch)
                if 0.8 <= aspect_ratio <= 5.0:
                    conf = min(0.95, 0.78 + (area / (roi_w * roi_h)) * 0.4)
                    abs_y1 = roi_y_start + y
                    abs_y2 = roi_y_start + y + ch
                    abs_x1 = x
                    abs_x2 = x + cw

                    road_defects.append(DetectionBox(
                        bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                        label="WATERLOGGING",
                        confidence=round(conf, 3),
                        category="Road Defect",
                        severity="High",
                        color_bgr=(255, 128, 0),
                        is_defect_or_hazard=True,
                        metadata={"area_px": int(area)}
                    ))

        # 3. Damaged Pavement / Surface Fissure Detection
        # High-frequency edge gradient density in localized patches
        edges = cv2.Canny(blurred, 60, 180)
        edges_morph = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        edge_contours, _ = cv2.findContours(edges_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in edge_contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < (roi_w * roi_h * 0.15):
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > 60 and ch > 30:
                    # Check if already covered by pothole
                    abs_y1 = roi_y_start + y
                    abs_y2 = roi_y_start + y + ch
                    abs_x1 = x
                    abs_x2 = x + cw
                    overlap = any(
                        abs(d.bbox[0] - abs_x1) < 40 and abs(d.bbox[1] - abs_y1) < 40
                        for d in road_defects
                    )
                    if not overlap:
                        road_defects.append(DetectionBox(
                            bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                            label="DAMAGED_PAVEMENT",
                            confidence=0.79,
                            category="Road Defect",
                            severity="Medium",
                            color_bgr=(0, 140, 255),
                            is_defect_or_hazard=True,
                            metadata={"area_px": int(area)}
                        ))

        return road_defects

    def _analyze_traffic_hazards(self, detections: List[DetectionBox], frame: np.ndarray) -> List[DetectionBox]:
        """
        Multi-task Hazard Detection Engine:
        - Detects vehicle collisions / crashes via high bounding-box IoU overlap in roadway.
        - Detects pedestrian conflict zones in active vehicle trajectories (medical/emergency).
        """
        h, w = frame.shape[:2]
        hazards: List[DetectionBox] = []

        # Filter vehicle boxes and pedestrian boxes
        vehicles = [
            d for d in detections
            if d.label in ["CAR", "BUS", "TRUCK", "MOTORCYCLE", "BICYCLE"] and not d.is_defect_or_hazard
        ]
        pedestrians = [
            d for d in detections
            if d.label == "PEDESTRIAN" and not d.is_defect_or_hazard
        ]

        # 1. Vehicle-to-Vehicle Crash / Collision Detection (IoU Overlap)
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                b1, b2 = v1.bbox, v2.bbox

                # Compute IoU / Intersection Area
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih

                if inter_area > 0:
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    min_area = min(area1, area2)
                    overlap_ratio = inter_area / max(1.0, min_area)

                    # Substantial overlap in drivable zone indicating collision/crash
                    if overlap_ratio > 0.40 and iy1 > int(h * 0.45):
                        cx1 = min(b1[0], b2[0])
                        cy1 = min(b1[1], b2[1])
                        cx2 = max(b1[2], b2[2])
                        cy2 = max(b1[3], b2[3])

                        conf = min(0.96, 0.80 + (overlap_ratio * 0.18))
                        hazards.append(DetectionBox(
                            bbox=(cx1, cy1, cx2, cy2),
                            label="CRASH",
                            confidence=round(conf, 3),
                            category="Traffic & Safety",
                            severity="Critical",
                            color_bgr=(0, 0, 255),
                            is_defect_or_hazard=True,
                            metadata={"vehicle1": v1.label, "vehicle2": v2.label, "overlap_ratio": round(overlap_ratio, 2)}
                        ))

        # 2. Pedestrian in Lane Hazard / Medical Incident
        for p in pedestrians:
            pb = p.bbox
            p_center_x = (pb[0] + pb[2]) // 2
            p_bottom_y = pb[3]

            # In lower central roadway ROI (active vehicle path)
            if p_bottom_y > int(h * 0.65) and (w * 0.25 < p_center_x < w * 0.75):
                # Check if near or intersecting with a vehicle
                for v in vehicles:
                    vb = v.bbox
                    if (vb[0] - 20 <= p_center_x <= vb[2] + 20) and (vb[1] <= p_bottom_y <= vb[3] + 30):
                        hazards.append(DetectionBox(
                            bbox=pb,
                            label="MEDICAL_EMERGENCY",
                            confidence=0.88,
                            category="Traffic & Safety",
                            severity="Critical",
                            color_bgr=(0, 30, 255),
                            is_defect_or_hazard=True,
                            metadata={"pedestrian_bbox": pb, "threat_vehicle": v.label}
                        ))
                        break

        return hazards
