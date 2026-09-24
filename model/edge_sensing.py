"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Main Edge Sensing & Real-Time AI Pipeline (edge_sensing.py)
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from config import (
    DEFAULT_BUS_ID,
    DEFAULT_ROUTE_ID,
    DEFAULT_MODEL_WEIGHTS,
    DEFECT_CONFIDENCE_THRESHOLD,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ALERTS_DIR,
    SNAPSHOTS_DIR,
    RECORDINGS_DIR,
)
try:
    from model.gps_simulator import GPSSimulator
    from model.detectors import EdgeDetector, DetectionBox
    from backend.alert_manager import AlertManager
    from model.hud_renderer import HUDRenderer
    from scripts.generate_demo_video import generate_synthetic_urban_video, OUTPUT_VIDEO_PATH
except ImportError:
    from gps_simulator import GPSSimulator
    from detectors import EdgeDetector, DetectionBox
    from alert_manager import AlertManager
    from hud_renderer import HUDRenderer
    from generate_demo_video import generate_synthetic_urban_video, OUTPUT_VIDEO_PATH


class EdgeSensingPipeline:
    """
    Main orchestration engine for Phase 1 of the SIH Mobile Urban Intelligence Platform.
    Streams video from live webcam or pre-recorded feeds, updates bus GPS route telemetry,
    runs YOLOv8 deep learning and computer vision defect analyzers, tracks vehicle density,
    renders real-time HUD overlays, and captures snapshot evidence + JSON alert payloads.
    """

    def __init__(
        self,
        source: str = "demo",
        weights: str = DEFAULT_MODEL_WEIGHTS,
        conf_threshold: float = DEFECT_CONFIDENCE_THRESHOLD,
        bus_id: str = DEFAULT_BUS_ID,
        route_id: str = DEFAULT_ROUTE_ID,
        camera_angle: str = "Front (Road Defect & Traffic)",
        headless: bool = False,
        save_video: bool = False,
        video_out_path: str = None,
        max_frames: int = -1,
        speed_factor: float = 1.0,
    ):
        self.source_arg = source
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.bus_id = bus_id
        self.route_id = route_id
        self.camera_angle = camera_angle
        self.headless = headless
        self.save_video = save_video
        self.max_frames = max_frames
        self.speed_factor = speed_factor

        # Initialize Modules
        print("=" * 75)
        print("  SMART INDIA HACKATHON 26124 | FLEET MOBILE URBAN INTELLIGENCE")
        print("  PHASE 1: REAL-TIME AI DETECTION & EDGE SENSING PIPELINE")
        print("=" * 75)
        print(f"  * Bus ID:            {self.bus_id}")
        print(f"  * Route ID:          {self.route_id}")
        print(f"  * Camera Feed Angle: {self.camera_angle}")
        print(f"  * Defect Conf Thresh: {self.conf_threshold * 100:.1f}%")
        print(f"  * Headless Mode:     {self.headless}")
        print(f"  * Video Source:      {self.source_arg}")
        print("=" * 75)

        self.gps = GPSSimulator(route_id=self.route_id, speed_factor=self.speed_factor)
        self.detector = EdgeDetector(
            model_weights=self.weights,
            conf_thresh=self.conf_threshold,
        )
        self.alert_manager = AlertManager(
            alerts_dir=ALERTS_DIR,
            snapshots_dir=SNAPSHOTS_DIR,
            bus_id=self.bus_id,
            route_id=self.route_id,
            conf_threshold=self.conf_threshold,
        )
        self.hud = HUDRenderer()

        # Video Capture Setup
        self.cap = None
        self._init_video_source()

        # Video Writer Setup
        self.writer = None
        if self.save_video:
            if not video_out_path:
                ts = time.strftime("%Y%m%d_%H%M%S")
                video_out_path = str(RECORDINGS_DIR / f"edge_feed_{self.bus_id}_{ts}.mp4")
            self.video_out_path = video_out_path
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.video_out_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))
            print(f"[EDGE] Recording output feed to: {self.video_out_path}")

        # Metrics & Telemetry
        self.frame_count = 0
        self.start_time = None
        self.fps = 30.0
        self.is_paused = False
        self.latest_alert_banner_text = None

    def _init_video_source(self):
        """Initialize webcam, wireless smartphone IP stream, video file, or synthetic demo feed."""
        source_str = str(self.source_arg).strip()

        if source_str.isdigit():
            # Live webcam or virtual smartphone camera (DroidCam, Iriun)
            cam_idx = int(source_str)
            print(f"[EDGE] Opening camera device index #{cam_idx}...")
            self.cap = cv2.VideoCapture(cam_idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        elif source_str.startswith(("http://", "https://", "rtsp://", "rtmp://")):
            # Wireless smartphone IP camera stream (e.g. Android IP Webcam app, DroidCam, RTSP stream)
            stream_url = source_str
            # Auto-normalize common Android IP Webcam or DroidCam endpoints (e.g., http://192.168.1.15:8080 -> /video)
            if stream_url.startswith(("http://", "https://")):
                from urllib.parse import urlparse
                parsed = urlparse(stream_url)
                if not parsed.path or parsed.path == "/":
                    if parsed.port in [8080, 4747, 8000, 8081]:
                        stream_url = stream_url.rstrip("/") + "/video"
            print(f"[EDGE] 📱 Connecting to Wireless Mobile IP Camera stream: {stream_url}...")
            self.cap = cv2.VideoCapture(stream_url)
            if not self.cap.isOpened() and not stream_url.endswith("/video") and stream_url.startswith("http"):
                fallback_url = stream_url.rstrip("/") + "/video"
                print(f"[EDGE] 🔄 Retrying connection with fallback MJPEG endpoint: {fallback_url}...")
                self.cap = cv2.VideoCapture(fallback_url)
        elif source_str in ["demo", "synthetic", "synth"]:
            # Synthetic urban driving simulation
            if not OUTPUT_VIDEO_PATH.exists():
                generate_synthetic_urban_video(OUTPUT_VIDEO_PATH)
            print(f"[EDGE] Streaming synthetic driving video: {OUTPUT_VIDEO_PATH}...")
            self.cap = cv2.VideoCapture(str(OUTPUT_VIDEO_PATH))
        else:
            # Video file path
            video_path = Path(source_str)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            print(f"[EDGE] Opening pre-recorded video: {video_path}...")
            self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {self.source_arg}\n"
                "Tip for Mobile IP Camera: If using 'IP Webcam' app on Android, use URL format: http://<PHONE_IP>:8080/video"
            )

    def run(self):
        """Main execution loop for real-time edge sensing."""
        self.start_time = time.time()
        prev_frame_time = time.time()
        window_name = f"SIH 26124 | Edge AI Sensing - Bus: {self.bus_id}"

        if not self.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1100, 620)

        print("[EDGE] Real-time edge pipeline active. Press 'q' to exit, 's' for manual alert, 'p' to pause.")

        try:
            while True:
                if not self.is_paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        # If video ended, loop if demo or break
                        if self.source_arg in ["demo", "synthetic", "synth"]:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                        if not ret:
                            print("[EDGE] Video feed stream ended.")
                            break

                    self.frame_count += 1
                    if self.max_frames > 0 and self.frame_count > self.max_frames:
                        print(f"[EDGE] Reached maximum requested frames ({self.max_frames}). Stopping.")
                        break

                    # Resize if needed to standard dimensions
                    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                    # 1. Update GPS Route Simulation
                    current_time = time.time()
                    dt = current_time - prev_frame_time
                    prev_frame_time = current_time
                    gps_reading = self.gps.update(dt=dt)

                    # 2. AI Multi-Modal Detection (YOLO + Road Defect Analyzer + Multi-Condition AI)
                    detections, traffic_state = self.detector.process_frame(frame, camera_angle=self.camera_angle)

                    # 3. Process Detections & Dispatch High-Confidence Alerts
                    for det in detections:
                        if det.is_defect_or_hazard:
                            alert = self.alert_manager.dispatch_alert(
                                frame=frame,
                                detection=det,
                                gps=gps_reading,
                                traffic=traffic_state,
                                all_detections=detections,
                                camera_angle=self.camera_angle,
                            )
                            if alert:
                                self.latest_alert_banner_text = (
                                    f"ALERT DISPATCHED: {alert.defect_type} ({int(alert.confidence_score*100)}%) | ID: {alert.alert_id}"
                                )

                    # 4. Smooth FPS calculation
                    instant_fps = 1.0 / max(0.001, dt)
                    self.fps = 0.9 * self.fps + 0.1 * instant_fps

                    # 4b. Sync Telemetry to FastAPI Server periodically (every ~30 frames)
                    if self.frame_count % 30 == 0:
                        self._sync_telemetry_to_backend(gps_reading, traffic_state)

                    # 5. Render High-Tech HUD Overlay
                    annotated_frame = self.hud.render(
                        frame=frame,
                        detections=detections,
                        gps=gps_reading,
                        traffic=traffic_state,
                        bus_id=self.bus_id,
                        route_id=self.route_id,
                        fps=self.fps,
                        total_alerts=self.alert_manager.total_alerts_dispatched,
                        latest_alert_text=self.latest_alert_banner_text
                    )

                    # 6. Save Video Frame if recording enabled
                    if self.writer is not None:
                        self.writer.write(annotated_frame)

                    # 7. Display in GUI Window if not headless
                    if not self.headless:
                        cv2.imshow(window_name, annotated_frame)
                    else:
                        if self.frame_count % 30 == 0:
                            bw = self.alert_manager.store_forward_queue.bandwidth_tracker.get_bandwidth_metrics()
                            print(
                                f"[FRAME {self.frame_count:04d}] "
                                f"FPS: {self.fps:.1f} | "
                                f"Angle: {self.camera_angle.split(' ')[0]} | "
                                f"GPS: ({gps_reading.latitude:.4f}, {gps_reading.longitude:.4f}) | "
                                f"Speed: {gps_reading.speed_kmh} km/h | "
                                f"Congestion: {traffic_state.congestion_level} ({traffic_state.total_vehicles} veh) | "
                                f"Alerts: {self.alert_manager.total_alerts_dispatched} | "
                                f"Bandwidth: {bw['cellular_rate_kb_per_hour']:.1f} KB/h ({bw['status']})"
                            )

                # Key controls
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]:  # 'q' or ESC
                        print("[EDGE] Quit key received. Exiting.")
                        break
                    elif key == ord('p'):  # 'p' pause
                        self.is_paused = not self.is_paused
                        print(f"[EDGE] {'Paused' if self.is_paused else 'Resumed'}.")
                    elif key == ord('s'):  # 's' manual alert snapshot
                        manual_det = DetectionBox(
                            bbox=(int(FRAME_WIDTH*0.4), int(FRAME_HEIGHT*0.6), int(FRAME_WIDTH*0.6), int(FRAME_HEIGHT*0.8)),
                            label="MANUAL_FLAGGED_HAZARD",
                            confidence=0.99,
                            category="Manual Inspection",
                            severity="High",
                            color_bgr=(0, 0, 255),
                            is_defect_or_hazard=True
                        )
                        self.alert_manager.dispatch_alert(
                            frame, manual_det, gps_reading, traffic_state, camera_angle=self.camera_angle
                        )

        except KeyboardInterrupt:
            print("\n[EDGE] Execution interrupted by user.")
        finally:
            self._cleanup(window_name)

    def _sync_telemetry_to_backend(self, gps, traffic):
        """Asynchronously send latest telemetry to FastAPI server and track bandwidth."""
        try:
            import requests
            import threading
            from config import SERVER_URL

            def _post():
                try:
                    payload = {
                        "bus_id": self.bus_id,
                        "route_id": self.route_id,
                        "camera_angle": self.camera_angle,
                        "gps_telemetry": gps.to_dict(),
                        "traffic_context": {
                            "total_vehicles": traffic.total_vehicles,
                            "congestion_level": traffic.congestion_level,
                            "congestion_score": traffic.congestion_score,
                            "breakdown": {
                                "cars": traffic.cars_count,
                                "buses": traffic.buses_count,
                                "trucks": traffic.trucks_count,
                                "two_wheelers": traffic.two_wheelers_count,
                            }
                        },
                        "fps": round(self.fps, 1),
                        "timestamp_utc": gps.timestamp,
                    }
                    payload_bytes = len(json.dumps(payload).encode("utf-8"))
                    self.alert_manager.store_forward_queue.bandwidth_tracker.record_bytes(
                        payload_bytes, payload_type="TELEMETRY_JSON"
                    )
                    requests.post(f"{SERVER_URL}/api/telemetry", json=payload, timeout=0.5)
                except Exception:
                    pass

            threading.Thread(target=_post, daemon=True).start()
        except Exception:
            pass

    def _cleanup(self, window_name: str):
        """Release resources and print final execution summary."""
        total_time = max(0.1, time.time() - (self.start_time or time.time()))
        avg_fps = self.frame_count / total_time
        bw_metrics = self.alert_manager.store_forward_queue.bandwidth_tracker.get_bandwidth_metrics()

        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if not self.headless:
            cv2.destroyAllWindows()

        print("\n" + "=" * 75)
        print("  EDGE SENSING PIPELINE EXECUTION SUMMARY")
        print("=" * 75)
        print(f"  * Total Frames Processed:    {self.frame_count}")
        print(f"  * Total Running Time:        {total_time:.2f} seconds")
        print(f"  * Average Pipeline FPS:      {avg_fps:.2f} FPS")
        print(f"  * Distance Monitored:        {self.gps.total_distance_traveled_m / 1000.0:.3f} km")
        print(f"  * Active Camera Angle:       {self.camera_angle}")
        print(f"  * Total Defect Alerts:       {self.alert_manager.total_alerts_dispatched}")
        print(f"  * Cellular Bandwidth Rate:   {bw_metrics['cellular_rate_kb_per_hour']:.2f} KB/h ({bw_metrics['cellular_rate_mb_per_hour']:.4f} MB/h)")
        print(f"  * Bandwidth Budget Status:   {bw_metrics['status']} (Budget: {bw_metrics['hourly_budget_mb']} MB/h)")
        print(f"  * Snapshots Saved To:        {SNAPSHOTS_DIR}")
        print(f"  * Alert JSONs Saved To:      {ALERTS_DIR}")
        print("=" * 75 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart India Hackathon 26124 - AI-Powered Mobile Urban Intelligence Edge Sensing Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="demo",
        help="Input source: '0' for laptop webcam, path to video file, or 'demo' for synthetic driving feed."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS,
        help=f"Path to YOLOv8 model weights (default: '{DEFAULT_MODEL_WEIGHTS}')."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFECT_CONFIDENCE_THRESHOLD,
        help=f"Defect confidence threshold for triggering alert snapshots and JSON (default: {DEFECT_CONFIDENCE_THRESHOLD})."
    )
    parser.add_argument(
        "--bus-id",
        type=str,
        default=DEFAULT_BUS_ID,
        help=f"Bus Identifier string (default: '{DEFAULT_BUS_ID}')."
    )
    parser.add_argument(
        "--route-id",
        type=str,
        default=DEFAULT_ROUTE_ID,
        help=f"Bus Route code (default: '{DEFAULT_ROUTE_ID}')."
    )
    parser.add_argument(
        "--camera-angle",
        type=str,
        default="Front (Road Defect & Traffic)",
        help="Camera feed perspective ('Front', 'Rear', 'Side', 'Cabin')."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (ideal for edge devices / servers)."
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video stream with HUD to data/recordings/."
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Custom output file path for recorded annotated video."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum frames to process before exiting (-1 for continuous)."
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="GPS simulation speed multiplier (default: 1.0)."
    )

    args = parser.parse_args()

    pipeline = EdgeSensingPipeline(
        source=args.source,
        weights_path=args.weights,
        conf_thresh=args.conf,
        bus_id=args.bus_id,
        route_id=args.route_id,
        camera_angle=args.camera_angle,
        headless=args.headless,
        save_video=args.save_video,
        output_video=args.output_video,
        max_frames=args.max_frames,
        speed_factor=args.speed_factor,
    )
    pipeline.run()

