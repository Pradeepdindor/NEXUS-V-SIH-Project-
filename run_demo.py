"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Master Demo Automation & Orchestration Launcher (run_demo.py)
"""

import os
import sys
import time
import signal
import shutil
import subprocess
import webbrowser
from pathlib import Path


def find_python_executable():
    """
    Intelligently find the Python executable on Windows that has all required
    packages (cv2, fastapi, ultralytics, streamlit) installed.
    """
    candidates = [
        r"D:\anaconda\python.exe",
        r"D:\anaconda3\python.exe",
        r"C:\Users\gauta\anaconda3\python.exe",
        r"C:\ProgramData\anaconda3\python.exe",
        sys.executable,
        shutil.which("python"),
        shutil.which("python3"),
    ]

    for cand in candidates:
        if cand and os.path.exists(cand):
            try:
                res = subprocess.run(
                    [cand, "-c", "import cv2, fastapi, streamlit, ultralytics; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "OK" in res.stdout:
                    return cand
            except Exception:
                continue

    return sys.executable


PYTHON_EXE = find_python_executable()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "data"
SAMPLE_VIDEO = DATA_DIR / "sample_driving_feed.mp4"

# Path definitions for modular components
FLASK_SCRIPT = BASE_DIR / "backend" / "flask_app.py"
FASTAPI_SCRIPT = BASE_DIR / "backend" / "server.py"
STREAMLIT_SCRIPT = BASE_DIR / "frontend" / "app.py"
EDGE_SCRIPT = BASE_DIR / "model" / "edge_sensing.py"
GEN_VIDEO_SCRIPT = BASE_DIR / "scripts" / "generate_demo_video.py"

processes = []


def print_banner():
    print("\n" + "=" * 80)
    print("  🚀 SMART INDIA HACKATHON (SIH) - PROBLEM STATEMENT 26124 / 24124")
    print("  AI-POWERED MOBILE URBAN INTELLIGENCE PLATFORM (TEAM: DIGITAL NOMADS)")
    print("  Master Prototype Orchestration Suite")
    print("=" * 80)
    print(f"  [RUNTIME] Using Python: {PYTHON_EXE}")
    print(f"  [ROOT]    Workspace:   {BASE_DIR}")
    print("=" * 80)
    print("  [1] 🌟 FULL DEMO (Synthetic Urban Driving Feed + Flask REST API + Leaflet GIS + Streamlit)")
    print("  [2] 📹 LIVE LAPTOP WEBCAM (Point Camera at Road / Dashcam Video on Phone/Screen)")
    print("  [3] 📱 WIRELESS MOBILE IP CAMERA / RTSP (Stream Smartphone via 'IP Webcam' / DroidCam)")
    print("  [4] 📂 PRE-RECORDED VIDEO DEMO (Select Custom Dashcam Video File .mp4/.avi)")
    print("  [5] 🗺️ FLASK REST API & LEAFLET GIS MAP ONLY (Port 5000)")
    print("  [6] 🖥️ FASTAPI & STREAMLIT BACKEND ONLY (Port 8000 & 8501)")
    print("  [7] 🗑️ RESET ALL DETECTED DATA (Clear PostGIS database, tickets & spatial records)")
    print("  [8] ❌ EXIT")
    print("=" * 80 + "\n")


def cleanup(sig=None, frame=None):
    """Gracefully terminate all spawned child processes."""
    print("\n[ORCHESTRATOR] Shutting down SIH Urban Intelligence Platform services...")
    for proc, name in processes:
        try:
            print(f"[ORCHESTRATOR] Stopping {name} (PID: {proc.pid})...")
            proc.terminate()
            time.sleep(0.3)
            if proc.poll() is None:
                proc.kill()
        except Exception as e:
            print(f"[ORCHESTRATOR] Error stopping {name}: {e}")

    print("[ORCHESTRATOR] All services stopped cleanly. Thank you for evaluating SIH 26124!\n")
    sys.exit(0)


def check_and_generate_demo_video():
    """Ensure sample synthetic driving video exists before launching."""
    if not SAMPLE_VIDEO.exists():
        print("[ORCHESTRATOR] Generating synthetic 720p urban driving simulation video...")
        cmd = [PYTHON_EXE, str(GEN_VIDEO_SCRIPT)]
        subprocess.run(cmd, check=True)


def start_flask_server():
    """Launch Flask REST API & Leaflet GIS Server."""
    print("[ORCHESTRATOR] Starting Flask REST API & Leaflet GIS Server (Port 5000)...")
    flask_cmd = [PYTHON_EXE, str(FLASK_SCRIPT)]
    proc = subprocess.Popen(flask_cmd, cwd=str(BASE_DIR))
    processes.append((proc, "Flask REST API & Leaflet GIS Server"))
    time.sleep(1.5)


def start_server():
    """Launch FastAPI Backend Server."""
    print("[ORCHESTRATOR] Starting FastAPI Central Command Backend (Port 8000)...")
    server_cmd = [PYTHON_EXE, str(FASTAPI_SCRIPT)]
    proc = subprocess.Popen(server_cmd, cwd=str(BASE_DIR))
    processes.append((proc, "FastAPI Backend Server"))
    time.sleep(1.8)  # Wait for server startup


def start_dashboard():
    """Launch Streamlit Demonstration Dashboard."""
    print("[ORCHESTRATOR] Starting Streamlit Judge Dashboard (Port 8501)...")
    dash_cmd = [
        PYTHON_EXE, "-m", "streamlit", "run",
        str(STREAMLIT_SCRIPT),
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    proc = subprocess.Popen(dash_cmd, cwd=str(BASE_DIR))
    processes.append((proc, "Streamlit Judge Dashboard"))
    time.sleep(2.0)  # Wait for dashboard startup


def start_edge_sensing(source="demo", conf=0.75):
    """Launch Edge Sensing AI Pipeline."""
    print(f"[ORCHESTRATOR] Starting Edge AI Sensing Pipeline (Source: {source}, Conf: {conf*100:.0f}%)...")
    edge_cmd = [
        PYTHON_EXE, str(EDGE_SCRIPT),
        "--source", str(source),
        "--conf", str(conf),
        "--save-video",
    ]
    proc = subprocess.Popen(edge_cmd, cwd=str(BASE_DIR))
    processes.append((proc, "Edge Sensing AI Pipeline"))


def perform_data_reset():
    """Directly reset PostGIS spatial events, SQLite database, and municipal tickets."""
    print("\n[ORCHESTRATOR] 🧹 Resetting detected data, PostGIS events, and municipal tickets...")
    try:
        from backend.spatial_db import SpatialEventDatabase
        from backend.ticket_engine import AutoTicketingEngine
        from config import TICKETS_DIR

        sdb = SpatialEventDatabase()
        sdb.reset_database(reseed=False)
        te = AutoTicketingEngine(tickets_dir=TICKETS_DIR)
        t_res = te.reset_tickets()
        print(f"[ORCHESTRATOR] ✅ Database reset complete! Cleared {t_res.get('deleted_tickets', 0)} tickets.")
        print("[ORCHESTRATOR] Ready for fresh sensing runs.")
    except Exception as e:
        print(f"[ORCHESTRATOR] ❌ Error resetting database: {e}")


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print_banner()
    choice = input("Enter option [1-8] (Default: 1): ").strip() or "1"

    if choice == "1":
        # Full Platform Launch with AI Landing Page (Default)
        check_and_generate_demo_video()
        start_flask_server()
        start_server()

        print("\n" + "=" * 75)
        print("  🚀 SIH-26124 | AI MOBILE URBAN INTELLIGENCE PLATFORM")
        print("  * LANDING PAGE:      http://127.0.0.1:5000/")
        print("  * AI DETECTION MODEL: [STANDBY]")
        print("  * Model inference activates ONLY when you run the Webcam Test!")
        print("=" * 75 + "\n")
        webbrowser.open("http://127.0.0.1:5000/")

    elif choice == "2":
        # Live Built-in Laptop Webcam Test Bench
        print("\n[INFO] Starting Live Laptop Webcam Test Bench...")
        print("[INFO] AI Model detection will run on live camera stream in browser...")
        start_flask_server()
        start_server()
        webbrowser.open("http://127.0.0.1:5000/demo#webcam")

    elif choice == "3":
        # Wireless Mobile IP Camera / RTSP Stream
        print("\n" + "=" * 70)
        print("  📱 WIRELESS MOBILE IP CAMERA SETUP GUIDE")
        print("=" * 70)
        print("  1. Install free 'IP Webcam' app on Android (or DroidCam / EpocCam).")
        print("  2. Connect phone & PC to the same Wi-Fi network.")
        print("  3. In 'IP Webcam', tap 'Start Server' at the bottom.")
        print("  4. Look at the IPv4 address shown on phone (e.g. http://192.168.1.15:8080).")
        print("=" * 70)
        mobile_url = input("Enter Phone IP Webcam URL (e.g. http://192.168.1.15:8080 or rtsp://...): ").strip()
        if not mobile_url:
            mobile_url = "http://192.168.1.100:8080/video"
            print(f"[INFO] Using default: {mobile_url}")
        
        # Normalize if user omitted /video
        if mobile_url.startswith(("http://", "https://")) and not ("://" in mobile_url and "/" in mobile_url.split("://")[1]):
            mobile_url = mobile_url.rstrip("/") + "/video"

        print(f"\n[INFO] Connecting to smartphone stream: {mobile_url}")
        start_flask_server()
        start_server()
        start_edge_sensing(source=mobile_url, conf=0.70)
        webbrowser.open("http://127.0.0.1:5000/demo")

    elif choice == "4":
        # Pre-recorded Video File
        video_path = input("Enter full path to driving video file (.mp4/.avi): ").strip()
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            sys.exit(1)
        start_flask_server()
        start_server()
        start_dashboard()
        start_edge_sensing(source=video_path, conf=0.75)
        webbrowser.open("http://127.0.0.1:5000")
        webbrowser.open("http://localhost:8501")

    elif choice == "5":
        # Flask & Leaflet GIS Only
        start_flask_server()
        webbrowser.open("http://127.0.0.1:5000")

    elif choice == "6":
        # FastAPI & Streamlit Only
        start_server()
        start_dashboard()
        webbrowser.open("http://localhost:8501")
        webbrowser.open("http://127.0.0.1:8000/docs")

    elif choice == "7":
        # Reset detected data
        perform_data_reset()
        input("\nPress Enter to return to main menu or exit...")
        sys.exit(0)

    elif choice == "8":
        print("[ORCHESTRATOR] Exiting.")
        sys.exit(0)
    else:
        print("[ERROR] Invalid choice.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("  ✅ SIH URBAN INTELLIGENCE PROTOTYPE SUITE IS LIVE!")
    print("  - Leaflet.js Web GIS Dashboard: http://127.0.0.1:5000")
    print("  - Flask Events REST API:       http://127.0.0.1:5000/events")
    print("  - Streamlit Judge Dashboard:   http://localhost:8501")
    print("  - FastAPI Central Command:     http://127.0.0.1:8000")
    print("  - Swagger API Documentation:   http://127.0.0.1:8000/docs")
    print("=" * 80)
    print("  Press Ctrl+C in this terminal anytime to stop all services.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()

