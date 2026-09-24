"""
Smart India Hackathon - Problem Statement 26124
Automated Cloudflare Public Tunnel Launcher & URL Extractor
"""

import os
import re
import sys
import time
import socket
import subprocess
import webbrowser
from pathlib import Path

# Safe Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
CLOUDFLARED_EXE = BASE_DIR / "cloudflared.exe"
URL_FILE = BASE_DIR / "public_url.txt"

def is_port_open(host="127.0.0.1", port=5000, timeout=1.5):
    """Check if the local Flask server is listening on port 5000."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def launch_tunnel():
    print("=" * 80, flush=True)
    print("  🚀 SIH 26124: LAUNCHING FREE WORLDWIDE PUBLIC HTTPS TUNNEL", flush=True)
    print("=" * 80, flush=True)

    if not CLOUDFLARED_EXE.exists():
        print(f"❌ Error: cloudflared.exe not found at {CLOUDFLARED_EXE}", flush=True)
        input("Press Enter to exit...")
        return

    # Check if Flask is running
    print("[1/3] Checking local web server on port 5000...", flush=True)
    if not is_port_open("127.0.0.1", 5000):
        print("⚠️  WARNING: Web server (run_flask.bat) is not currently detected on port 5000.", flush=True)
        print("   Please make sure 'run_flask.bat' is started in another terminal window!", flush=True)
    else:
        print("✅ Local web server is running on http://127.0.0.1:5000", flush=True)

    print("[2/3] Connecting to Cloudflare global network via HTTP/2 (Firewall Bypass)...", flush=True)
    print("      (Please wait 5 to 10 seconds for the secure public link to generate)...", flush=True)

    cmd = [
        str(CLOUDFLARED_EXE),
        "tunnel",
        "--protocol", "http2",
        "--url", "http://127.0.0.1:5000"
    ]

    # Run cloudflared and capture output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    public_url = None

    try:
        for line in iter(process.stdout.readline, ""):
            # Check if this line contains the trycloudflare URL
            match = url_pattern.search(line)
            if match and not public_url:
                public_url = match.group(0)
                
                # Write to public_url.txt in root
                with open(URL_FILE, "w", encoding="utf-8") as f:
                    f.write(public_url + "\n")

                print("\n" + "=" * 80, flush=True)
                print("  🎉 YOUR FREE PUBLIC HTTPS LINK IS READY & LIVE!", flush=True)
                print("=" * 80, flush=True)
                print(f"\n  👉 PUBLIC URL: {public_url}\n", flush=True)
                print("=" * 80, flush=True)
                print(f"  📁 Saved link to: {URL_FILE}", flush=True)
                print("  🌐 Anyone on any smartphone, tablet, or PC worldwide can open this link!", flush=True)
                print("  🛑 To stop the tunnel, press CTRL + C in this window.", flush=True)
                print("=" * 80 + "\n", flush=True)

                # Try to open in user's browser
                try:
                    webbrowser.open(public_url)
                except Exception:
                    pass

            # Filter noise lines once URL is found to keep console super clean
            if public_url:
                if "Registered tunnel connection" in line or "location=" in line:
                    print(f"  [CONNECTED] {line.strip()}", flush=True)
                elif "ERR" in line:
                    print(f"  [LOG] {line.strip()}", flush=True)
            else:
                # Show initialization progress
                if "Requesting new quick Tunnel" in line:
                    print("   * Negotiating secure tunnel token with Cloudflare Edge...", flush=True)

        process.wait()

    except KeyboardInterrupt:
        print("\n[STOPPING] Shutting down Cloudflare Tunnel...", flush=True)
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        print("[DONE] Tunnel closed safely.", flush=True)

if __name__ == "__main__":
    launch_tunnel()
