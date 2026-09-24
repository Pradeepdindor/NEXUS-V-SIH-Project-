"""
Smart India Hackathon - Problem Statement 26124
Export Road Hazard Spatial Database to Microsoft Excel / CSV
"""

import os
import sys
import csv
import sqlite3
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "dataset" / "data" / "urban_intelligence_spatial.db"
CSV_OUT = BASE_DIR / "dataset" / "data" / "road_hazards_dataset.csv"
ROOT_CSV = BASE_DIR / "road_hazards_dataset.csv"

def export_database_to_csv():
    print("=" * 80)
    print("  [DATASET EXPORT] EXPORTING ROAD HAZARDS DATABASE TO EXCEL / CSV")
    print(f"  Source Database: {DB_PATH}")
    print("=" * 80)

    if not DB_PATH.exists():
        print(f"❌ Database not found at: {DB_PATH}")
        return None

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            event_id AS "Event ID",
            ticket_id AS "Municipal Ticket ID",
            event_type AS "Defect Type",
            category AS "Category",
            ROUND(confidence_score * 100, 1) || '%' AS "AI Confidence",
            severity AS "Severity Level",
            verification_count AS "Multi-Bus Verifications",
            bus_id AS "Primary Reporting Bus",
            route_id AS "Transit Route",
            ROUND(latitude, 6) AS "Latitude",
            ROUND(longitude, 6) AS "Longitude",
            assigned_department AS "Assigned Department",
            status AS "Resolution Status",
            '₹' || estimated_cost_inr AS "Estimated Cost (INR)",
            action_required AS "Required PWD Action",
            snapshot_filename AS "Evidence Photo",
            created_at_utc AS "Detected UTC Timestamp"
        FROM events
        ORDER BY created_at_utc DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("⚠️ Database currently has 0 rows.")
        return None

    headers = rows[0].keys()

    for out_path in [CSV_OUT, ROOT_CSV]:
        with open(out_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(list(row))

    print(f"[SUCCESS] Successfully exported {len(rows)} detected road hazards to Excel CSV!")
    print(f"   Saved to: {ROOT_CSV}")
    print(f"   Saved to: {CSV_OUT}")
    print("=" * 80)

    # Try to open directly in Microsoft Excel on Windows
    try:
        print("\n[LAUNCH] Opening dataset in Microsoft Excel...")
        os.startfile(str(ROOT_CSV))
    except Exception:
        pass

    return str(ROOT_CSV)

if __name__ == "__main__":
    export_database_to_csv()
