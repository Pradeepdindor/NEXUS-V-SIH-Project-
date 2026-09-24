"""
Smart India Hackathon - Problem Statement 26124
Interactive Database Viewer & SQL Console
"""

import os
import sys
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "dataset" / "data" / "urban_intelligence_spatial.db"

def print_banner():
    print("=" * 85)
    print("  🔍 SIH 26124: ROAD HAZARD SPATIAL DATABASE VIEWER")
    print(f"  Database File: {DB_PATH}")
    print("=" * 85)

def display_events():
    if not DB_PATH.exists():
        print(f"❌ Database file not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event_id, event_type, confidence_score, severity, 
               verification_count, latitude, longitude, assigned_department, 
               ticket_id, snapshot_filename 
        FROM events 
        ORDER BY created_at_utc DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    print(f"\n📊 TOTAL DETECTED ROAD HAZARDS STORED: {len(rows)}\n")
    if not rows:
        print("  (Database currently has 0 events. Run a demo or webcam detection to populate!)")
        return

    # Table Header
    header = f"{'#':<3} | {'EVENT ID':<22} | {'TYPE':<16} | {'CONF':<7} | {'SEVERITY':<8} | {'VERIF':<5} | {'GPS LAT, LON':<20} | {'TICKET ID':<14} | {'SNAPSHOT'}"
    print(header)
    print("-" * len(header))

    for idx, r in enumerate(rows, 1):
        lat_lon = f"{r['latitude']:.5f}, {r['longitude']:.5f}"
        conf = f"{r['confidence_score']*100:.1f}%"
        snap = r['snapshot_filename'] or 'N/A'
        ticket = r['ticket_id'] or 'PENDING'
        print(f"{idx:<3} | {r['event_id']:<22} | {r['event_type']:<16} | {conf:<7} | {r['severity']:<8} | {r['verification_count']:<5} | {lat_lon:<20} | {ticket:<14} | {snap}")

    print("-" * len(header))

def interactive_query():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n💡 Tip: Type any custom SQL query (e.g., SELECT * FROM events WHERE event_type='POTHOLE';)")
    print("   Or type 'all' to refresh table, 'exit' or 'q' to quit.\n")

    while True:
        try:
            query = input("SQL> ").strip()
            if not query:
                continue
            if query.lower() in ('exit', 'quit', 'q'):
                break
            if query.lower() in ('all', 'refresh'):
                display_events()
                continue

            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                results = cursor.fetchall()
                print(f"-> Returned {len(results)} row(s):")
                for row in results:
                    print(dict(row))
            else:
                conn.commit()
                print(f"-> Query executed successfully. Rows affected: {cursor.rowcount}")
        except Exception as e:
            print(f"❌ SQL Error: {e}")

    conn.close()

if __name__ == "__main__":
    print_banner()
    display_events()
    interactive_query()
