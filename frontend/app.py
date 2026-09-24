"""
Smart India Hackathon - Problem Statement 26124
AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Module: Interactive Judge Demonstration Dashboard (app.py)
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

from config import (
    SERVER_URL,
    TICKETS_DIR,
    SNAPSHOTS_DIR,
    ALERTS_DIR,
    DEFAULT_BUS_ID,
    DEFAULT_ROUTE_ID,
    FLEET_BUS_OPTIONS,
    CAMERA_ANGLES,
    SIMULATED_ROUTE_WAYPOINTS,
    DEFECT_TYPES,
)

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="AI Mobile Urban Intelligence | SIH-26124",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Glassmorphism CSS
st.markdown("""
<style>
    /* Dark Glassmorphic Theme */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0a0d13 100%);
        color: #e6edf3;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Top Header Bar */
    .header-box {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 139, 253, 0.25);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 24px;
        font-weight: 700;
        color: #58a6ff;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .header-sub {
        font-size: 13px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .metric-num {
        font-size: 26px;
        font-weight: 800;
        color: #3fb950;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Badges */
    .badge-critical { background: #f85149; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .badge-high { background: #d29922; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .badge-medium { background: #388bfd; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 11px; }
    .badge-low { background: #3fb950; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 11px; }

    /* Section Cards */
    .card-container {
        background: rgba(22, 27, 34, 0.80);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# BACKEND API CLIENT HELPERS
# ==========================================
def check_backend_health():
    try:
        r = requests.get(f"{SERVER_URL}/", timeout=0.6)
        return r.status_code == 200
    except Exception:
        return False


def fetch_tickets(status=None, severity=None):
    try:
        params = {}
        if status and status != "ALL":
            params["status"] = status
        if severity and severity != "ALL":
            params["severity"] = severity
        r = requests.get(f"{SERVER_URL}/api/tickets", params=params, timeout=1.0)
        if r.status_code == 200:
            return r.json().get("tickets", [])
    except Exception:
        pass

    # Local fallback if server offline
    tickets = []
    if TICKETS_DIR.exists():
        for f in sorted(TICKETS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as tf:
                    tickets.append(json.load(tf))
            except Exception:
                pass
    return tickets


def fetch_live_telemetry():
    try:
        r = requests.get(f"{SERVER_URL}/api/telemetry/live", timeout=0.8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"fleet": {}, "recent_alerts": []}


def fetch_analytics():
    try:
        r = requests.get(f"{SERVER_URL}/api/analytics/summary", timeout=0.8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback compute
    tickets = fetch_tickets()
    return {
        "total_tickets_generated": len(tickets),
        "open_tickets": len(tickets),
        "critical_severity_count": sum(1 for t in tickets if t.get("severity") == "Critical"),
        "high_severity_count": sum(1 for t in tickets if t.get("severity") == "High"),
        "total_estimated_repair_cost_inr": sum(t.get("estimated_repair_cost_inr", 0) for t in tickets),
        "city_road_health_index": 82.4,
    }


def reset_platform_data():
    """Reset database and tickets via API or direct engine call."""
    success = False
    try:
        r = requests.post(f"{SERVER_URL}/api/reset", timeout=1.5)
        if r.status_code == 200:
            success = True
    except Exception:
        pass
    
    if not success:
        # Local direct fallback
        try:
            try:
                from backend.spatial_db import SpatialEventDatabase
                from backend.ticket_engine import AutoTicketingEngine
            except ImportError:
                from spatial_db import SpatialEventDatabase
                from ticket_engine import AutoTicketingEngine
            sdb = SpatialEventDatabase()
            sdb.reset_database(reseed=False)
            te = AutoTicketingEngine(tickets_dir=TICKETS_DIR)
            te.reset_tickets()
            success = True
        except Exception as e:
            print(f"Local reset error: {e}")
    return success


# ==========================================
# SIDEBAR: JUDGE PARAMETER CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Judge Control Suite")
    st.markdown("Real-time edge parameters & fleet selector")

    # Backend Connection Status
    is_server_up = check_backend_health()
    if is_server_up:
        st.success("🟢 Central Command Backend: ONLINE")
    else:
        st.warning("🟠 Backend: STANDALONE MODE (Local Storage Active)")

    st.markdown("---")

    # 1. Model Confidence Threshold Slider
    conf_threshold = st.slider(
        "🎯 AI Defect Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Detections exceeding this confidence level automatically dispatch PWD/BBMP repair tickets."
    )

    # 2. Camera View Selector
    camera_view = st.selectbox(
        "📹 Mobile Sensing Camera Perspective",
        options=CAMERA_ANGLES,
        index=0,
        help="Switch between multi-angle bus-mounted cameras."
    )

    # 3. Bus Fleet & Route Selector
    bus_choice = st.selectbox(
        "🚌 Active Transit Fleet Bus",
        options=FLEET_BUS_OPTIONS,
        format_func=lambda x: f"{x['bus_id']} ({x['route_id']})",
        index=0
    )
    selected_bus_id = bus_choice["bus_id"]
    selected_route_id = bus_choice["route_id"]

    # 4. Hazard Category Filters
    st.markdown("##### 🔍 Hazard Filter")
    cat_filter = st.multiselect(
        "Filter Categories",
        options=["Road Defect", "Missing Infrastructure", "Traffic & Safety"],
        default=["Road Defect", "Missing Infrastructure", "Traffic & Safety"]
    )

    # 5. Mobile IP Camera Setup Helper
    with st.expander("📱 Wireless Mobile IP Camera (IP Webcam)"):
        st.markdown("""
        **Turn Any Smartphone into an Onboard Transit Camera:**
        1. Install free **IP Webcam** (Android) or **DroidCam**.
        2. Connect phone & PC to same Wi-Fi.
        3. Tap **Start Server** in the app.
        4. Copy the IP address shown (e.g. `http://192.168.1.15:8080/video`).
        5. Run `python run_demo.py` and select Option **[3]**.
        """)

    # 6. Reset Data Button
    st.markdown("##### 🗑️ Data Reset")
    if st.button("🗑️ Reset All Detected Data & Events", use_container_width=True, type="secondary"):
        if reset_platform_data():
            st.toast("🧹 All detected road events, PostGIS records & tickets cleared!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Failed to reset platform data.")

    # Auto Refresh
    auto_refresh = st.checkbox("⚡ Live Telemetry Auto-Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Polling Interval (sec)", 1, 10, 3)

    st.markdown("---")
    st.caption("Smart India Hackathon 2026 | Problem Statement 26124")


# ==========================================
# TOP HEADER BAR
# ==========================================
st.markdown(f"""
<div class="header-box">
    <div>
        <div class="header-title">🚌 AI-POWERED MOBILE URBAN INTELLIGENCE PLATFORM</div>
        <div class="header-sub">Public Transport Fleet Edge Sensing & Automated Municipal PWD Repair Dispatcher</div>
    </div>
    <div style="text-align: right;">
        <span style="background: rgba(56, 139, 253, 0.15); border: 1px solid #388bfd; color: #58a6ff; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 13px;">
            BUS: {selected_bus_id}
        </span>
        <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">Route: {selected_route_id} | {bus_choice['name']}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# FETCH DATA
# ==========================================
analytics = fetch_analytics()
tickets = fetch_tickets()
telemetry_data = fetch_live_telemetry()
active_fleet = telemetry_data.get("fleet", {})
bus_telemetry = active_fleet.get(selected_bus_id, {})

# Default metrics fallback
current_speed = bus_telemetry.get("gps_telemetry", {}).get("speed_kmh", 32.4)
current_landmark = bus_telemetry.get("gps_telemetry", {}).get("current_landmark", "MG Road Metro Station")
current_lat = bus_telemetry.get("gps_telemetry", {}).get("latitude", 12.9756)
current_lon = bus_telemetry.get("gps_telemetry", {}).get("longitude", 77.6095)
traffic_ctx = bus_telemetry.get("traffic_context", {})
congestion_level = traffic_ctx.get("congestion_level", "Medium")
total_vehicles = traffic_ctx.get("total_vehicles", 4)


# ==========================================
# METRIC CARDS ROW
# ==========================================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">City Road Health</div>
        <div class="metric-num" style="color: #3fb950;">{analytics.get('city_road_health_index', 84.5)} / 100</div>
        <div style="font-size: 11px; color: #8b949e;">AI Surface Quality Score</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Municipal Tickets</div>
        <div class="metric-num" style="color: #58a6ff;">{analytics.get('total_tickets_generated', len(tickets))}</div>
        <div style="font-size: 11px; color: #8b949e;">Auto-Dispatched to PWD</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Critical / High Alerts</div>
        <div class="metric-num" style="color: #f85149;">{analytics.get('critical_severity_count', 0) + analytics.get('high_severity_count', 0)}</div>
        <div style="font-size: 11px; color: #8b949e;">Urgent SLA Interventions</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    cg_color = "#3fb950" if congestion_level == "Low" else ("#d29922" if congestion_level == "Medium" else "#f85149")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Live Congestion</div>
        <div class="metric-num" style="color: {cg_color};">{congestion_level.upper()}</div>
        <div style="font-size: 11px; color: #8b949e;">{total_vehicles} Active Vehicles in ROI</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Hotspots</div>
        <div class="metric-num" style="color: #d29922;">{analytics.get('total_tickets_generated', len(fetch_tickets()))}</div>
        <div style="font-size: 11px; color: #8b949e;">Geotagged Priority Zones</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
tab_live, tab_map, tab_tickets, tab_analytics = st.tabs([
    "📹 Live Edge Sensing & Vision Feed",
    "🗺️ City Intelligence GIS Map",
    "📋 Municipal Repair Work Orders (PWD Log)",
    "📊 Fleet & Infrastructure Analytics"
])


# -------------------------------------------------------------
# TAB 1: LIVE EDGE SENSING FEED
# -------------------------------------------------------------
with tab_live:
    col_feed, col_telemetry = st.columns([1.6, 1.0])

    with col_feed:
        st.markdown("##### 📡 Real-Time Dashcam & Vision Feed")

        # Check for latest snapshot or sample frame
        snapshots = sorted(list(SNAPSHOTS_DIR.glob("*.jpg")), key=os.path.getmtime, reverse=True)
        if snapshots:
            latest_snap = snapshots[0]
            st.image(str(latest_snap), caption=f"Live Feed Snapshot ({selected_bus_id} | {camera_view})", use_container_width=True)
        else:
            st.info("Streaming initialized. Waiting for incoming edge camera frames...")

        st.caption(f"📍 Waypoint: **{current_landmark}** | GPS: `({current_lat:.5f}, {current_lon:.5f})` | Speed: **{current_speed} km/h**")

    with col_telemetry:
        st.markdown("##### 🚗 Live Traffic Density & Vehicle Breakdown")

        breakdown = traffic_ctx.get("breakdown", {"cars": 3, "buses": 1, "trucks": 0, "two_wheelers": 2})

        # Plotly Bar Chart for Vehicles Breakdown
        df_veh = pd.DataFrame([
            {"Vehicle Type": "Cars", "Count": breakdown.get("cars", 3), "Color": "#58a6ff"},
            {"Vehicle Type": "Buses", "Count": breakdown.get("buses", 1), "Color": "#3fb950"},
            {"Vehicle Type": "Trucks", "Count": breakdown.get("trucks", 0), "Color": "#d29922"},
            {"Vehicle Type": "2-Wheelers", "Count": breakdown.get("two_wheelers", 2), "Color": "#a371f7"},
        ])

        fig_veh = px.bar(
            df_veh,
            x="Count",
            y="Vehicle Type",
            orientation="h",
            color="Vehicle Type",
            color_discrete_map={
                "Cars": "#58a6ff",
                "Buses": "#3fb950",
                "Trucks": "#d29922",
                "2-Wheelers": "#a371f7",
            },
            text="Count"
        )
        fig_veh.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_veh, use_container_width=True)

        # Traffic Congestion Gauge
        st.markdown("##### 🚦 Congestion Level Meter")
        score = traffic_ctx.get("congestion_score", 0.45)
        st.progress(float(score))
        st.caption(f"Status: **{congestion_level.upper()}** (Density Index: {score*100:.1f}%)")

        st.markdown("---")
        st.markdown("##### ⚠️ Latest Edge Defect Alert")
        if tickets:
            latest_t = tickets[0]
            st.markdown(f"""
            **{latest_t.get('hazard_type')}** ({latest_t.get('severity')} Severity)
            - **Ticket ID:** `{latest_t.get('ticket_id')}`
            - **Assigned Dept:** {latest_t.get('assigned_department')}
            - **SLA Target:** {latest_t.get('sla_hours')} Hours
            """)
        else:
            st.write("No defect alerts recorded yet.")


# -------------------------------------------------------------
# TAB 2: CITY INTELLIGENCE GIS MAP
# -------------------------------------------------------------
with tab_map:
    map_col1, map_col2 = st.columns([1.5, 1.0])
    with map_col1:
        st.markdown("##### 🗺️ Geotagged Hazard Clusters & Transit Route Corridors")
    with map_col2:
        enable_heatmap = st.checkbox("🔥 Enable Road Damage Heatmap Layer (Weighted Density)", value=True)

    # Center map around Bangalore route
    m = folium.Map(
        location=[12.9750, 77.6200],
        zoom_start=13,
        tiles="CartoDB dark_matter"
    )

    # Plot Transit Route Line
    route_coords = [[wp["lat"], wp["lon"]] for wp in SIMULATED_ROUTE_WAYPOINTS]
    folium.PolyLine(
        route_coords,
        color="#388bfd",
        weight=4,
        opacity=0.8,
        tooltip=f"Active Bus Route: {selected_route_id}"
    ).add_to(m)

    # Plot Waypoint Stops
    for wp in SIMULATED_ROUTE_WAYPOINTS:
        folium.CircleMarker(
            location=[wp["lat"], wp["lon"]],
            radius=4,
            color="#58a6ff",
            fill=True,
            fill_color="#58a6ff",
            fill_opacity=0.7,
            popup=f"<b>Stop:</b> {wp['name']}<br><b>Ward:</b> {wp.get('ward', 'Central')}"
        ).add_to(m)

    # Heatmap Data Aggregation
    heat_data = []

    # Plot Defect & Hazard Tickets
    for t in tickets:
        loc = t.get("location", {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat and lon:
            sev = t.get("severity", "Medium")
            pin_color = "red" if sev == "Critical" or sev == "High" else ("orange" if sev == "Medium" else "green")
            ver_count = t.get("verification_count", 1)

            # Add to heatmap
            base_weight = 1.0 if sev == "Critical" else (0.75 if sev == "High" else 0.5)
            heat_data.append([lat, lon, base_weight * min(1.5, 1.0 + 0.2 * (ver_count - 1))])

            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 220px;">
                <h4 style="margin:0; color: #d9534f;">{t.get('hazard_type')}</h4>
                <p style="margin:4px 0;"><b>Ticket:</b> {t.get('ticket_id')}</p>
                <p style="margin:4px 0;"><b>Severity:</b> {sev} | <b>Conf:</b> {t.get('detection_confidence', 0.85)*100:.1f}%</p>
                <p style="margin:4px 0;"><b>Dept:</b> {t.get('assigned_department')}</p>
                <p style="margin:4px 0;"><b>Ward:</b> {t.get('ward_zone')}</p>
                <p style="margin:4px 0;"><b>Confirmations:</b> <span style="color:#00d2ff; font-weight:bold;">{ver_count} Fleet Buses</span></p>
                <p style="margin:4px 0; font-size: 11px; color: #5cb85c;"><b>Privacy Anonymized:</b> Face/Plate Blurred</p>
                <a href="{t.get('google_maps_url')}" target="_blank" style="color:#0275d8;">Open in Google Maps</a>
            </div>
            """
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=pin_color, icon="exclamation-sign")
            ).add_to(m)

    # Add Folium HeatMap Layer if enabled
    if enable_heatmap and heat_data:
        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            max_zoom=16,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)

    # Active Bus Location Marker
    folium.Marker(
        location=[current_lat, current_lon],
        popup=f"<b>Active Fleet:</b> {selected_bus_id}<br>Speed: {current_speed} km/h",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    folium_static(m, width=1150, height=520)


# -------------------------------------------------------------
# TAB 3: MUNICIPAL REPAIR TICKETS (PWD LOG)
# -------------------------------------------------------------
with tab_tickets:
    st.markdown("##### 📋 Auto-Generated Municipal Work Orders (PWD / BBMP Dispatch)")

    # Filter row
    f1, f2 = st.columns([1, 1])
    with f1:
        status_filter = st.selectbox("Filter Status", ["ALL", "DISPATCHED_TO_CONTRACTOR", "IN_PROGRESS", "RESOLVED"])
    with f2:
        sev_filter = st.selectbox("Filter Severity", ["ALL", "Critical", "High", "Medium", "Low"])

    filtered_tickets = [
        t for t in tickets
        if (status_filter == "ALL" or t.get("status") == status_filter) and
           (sev_filter == "ALL" or t.get("severity") == sev_filter)
    ]

    if filtered_tickets:
        # Table Overview
        table_rows = []
        for t in filtered_tickets:
            table_rows.append({
                "Ticket ID": t.get("ticket_id"),
                "Hazard": t.get("hazard_type"),
                "Severity": t.get("severity"),
                "Department": t.get("assigned_department", "")[:35] + "...",
                "Ward": t.get("ward_zone"),
                "SLA Target": f"{t.get('sla_hours')}h",
                "Status": t.get("status"),
                "Created At": t.get("created_at_utc", "")[:19].replace("T", " "),
            })

        df_tickets = pd.DataFrame(table_rows)
        st.dataframe(df_tickets, use_container_width=True, height=240)

        st.markdown("---")
        st.markdown("##### 🔍 Ticket Detail Inspector & Contractor Resolution")

        selected_tid = st.selectbox("Select Ticket to Inspect / Update", [t["ticket_id"] for t in filtered_tickets])
        target_t = next((t for t in filtered_tickets if t["ticket_id"] == selected_tid), None)

        if target_t:
            t_col1, t_col2 = st.columns([1.2, 1.0])

            with t_col1:
                st.markdown(f"### Ticket: `{target_t['ticket_id']}`")
                st.markdown(f"**Hazard:** {target_t.get('hazard_type')} | **Severity:** `{target_t.get('severity')}`")
                st.markdown(f"**Assigned Authority:** {target_t.get('assigned_department')}")
                st.markdown(f"**Ward:** {target_t.get('ward_zone')} | **Location:** `{target_t.get('location', {}).get('current_landmark')}`")
                st.markdown(f"**Action Required:** {target_t.get('action_required')}")
                st.markdown(f"**Google Maps:** [Open Direct Coordinate Pin]({target_t.get('google_maps_url')})")

                # Status Updater
                st.markdown("###### Update Work Order Status")
                new_st = st.selectbox("Change Status", ["DISPATCHED_TO_CONTRACTOR", "IN_PROGRESS", "RESOLVED", "CLOSED"], index=0)
                contractor_input = st.text_input("Assigned Contractor", value=target_t.get("contractor_assigned", "Govt Urban Works Empanelled Contractor A"))
                res_notes = st.text_area("Resolution Notes / Inspection Memo", value="Repair crew mobilized for surface patching.")

                if st.button("💾 Save Status Update"):
                    if is_server_up:
                        try:
                            requests.patch(
                                f"{SERVER_URL}/api/tickets/{target_t['ticket_id']}",
                                json={"status": new_st, "contractor_assigned": contractor_input, "resolution_notes": res_notes},
                                timeout=1.0
                            )
                            st.success(f"Ticket {target_t['ticket_id']} successfully updated to '{new_st}'!")
                        except Exception as e:
                            st.error(f"Failed to update via API: {e}")
                    else:
                        ticket_engine = AutoTicketingEngine(tickets_dir=TICKETS_DIR)
                        ticket_engine.update_ticket_status(target_t['ticket_id'], new_st, contractor_input, res_notes)
                        st.success(f"Ticket {target_t['ticket_id']} locally updated to '{new_st}'!")

            with t_col2:
                st.markdown("##### 📸 Photographic Defect Evidence")
                snap_file = target_t.get("snapshot_filename")
                snap_path = SNAPSHOTS_DIR / snap_file if snap_file else None
                if snap_path and snap_path.exists():
                    st.image(str(snap_path), caption=f"Evidence Snapshot: {target_t['ticket_id']}", use_container_width=True)
                else:
                    st.info("Photo snapshot stored on edge archive.")

                st.markdown("###### 📜 Automated Audit Trail")
                for audit in target_t.get("audit_trail", []):
                    if isinstance(audit, dict):
                        st.markdown(f"- **{audit.get('action', 'Action')}** ({str(audit.get('timestamp', ''))[:19].replace('T', ' ')}) — *{audit.get('actor', 'System')}*: {audit.get('details', '')}")
                    else:
                        st.markdown(f"- {audit}")
    else:
        st.info("No tickets match current filters.")

    st.markdown("---")
    st.markdown("##### 🗑️ Work Orders & Storage Maintenance")
    if st.button("🗑️ Reset All Work Orders & Spatial Events", key="tab3_reset_btn", type="secondary"):
        if reset_platform_data():
            st.toast("🧹 All municipal tickets and spatial events cleared!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Failed to reset tickets.")



# -------------------------------------------------------------
# TAB 4: FLEET & INFRASTRUCTURE ANALYTICS
# -------------------------------------------------------------
with tab_analytics:
    st.markdown("##### 📊 Municipal Road Quality & Fleet Distribution Analytics")

    a_col1, a_col2 = st.columns(2)

    with a_col1:
        st.markdown("###### Hazard Type Distribution")
        hazard_dist = analytics.get("hazard_type_distribution", {"Pothole": 4, "Damaged Pavement": 3, "Waterlogging": 3})
        df_hz = pd.DataFrame([{"Hazard": k, "Count": v} for k, v in hazard_dist.items()])
        if not df_hz.empty:
            fig_hz = px.pie(df_hz, names="Hazard", values="Count", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_hz.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280)
            st.plotly_chart(fig_hz, use_container_width=True)

    with a_col2:
        st.markdown("###### Municipal Department Workload")
        dept_dist = analytics.get("department_distribution", {
            "PWD Road Maintenance": 7,
            "Urban Drainage (BWSSB)": 3,
            "Traffic Engineering Cell": 2
        })
        df_dp = pd.DataFrame([{"Department": k[:25] + "...", "Tickets": v} for k, v in dept_dist.items()])
        if not df_dp.empty:
            fig_dp = px.bar(df_dp, x="Department", y="Tickets", color="Tickets", color_continuous_scale="Viridis")
            fig_dp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280)
            st.plotly_chart(fig_dp, use_container_width=True)


# ==========================================
# AUTO REFRESH LOOP
# ==========================================
if auto_refresh and hasattr(st, "runtime") and st.runtime.exists():
    time.sleep(refresh_interval)
    st.rerun()
