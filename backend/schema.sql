-- ==============================================================================
-- SMART INDIA HACKATHON 2026 - PROBLEM STATEMENT 26124
-- Mobile Urban Intelligence Platform Using Public Transport Fleet
-- PostgreSQL + PostGIS Spatial Schema Definition (schema.sql)
-- ==============================================================================

-- 1. Enable PostGIS Spatial Extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 2. Create Events & Defect Incidents Table with Spatial Geometry
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR(64) PRIMARY KEY,
    ticket_id VARCHAR(64) UNIQUE,
    tracking_code VARCHAR(32),
    bus_id VARCHAR(64) NOT NULL,
    route_id VARCHAR(64) NOT NULL,
    category VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    confidence_score REAL NOT NULL,
    severity VARCHAR(32) NOT NULL,
    assigned_department TEXT NOT NULL,
    ward_zone VARCHAR(128),
    status VARCHAR(64) DEFAULT 'DISPATCHED_TO_CONTRACTOR',
    action_required TEXT,
    estimated_cost_inr INTEGER DEFAULT 0,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    altitude_m REAL DEFAULT 920.0,
    speed_kmh REAL DEFAULT 30.0,
    landmark VARCHAR(255),
    snapshot_filename VARCHAR(255),
    snapshot_url TEXT,
    google_maps_url TEXT,
    anonymized BOOLEAN DEFAULT TRUE,
    verification_count INTEGER DEFAULT 1,
    reporting_buses TEXT, -- JSON string or array of reporting bus IDs
    traffic_context TEXT, -- JSON payload of surrounding traffic density
    audit_trail TEXT,     -- JSON array of workflow lifecycle transitions
    created_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- PostGIS Geometry Column (WGS84 EPSG:4326)
    geom geometry(Point, 4326)
);

-- 3. Spatial & Attribute Indexes for Ultra-Fast Geospatial Queries
CREATE INDEX IF NOT EXISTS idx_events_geom ON events USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
CREATE INDEX IF NOT EXISTS idx_events_department ON events(assigned_department);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at_utc);

-- 4. Automatic Geometry Synchronization Trigger
CREATE OR REPLACE FUNCTION sync_event_geom()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    NEW.updated_at_utc = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sync_event_geom ON events;
CREATE TRIGGER trg_sync_event_geom
BEFORE INSERT OR UPDATE ON events
FOR EACH ROW EXECUTE FUNCTION sync_event_geom();
