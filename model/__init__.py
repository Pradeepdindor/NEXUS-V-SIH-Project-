"""
Model Package: AI Computer Vision Detectors, HUD Renderer, GPS Simulator & Edge Sensing Pipeline
"""

try:
    from model.detectors import EdgeDetector, DetectionBox, TrafficState
    from model.gps_simulator import GPSSimulator, GPSReading, haversine_distance_meters
    from model.hud_renderer import HUDRenderer
    from model.edge_sensing import EdgeSensingPipeline
except ImportError:
    pass
