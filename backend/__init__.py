"""
Backend Package: Central Command Server, REST APIs, Spatial Database & Automated Ticketing
"""

try:
    from backend.spatial_db import SpatialEventDatabase
    from backend.ticket_engine import AutoTicketingEngine, MunicipalTicket
    from backend.alert_manager import AlertManager, EdgeAnonymizer, StoreAndForwardQueue
except ImportError:
    pass
