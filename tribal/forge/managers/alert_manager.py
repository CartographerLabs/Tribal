import os
import tempfile
from rich.console import Console
import hashlib
from threading import Lock
from time import time

class AlertManager:
    ALERT_FILE = os.path.join(tempfile.gettempdir(), 'tribal_forge_alerts.txt')
    
    def __init__(self):
        self.console = Console()
        self.recent_alerts = {}  # Track recent alerts with timestamps
        self.lock = Lock()
        self.alert_timeout = 1.0  # Seconds to consider an alert as duplicate
        # Create or clear the alert file
        open(self.ALERT_FILE, 'w').close()

    def _get_alert_hash(self, message, origin):
        """Create a hash of the alert to identify duplicates"""
        alert_str = f"{origin}:{str(message)}"
        return hashlib.md5(alert_str.encode()).hexdigest()

    def _is_duplicate(self, alert_hash):
        """Check if alert is a duplicate and clean old alerts"""
        current_time = time()
        
        # Clean old alerts
        self.recent_alerts = {
            k: v for k, v in self.recent_alerts.items()
            if current_time - v < self.alert_timeout
        }
        
        # Check if alert is recent
        if alert_hash in self.recent_alerts:
            return True
        
        self.recent_alerts[alert_hash] = current_time
        return False

    def send_alert(self, message, origin):
        alert_hash = self._get_alert_hash(message, origin)
        with self.lock:
            if self._is_duplicate(alert_hash):
                return
                
            alert_msg = f"ALERT from {origin}: {message}\n"
            self.console.print(alert_msg.strip(), style="bold red")
            with open(self.ALERT_FILE, 'a') as f:
                f.write(alert_msg)

alert_manager = AlertManager()