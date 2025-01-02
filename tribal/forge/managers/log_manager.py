import os
import tempfile
from rich.console import Console
import hashlib
from threading import Lock
from time import time

class LogManager:
    LOG_FILE = os.path.join(tempfile.gettempdir(), 'tribal_forge_logs.txt')
    
    def __init__(self, mode="debug"):
        self.console = Console()
        self.mode = mode
        self.recent_messages = {}  # Track recent messages with timestamps
        self.lock = Lock()  # Add thread lock
        self.message_timeout = 1.0  # Seconds to consider a message as duplicate
        # Create or clear the log file
        open(self.LOG_FILE, 'w').close()

    def _get_message_hash(self, message, origin, priority):
        """Create a hash of the message to identify duplicates"""
        msg_str = f"{origin}:{str(message)}:{priority}"
        return hashlib.md5(msg_str.encode()).hexdigest()

    def _is_duplicate(self, msg_hash):
        """Check if message is a duplicate and clean old messages"""
        current_time = time()
        
        # Clean old messages
        self.recent_messages = {
            k: v for k, v in self.recent_messages.items()
            if current_time - v < self.message_timeout
        }
        
        # Check if message is recent
        if msg_hash in self.recent_messages:
            return True
        
        self.recent_messages[msg_hash] = current_time
        return False

    def log_send_broadcast(self, message, origin, priority="normal"):
        if self.mode not in ["info", "debug"]:
            return

        msg_hash = self._get_message_hash(message, origin, priority)
        with self.lock:
            if self._is_duplicate(msg_hash):
                return
            
            log_msg = f"Sent: {origin} -> {message} [Priority: {priority}]\n"
            self.console.print(log_msg.strip(), style="bold green")
            with open(self.LOG_FILE, 'a') as f:
                f.write(log_msg)

    def log(self, message, style="bold white"):
        if self.mode not in ["info", "debug"]:
            return

        with self.lock:
            log_msg = f"{message}\n"
            self.console.print(log_msg.strip(), style=style)
            with open(self.LOG_FILE, 'a') as f:
                f.write(log_msg)
    
    def log_receive_broadcast(self, message, origin, priority="normal"):
        if self.mode not in ["info", "debug"]:
            return

        msg_hash = self._get_message_hash(message, origin, priority)
        with self.lock:
            if self._is_duplicate(msg_hash):
                return
            
            log_msg = f"Received: {origin} -> {message} [Priority: {priority}]\n"
            self.console.print(log_msg.strip(), style="bold blue")
            with open(self.LOG_FILE, 'a') as f:
                f.write(log_msg)

    def set_mode(self, mode):
        self.mode = mode

log_manager = LogManager()
