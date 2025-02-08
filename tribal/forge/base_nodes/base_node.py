import threading
import time
from tribal.forge.managers.log_manager import log_manager

MESSAGE_FORMAT = {
"ORIGIN": "ORIGIN",
"MESSAGE": "MESSAGE",
"TYPE": "TYPE"
}

class BaseNode:

    def __init__(self, name, broadcast_manager):
        self.name = name
        self.broadcast_manager = broadcast_manager
        self.broadcast_manager.add_node(self)
        self._implicit_receivers = []  # Move to instance level
        self._explicit_receivers = []  # Move to instance level
        self.message_type = None
        self.last_send_time = 0
        self.last_receive_time = 0

    def set_message_type(self, message_type):
        self.message_type = message_type

    def _construct_message(self, origin, message, message_type = None):

        if message_type is None:
            if self.message_type is None:
                self.message_type = self.name

            message_type = self.message_type

        return {
            MESSAGE_FORMAT["ORIGIN"]: origin,
            MESSAGE_FORMAT["MESSAGE"]: message,
            MESSAGE_FORMAT["TYPE"]: message_type
        }


    def add_broadcast_implicit_receiver(self, origin_type):
        self._implicit_receivers.append(origin_type)

    def add_explicit_receiver(self, origin_name):
        self._explicit_receivers.append(origin_name)

    def remove_implicit_receiver(self, origin_type):
        if origin_type in self._implicit_receivers:
            self._implicit_receivers.remove(origin_type)

    def remove_explicit_receiver(self, origin_name):
        if origin_name in self._explicit_receivers:
            self._explicit_receivers.remove(origin_name)

    def send_broadcast(self, message, origin):
        self.last_send_time = time.time()
        threading.Thread(target=self.broadcast_manager.send_broadcast, args=(message,origin,)).start()
        
    def _check_received_broadcasts(self, message):

        for receiver in self._implicit_receivers:
            if message[MESSAGE_FORMAT["TYPE"]] == receiver:
                return True
            
        for receiver in self._explicit_receivers:
            if message[MESSAGE_FORMAT["ORIGIN"]] == receiver:
                return True
            
        return False

    def _process_broadcast(self, message):
        raise NotImplementedError("process_broadcast not implemented")

    def receive_broadcast(self, message, origin):
        if self._check_received_broadcasts(message):
            self.last_receive_time = time.time()
            log_manager.log_receive_broadcast(message, origin)
            self._process_broadcast(message)