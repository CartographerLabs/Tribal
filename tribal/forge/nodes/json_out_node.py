import json
import threading
from tribal.forge.base_nodes import MESSAGE_FORMAT, BaseEndNode

class JsonOutNode(BaseEndNode):
    def __init__(self, broadcast_manager, json_file):
        
        super().__init__("Json Reporting", broadcast_manager)
        self.lock = threading.Lock()
        self.json_file = json_file

    def _process_broadcast(self, message):

        message = message[MESSAGE_FORMAT["MESSAGE"]]

        with self.lock:
            try:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []

            data.append(message)

            with open(self.json_file, 'w') as f:
                json.dump(data, f, indent=4)