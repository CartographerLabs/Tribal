from tribal.forge.base_nodes import MESSAGE_FORMAT, BaseSourceNode
import json
import time

class JsonReaderNode(BaseSourceNode):
    def __init__(self, broadcast_manager, json_path, sleep=0.1):
        self.sleep = float(sleep)
        self.json_path = json_path
        super().__init__("JSON Reader", broadcast_manager)

    def _run(self):
        with open(self.json_path, mode='r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            for row in data:
                time.sleep(self.sleep)

                message = self._construct_message(self.name, row)
                self.send_broadcast(message, self.name)
