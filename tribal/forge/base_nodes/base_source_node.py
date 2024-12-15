from tribal.forge.base_nodes.base_node import BaseNode
import threading

class BaseSourceNode(BaseNode):

    def __init__(self, name, broadcast_manager):
        super().__init__(name, broadcast_manager)

    def start(self):
        threading.Thread(target=self._run).start()

    def _run(self):
        raise NotImplementedError("run not implemented in source node")