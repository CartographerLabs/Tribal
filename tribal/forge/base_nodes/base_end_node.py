from tribal.forge.base_nodes.base_node import BaseNode

class BaseEndNode(BaseNode):

    def __init__(self, name, broadcast_manager):
        super().__init__(name, broadcast_manager)

    def send_broadcast(self, message, origin):
        raise NotImplementedError("send_broadcast not implemented in end node")