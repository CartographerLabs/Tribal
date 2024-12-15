from tribal.forge.base_nodes.base_node import BaseNode

class BaseProcessorNode(BaseNode):
    def __init__(self, name, broadcast_manager):
        super().__init__(name, broadcast_manager)
    pass