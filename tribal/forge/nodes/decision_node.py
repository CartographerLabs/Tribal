from tribal.forge.base_nodes import MESSAGE_FORMAT, BaseProcessorNode

class DecisionNode(BaseProcessorNode):
    def __init__(self, broadcast_manager, expression):
        super().__init__("Decision Node", broadcast_manager)
        self.expression = expression

    def _process_broadcast(self, act_message):
        message = act_message[MESSAGE_FORMAT["MESSAGE"]]
        local_context = {"message": message, **locals()}

        if eval(self.expression, local_context, local_context):
            message = self._construct_message(self.name, message)
            self.send_broadcast(message, self.name)
