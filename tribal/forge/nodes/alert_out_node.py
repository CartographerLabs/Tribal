
from tribal.forge.base_nodes import MESSAGE_FORMAT,BaseEndNode
from tribal.forge.managers.alert_manager import alert_manager

class AlertOutNode(BaseEndNode):
    def __init__(self, broadcast_manager):
        super().__init__("Alert Reporting", broadcast_manager)

    def _process_broadcast(self, message):
        alert_manager.send_alert(f"Received message: {message[MESSAGE_FORMAT["MESSAGE"]]}", self.name)