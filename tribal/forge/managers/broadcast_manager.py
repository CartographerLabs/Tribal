from tribal.forge.managers.log_manager import LogManager
import time

class BroadcastManager():

    nodes = []

    def __init__(self):
        pass

    def send_broadcast(self, message, origin):
        LogManager().log_send_broadcast(message, origin)

        for node in self.nodes:
            node.receive_broadcast(message, node.name)
    
    def add_node(self, node):
        LogManager().log(f"Added node: {node.name} - all nodes: {str(self.nodes)}")
        self.nodes.append(node)

    def get_node_config(self):
        return [{
            'id': node.name,
            'type': node.__class__.__name__,
            'receivers': self._get_node_receivers(node),
            'message_type': node.message_type,
            'params': node.get_params(),
            'hasStart': hasattr(node, 'start'),
            'state': getattr(node, 'state', 'stopped')
        } for node in self.nodes]

    def _get_node_receivers(self, node):
        return [
            {'type': 'implicit', 'value': r} for r in node._implicit_receivers
        ] + [
            {'type': 'explicit', 'value': r} for r in node._explicit_receivers
        ]

    def remove_node(self, node_id):
        LogManager().log(f"Removed node: {node_id}")
        self.nodes = [n for n in self.nodes if n.name != node_id]

    def add_receiver(self, sender_node, receiver, receiver_type):
        LogManager().log(f"Added receiver: {receiver} to {sender_node.name} ({receiver_type})")
        if receiver_type == 'explicit':
            receiver_node = next((n for n in self.nodes if n.name == receiver), None)
            if receiver_node:
                sender_node.add_explicit_receiver(receiver_node.name)
        else:  # implicit
            sender_node.add_broadcast_implicit_receiver(receiver)

    def get_message_types(self):
        """Get all unique message types from nodes"""
        types = set()
        for node in self.nodes:
            if hasattr(node, 'message_type') and node.message_type:
                types.add(node.message_type)
            # Also include all implicit receivers as valid message types
            types.update(node._implicit_receivers)
        return types

    def get_node_by_name(self, name):
        """Get a node instance by its name"""
        return next((n for n in self.nodes if n.name == name), None)

    def remove_receiver(self, node, receiver):
        LogManager().log(f"Removed receiver: {receiver} from {node.name}")
        """Remove a receiver from a node"""
        # Try removing both implicit and explicit - only one will succeed
        node.remove_implicit_receiver(receiver)
        node.remove_explicit_receiver(receiver)

    def get_nodes_status(self):
        current_time = time.time()
        return [{
            'id': node.name,
            'recent_send': hasattr(node, 'last_send_time') and (current_time - node.last_send_time) <= 1,
            'recent_receive': hasattr(node, 'last_receive_time') and (current_time - node.last_receive_time) <= 1
        } for node in self.nodes]