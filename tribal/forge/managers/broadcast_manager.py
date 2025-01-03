from tribal.forge.managers.log_manager import log_manager
import time
import threading
from collections import defaultdict
from queue import Queue

class BroadcastManager:
    nodes = []

    def __init__(self, num_simultaneous_same_node_running=1):
        self.num_simultaneous_same_node_running = num_simultaneous_same_node_running
        self.node_type_locks = defaultdict(threading.Semaphore)
        self.node_queues = defaultdict(Queue)
        self.node_workers = {}

        if num_simultaneous_same_node_running > 0:
            for node_type in self.node_type_locks:
                self.node_type_locks[node_type] = threading.Semaphore(num_simultaneous_same_node_running)

    def _process_node_queue(self, node_type):
        """Worker function for processing queued tasks for a specific node type."""
        while True:
            try:
                task = self.node_queues[node_type].get()
                if task is None:  # Sentinel to end the worker thread
                    break
                func, args = task
                func(*args)
            finally:
                self.node_queues[node_type].task_done()

    def _start_worker_if_needed(self, node_type):
        """Ensure a worker thread is running for the given node type."""
        if node_type not in self.node_workers:
            worker_thread = threading.Thread(target=self._process_node_queue, args=(node_type,), daemon=True)
            self.node_workers[node_type] = worker_thread
            worker_thread.start()

    def send_broadcast(self, message, origin):
        log_manager.log_send_broadcast(message, origin)

        for node in self.nodes:
            if node._check_received_broadcasts(message):
                node_type = type(node).__name__
                self._start_worker_if_needed(node_type)
                self.node_queues[node_type].put((self._thread_safe_receive_broadcast, (node, message)))

    def _thread_safe_receive_broadcast(self, node, message):
        node_type = type(node).__name__
        lock = self.node_type_locks[node_type]

        with lock:
            node.receive_broadcast(message, node.name)

    def add_node(self, node):
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
        self.nodes = [n for n in self.nodes if n.name != node_id]

    def add_receiver(self, sender_node, receiver, receiver_type):
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
