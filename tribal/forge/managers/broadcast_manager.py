from tribal.forge.managers.log_manager import log_manager, LogManager
import time
import threading
from collections import defaultdict
from queue import Queue


class BroadcastManager:
    nodes = []

    def __init__(self, num_simultaneous_same_node_running=0):
        self.num_simultaneous_same_node_running = num_simultaneous_same_node_running
        self.node_type_locks = defaultdict(threading.Semaphore)
        self.node_queues = defaultdict(Queue)
        self.node_workers = {}

        if num_simultaneous_same_node_running > 0:
            for node_type in self.node_type_locks:
                self.node_type_locks[node_type] = threading.Semaphore(num_simultaneous_same_node_running)

        LogManager().log(f"BroadcastManager initialized with simultaneous node cap: {num_simultaneous_same_node_running}")

    def _process_node_queue(self, node_type):
        """Worker function for processing queued tasks for a specific node type."""
        LogManager().log(f"Worker started for node type: {node_type}")

        while True:
            try:
                task = self.node_queues[node_type].get()
                if task is None:  # Sentinel to end the worker thread
                    LogManager().log(f"Stopping worker for node type: {node_type}")
                    break
                func, args = task
                LogManager().log(f"Dequeuing task for {node_type}, executing {func.__name__} with args {args}")
                func(*args)
            except Exception as e:
                LogManager().log(f"Error processing task for {node_type}: {str(e)}")
            finally:
                self.node_queues[node_type].task_done()
                LogManager().log(f"Task for {node_type} completed")

    def _start_worker_if_needed(self, node_type):
        """Ensure a worker thread is running for the given node type."""
        if node_type not in self.node_workers or not self.node_workers[node_type].is_alive():
            worker_thread = threading.Thread(target=self._process_node_queue, args=(node_type,), daemon=True)
            self.node_workers[node_type] = worker_thread
            worker_thread.start()
            LogManager().log(f"Worker thread started for node type: {node_type}")

    def send_broadcast(self, message, origin):
        log_manager.log_send_broadcast(message, origin)
        LogManager().log(f"Sending broadcast from {origin}: {message}")

        for node in self.nodes:
            if node._check_received_broadcasts(message):
                node_type = type(node).__name__
                LogManager().log(f"Node {node.name} of type {node_type} queued for message processing")
                self._start_worker_if_needed(node_type)
                self.node_queues[node_type].put((self._thread_safe_receive_broadcast, (node, message)))
                LogManager().log(f"Task queued for {node_type}: {message}")

    def _thread_safe_receive_broadcast(self, node, message):
        node_type = type(node).__name__
        lock = self.node_type_locks[node_type]

        with lock:
            LogManager().log(f"Node {node.name} of type {node_type} executing receive_broadcast")
            node.receive_broadcast(message, node.name)
            LogManager().log(f"Node {node.name} finished processing broadcast: {message}")

    def add_node(self, node):
        self.nodes.append(node)
        LogManager().log(f"Node {node.name} of type {type(node).__name__} added to BroadcastManager")

    def get_node_config(self):
        config = [{
            'id': node.name,
            'type': node.__class__.__name__,
            'receivers': self._get_node_receivers(node),
            'message_type': node.message_type,
            'params': node.get_params(),
            'hasStart': hasattr(node, 'start'),
            'state': getattr(node, 'state', 'stopped')
        } for node in self.nodes]
        LogManager().log(f"Node configuration fetched: {config}")
        return config

    def _get_node_receivers(self, node):
        return [
            {'type': 'implicit', 'value': r} for r in node._implicit_receivers
        ] + [
            {'type': 'explicit', 'value': r} for r in node._explicit_receivers
        ]

    def remove_node(self, node_id):
        self.nodes = [n for n in self.nodes if n.name != node_id]
        LogManager().log(f"Node {node_id} removed from BroadcastManager")

    def add_receiver(self, sender_node, receiver, receiver_type):
        if receiver_type == 'explicit':
            receiver_node = next((n for n in self.nodes if n.name == receiver), None)
            if receiver_node:
                sender_node.add_explicit_receiver(receiver_node.name)
                LogManager().log(f"Explicit receiver {receiver_node.name} added to {sender_node.name}")
        else:  # implicit
            sender_node.add_broadcast_implicit_receiver(receiver)
            LogManager().log(f"Implicit receiver {receiver} added to {sender_node.name}")

    def get_message_types(self):
        types = set()
        for node in self.nodes:
            if hasattr(node, 'message_type') and node.message_type:
                types.add(node.message_type)
            types.update(node._implicit_receivers)
        LogManager().log(f"Fetched unique message types: {types}")
        return types

    def get_node_by_name(self, name):
        node = next((n for n in self.nodes if n.name == name), None)
        LogManager().log(f"Node fetched by name {name}: {node}")
        return node

    def remove_receiver(self, node, receiver):
        node.remove_implicit_receiver(receiver)
        node.remove_explicit_receiver(receiver)
        LogManager().log(f"Receiver {receiver} removed from node {node.name}")

    def get_nodes_status(self):
        current_time = time.time()
        status = [{
            'id': node.name,
            'recent_send': hasattr(node, 'last_send_time') and (current_time - node.last_send_time) <= 1,
            'recent_receive': hasattr(node, 'last_receive_time') and (current_time - node.last_receive_time) <= 1
        } for node in self.nodes]
        LogManager().log(f"Fetched node statuses: {status}")
        return status
