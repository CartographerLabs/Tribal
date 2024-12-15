from flask import Flask, jsonify, request
from tribal.forge.nodes import __node_types__, CsvReaderNode, JsonReaderNode, FeatureExtractorNode, JsonOutNode, DecisionNode, AlertOutNode
from tribal.forge.managers import BroadcastManager
import inspect
from tribal.forge.managers.log_manager import log_manager, LogManager
from tribal.forge.managers.alert_manager import alert_manager, AlertManager
import time
import os
import json

app = Flask(__name__)

# Initialise broadcast manager as in the test file
broadcast_manager = BroadcastManager()

# Keep track of nodes and their properties
data = {
    "nodes": {},  # Will now store actual node instances too
    "connections": [],
    "message_types": set()
}

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(TEMPLATE_DIR, exist_ok=True)

def get_available_node_types():
    return __node_types__

@app.route('/get_node_types', methods=['GET'])
def get_node_types():
    return jsonify({"node_types": get_available_node_types()})

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tribal Forge - Node Manager</title>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #1a73e8;
            --background-color: #f8f9fa;
            --card-background: #ffffff;
            --text-color: #202124;
            --secondary-text: #5f6368;
            --border-color: #dadce0;
            --hover-color: #f1f3f4;
            --console-bg: #202124;
            --console-text: #e8eaed;
        }

        body {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Roboto', sans-serif;
        }

        /* Modern Sidebar */
        .sidenav {
            width: 250px;
            background-color: var(--card-background);
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }

        .sidenav .brand-logo {
            padding: 20px 15px;
            font-size: 20px;
            color: var(--primary-color);
            border-bottom: 1px solid var(--border-color);
        }

        /* Main Content */
        .main-content {
            margin-left: 250px;
            padding: 20px;
        }

        /* Modern Cards */
        .node-card {
            border-radius: 8px;
            box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
            transition: box-shadow 0.3s;
            background-color: var(--card-background);
            margin: 16px 0;
        }

        .node-card:hover {
            box-shadow: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
        }

        .card .card-title {
            font-size: 16px;
            font-weight: 500;
            color: var(--text-color);
        }

        /* Modern Console */
        .console-wrapper {
            border-radius: 8px 8px 0 0;
            background-color: var(--console-bg);
            color: var(--console-text);
            box-shadow: 0 -2px 10px rgba(0,0,0,0.2);
        }

        .console-header {
            background-color: #2d2d2d;
            border-radius: 8px 8px 0 0;
        }

        .console-content {
            background-color: var(--console-bg);
            color: var(--console-text);
            font-family: 'Roboto Mono', monospace;
        }

        /* Modern Form Elements */
        .input-field input[type=text]:focus {
            border-bottom: 2px solid var(--primary-color) !important;
            box-shadow: none !important;
        }

        .btn {
            background-color: var(--primary-color);
            border-radius: 4px;
            text-transform: none;
            font-weight: 500;
            box-shadow: none;
        }

        .btn:hover {
            background-color: #1557b0;
            box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3);
        }

        /* Chip Styling */
        .chip {
            background-color: var(--hover-color);
            border-radius: 16px;
            font-size: 13px;
        }

        /* Status Indicators */
        .node-status i {
            padding: 4px;
            border-radius: 50%;
            transition: background-color 0.3s;
        }

        .node-status i[title="Sending"].active,
        .node-status i[title="Receiving"].active {
            color: var(--primary-color);
            background-color: rgba(26,115,232,0.1);
        }

        /* Modal Styling */
        .modal {
            border-radius: 8px;
            background-color: var(--card-background);
        }

        .modal .modal-footer {
            background-color: var(--card-background);
        }
        
        /* Fixed Console Styling */
        .console-wrapper {
            position: fixed;
            bottom: 0;
            left: 250px;  /* Match sidenav width */
            right: 0;
            height: 300px;
            background: var(--console-bg);
            z-index: 1000;
            transform: translateY(260px);
            transition: transform 0.3s ease;
            border-radius: 8px 8px 0 0;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.2);
        }

        .console-wrapper:hover {
            transform: translateY(0);
        }

        .console-header {
            padding: 8px 16px;
            background: #2d2d2d;
            border-radius: 8px 8px 0 0;
            cursor: grab;
        }

        .console-content {
            height: calc(100% - 40px);
            overflow-y: auto;
            padding: 12px;
            font-family: 'Roboto Mono', monospace;
            font-size: 12px;
            line-height: 1.5;
            color: var(--console-text);
        }

        .log-entry {
            color: #4CAF50;
            margin: 4px 0;
        }

        .alert-entry {
            color: #f44336;
            margin: 4px 0;
        }

        /* Section Styling */
        .section {
            display: none;
            padding: 20px;
        }

        .section.active {
            display: block;
        }

        /* Active Navigation Item */
        .sidenav .active {
            background-color: rgba(26,115,232,0.1);
        }
        
        .sidenav a {
            color: var(--text-color) !important;
        }
        
        .sidenav a.active {
            color: var(--primary-color) !important;
        }

        /* Console Tab Styling */
        .console-tabs {
            display: flex;
            padding: 0 10px;
        }

        .console-tab {
            padding: 8px 16px;
            cursor: pointer;
            color: #9e9e9e;
            border-bottom: 2px solid transparent;
            margin: 0 4px;
        }

        .console-tab.active {
            color: var(--primary-color);
            border-bottom: 2px solid var(--primary-color);
        }

        .console-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0;
            height: 40px;
        }

        .console-controls {
            padding: 0 16px;
        }

        .console-controls i {
            cursor: pointer;
            opacity: 0.7;
        }

        .console-controls i:hover {
            opacity: 1;
        }

        .console-content {
            display: none;
        }

        .console-content.active {
            display: block;
        }
        
        /* Template List Styling */
        #template-list .template-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 16px;
        }

        #template-list .template-item a {
            display: flex;
            align-items: center;
            color: var(--secondary-text) !important;
            flex-grow: 1;
            white-space: nowrap;  // Add this line
            overflow: hidden;  // Add this line
            text-overflow: ellipsis;  // Add this line
        }

        #template-list .template-item a i {
            color: var(--secondary-text) !important;
            margin-right: 8px;
            margin-left: 0;  // Add this line to align the icons to the left
        }

        #template-list .template-item .delete-icon {
            color: var(--secondary-text) !important;
            margin-left: auto;
        }

        #template-list .template-item a:hover .delete-icon {
            color: var(--primary-color) !important;
        }

        .sidenav .template-item a i {
            margin-right: 16px;
            margin-left: 0;  // Add this line to align the icons to the left
        }
    </style>
</head>
<body>
    <!-- Sidebar Navigation -->
    <ul id="slide-out" class="sidenav sidenav-fixed">
        <div class="brand-logo">
            <i class="material-icons">device_hub</i>
            Tribal Forge
        </div>
        <li><a href="#!" class="waves-effect active" onclick="showSection('nodes')"><i class="material-icons">widgets</i>Nodes</a></li>
        <li><a href="#!" class="waves-effect" onclick="showSection('console')"><i class="material-icons">terminal</i>Console</a></li>
        <li class="divider"></li>
        <li><div class="subheader">Templates</div></li>
        <div id="template-list">
            <!-- Templates will be loaded here -->
        </div>
        <li><a href="#!" class="waves-effect" onclick="saveTemplate()"><i class="material-icons">save</i>Save as Template</a></li>
    </ul>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Nodes Section -->
        <div id="nodes-section" class="section active">
            <div class="row">
                <div class="col s12">
                    <div class="card">
                        <div class="card-content">
                            <span class="card-title">Add New Node</span>
                            <form id="add-node-form">
                                <div class="row">
                                    <div class="input-field col s6">
                                        <input id="node_name" type="text" required>
                                        <label for="node_name">Node Name</label>
                                    </div>
                                    <div class="input-field col s6">
                                        <select id="node_type" required>
                                            <option value="" disabled selected>Choose Node Type</option>
                                        </select>
                                        <label>Node Type</label>
                                    </div>
                                </div>
                                <div id="dynamic-fields" class="row"></div>
                                <button class="btn waves-effect waves-light" type="submit">
                                    Create Node
                                    <i class="material-icons right">add_circle</i>
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row" id="nodes-container">
                <!-- Nodes will be dynamically inserted here -->
            </div>
        </div>

        <!-- Console Section -->
        <div id="console-section" class="section">
            <h5>Console Output</h5>
            <p class="grey-text">Live logs and alerts will appear here</p>
        </div>
    </div>

    <!-- Enable notifications button -->
    <div id="notification-prompt" style="display: none; position: fixed; bottom: 20px; right: 20px; padding: 15px; background: var(--primary-color); color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        Would you like to enable notifications?
        <button onclick="requestNotificationPermission()" class="btn-flat waves-effect waves-light white-text">Enable</button>
        <button onclick="closeNotificationPrompt()" class="btn-flat waves-effect waves-light white-text">No thanks</button>
    </div>

    <!-- Modal for adding receivers -->
    <div id="receiver-modal" class="modal">
        <div class="modal-content">
            <h4>Add Receiver</h4>
            <div class="row">
                <div class="input-field col s12">
                    <select id="receiver-type" onchange="updateReceiverOptions()">
                        <option value="explicit">Explicit (Node)</option>
                        <option value="implicit">Implicit (Message Type)</option>
                    </select>
                    <label>Receiver Type</label>
                </div>
                <div class="input-field col s12">
                    <select id="receiver-name">
                        <!-- Options will be populated dynamically -->
                    </select>
                    <label>Select Receiver</label>
                </div>
            </div>
        </div>
        <div class="modal-footer">
            <a href="#!" class="modal-close waves-effect waves-red btn-flat">Cancel</a>
            <a href="#!" id="save-receiver" class="waves-effect waves-green btn">Add</a>
        </div>
    </div>

    <div class="console-wrapper" id="consoleWrapper">
        <div class="console-header" id="consoleHeader">
            <div class="console-tabs">
                <div class="console-tab active" onclick="switchConsole('logs')">Logs</div>
                <div class="console-tab" onclick="switchConsole('alerts')">Alerts</div>
            </div>
            <div class="console-controls">
                <i class="material-icons" onclick="clearConsole()" title="Clear current console">delete_sweep</i>
                <span class="drag-handle">⋮⋮</span>
            </div>
        </div>
        <div id="logs" class="console-content active"></div>
        <div id="alerts" class="console-content"></div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
    <script>
        let currentNodeName = '';
        let notificationCounter = 0;
        let windowFocused = true;

        document.addEventListener('DOMContentLoaded', function() {
            M.AutoInit();
            refreshNodes();
            loadNodeTypes();
            initializeConsole();
            checkNotificationPermission();
            loadTemplateList();  // Add this line
        });

        // Add window focus tracking
        window.addEventListener('focus', () => {
            windowFocused = true;
            notificationCounter = 0;
        });
        
        window.addEventListener('blur', () => {
            windowFocused = false;
        });

        function checkNotificationPermission() {
            if (!('Notification' in window)) {
                return;
            }

            if (Notification.permission === 'default') {
                document.getElementById('notification-prompt').style.display = 'block';
            }
        }

        function closeNotificationPrompt() {
            document.getElementById('notification-prompt').style.display = 'none';
        }

        async function requestNotificationPermission() {
            try {
                const permission = await Notification.requestPermission();
                if (permission === 'granted') {
                    M.toast({html: 'Notifications enabled!'});
                }
            } catch (err) {
                console.error('Error requesting notification permission:', err);
            }
            closeNotificationPrompt();
        }

        function showNotification(message) {
            if (!windowFocused && Notification.permission === 'granted') {
                notificationCounter++;
                const notification = new Notification('Tribal Forge Alerts', {
                    body: `You have ${notificationCounter} unread alert${notificationCounter > 1 ? 's' : ''}`,
                    icon: '/favicon.ico',
                    badge: '/favicon.ico',
                    tag: 'tribal-forge-alert',
                    renotify: true
                });

                notification.onclick = function() {
                    window.focus();
                    showSection('console');
                    switchConsole('alerts');
                    notification.close();
                };
            }
        }

        function initializeConsole() {
            const logSource = new EventSource('/stream_logs');
            const alertSource = new EventSource('/stream_alerts');
            
            logSource.onmessage = function(event) {
                const logsDiv = document.getElementById('logs');
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.textContent = event.data;
                logsDiv.appendChild(entry);
                logsDiv.scrollTop = logsDiv.scrollHeight;
            };
            
            alertSource.addEventListener('message', function(event) {
                // Parse the alert data
                const alertData = event.data.trim();
                if (!alertData) return;
                
                // Update the alerts console
                const alertsDiv = document.getElementById('alerts');
                const entry = document.createElement('div');
                entry.className = 'alert-entry';
                entry.textContent = alertData;
                alertsDiv.appendChild(entry);
                alertsDiv.scrollTop = alertsDiv.scrollHeight;
                
                // Show desktop notification
                if (!document.hasFocus()) {
                    showNotification();  // No need to pass message since we're just showing count
                }
            });
        }

        function switchConsole(type) {
            // Update content visibility
            document.querySelectorAll('.console-content').forEach(el => {
                el.classList.remove('active');
            });
            document.getElementById(type).classList.add('active');
            
            // Update tab styling
            document.querySelectorAll('.console-tab').forEach(el => {
                el.classList.remove('active');
            });
            document.querySelector(`.console-tab[onclick*="${type}"]`).classList.add('active');
        }

        function clearConsole() {
            const activeTab = document.querySelector('.console-tab.active').textContent.toLowerCase();
            document.getElementById(activeTab).innerHTML = '';
            
            // Also clear the file on the server
            fetch(`/clear_${activeTab}`, {
                method: 'POST'
            }).then(response => {
                if (!response.ok) {
                    M.toast({html: 'Failed to clear console'});
                }
            });
        }

        function loadNodeTypes() {
            $.get('/get_node_types', function(response) {
                const select = $('#node_type');
                response.node_types.forEach(type => {
                    select.append(`<option value="${type}">${type}</option>`);
                });
                select.formSelect();
            });
        }

        function refreshNodes() {
            $.get('/get_nodes', function(response) {
                const container = $('#nodes-container');
                container.empty();
                
                response.nodes.forEach(node => {
                    container.append(createNodeCard(node));
                });
                
                M.Tooltip.init(document.querySelectorAll('.tooltipped'));
            });
        }

        function createNodeCard(node) {
            return `
                <div class="col s12 m6 l4">
                    <div class="card node-card">
                        <div class="card-content">
                            <div style="display: flex; justify-content: space-between; align-items: center">
                                <span class="card-title">${node.name}</span>
                                <div class="node-status" data-node="${node.name}">
                                    <i class="material-icons tiny" title="Sending">call_made</i>
                                    <i class="material-icons tiny" title="Receiving">call_received</i>
                                </div>
                            </div>
                            <p><i class="material-icons tiny">category</i> ${node.type}</p>
                            <p><i class="material-icons tiny">label</i> ${node.message_type || 'No message type'}</p>
                            <div class="chip-container">
                                ${node.receivers.map(r => `
                                    <div class="chip">
                                        ${r}
                                        <i class="close material-icons" onclick="removeReceiver('${node.name}', '${r}')">close</i>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        <div class="card-action">
                            <a href="#!" onclick="openAddReceiverModal('${node.name}')">
                                Add Receiver
                            </a>
                            <a href="#!" onclick="setMessageType('${node.name}')">
                                Set Message Type
                            </a>
                            <a href="#!" class="red-text" onclick="deleteNode('${node.name}')">
                                Delete Node
                            </a>
                            ${node.has_start ? `<a href="#!" class="green-text" onclick="startNode('${node.name}')">Start Node</a>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }

        function setCurrentNode(nodeName) {
            currentNodeName = nodeName;
        }

        $('#save-receiver').click(function() {
            const type = $('#receiver-type').val();
            const name = $('#receiver-name').val();
            
            if (!name) {
                M.toast({html: 'Please select a receiver'});
                return;
            }

            $.ajax({
                url: '/add_receiver',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    sender: currentNodeName,
                    receiver: name,
                    type: type
                }),
                success: function(response) {
                    M.toast({html: response.message});
                    refreshNodes();
                    $('#receiver-modal').modal('close');
                },
                error: function(xhr) {
                    M.toast({html: xhr.responseJSON.error});
                }
            });
        });

        function setMessageType(nodeName) {
            const messageType = prompt('Enter message type:');
            if (!messageType) return;

            $.ajax({
                url: '/set_message_type',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    node: nodeName,
                    message_type: messageType
                }),
                success: function(response) {
                    M.toast({html: response.message});
                    refreshNodes();
                },
                error: function(xhr) {
                    M.toast({html: xhr.responseJSON.error});
                }
            });
        }

        $("#node_type").change(function() {
            const selectedType = $(this).val();
            if (selectedType) {
                $.get(`/get_node_arguments?type=${selectedType}`, function(data) {
                    const dynamicFields = $("#dynamic-fields");
                    dynamicFields.empty();
                    data.arguments.forEach(arg => {
                        dynamicFields.append(`
                            <div class="input-field col s12 m6">
                                <input id="${arg}" type="text" required>
                                <label for="${arg}">${arg}</label>
                            </div>
                        `);
                    });
                });
            }
        });

        $("#add-node-form").submit(function(event) {
            event.preventDefault();
            const formData = {
                name: $("#node_name").val(),
                type: $("#node_type").val(),
                args: {}
            };

            $("#dynamic-fields input").each(function() {
                formData.args[$(this).attr("id")] = $(this).val();
            });

            $.ajax({
                url: "/add_node",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(formData),
                success: function(response) {
                    M.toast({html: response.message});
                    refreshNodes();
                    $("#add-node-form")[0].reset();
                    $("#dynamic-fields").empty();
                },
                error: function(xhr) {
                    M.toast({html: xhr.responseJSON.error});
                }
            });
        });

        function updateReceiverOptions() {
            const receiverType = $('#receiver-type').val();
            const receiverSelect = $('#receiver-name');
            
            $.get(`/get_available_receivers?type=${receiverType}`, function(data) {
                receiverSelect.empty();
                receiverSelect.append('<option value="" disabled selected>Choose receiver</option>');
                
                if (receiverType === 'explicit') {
                    data.receivers.forEach(node => {
                        receiverSelect.append(
                            `<option value="${node.name}">${node.name} (${node.type})</option>`
                        );
                    });
                } else {
                    data.receivers.forEach(msgType => {
                        receiverSelect.append(
                            `<option value="${msgType}">${msgType}</option>`
                        );
                    });
                }
                receiverSelect.formSelect();
            });
        }

        function openAddReceiverModal(nodeName) {
            currentNodeName = nodeName;
            const modal = $('#receiver-modal');
            updateReceiverOptions();
            modal.modal('open');
        }

        // Add a new function to remove receivers
        function removeReceiver(nodeName, receiverName) {
            $.ajax({
                url: '/remove_receiver',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    sender: nodeName,
                    receiver: receiverName
                }),
                success: function(response) {
                    M.toast({html: response.message});
                    refreshNodes();
                },
                error: function(xhr) {
                    M.toast({html: xhr.responseJSON.error});
                }
            });
        }

        function deleteNode(nodeName) {
            if (confirm('Are you sure you want to delete this node?')) {
                $.ajax({
                    url: '/delete_node',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({
                        node: nodeName
                    }),
                    success: function(response) {
                        M.toast({html: response.message});
                        refreshNodes();
                    },
                    error: function(xhr) {
                        M.toast({html: xhr.responseJSON.error});
                    }
                });
            }
        }

        function startNode(nodeName) {
            $.ajax({
                url: '/start_node',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    node: nodeName
                }),
                success: function(response) {
                    M.toast({html: response.message});
                    refreshNodes();
                },
                error: function(xhr) {
                    M.toast({html: xhr.responseJSON.error});
                }
            });
        }

        function updateNodeStatuses() {
            $.get('/node_status', function(response) {
                response.statuses.forEach(status => {
                    const statusDiv = $(`.node-status[data-node="${status.id}"]`);
                    statusDiv.find('i[title="Sending"]').css('color', status.recent_send ? '#2196F3' : 'grey');
                    statusDiv.find('i[title="Receiving"]').css('color', status.recent_receive ? '#2196F3' : 'grey');
                });
            });
        }

        // Add status polling
        setInterval(updateNodeStatuses, 100);
        
        function showSection(sectionId) {
            // Hide all sections
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Remove active class from nav items
            document.querySelectorAll('.sidenav a').forEach(item => {
                item.classList.remove('active');
            });
            
            // Show selected section
            document.getElementById(`${sectionId}-section`).classList.add('active');
            
            // Add active class to nav item
            document.querySelector(`.sidenav a[onclick*="${sectionId}"]`).classList.add('active');
            
            // Adjust console visibility
            const consoleWrapper = document.getElementById('consoleWrapper');
            if (sectionId === 'console') {
                consoleWrapper.style.transform = 'translateY(0)';
                consoleWrapper.style.height = 'calc(100vh - 64px)';
            } else {
                consoleWrapper.style.transform = 'translateY(260px)';
                consoleWrapper.style.height = '300px';
            }
        }

        function loadTemplateList() {
            fetch('/get_templates')
                .then(response => response.json())
                .then(data => {
                    const templateList = document.getElementById('template-list');
                    templateList.innerHTML = '';
                    
                    data.templates.forEach(template => {
                        const li = document.createElement('li');
                        li.className = 'template-item';
                        li.innerHTML = `
                            <a href="#!" class="waves-effect" onclick="loadTemplate('${template}')">
                                <i class="material-icons">description</i>${template.replace('.json', '')}
                            </a>
                            <a href="#!" class="waves-effect delete-icon" onclick="deleteTemplate('${template}')">
                                <i class="material-icons">close</i>
                            </a>
                        `;
                        templateList.appendChild(li);
                    });
                });
        }

        function deleteTemplate(name) {
            if (!confirm('Are you sure you want to delete this template?')) {
                return;
            }
            
            fetch('/delete_template', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name: name })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    M.toast({html: data.error});
                } else {
                    M.toast({html: data.message});
                    loadTemplateList();
                }
            });
        }

        function saveTemplate() {
            const name = prompt('Enter template name:');
            if (!name) return;
            
            fetch('/save_template', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name: name })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    M.toast({html: data.error});
                } else {
                    M.toast({html: data.message});
                    loadTemplateList();
                }
            });
        }

        function loadTemplate(name) {
            if (!confirm('Loading a template will replace your current configuration. Continue?')) {
                return;
            }
            
            fetch('/load_template', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name: name })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    M.toast({html: data.error});
                } else {
                    M.toast({html: data.message});
                    refreshNodes();
                }
            });
        }
    </script>
</body>
</html>"""

def sanitize_name(name):
    return name.replace(' ', '_')

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/get_node_arguments', methods=['GET'])
def get_node_arguments():
    node_type = request.args.get('type')
    node_class = globals().get(node_type)
    if not node_class:
        return jsonify({"arguments": []})

    # Only look at the constructor parameters
    constructor = inspect.signature(node_class.__init__)
    # Skip self and broadcast_manager, but keep node-specific parameters
    arguments = list(constructor.parameters.keys())[2:]
    # Remove llm if it exists in the arguments
    if 'llm' in arguments:
        arguments.remove('llm')

    return jsonify({"arguments": arguments})

@app.route('/get_nodes', methods=['GET'])
def get_nodes():
    nodes = [
        {
            "name": name,
            "type": details["type"],
            "receivers": details.get("receivers", []),
            "message_type": details.get("message_type"),
            "has_start": details.get("has_start", False),
            "state": details.get("state", "stopped")
        }
        for name, details in data['nodes'].items()
    ]
    return jsonify({"nodes": nodes})

@app.route('/add_node', methods=['POST'])
def add_node():
    node_data = request.json
    node_type = node_data['type']
    node_name = sanitize_name(node_data['name'])
    args = node_data['args']

    node_class = globals().get(node_type)
    if not node_class:
        return jsonify({"error": "Invalid node type"}), 400

    try:
        node = node_class(broadcast_manager, **args)
        node.name = node_name
        broadcast_manager.add_node(node)  # Register node with broadcast manager
        
        data['nodes'][node_name] = {
            "type": node_type,
            "details": node_data,
            "receivers": [],
            "message_type": None,
            "instance": node,
            "has_start": hasattr(node, 'start'),
            "state": "stopped"
        }
        
        return jsonify({"message": "Node added successfully", "node": node_name})
    except Exception as e:
        return jsonify({"error": f"Failed to initialize node: {str(e)}"}), 400

@app.route('/add_receiver', methods=['POST'])
def add_receiver():
    connection_data = request.json
    sender_name = connection_data['sender']
    receiver = connection_data['receiver']
    receiver_type = connection_data['type']

    if sender_name not in data['nodes']:
        return jsonify({"error": "Invalid sender node"}), 400

    sender_node = data['nodes'][sender_name]['instance']
    broadcast_manager.add_receiver(sender_node, receiver, receiver_type)

    # Update the UI tracking
    if receiver not in data['nodes'][sender_name]["receivers"]:
        data['nodes'][sender_name]["receivers"].append(receiver)

    return jsonify({"message": "Receiver added successfully"})

@app.route('/get_available_receivers', methods=['GET'])
def get_available_receivers():
    receiver_type = request.args.get('type')
    if (receiver_type == 'explicit'):
        return jsonify({
            "receivers": [
                {"name": name, "type": details["type"]} 
                for name, details in data['nodes'].items()
            ]
        })
    else:  # implicit
        return jsonify({
            "receivers": list(broadcast_manager.get_message_types())
        })

@app.route('/set_message_type', methods=['POST'])
def set_message_type():
    msg_data = request.json
    node_name = msg_data['node']
    message_type = sanitize_name(msg_data['message_type'])

    if node_name not in data['nodes']:
        return jsonify({"error": "Node not found"}), 400

    node = data['nodes'][node_name]['instance']
    node.set_message_type(message_type)
    data['nodes'][node_name]['message_type'] = message_type
    
    return jsonify({"message": "Message type set successfully"})

@app.route('/get_message_types', methods=['GET'])
def get_message_types():
    return jsonify({"message_types": list(data['message_types'])})

@app.route('/remove_receiver', methods=['POST'])
def remove_receiver():
    req_data = request.json
    sender_name = req_data['sender']
    receiver = req_data['receiver']

    if sender_name not in data['nodes']:
        return jsonify({"error": "Invalid sender node"}), 400

    sender = data['nodes'][sender_name]
    if receiver in sender["receivers"]:
        # Remove from UI tracking
        sender["receivers"].remove(receiver)
        
        # Remove from broadcast manager
        node = sender['instance']
        broadcast_manager.remove_receiver(node, receiver)
            
        return jsonify({"message": "Receiver removed successfully"})
    
    return jsonify({"error": "Receiver not found"}), 400

@app.route('/delete_node', methods=['POST'])
def delete_node():
    node_name = request.json.get('node')
    
    if node_name not in data['nodes']:
        return jsonify({"error": "Node not found"}), 400

    # Remove node from broadcast manager
    node = data['nodes'][node_name]['instance']
    broadcast_manager.remove_node(node.name)

    # Remove from UI tracking
    for node_data in data['nodes'].values():
        if node_name in node_data["receivers"]:
            node_data["receivers"].remove(node_name)

    del data['nodes'][node_name]
    return jsonify({"message": "Node deleted successfully"})

@app.route('/start_node', methods=['POST'])
def start_node():
    node_name = request.json.get('node')
    
    if (node_name not in data['nodes']):
        return jsonify({"error": "Node not found"}), 400

    node = data['nodes'][node_name]['instance']
    try:
        node.start()
        return jsonify({"message": "Node started successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to start node: {str(e)}"}), 400

@app.route('/node_status')
def node_status():
    return jsonify({"statuses": broadcast_manager.get_nodes_status()})

@app.route('/stream_logs')
def stream_logs():
    def generate():
        with open(LogManager.LOG_FILE) as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    time.sleep(0.1)
    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/stream_alerts')
def stream_alerts():
    def generate():
        with open(AlertManager.ALERT_FILE) as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    time.sleep(0.1)
    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    try:
        open(LogManager.LOG_FILE, 'w').close()
        return '', 204
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_alerts', methods=['POST'])
def clear_alerts():
    try:
        open(AlertManager.ALERT_FILE, 'w').close()
        return '', 204
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_template', methods=['POST'])
def save_template():
    template_data = request.json
    template_name = template_data.get('name')
    if not template_name.endswith('.json'):
        template_name += '.json'
    
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    
    # Save current configuration
    template = {
        'nodes': {},
        'connections': []
    }
    
    # Save node configurations with receiver types and all necessary data
    for name, node_data in data['nodes'].items():
        receivers_info = []
        for receiver in node_data['receivers']:
            # Check if receiver is a node name (explicit) or message type (implicit)
            receiver_type = 'explicit' if receiver in data['nodes'] else 'implicit'
            receivers_info.append({
                'name': receiver,
                'type': receiver_type
            })
        
        # Ensure args exists and is properly structured
        args = node_data['details'].get('args', {})
        if not isinstance(args, dict):
            args = {}
        
        template['nodes'][name] = {
            'type': node_data['type'],
            'args': args,  # Save cleaned args
            'receivers': receivers_info,
            'message_type': node_data['message_type']
        }
    
    try:
        os.makedirs(TEMPLATE_DIR, exist_ok=True)  # Ensure directory exists
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        return jsonify({"message": "Template saved successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to save template: {str(e)}"}), 500

@app.route('/get_templates', methods=['GET'])
def get_templates():
    try:
        # Ensure directory exists
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        templates = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.json')]
        return jsonify({"templates": templates})
    except Exception as e:
        return jsonify({"error": f"Failed to list templates: {str(e)}"}), 500

@app.route('/load_template', methods=['POST'])
def load_template():
    template_name = request.json.get('name')
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    
    try:
        with open(template_path, 'r') as f:
            template = json.load(f)
            
        # Clear current configuration
        for node_name in list(data['nodes'].keys()):
            node = data['nodes'][node_name]['instance']
            broadcast_manager.remove_node(node.name)
        data['nodes'].clear()
        
        # Load template configuration
        for name, node_data in template['nodes'].items():
            # Create node
            node_type = node_data['type']
            node_args = node_data['args'].copy()  # Make a copy to avoid modifying original
            node_class = globals().get(node_type)
            
            if node_class:
                # Only add llm=None if the node class accepts it
                sig = inspect.signature(node_class.__init__)
                if 'llm' in sig.parameters:
                    node_args['llm'] = None
                
                # Create node instance
                node = node_class(broadcast_manager, **node_args)
                node.name = name
                broadcast_manager.add_node(node)
                
                # Store node data
                data['nodes'][name] = {
                    "type": node_type,
                    "details": {"args": node_args},
                    "receivers": [],
                    "message_type": node_data['message_type'],
                    "instance": node,
                    "has_start": hasattr(node, 'start'),
                    "state": "stopped"
                }
                
                # Set message type if exists
                if node_data['message_type']:
                    node.set_message_type(node_data['message_type'])
                
                # Add receivers with their types
                for receiver_info in node_data['receivers']:
                    receiver_name = receiver_info['name']
                    receiver_type = receiver_info['type']
                    try:
                        broadcast_manager.add_receiver(node, receiver_name, receiver_type)
                        data['nodes'][name]["receivers"].append(receiver_name)
                    except Exception as e:
                        print(f"Warning: Could not add receiver {receiver_name} to node {name}: {str(e)}")
                    
        return jsonify({"message": "Template loaded successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to load template: {str(e)}"}), 500

@app.route('/delete_template', methods=['POST'])
def delete_template():
    template_name = request.json.get('name')
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    
    try:
        if os.path.exists(template_path):
            os.remove(template_path)
            return jsonify({"message": "Template deleted successfully"})
        return jsonify({"error": "Template not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to delete template: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)