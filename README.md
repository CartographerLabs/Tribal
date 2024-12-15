| :warning: This repository is based on PhD research that seeks to identify radicalisation on online platforms. Due to this; text, themes, and content relating to far-right extremism are present in this repository. Please continue with care. :warning:   <br />  <br />  [Samaritans](http://www.samaritans.org) - Call 116 123 \| [ACT Early](https://www.act.campaign.gov.uk) \| [actearly.uk](https://www.actearly.uk) \| Prevent advice line 0800 011 3764|
| --- |

<p align="center">
  <img width="100%" src="https://github.com/CartographerLabs/Tribal/blob/old/tribal-logo.gif">
</p>
<p align="center">🤖 Social Network Data Pipeline Framework 🌐</p>

## Overview

Tribal Forge is a utility for building and managing data pipelines using interconnected nodes. Each node performs specific tasks like reading data, extracting features, making decisions, and exporting results. Built with Python and Flask, it offers an intuitive web interface for seamless workflow management.

![Tribal Forge Screenshot](tf_demo.gif)

## ✨ Features
Tribal Forge is configured into modular nodes, that can be written for varying tasks. The pre-configured nodes and features include:
-  **CSV and JSON Data Reading**: Nodes to read data from CSV and JSON files.
- **Feature Extraction**: Nodes to extract various features from data, including sentiment, toxicity, lexical diversity, and more.
- **Decision Making**: Nodes to make decisions based on extracted features.
- **Data Output**: Nodes to output data to JSON files or send alerts.

## 📦 Installation

1. Clone the repository:
    ```
    git clone https://github.com/your-repo/tribal-forge.git
    cd tribal-forge
    ```

2. Install the required dependencies:
    ```
   pip install -r requirements.txt
    ```

4. Run the Flask application:
    ```
   python app.py
    ```

## 🚀 Usage

### Adding Nodes

Nodes can be added via the web interface or programmatically. Each node type has specific parameters required for initialization.

Example of adding a CSV Reader Node:
```
csv_node = CsvReaderNode(broadcast_manager, "path/to/your/csvfile.csv")
csv_node.set_message_type("post")
broadcast_manager.add_node(csv_node)
```

### Connecting Nodes

Nodes can be connected to send and receive messages. Connections can be implicit (based on message types) or explicit (direct connections).

Example of connecting nodes:
```
feature_extractor_node.add_broadcast_implicit_receiver("post")
json_out_node.add_explicit_receiver(feature_extractor_node.name)
```

### Starting Nodes

Nodes can be started to begin processing data.
```
csv_node.start()
```

### 🌐 Web Interface

The web interface provides endpoints to manage nodes and their connections. Some of the available endpoints include:

- ```/get_node_types```: Get available node types.
- ```/add_node```: Add a new node.
- ```/add_receiver```: Add a receiver to a node.
- ```/remove_receiver```: Remove a receiver from a node.
- ```/delete_node```: Delete a node.
- ```/start_node```: Start a node.
- ```/get_message_types```: Get all unique message types.

## 📝 Example

Here is an example of setting up nodes and starting the CSV Reader Node:
```
from tribal.forge.nodes import *
from tribal.forge.managers import BroadcastManager

broadcast_manager = BroadcastManager()

csv_node = CsvReaderNode(broadcast_manager, "test_posts.csv")
csv_node.set_message_type("post")

feature_extractor_node = FeatureExtractorNode(broadcast_manager)
feature_extractor_node.add_broadcast_implicit_receiver("post")
feature_extractor_node.set_message_type("feature_extracted_posts")

json_out_node = JsonOutNode(broadcast_manager, "out.json")
json_out_node.add_explicit_receiver(feature_extractor_node.name)

csv_node.start()
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
