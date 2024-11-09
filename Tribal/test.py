import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
import networkx as nx
from objects.user import UserObject
from pprint import pprint 
from utils.feature_extractor import FeatureExtractor
from utils.config_manager import ConfigManager
from data_set_managers.json_dataset_manager import JsonDatasetManager
from objects.timeline import TimelineObject
import json 
import pickle
from datetime import datetime, timedelta
from torch_geometric.utils import from_networkx
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Example Data Loading (You would replace this part with your actual data loading)
# Load the dataset and extract the user features from the dataset
# Here we assume features like centrality, sentiment, toxicity, theme, and operational status are present in the dataset

print("Loading telegram post data...")
telegram_post_data = JsonDatasetManager(str(ConfigManager().location_of_post_data))

print("Loading ground truth extremist post data...")
ground_truth_extremist_post_data = JsonDatasetManager(ConfigManager().location_of_ground_truth_extremist_post_data)

# Initialize feature extractor
print("Initializing FeatureExtractor...")
feature_extractor = FeatureExtractor(telegram_post_data.get_list_of_posts(), ground_truth_extremist_post_data.get_list_of_posts())

users = telegram_post_data.get_all_user_data()

# Build the graph based on users' connections and features
graph = nx.Graph()

# Initialize empty mappings for themes
theme_to_num = {}
num_to_theme = {}
current_index = 0

node_labels = []  # Placeholder for node labels
node_features = []  # Placeholder for feature vectors

CHOSEN_OVERALL_WINDOW_START, CHOSEN_OVERALL_WINDOW_END = (
    telegram_post_data.get_timeframe()
)
posts = telegram_post_data.get_all_user_data(
    start_time=CHOSEN_OVERALL_WINDOW_START, end_time=CHOSEN_OVERALL_WINDOW_END
)

timeline = TimelineObject(feature_extractor)
timeline.posts = posts
start, end = timeline.get_current_start_and_end_dates()
timeline.make_new_window(CHOSEN_OVERALL_WINDOW_START, CHOSEN_OVERALL_WINDOW_END)
window = timeline.windows[0]
conversations = window.conversations
# Process each user and combine their profile data into a unified feature vector
for post in timeline.posts:
    user_data = post.get_dict()
    # Extract features from the dataset (e.g., centrality, sentiment, toxicity)
    centrality = user_data["centrality"]
    avrg_sentiment = user_data["avrg_sentiment"]
    avrg_toxicity = user_data["avrg_toxicity"]

    # Convert boolean to integer for 'is_operational'
    avrg_is_operational_num = int(user_data["avrg_is_operational"])

    # Convert theme to a number using dynamic mapping
    theme = user_data["avrg_theme"]
    if theme not in theme_to_num:
        theme_to_num[theme] = current_index
        num_to_theme[current_index] = theme
        current_index += 1

    avrg_theme_num = theme_to_num[theme]

    # Combine all features into a feature vector
    feature_vector = [centrality, avrg_sentiment, avrg_toxicity, avrg_is_operational_num, avrg_theme_num]
    node_features.append(feature_vector)

    # Append user label (e.g., extremist = 1, non-extremist = 0)
    node_labels.append(user_data["label"])

    # Add node to the graph with profile (features)
    graph.add_node(user_id, profile=torch.tensor(feature_vector, dtype=torch.float))

# Add edges to the graph (representing user connections)
# You need to implement this based on your dataset; here is a placeholder example:
for user_id, user_data in users.items():
    connections = user_data.get('connections', [])
    for connected_user in connections:
        graph.add_edge(user_id, connected_user)

# Convert NetworkX graph to PyTorch Geometric format
data = from_networkx(graph)
data.x = torch.stack([graph.nodes[node]['profile'] for node in graph.nodes])  # Node features
data.y = torch.tensor(node_labels, dtype=torch.long)  # Labels for each node (extremist or non-extremist)

# Now that the data is ready, we will split it into train, validation, and test sets
train_mask, val_test_mask = train_test_split(range(data.num_nodes), test_size=0.4, random_state=42)
val_mask, test_mask = train_test_split(val_test_mask, test_size=0.5, random_state=42)

# Create mask tensors for train, validation, and test sets
train_mask = torch.tensor([i in train_mask for i in range(data.num_nodes)], dtype=torch.bool)
val_mask = torch.tensor([i in val_mask for i in range(data.num_nodes)], dtype=torch.bool)
test_mask = torch.tensor([i in test_mask for i in range(data.num_nodes)], dtype=torch.bool)

# Define a GCN model for node classification
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the GCN model, optimizer, and loss function
model = GCN(in_channels=data.x.shape[1], hidden_channels=16, out_channels=2)  # Assuming binary classification (extremist vs non-extremist)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Function to evaluate the model on a specific mask (train, validation, or test)
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data)  # Forward pass
        pred = out.argmax(dim=1)  # Get predicted classes
        correct = pred[mask].eq(data.y[mask]).sum().item()  # Count correct predictions
        accuracy = correct / mask.sum().item()  # Compute accuracy
    return accuracy

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)
    
    # Compute loss only for training nodes
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # Every 10 epochs, evaluate on validation set
    if epoch % 10 == 0:
        train_acc = evaluate(train_mask)
        val_acc = evaluate(val_mask)
        print(f'Epoch {epoch}, Loss: {loss.item()}, Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}')

# After training, evaluate on the test set
test_acc = evaluate(test_mask)
print(f'Test Accuracy: {test_acc}')

# Function to predict the effect of adding a user (using GCN)
def predict_add_user_gnn(new_user_id, new_user_features, connections, graph, model):
    print(f"Predicting changes by adding user {new_user_id}...")

    # Add the new user node and connections
    graph.add_node(new_user_id, profile=torch.tensor(new_user_features, dtype=torch.float))
    for connection in connections:
        graph.add_edge(new_user_id, connection)

    # Convert updated graph to PyTorch Geometric format
    data = from_networkx(graph)
    data.x = torch.stack([graph.nodes[node]['profile'] for node in graph.nodes])
    data.edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Predict using the GCN
    model.eval()
    with torch.no_grad():
        out = model(data)

    # Output the predicted features or labels for the new user
    predicted_features = out[new_user_id]
    print(f"Predicted features for new user {new_user_id}: {predicted_features}")
    return predicted_features

# Function to predict the effect of removing a user (using GCN)
def predict_remove_user_gnn(user_id, graph, model):
    print(f"Predicting changes by removing user {user_id}...")

    # Remove the user node and edges
    if graph.has_node(user_id):
        graph.remove_node(user_id)
    else:
        print(f"User {user_id} does not exist in the graph.")
        return None

    # Convert the updated graph to PyTorch Geometric format
    data = from_networkx(graph)
    data.x = torch.stack([graph.nodes[node]['profile'] for node in graph.nodes])
    data.edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Predict using the GCN
    model.eval()
    with torch.no_grad():
        out = model(data)

    print(f"Predicted features for the remaining nodes after removing user {user_id}: {out}")
    return out

# Example usage of prediction functions
new_user_id = 9999
new_user_features = [0.6, 0.5, 0.4, 1, theme_to_num['example_theme']]  # Example features
connections = [1, 2]  # Example connections to other users

# Predict effect of adding a user
predicted_features = predict_add_user_gnn(new_user_id, new_user_features, connections, graph, model)

# Predict effect of removing a user
removed_user_predictions = predict_remove_user_gnn(2, graph, model)

# Function to predict missing features after removing and re-adding a user
def predict_removed_user_features(user_id, graph, model):
    if not graph.has_node(user_id):
        print(f"User {user_id} does not exist in the graph.")
        return None

    # Remove user from the graph
    print(f"Removing user {user_id} from the graph...")
    user_profile = graph.nodes[user_id]['profile']  # Keep the original features
    graph.remove_node(user_id)
    
    # Strip some features (set sentiment and toxicity to NaN)
    incomplete_profile = user_profile.clone()
    incomplete_profile[1:3] = float('nan')
    print(f"Stripped features (sentiment and toxicity) for user {user_id}: {incomplete_profile}")

    # Re-add the user with incomplete features
    print(f"Re-adding user {user_id} with missing features...")
    graph.add_node(user_id, profile=incomplete_profile)

    # Recreate edges (connections)
    original_connections = users[user_id].get('connections', [])
    for connection in original_connections:
        graph.add_edge(user_id, connection)

    # Convert the updated graph back to PyTorch Geometric format
    data = from_networkx(graph)
    data.x = torch.stack([graph.nodes[node]['profile'] for node in graph.nodes])
    data.edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Predict the missing features using the model
    model.eval()
    with torch.no_grad():
        predicted_features = model(data)

    original_features = user_profile  # Original complete profile
    predicted_features_for_user = predicted_features[user_id]
    print(f"Original features: {original_features}")
    print(f"Predicted features: {predicted_features_for_user}")

    return original_features, predicted_features_for_user

# Example usage: Remove user, strip features, and predict missing features
user_to_remove = 2  # Specify the user ID to remove and test
original, predicted = predict_removed_user_features(user_to_remove, graph, model)

print(f"Preidcted features {predicted}")

# Calculate the accuracy of the prediction for the missing features
missing_indices = [1, 2]  # Indices of the stripped features (sentiment and toxicity)
error = torch.abs(original[missing_indices] - predicted[missing_indices])
print(f"Prediction error for missing features: {error}")