import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, ServersideOutputTransform
import networkx as nx
from pyvis.network import Network
from datetime import datetime, timedelta
from dash.exceptions import PreventUpdate
from objects.user import UserObject
from utils.feature_extractor import FeatureExtractor
from utils.config_manager import ConfigManager
from data_set_managers.json_dataset_manager import JsonDatasetManager
from objects.timeline import TimelineObject
import base64
import io
import json
import tempfile
from rich.console import Console
from rich.progress import track

# Initialize the console for rich logging
console = Console()

# Initialize the Dash app with Bootstrap theme and server-side output transform
app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    transforms=[ServersideOutputTransform()],
    suppress_callback_exceptions=True,
)


def initialize_data(post_data_path):
    global telegram_post_data, ground_truth_extremist_post_data, feature_extractor, timeline, conversations

    console.log("[bold green]Initializing data...[/bold green]")

    telegram_post_data = JsonDatasetManager(post_data_path)
    ground_truth_extremist_post_data = JsonDatasetManager(
        ConfigManager().location_of_ground_truth_extremist_post_data
    )

    feature_extractor = FeatureExtractor(
        telegram_post_data.get_list_of_posts(),
        ground_truth_extremist_post_data.get_list_of_posts(),
    )

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

    console.log(
        f"[bold green]Data initialized successfully with start: {start} and end: {end}[/bold green]"
    )

    return start, end


# Define initial data
start, end = initialize_data(str(ConfigManager().location_of_post_data))

# Define the edge colors
edge_colors = {"reply": "blue", "mention": "green", "temporal": "red"}

# Define the layout of the app
app.layout = html.Div(
    style={
        "backgroundColor": "#f8f9fa",
        "color": "#333",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        dbc.NavbarSimple(
            children=[dbc.NavItem(dbc.NavLink("Social Network Analysis", href="#"))],
            brand="Tribal",
            brand_href="#",
            color="primary",
            dark=True,
            fluid=True,
        ),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.Label(
                                                    "Select Date Window:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.RangeSlider(
                                                    id="date-range-slider",
                                                    min=int(start.timestamp()),
                                                    max=int(end.timestamp()),
                                                    value=[
                                                        int(start.timestamp()),
                                                        int(end.timestamp()),
                                                    ],
                                                    marks={
                                                        i: datetime.fromtimestamp(
                                                            i
                                                        ).strftime("%Y-%m-%d")
                                                        for i in range(
                                                            int(start.timestamp()),
                                                            int(end.timestamp())
                                                            + 86400,
                                                            86400,
                                                        )
                                                    },
                                                    step=86400,
                                                ),
                                            ]
                                        )
                                    ],
                                    className="mb-4",
                                )
                            ],
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.Label(
                                                    "Select Conversation:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.Slider(
                                                    id="date-slider",
                                                    min=0,
                                                    max=len(conversations) - 1,
                                                    value=0,
                                                    marks={
                                                        i: str(i)
                                                        for i in range(
                                                            len(conversations)
                                                        )
                                                    },
                                                    step=1,
                                                ),
                                            ]
                                        )
                                    ],
                                    className="mb-4",
                                )
                            ],
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.Iframe(
                                                    id="network-graph",
                                                    style={
                                                        "width": "100%",
                                                        "height": "750px",
                                                        "border": "none",
                                                    },
                                                )
                                            ]
                                        )
                                    ],
                                    className="mb-4",
                                )
                            ],
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                dcc.Upload(
                                                    id="upload-data",
                                                    children=[
                                                        html.Button(
                                                            "Upload New Data",
                                                            id="upload-button",
                                                        )
                                                    ],
                                                    style={
                                                        "width": "100%",
                                                        "border": "1px solid #ccc",
                                                        "borderRadius": "5px",
                                                        "padding": "10px",
                                                        "textAlign": "center",
                                                        "cursor": "pointer",
                                                    },
                                                ),
                                                html.Div(id="upload-output"),
                                            ]
                                        )
                                    ],
                                    className="mb-4",
                                )
                            ],
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
        ),
    ],
)

# JavaScript to trigger file upload dialog
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            document.getElementById('upload-data').click();
        }
        return null;
    }
    """,
    Output("upload-data", "contents"),
    Input("upload-button", "n_clicks"),
)


@app.callback(
    Output("upload-output", "children"),
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
)
def update_file(contents, filename):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    # Read the file and update data
    # Assuming decoded is your data in bytes
    decoded_data = decoded.decode("utf-8")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", encoding="utf-8"
    ) as temp_file:
        # Write the decoded data to the temporary file
        temp_file.write(decoded_data)

        # Get the name of the temporary file
        temp_file_name = temp_file.name

        # Reinitialize dataset managers and timeline with new data
        start, end = initialize_data(temp_file_name)

        # Update the range slider
        console.log(
            f"[bold green]File '{filename}' uploaded and data updated successfully.[/bold green]"
        )
        return f"File '{filename}' uploaded and data updated successfully."


@app.callback(Output("network-graph", "srcDoc"), [Input("date-slider", "value")])
def update_graph(selected_date_idx):
    html_file = create_network_graph(selected_date_idx)

    with open(html_file, "r") as f:
        return f.read()


@app.callback(
    [
        Output("date-slider", "max"),
        Output("date-slider", "marks"),
        Output("date-slider", "value"),
    ],
    [Input("date-range-slider", "value")],
)
def update_slider_range(date_range):
    if date_range is None or len(date_range) != 2:
        raise PreventUpdate

    start_date = datetime.fromtimestamp(date_range[0])
    end_date = datetime.fromtimestamp(date_range[1])

    timeline.clear_windows()
    timeline.make_new_window(start_date, end_date)
    window = timeline.windows[0]
    global conversations
    conversations = window.conversations

    slider_range = range(len(conversations))
    new_value = 0 if len(slider_range) > 0 else None

    return len(slider_range) - 1, {i: str(i) for i in slider_range}, new_value


def create_network_graph(selected_index):
    console.log("[bold green]Creating network graph...[/bold green]")

    G = conversations[selected_index].graph
    new_graph = nx.MultiDiGraph()
    new_graph.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        edge_type = data.get("type", "unknown")
        color = edge_colors.get(edge_type, "black")
        new_graph.add_edge(u, v, color=color, title=edge_type)

    net = Network(
        notebook=False,
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
    )
    net.from_nx(new_graph)

    for node in new_graph.nodes(data=True):
        username = node[0]
        user_info = None
        for user in conversations[selected_index]._users:
            if user.username == username:
                user_info = "-".join(user.get_dict(True))
                break
        tooltip_content = f"{user_info}" if user_info else username
        net.get_node(username)["title"] = tooltip_content

    net.set_options(
        """
    var options = {
      "nodes": {
        "borderWidth": 2,
        "size": 20,
        "color": {
          "background": "#333",
          "border": "#666",
          "highlight": {
            "background": "#444",
            "border": "#888"
          }
        },
        "font": {
          "color": "#fff"
        }
      },
      "edges": {
        "color": {
          "inherit": false,
          "color": "#333"
        },
        "smooth": {
          "type": "continuous"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1
          }
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true
      }
    }
    """
    )

    html_file = "network_graph.html"
    net.write_html(html_file)

    console.log("[bold green]Network graph created successfully.[/bold green]")
    return html_file


if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    except Exception as e:
        console.log(f"[bold red]Error running the server: {e}[/bold red]")
