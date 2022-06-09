import os
import dash
import torch
import dash_bootstrap_components as dbc
import plotly.express as px

from dash import html
from dash import dcc
from torchmetrics import Precision, Recall, F1Score, Accuracy


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def metrics_collection(y_true, y_predict):
    pass


def create_figure(ground_truth_image, title_text):
    """creates the ground truth image"""
    fig = px.imshow(ground_truth_image.view(28, 28))
    fig.update_layout(
        title=dict(
            text=title_text,
            font_family="Ucityweb, sans-serif",
            font=dict(size=24),
            y=0.05,
            yanchor="bottom",
            x=0.5,
        )
    )
    return fig


#### DATA ####
predictions_fname = os.path.join("data", "predictions", "predictions.pt")
predictions = torch.load(predictions_fname)
ground_truths_fname = os.path.join("data", "training_split", "val.pt")
ground_truths = torch.load(ground_truths_fname)
sample_idx = 10


#### APP LAYOUT ####
NavBar = dbc.NavbarSimple(
    brand="Application Name",
    color="#792ee5",
    dark=True,
    fluid=True,
    className="app-title",
)

ModelCard = dbc.Card(
    dbc.CardBody(
        [
            html.H1(f"Model Card", id="model_card", className="card-title"),
            html.P(
                f"Model Info 1: {'lorem ipsum dolor sit'}",
                id="model_info_1",
                className="model-card-text",
            ),
            html.P(
                f"Model Info 2: {'lorem ipsum dolor sit'}",
                id="model_info_2",
                className="model-card-text",
            ),
            html.P(
                f"Model Info 3: {'lorem ipsum dolor sit'}",
                id="model_info_3",
                className="model-card-text",
            ),
            html.P(
                f"Model Info 4: {'lorem ipsum dolor sit'}",
                id="model_info_4",
                className="model-card-text",
            ),
        ]
    ),
    className="model-card-container",
)

SideBar = dbc.Col(
    [ModelCard],
    width=3,
)

GroundTruth = dcc.Graph(
    id="left-fig",
    figure=create_figure(ground_truths[sample_idx][0], "Ground Truth"),
    config={
        "responsive": True,  # dynamically resizes Graph with browser winder
        "displayModeBar": True,  # always show the Graph tools
        "displaylogo": False,  # remove the plotly logo
    },
)

Predictions = dcc.Graph(
    id="right-fig",
    figure=create_figure(predictions[sample_idx][0], "Decoded"),
    config={
        "responsive": True,  # dynamically resizes Graph with browser winder
        "displayModeBar": True,  # always show the Graph tools
        "displaylogo": False,  # remove the plotly logo
    },
)

Metrics = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 1", className="card-title"),
                        html.P(0.01, id="metric_1_text", className="metric-card-text"),
                    ],
                    id="metric_1_card",
                    className="metric-container",
                )
            ],
            width=3,
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 2", className="card-title"),
                        html.P(0.01, id="metric_2_text", className="metric-card-text"),
                    ],
                    id="metric_2_card",
                    className="metric-container",
                )
            ],
            width=3,
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 3", className="card-title"),
                        html.P(0.01, id="metric_3_text", className="metric-card-text"),
                    ],
                    id="metric_3_card",
                    className="metric-container",
                )
            ],
            width=3,
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 4", className="card-title"),
                        html.P(0.01, id="metric_4_text", className="metric-card-text"),
                    ],
                    id="metric_4_card",
                    className="metric-container",
                )
            ],
            width=3,
        ),
    ],
    id="metrics",
    justify="center",
)

Graphs = dbc.Row(
    [
        dbc.Col([GroundTruth], className="pretty-container", width=5),
        dbc.Col(width=1),
        dbc.Col([Predictions], className="pretty-container", width=5),
    ],
    justify="center",
)

MainArea = dbc.Col([Metrics, Graphs])

Body = dbc.Container([dbc.Row([SideBar, MainArea])], fluid=True)

#### PASS LAYOUT TO DASH ####
app.layout = html.Div(
    [
        NavBar,
        html.Br(),  # hacky way to create space between header (navbar) and body
        Body,
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True)
