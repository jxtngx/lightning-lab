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
    metrics = {
        "Precision": Precision(y_true, y_predict),
        "Recall": Recall(y_true, y_predict),
        "F1": F1Score(y_true, y_predict),
        "Accuracy": Accuracy(y_true, y_predict),
    }
    return metrics


def leftside_figure(ground_truth_image):
    """creates the ground truth image"""
    fig = px.imshow(ground_truth_image.view(28, 28))
    fig.update_layout(title=dict(text="Ground Truth"))
    return fig


def rightside_figure(prediction_image):
    """creates the decoded image"""
    fig = px.imshow(prediction_image.view(28, 28))
    fig.update_layout(title=dict(text="Decoded"))
    return fig


#### DATA ####
predictions_fname = os.path.join("data", "predictions", "predictions.pt")
predictions = torch.load(predictions_fname)
ground_truths_fname = os.path.join("data", "training_split", "val.pt")
ground_truths = torch.load(ground_truths_fname)
sample_idx = 10


#### APP LAYOUT ####
NavBar = dbc.NavbarSimple(
    brand="MNIST Encoder-Decoder",
    color="#792ee5",
    dark=True,
    fluid=True,
)

ModelCard = dbc.Card(
    dbc.CardBody(
        [
            html.H1(f"Model Card", id="model_name", className="card-title"),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_1",
                className="modelcard-text",
            ),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_2",
                className="modelcard-text",
            ),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_3",
                className="modelcard-text",
            ),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_4",
                className="modelcard-text",
            ),
        ]
    ),
    className="info-container",
)

SideBar = dbc.Col(
    [ModelCard],
    width=3,
)

GroundTruth = dcc.Graph(
    id="leftside_figure",
    figure=leftside_figure(ground_truths[sample_idx][0]),
    config={
        "responsive": True,  # dynamically resizes Graph with browser winder
        "displayModeBar": True,  # always show the Graph tools
        "displaylogo": False,  # remove the plotly logo
    },
)

Predictions = dcc.Graph(
    id="rightside_figure",
    figure=rightside_figure(predictions[sample_idx][0]),
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
                        html.H4("Metric 1", style={"font-weight": "bold"}),
                        html.H6(0.01, id="metric_1_text"),
                    ],
                    id="metric_1_card",
                    className="mini-container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 2", style={"font-weight": "bold"}),
                        html.H6(0.01, id="metric_2_text"),
                    ],
                    id="metric_2_card",
                    className="mini-container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 3", style={"font-weight": "bold"}),
                        html.H6(0.01, id="metric_3_text"),
                    ],
                    id="metric_3_card",
                    className="mini-container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 4", style={"font-weight": "bold"}),
                        html.H6(0.01, id="metric_4_text"),
                    ],
                    id="metric_4_card",
                    className="mini-container",
                )
            ]
        ),
    ],
    id="scores_card",
)

Graphs = dbc.Row(
    [
        dbc.Col([GroundTruth], className="pretty-container", width=5),
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
