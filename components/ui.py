import os
import dash
import torch
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import html
from dash import dcc
from torchmetrics import Precision, Recall, F1Score, Accuracy
from pytorch_lightning import Trainer
from lightning_agents.agents.learning_agents.linear.module import (
    LinearEncoderDecoder as LitModel,
)
from lightning_agents.pipeline.datamodule import LitDataModule


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
            html.H4(f"Model Card", id="model_name", className="card-text"),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_1",
                className="card-text",
                style={"font-size": "80%"},
            ),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_2",
                className="card-text",
                style={"font-size": "80%"},
            ),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_3",
                className="card-text",
                style={"font-size": "80%"},
            ),
            html.P(
                f"Some Model Info: {0}",
                id="modelcard_4",
                className="card-text",
                style={"font-size": "80%"},
            ),
        ]
    ),
    className="pretty_container",
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
                        html.P("Metric 1", style={"font-weight": "bold"}),
                        html.H6(0.01, id="precision-score", style={"font-size": "80%"}),
                    ],
                    id="metric_1",
                    className="mini_container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.P("Metric 2", style={"font-weight": "bold"}),
                        html.H6(0.01, id="recall-score", style={"font-size": "80%"}),
                    ],
                    id="metric_2",
                    className="mini_container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.P("Metric 3", style={"font-weight": "bold"}),
                        html.H6(0.01, id="f1-score", style={"font-size": "80%"}),
                    ],
                    id="metric_3",
                    className="mini_container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.P("Metric 4", style={"font-weight": "bold"}),
                        html.H6(0.01, id="accuracy-score", style={"font-size": "80%"}),
                    ],
                    id="metric_4",
                    className="mini_container",
                )
            ]
        ),
    ],
    id="metrics_card",
)

MainArea = dbc.Col(
    [
        Metrics,
        dbc.Row(
            [
                dbc.Col([GroundTruth], className="pretty_container", width=5),
                dbc.Col([Predictions], className="pretty_container", width=5),
            ],
            justify="center",
        ),
    ]
)

Body = dbc.Container([dbc.Row([SideBar, MainArea])], fluid=True)


if __name__ == "__main__":
    app.run_server(debug=True)
