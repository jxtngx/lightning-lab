import os
import dash
import torch
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import html
from dash import dcc
from torchmetrics import Precision, Recall, F1Score, Accuracy
from pytorch_lightning import Trainer
from lightning_pod.network.module import LitModel
from lightning_pod.pipeline.datamodule import LitDataModule


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def scores_report(y_true, y_predict):
    scores = {
        "Precision": Precision(y_true, y_predict),
        "Recall": Recall(y_true, y_predict),
        "F1": F1Score(y_true, y_predict),
        "Accuracy": Accuracy(y_true, y_predict),
    }
    return scores


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
datamodule = LitDataModule()
datamodule.prepare_data()
datamodule.setup()
val_dataloader = datamodule.val_dataloader()

#### MODEL ####
checkpoint = "models/checkpoints/model.ckpt"
model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint)
model.eval()

#### TRAINER ####
trainer = Trainer(enable_progress_bar=False)

#### PREDICTIONS ####
# predictions = trainer.predict(model, datamodule.val_dataloader())
sample_idx = 0
ground_truth = val_dataloader.dataset[sample_idx][0]
with torch.no_grad():
    prediction = model(ground_truth)


#### APP LAYOUT ####
NAVBAR = dbc.NavbarSimple(
    brand="MNIST Encoder-Decoder",
    color="#792ee5",
    dark=True,
    fluid=True,
)

MODEL_CARD = dbc.Card(
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

SIDEBAR = dbc.Col(
    [MODEL_CARD],
    width=3,
)

GROUNDTRUTH = dcc.Graph(
    id="leftside_figure",
    figure=leftside_figure(ground_truth),
    config={
        "responsive": True,  # dynamically resizes Graph with browser winder
        "displayModeBar": True,  # always show the Graph tools
        "displaylogo": False,  # remove the plotly logo
    },
)

PREDICTIONS = dcc.Graph(
    id="rightside_figure",
    figure=rightside_figure(prediction),
    config={
        "responsive": True,  # dynamically resizes Graph with browser winder
        "displayModeBar": True,  # always show the Graph tools
        "displaylogo": False,  # remove the plotly logo
    },
)

SCORES = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.P("Precision", style={"font-weight": "bold"}),
                        html.H6(0.01, id="precision-score", style={"font-size": "80%"}),
                    ],
                    id="R2",
                    className="mini_container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.P("Recall", style={"font-weight": "bold"}),
                        html.H6(0.01, id="recall-score", style={"font-size": "80%"}),
                    ],
                    id="RMSE",
                    className="mini_container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.P("F1", style={"font-weight": "bold"}),
                        html.H6(0.01, id="f1-score", style={"font-size": "80%"}),
                    ],
                    id="MPD",
                    className="mini_container",
                )
            ]
        ),
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.P("Accuracy", style={"font-weight": "bold"}),
                        html.H6(0.01, id="accuracy-score", style={"font-size": "80%"}),
                    ],
                    id="MGD",
                    className="mini_container",
                )
            ]
        ),
    ],
    id="scores_card",
)

MAIN_AREA = dbc.Col(
    [
        SCORES,
        dbc.Row(
            [
                dbc.Col([GROUNDTRUTH], className="pretty_container", width=5),
                dbc.Col([PREDICTIONS], className="pretty_container", width=5),
            ],
            justify="center",
        ),
    ]
)

BODY = dbc.Container([dbc.Row([SIDEBAR, MAIN_AREA])], fluid=True)


if __name__ == "__main__":
    app.run_server(debug=True)
