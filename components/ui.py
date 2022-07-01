import os
import dash
import torch
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

from dash import dcc, html, dash_table
from pytorch_lightning.utilities.model_summary import ModelSummary
from lightning_pod.core.module import LitModel


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def metrics_collection(y_true, y_predict):
    pass


def create_figure(image, title_text):
    fig = px.imshow(image.view(28, 28))
    fig.update_layout(
        title=dict(
            text=title_text,
            font_family="Ucityweb, sans-serif",
            font=dict(size=24),
            y=0.05,
            yanchor="bottom",
            x=0.5,
        ),
        height=300,
    )
    return fig


def make_model_layer_table(model_summary: list):
    model_layers = model_summary[:-4]
    model_layers = [i for i in model_layers if not all(j == "-" for j in i)]
    model_layers = [i.split("|") for i in model_layers]
    model_layers = [[j.strip() for j in i] for i in model_layers]
    model_layers[0][0] = "Layer"
    header = model_layers[0]
    body = model_layers[1:]
    table = pd.DataFrame(body, columns=header)
    table = dash_table.DataTable(
        data=table.to_dict("records"),
        columns=[{"name": i, "id": i} for i in table.columns],
        style_cell={
            "textAlign": "left",
            "font-family": "FreightSans, Helvetica Neue, Helvetica, Arial, sans-serif",
        },
        style_as_list_view=True,
        style_table={
            "overflow-x": "auto",
        },
        style_header={"border": "0px solid black"},
    )
    return table


def make_model_param_text(model_summary: list):
    model_params = model_summary[-4:]
    model_params = [i.split("  ") for i in model_params]
    model_params = [[i[0]] + [i[-1]] for i in model_params]
    model_params = [[j.strip() for j in i] for i in model_params]
    model_params = [i[::-1] for i in model_params]
    model_params[-1][0] = "Est. params size (MB)"
    model_params = ["".join([i[0], ": ", i[-1]]) for i in model_params]
    return model_params


def make_model_summary(model_summary: ModelSummary):
    model_summary = model_summary.__str__().split("\n")
    model_layers = make_model_layer_table(model_summary)
    model_params = make_model_param_text(model_summary)
    return model_layers, model_params


def find_index(dataset, label=0, label_idx=1):
    for i in range(len(dataset)):
        if dataset[i][label_idx] == label:
            return i


#### DATA ####
## images ##
predictions_fname = os.path.join("data", "predictions", "predictions.pt")
predictions = torch.load(predictions_fname)
ground_truths_fname = os.path.join("data", "training_split", "val.pt")
ground_truths = torch.load(ground_truths_fname)
ground_truth_labels = list(set(i[1] for i in ground_truths))
# find first zero
zero_idx = find_index(ground_truths, label=0, label_idx=1)


## model summary ##
chkptdir = os.path.join("models", "checkpoints")
available_checkpoints = os.listdir(chkptdir)
available_checkpoints.remove("README.md")
latest_checkpoint = available_checkpoints[0]
chkpt_fname = os.path.join("models", "checkpoints", "model.ckpt")
model = LitModel.load_from_checkpoint(chkpt_fname)
summary = ModelSummary(model)
model_layers, model_params = make_model_summary(summary)

#### APP LAYOUT ####
NavBar = dbc.NavbarSimple(
    brand="Linear Encoder-Decoder",
    color="#792ee5",
    dark=True,
    fluid=True,
    className="app-title",
)

Control = dbc.Card(
    dbc.CardBody(
        [
            html.H1("Label", className="card-title"),
            dcc.Dropdown(
                options=ground_truth_labels,
                value=0,
                multi=False,
                id="dropdown",
                searchable=True,
            ),
        ]
    ),
    className="model-card-container",
)

ModelCard = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H1("Model Card", id="model_card", className="card-title"),
                html.Br(),
                html.H3("Layers", className="card-title"),
                model_layers,
                html.Br(),
                html.H3("Parameters", className="card-title"),
                html.P(
                    f"{model_params[0]}",
                    id="model_info_1",
                    className="model-card-text",
                ),
                html.P(
                    f"{model_params[1]}",
                    id="model_info_2",
                    className="model-card-text",
                ),
                html.P(
                    f"{model_params[2]}",
                    id="model_info_3",
                    className="model-card-text",
                ),
                html.P(
                    f"{model_params[3]}",
                    id="model_info_4",
                    className="model-card-text",
                ),
            ]
        ),
    ],
    className="model-card-container",
)

SideBar = dbc.Col(
    [
        Control,
        html.Br(),
        ModelCard,
    ],
    width=3,
)

GroundTruth = dcc.Graph(
    id="left-fig",
    figure=create_figure(ground_truths[zero_idx][0], "Ground Truth"),
    config={
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
    },
)

Predictions = dcc.Graph(
    id="right-fig",
    figure=create_figure(predictions[zero_idx][0], "Decoded"),
    config={
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
    },
)

Metrics = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Card(
                    [
                        html.H4("Metric 1", className="card-title"),
                        html.P(
                            "0.xx", id="metric_1_text", className="metric-card-text"
                        ),
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
                        html.P(
                            "0.xx", id="metric_2_text", className="metric-card-text"
                        ),
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
                        html.P(
                            "0.xx", id="metric_3_text", className="metric-card-text"
                        ),
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
                        html.P(
                            "0.xx", id="metric_4_text", className="metric-card-text"
                        ),
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
        dbc.Col([GroundTruth], className="pretty-container", width=4),
        dbc.Col(width=1),
        dbc.Col([Predictions], className="pretty-container", width=4),
    ],
    justify="center",
    align="middle",
    className="graph-row",
)

MainArea = dbc.Col([Metrics, html.Br(), Graphs])

Body = dbc.Container([dbc.Row([SideBar, MainArea])], fluid=True)

#### PASS LAYOUT TO DASH ####
app.layout = html.Div(
    [
        NavBar,
        html.Br(),
        Body,
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True)
