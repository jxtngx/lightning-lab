import os
import dash
import torch
import lightning as L
import dash_bootstrap_components as dbc

from dash import html
from dash.dependencies import Input, Output
from components.ui import create_figure, find_index
from components.ui import NavBar, Body

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import TensorDataset


@dataclass
class Data:
    """class for storing ground truth and prediction torch.TensorDatasets"""

    predictions_fname: Path = os.path.join("data", "predictions", "predictions.pt")
    predictions: TensorDataset = torch.load(predictions_fname)
    ground_truths_fname: Path = os.path.join("data", "training_split", "val.pt")
    ground_truths: TensorDataset = torch.load(ground_truths_fname)


class DashWorker(L.LightningWork):
    def run(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = html.Div(
            [
                NavBar,
                html.Br(),
                Body,
            ]
        )

        @app.callback(
            [Output("left-fig", "figure"), Output("right-fig", "figure")],
            [Input("dropdown", "value")],
        )
        def update_figure(label_value):
            xidx = 0
            idx = find_index(Data.ground_truths, label=label_value, label_idx=1)
            gt = Data.ground_truths[idx][xidx]
            pred = Data.predictions[idx][xidx]
            ground_truth_fig = create_figure(gt, "Ground Truth")
            prediction_fig = create_figure(pred, "Decoded")
            return ground_truth_fig, prediction_fig

        app.run_server(host=self.host, port=self.port)


class DashFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_dash = DashWorker(parallel=True)

    def run(self):
        self.lit_dash.run()

    def configure_layout(self):
        tab1 = {"name": "home", "content": self.lit_dash}
        return tab1


app = L.LightningApp(DashFlow())
