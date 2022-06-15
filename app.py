import os
import dash
import torch
import lightning as L
import dash_bootstrap_components as dbc

from dash import html
from dash.dependencies import Input, Output
from components.ui import create_figure
from components.ui import NavBar, Body


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
        def update_figure(digit_value):
            predictions_fname = os.path.join("data", "predictions", "predictions.pt")
            predictions = torch.load(predictions_fname)
            ground_truths_fname = os.path.join("data", "training_split", "val.pt")
            ground_truths = torch.load(ground_truths_fname)

            for i in range(len(ground_truths)):
                if ground_truths[i][1] == digit_value:
                    sample_idx = i
                    break

            ground_truth_fig = create_figure(
                ground_truths[sample_idx][0], "Ground Truth"
            )
            prediction_fig = create_figure(predictions[sample_idx][0], "Decoded")
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
