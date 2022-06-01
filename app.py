# market classification app
import os
import lightning as L
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash_app import NAVBAR, BODY


DATAPATH = os.path.join("data", "cache")


class DashWorker(L.LightningWork):
    def run(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        # server = app.server
        app.layout = html.Div(
            [
                NAVBAR,
                html.Br(),  # hacky way to create space between header (navbar) and body
                BODY,
            ]
        )
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
