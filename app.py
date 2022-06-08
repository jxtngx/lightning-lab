# market classification app
import os
import lightning as L
import dash
import dash_bootstrap_components as dbc
from dash import html
from components.ui import NavBar, Body


class DashWorker(L.LightningWork):
    def run(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        # server = app.server
        app.layout = html.Div(
            [
                NavBar,
                html.Br(),  # hacky way to create space between header (navbar) and body
                Body,
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
