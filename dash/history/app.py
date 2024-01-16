import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output
from components import NavBar, Body

from utilities import create_figure


this_file = Path(__file__)
this_studio_idx = [i for i, j in enumerate(this_file.parents) if j.name.endswith("this_studio")][0]
this_studio = this_file.parents[this_studio_idx]
csvlogs = os.path.join(this_studio, "vision-lab", "logs", "csv")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([NavBar, html.Br(), Body])


@app.callback(
    Output("metric-graph", "figure"),
    [Input("dropdown", "value")],
)
def update_figure(label_value):
    print(label_value)
    fig = create_figure(os.path.join(csvlogs, label_value, "metrics.csv"))
    return fig


app.run_server(port=8000)
