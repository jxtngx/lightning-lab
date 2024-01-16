import os
from pathlib import Path
from time import sleep

import dash_bootstrap_components as dbc
from dash import dcc, html

from utilities import create_figure

this_file = Path(__file__)
this_studio_idx = [i for i, j in enumerate(this_file.parents) if j.name.endswith("this_studio")][0]
this_studio = this_file.parents[this_studio_idx]
csvlogs = os.path.join(this_studio, "vision-lab", "logs", "csv")

runs = os.listdir(csvlogs)
numruns = len(runs)
tgtrun = numruns - 1


NavBar = dbc.NavbarSimple(
    brand="VisionTransformer Base 32 Run Metrics",
    color="#792ee5",
    dark=True,
    fluid=True,
    className="app-title",
)


Graph = dbc.Col(
    [
        dcc.Graph(
            id="metric-graph",
            figure=create_figure(os.path.join(csvlogs, runs[tgtrun], "metrics.csv")),
            config={
                "responsive": True,
                "displayModeBar": True,
                "displaylogo": False,
            },
        ),
        dcc.Interval(id="interval-component", interval=1 * 1000, n_intervals=0),  # in milliseconds
    ]
)

Body = dbc.Container(dbc.Row([Graph]), fluid=True)
