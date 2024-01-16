import os
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

from utilities import create_figure

this_file = Path(__file__)
this_studio_idx = [i for i, j in enumerate(this_file.parents) if j.name.endswith("this_studio")][0]
this_studio = this_file.parents[this_studio_idx]
csvlogs = os.path.join(this_studio, "vision-lab", "logs", "csv")

RUNS = os.listdir(csvlogs)

NavBar = dbc.NavbarSimple(
    brand="VisionTransformer Base 32 Run Metrics",
    color="#792ee5",
    dark=True,
    fluid=True,
    className="app-title",
)

Control = dbc.Card(
    dbc.CardBody(
        [
            html.H1("Run Version", className="card-title"),
            dcc.Dropdown(
                options=RUNS,
                value=RUNS[0],
                multi=False,
                id="dropdown",
                searchable=True,
            ),
        ]
    ),
    className="model-card-container",
)

SideBar = dbc.Col([Control], width=3)

Graph = dbc.Col(
    dcc.Loading(
        [
            dcc.Graph(
                id="metric-graph",
                figure=create_figure(os.path.join(csvlogs, RUNS[0], "metrics.csv")),
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            ),
            dcc.Interval(id="interval-component", interval=1 * 1000, n_intervals=0),  # in milliseconds
        ]
    )
)

Body = dbc.Container(dbc.Row([SideBar, Graph]), fluid=True)
