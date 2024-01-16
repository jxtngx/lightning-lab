# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dash
import dash_bootstrap_components as dbc
import torch
from dash import html
from dash.dependencies import Input, Output
from components import Body, create_figure, find_index, NavBar

from visionlab import config

PREDICTIONS = torch.load(config.Paths.predictions)
DATASET = torch.load(config.Paths.test_split)
LABELNAMES = DATASET.classes


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        NavBar,
        html.Br(),
        Body,
    ]
)


@app.callback(
    [Output("gt-fig", "figure"), Output("pred-card", "children")],
    [Input("dropdown", "value")],
)
def update_figure(label_value):
    xidx = 0
    labelidx = 1
    idx = find_index(DATASET, label=LABELNAMES.index(label_value), label_idx=labelidx)
    gt = DATASET[idx][xidx]
    pred = LABELNAMES[torch.argmax(PREDICTIONS[idx][labelidx])]
    fig = create_figure(gt, "Ground Truth")
    return fig, pred


app.run_server(port=8000)
