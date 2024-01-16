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

import os

import numpy as np
import pandas as pd
import plotly.express as px
from dash import dash_table
from lightning.pytorch.utilities.model_summary import ModelSummary

from visionlab import config, VisionTransformer


def make_metrics_summary():
    logsdir = os.path.join("logs", "csv")
    logs = os.listdir(logsdir)
    most_recent = os.path.join(logsdir, f"version_{len(logs)-1}", "metrics.csv")
    summary = pd.read_csv(most_recent)
    if not pd.isna(summary["val-loss"].iloc[-1]):
        index = -1
    elif pd.isna(summary["val-loss"].iloc[-1]):
        index = -2
    collection = {
        "Training Loss": summary["training-loss"].iloc[index - 1],
        "Val Loss": summary["val-loss"].iloc[index],
        "Val Acc": summary["val-acc"].iloc[index],
    }
    return collection


def create_figure(image, title_text):
    image = np.transpose(image.numpy(), (1, 2, 0))
    fig = px.imshow(image)
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


def make_model_summary():
    available_checkpoints = os.listdir(config.Paths.ckpts)
    available_checkpoints.remove("README.md")
    latest_checkpoint = available_checkpoints[0]
    chkpt_filename = os.path.join(config.Paths.ckpts, latest_checkpoint)
    model = VisionTransformer.load_from_checkpoint(chkpt_filename)
    model_summary = ModelSummary(model)
    model_summary = model_summary.__str__().split("\n")
    model_layers = make_model_layer_table(model_summary)
    model_params = make_model_param_text(model_summary)
    return {"layers": model_layers, "params": model_params}


def find_index(dataset, label, label_idx):
    for i in range(len(dataset)):
        if dataset[i][label_idx] == label:
            return i
