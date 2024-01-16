from pathlib import Path
from typing import Union

import pandas as pd
import plotly.graph_objects as go


def create_figure(path: Union[str, Path]):
    if isinstance(path, str):
        run_name = path.split("/")[-2]
    else:
        run_name = path.parent.name
    run_name = " ".join(run_name.split("_")).title()

    data = pd.read_csv(path).drop("step", axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["training-loss"]))
    fig.update_layout(
        title=dict(
            text=f"Run Metrics: {run_name}",
            font_family="Ucityweb, sans-serif",
            font=dict(size=24),
            y=0.90,
            yanchor="bottom",
            x=0.5,
        )
    )
    return fig
