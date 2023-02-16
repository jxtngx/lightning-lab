from pathlib import Path

from lightning import LightningApp, LightningFlow
from lightning.app.frontend import StaticWebFrontend


class ReactUI(LightningFlow):
    def __init__(self):
        super().__init__()

    def configure_layout(self):
        return StaticWebFrontend(Path(__file__).parent / "reactui/dist")


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.react_ui = ReactUI()

    def run(self):
        self.react_ui.run()

    def configure_layout(self):
        return [{"name": "ReactUI", "content": self.react_ui}]


app = LightningApp(RootFlow())
