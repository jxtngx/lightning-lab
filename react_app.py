from pathlib import Path

from lightning import LightningApp, LightningFlow
from lightning.app.frontend import StaticWebFrontend


class ReactUI(LightningFlow):
    def __init__(self):
        super().__init__()
        # self.message_to_print = "Hello World!"
        # self.should_print = False

    def configure_layout(self):
        return StaticWebFrontend(Path(__file__).parent / "reactui/dist")


class HelloWorld(LightningFlow):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.react_ui = ReactUI()

    def run(self):
        self.react_ui.run()
        # if self.react_ui.should_print:
        # print(f"{self.counter}: {self.react_ui.message_to_print}")
        # self.counter += 1

    def configure_layout(self):
        return [{"name": "ReactUI", "content": self.react_ui}]


app = LightningApp(HelloWorld())
