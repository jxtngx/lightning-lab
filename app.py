import lightning as L


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        # ############################################ #
        # initialize your workers and child flows here #
        # ############################################ #

    def run(self):
        # ##################################### #
        # run your workers and child flows here #
        # ##################################### #
        ...

    def configure_layout(self):
        # ########################### #
        # create your app layout here #
        # ########################### #
        ...


app = L.LightningApp(RootFlow())
