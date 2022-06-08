<div align="center">

# Lightning App

A template environment and system architecture for [Lightning](https://www.pytorchlightning.ai/) Apps.

![](https://img.shields.io/badge/PyTorch_Lightning-Ecosystem-informational?style=flat&logo=pytorchlightning&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Grid.ai-Cloud_Compute-informational?style=flat&logo=grid.ai&logoColor=white&color=2bbc8a)

<!-- [![codecov](https://codecov.io/gh/JustinGoheen/lightning-app/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-app)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-app.svg?style=shield) -->


[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-app)

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/JustinGoheen/lightning-app)

</div>

## Overview
The project on the `main` branch only shows a static Dash app, displaying a prediction result saved during a local training process.

Extensions of this version will be created on separate branches, with the prefix "alt-app/" for the branch i.e. the next iteration of this example is called ["alt-app/predict-from-checkpoint"](https://github.com/JustinGoheen/lightning-app/tree/alt-app/predict-from-checkpoint). As the name implies, this version will show users how to load a model from a checkpoint and predict on new data.

Current Lightning App (deployed to cloud) is shown below:

<img width="1505" alt="Screen Shot 2022-06-03 at 9 21 47 AM" src="https://user-images.githubusercontent.com/26209687/171926909-022c4ae8-9574-4cd4-b381-a42cf495dbc8.png">

## Potential roadblocks

I am not sure if the Dash [callbacks](https://dash.plotly.com/basic-callbacks) will "talk" to the Lightning Flow/Work state to update the UI as intended (as it should if using Dash only).

## Additional features:
- Once Lightning becomes generally available, users will be able to open a reproducible environment in either Gitpod or GitHub CodeSpaces.
- Users are able to view example training logs and PyTorch profiler logs in TensorBoard from within the Gitpod Workspace or CodeSpace.
- Users can reference the lightning-pod [wiki](https://github.com/JustinGoheen/lightning-pod/wiki) for additional information and guides.