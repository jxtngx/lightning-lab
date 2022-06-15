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
The project on the `main` branch is a Dash app with a single callback.

Additional apps are on separate branches with the prefix `alt-app/` followed by a short description of the app's learning point e.g. `alt-app/predict-from-checkpoint`, `alt-app/multiple-dash-callbacks`, or `alt-app/scheduled-predictions` etc.


App UI preview:

![](assets/sample_app.png)

## Potential roadblocks

None for now.

## Additional features:
- Once Lightning Apps becomes generally available, users will be able to open a reproducible environment in either Gitpod or GitHub CodeSpaces.
- Users are able to view example training logs and PyTorch profiler logs in TensorBoard from within the Gitpod Workspace or CodeSpace.
- Users can reference the lightning-pod [wiki](https://github.com/JustinGoheen/lightning-pod/wiki) for additional information and guides.