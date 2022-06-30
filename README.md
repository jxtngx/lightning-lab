<div align="center">

# Lightning Pod


![](https://img.shields.io/badge/Lightning.ai-Ecosystem-informational?style=flat&logo=pytorchlightning&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Grid.ai-Cloud_Compute-informational?style=flat&logo=grid.ai&logoColor=white&color=2bbc8a)

[![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-pod.svg?style=shield)


</div>

## Overview

Lightning Pod is a template Python environment, tooling, and system architecture for [Lightning OS](https://www.pytorchlightning.ai/) that culminates with a Plotly Dash [UI](https://01g6bdbc5e55wc5ffgj11gtkxj.litng-ai-03.litng.ai/view/home) deployed to the Lightning platform.

The main focus of this project is to provide a high-level a framework to students and researchers needing to incorporate deep learning into a project.

The code that facilitaties building Torch nn.Modules, a Lightning Module, and Lightning Trainer is in `lightning_pod.core`.

The code that facilitates raw data preprocessing, building a Torch Dataset, and Lightning DataModule is in `lightning_pod.pipeline`.

The configs for Lightning Apps and Grid.ai are in `.lightningai`.

### Using the Template

The intent is that users [create a new repo from the template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) in GitHub's web interface and then clone the newly created repo in their personal account to their local machine. Following this recommendation will provide users with all code and a clean git tree.

#### Prepping for Use
A [CLI](https://github.com/JustinGoheen/lightning-pod/blob/main/lightning_pod/cli/console.py) `pod` has been provided to assist with basic tasks.


`pod teardown` will destroy the example data splits, saved predictions, logs, profilers, checkpoints, and ONNX. <br>
`pod trainer run` runs the provided example Trainer. <br>
`pod seed` executes teardown, moves example code provided in `lightning_pod` to a new directory `examples` in the project root directory, and then creates a new `trainer.py` `trainer.yaml` and `module.py` in `lightning_pod/core`.

The flow for creating new checkpoints and an ONNX model from the provided encoder-decoder looks like:

```sh
cd {{ path to clone }}
{{ create virtual environment using provided files }}
{{ activate virtual environment }}
pip install lightning # if using conda
pip install -e .
pod teardown
pod trainer run
```

> miniconda (on macOS) is not installing lighting from the provided environment.yml; which is why the above shows a call to `pip install` after activating the `lightning-ai` conda environment

Once the new Trainer has finished, the app can be viewed by running the following in terminal:

```sh
lightning run app app.py
```

#### Full Tear Down
The CLI command shown below will remove boilerplate to allow users to begin their own projects:

```sh
pod seed
```

The example code will be preserved in a new directory `examples` after running the above. This `examples` directory can safely be deleted if it is not needed.

Files removed:

- cached MNIST data found in `data/cache/LitDataSet`
- training splits found in `data/training_split`
- saved predictions found in `data/predictions`
- PyTorch Profiler logs found in `logs/profiler`
- TensorBoard logs found in `logs/logger`
- model checkpoints found in `models/checkpoints`
- persisted ONNX model found in `models/onnx`

## Deploying to Lightning Cloud

Deploying finished applications to Lightning is simple. If you haven't done so, create an account on [Lightning.ai](https://www.pytorchlightning.ai/). Once an account has been created, one needs only to add an additional flag to `lightning run` as shown below:

```sh
lightning run app app.py --cloud
```

This will load the app to your account, build services, and then run the app on Lightning's platform. An `Open App` button will be shown in the Lightning Web UI when your app is ready to be launched and viewed in the browser.

The name of the app loaded to Lightning can be changed in the [`.lightningai/framework/.lightning`](https://github.com/JustinGoheen/lightning-pod/tree/main/.lightningai/framework/.lightning) file or with

```sh
lightning run app app.py --cloud --name="what ever name you choose"
```

## Skills

_New to ML and software engineering students ..._

Do not be overwhelmed by the amount of files contained in the repo. The directories other than lightning_pod are a collection of "Hello, World!" like examples meant to help you begin to understand basic CI-CD, testing, documentation etc. 

If you only need to process data and implement an algorithm from a paper or pseudcode, you can focus on `lightning_pod.core` and `lightning_pod.pipeline` and ignore the rest of the code, so long as you follow the basic class and function naming conventions I've provided. _Altering the naming conventions will cause breaking changes_.

### Software Engineering

The Lightning team has created a series of [Engineering for Researchers](https://www.pytorchlightning.ai/edu/engineering-class) videos to help individuals become familiar with software engineering best practices.

### Deep Learning

Grant Sanderson, also known as 3blue1brown on YouTube, has provided a very useful, high level [introduction to neural networks](https://www.3blue1brown.com/topics/neural-networks). Grant's [other videos](https://www.3blue1brown.com/#lessons) are also useful for computer and data science, and mathematics in general.

NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning. Additionally, Professor Canziani was kind enough to make his course materials public [on GitHub](https://github.com/Atcold/NYU-DLSP21).

The book [Dive into Deep Learning](http://d2l.ai/#), created by a team of Amazon engineers, is availlable for free.

DeepMind has shared several lectures series created for UCL [on YouTube](https://www.youtube.com/c/DeepMind/playlists?view=50&sort=dd&shelf_id=9).

OpenAI has created [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), an introductory series in reinforcement learning and deep learning.

### Additional Resources

Aside from the above, I've started a [wiki](https://justingoheen.github.io/lightning-engineer/) to help guide individuals through some of the concepts and tooling discussed in this document.

## Tooling

The tooling i.e. the dependencies, or stack, was selected by referring to the Lightning ecosystem repos: PyTorch Lightning, Lightning Flash, torchmetrics etc. Tooling not used by the Lightning team is also used, and is described below briefly, and in the wiki in greater detail.

### Lightning Stack

The lightning team typically uses DeepSource, CircleCI, GitHub Actions, and Azure Pipelines for top level CI/CD management. At a deeper level, the team uses PyTest + coverage + CodeCov for unit testing, mypy for type checking, flake8 + Black for linting and formatting, pre-commit to for git commit QA, and mergify for automating PR merges which pass all CI/CD checks.

Azure Pipelines, pre-commit, and mergify are not used in this project repo.

### Extras

This repo uses a GitHub Action for GitHub CodeQL security analysis; this action is the default action set by GitHub when enabling [code scanning](https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/about-code-scanning) for any repo.

## Cloud Development

Lightning Pod enables collaborative development with Gitpod and GitHub CodeSpaces. Please note that these tools have only been tested on creating and training a custom LightningModule i.e. it is necessary to debug Lightning and Dash apps locally. Lastly, GitHub CodeSpaces is still in beta for individual pro accounts. Gitpod offers 50 free hours per month. Support for [Grid Sessions](https://docs.grid.ai/features/sessions) is planned.

<div align="center">

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/JustinGoheen/lightning-pod)

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

Gitpod and CodeSpaces uses pyenv instead of conda ... meaning the terminal commands to use the CLI's are slightly different.

Once the workspace image has finished building, do the following to teardown the example and run a trainer of your own from the provided example LightningModule:

```sh
pod teardown
pod trainer run
```

If using VS Code (in browser or on desktop), it is possible to view PyTorch Profiler and TensorBoard logs when using Gitpod or CodeSpaces. Access the VS Code command palette and enter `>Python: Launch TensorBoard`. A new port will start; TensorBoard will launch once the new port is active. If the TensorBoard window remains blank, close it and restart the TensorBoard session.

## Getting Help

Please join the [Lightning Community Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-19m2xnz2o-hC80K2vGCoGCpP4vTh6T1g) for questions about the Lightning ecosystem. Feel free to @ me in Slack if you have a question specific to this repo.

## Contributing

There is no need to submit an issue or PR to this repo. This template is exactly that – a template for others to fork or clone and improve on, and share with the community. My hopes in sharing this template is that new to ML students or PhD researchers in any domain can quickly form a project from trustworthy boilerplate.