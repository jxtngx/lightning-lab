<div align="center">

# Lightning Pod

[![](https://img.shields.io/badge/Python-Language-informational?style=flat&logo=python&logoColor=white&color=2bbc8a)](#)
[![](https://img.shields.io/badge/Lightning.ai-Ecosystem-informational?style=flat&logo=pytorchlightning&logoColor=white&color=2bbc8a)](#)

[![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-pod.svg?style=shield)


</div>

## Overview

Lightning Pod is a template Python environment, tooling, and system architecture for the [Lightning AI](https://www.pytorchlightning.ai/) ecosystem that culminates with a Plotly Dash [UI](https://01g6bdbc5e55wc5ffgj11gtkxj.litng-ai-03.litng.ai/view/home) deployed to the Lightning platform.

The main focus of this project is to provide users with high-level (basic to intermediate) research boilerplate; inclusive of CI/CD, testing, code quality, and packaging examples.

<details>
  <summary>Core AI/ML Ecosystem</summary>

  These are the base frameworks. Many other tools (numpy, pyarrow etc) are installed as dependencies when installing the core dependencies.

  - pytorch
  - pytorch-lightning
  - lightning-app
  - lightning-hpo
  - torchmetrics
  - weights and biases
  - optuna
  - hydra
  - plotly
  - dash

</details>

<details>
  <summary>Testing and Code Quality</summary>

  - PyTest
  - coverage
  - MyPy
  - Black
  - isort
  - pre-commit

</details>

<details>
  <summary>Packaging</summary>

  - setuptools
  - build
  - twine
  - poetry

</details>

<details>
  <summary>CI/CD</summary>

  - CircleCI
  - Deepsource
  - GitHub Actions
  - Mergify

</details>

<details>
  <summary>Notable Extras</summary>

  These frameworks and libraries are installed when creating an environment from the provided poetry and conda utilities.

  - fastapi
  - pydantic
  - gunicorn
  - uvicorn
  - click
  - rich
  - pyarrow
  - numpy

</details>

### Core Code

`lightning_pod.core` contains code for Lightning Modules and Trainer.

`lightning_pod.pipeline` contains code for data preprocessing, building a Torch Dataset, and LightningDataModule.

`.lightningai` contains configs for Lightning Platform.

If you only need to process data and implement an algorithm from a paper or pseudcode, you can focus on `lightning_pod.core` and `lightning_pod.pipeline` and ignore the rest of the code, so long as you follow the basic class and function naming conventions I've provided.

> Altering the naming conventions will cause the flow to break. Be sure to refactor correctly.

### Using the Template

The intent is that users [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repo, set that fork as a [template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository), then [create a new repo from their template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template), and lastly [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) their newly created repo created from the template.

> it is recommended to keep your fork of lightning-pod free of changes and synced with the lightning-pod source repo, as this ensures new features become available immediately after release

#### Creating an Environment

Base dependencies can be viewed in [pyproject.toml](https://github.com/JustinGoheen/lightning-pod/blob/main/pyproject.toml).

Instructions for creating a new environment are shown below.

<details>
  <summary>poetry</summary>

Install [Poetry](https://python-poetry.org/docs/master/#installing-with-the-official-installer) if you do not already have it installed.

```sh
cd {{ path to clone }}
poetry install
# if desired, install extras
poetry shell
pip install -r requirements/extras.txt
{{ set interpreter in IDE }}
```
</details>

<details>
  <summary>conda</summary>

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) if you do not already have it installed.

> m-series macOS users, it is recommended to use the `Miniconda3 macOS Apple M1 64-bit bash` installation

```sh
cd {{ path to clone }}
conda env create -f environment.yml
conda activate lightning-ai
pip install -e .
# if desired, install extras
pip install -r requirements/extras.txt
{{ set interpreter in IDE }}
```
</details>

<details>
  <summary>venv</summary>

[venv](https://docs.python.org/3/library/venv.html) is not something that needs to be installed; it is part of Python standard.

```sh
cd {{ path to clone }}
python3 -m venv venv/
# to activate on windows
venv\Scripts\activate.bat
# to activate on macos and Unix
source venv/bin/activate
# install lightning-pod
pip install -e .
# if desired, install extras
pip install -r requirements/extras.txt
{{ set interpreter in IDE }}
```
</details>

#### Command Line Interface

A [CLI](https://github.com/JustinGoheen/lightning-pod/blob/main/lightning_pod/cli/console.py) `pod` is provided to assist with certain project tasks and to interact with Trainer. The commands for `pod` and their affects are shown below.

<details>
  <summary>pod</summary>

`pod teardown` will destroy any existing data splits, saved predictions, logs, profilers, checkpoints, and ONNX. <br>

`pod trainer run` runs the Trainer. <br>

`pod bug-report` creates a bug report to [submit issues on GitHub](https://github.com/Lightning-AI/lightning/issues) for Lightning. the report is printed to screen in terminal, and generated as a markdown file for easy submission.

`pod seed` will remove boilerplate to allow users to begin their own projects.

Files removed by `pod seed`:

- cached MNIST data found in `data/cache/LitDataSet`
- training splits found in `data/training_split`
- saved predictions found in `data/predictions`
- PyTorch Profiler logs found in `logs/profiler`
- TensorBoard logs found in `logs/logger`
- model checkpoints found in `models/checkpoints`
- persisted ONNX model found in `models/onnx`

The flow for creating new checkpoints and an ONNX model from the provided encoder-decoder looks like:

```sh
pod teardown
pod trainer run
```

Once the new Trainer has finished, the app can be viewed by running the following in terminal:

```sh
lightning run app app.py
```

</details>

## Deploying to Lightning Cloud

Deploying finished applications to Lightning is simple. If you haven't done so, create an account on [Lightning.ai](https://www.pytorchlightning.ai/). Once an account has been created, one needs only to add an additional flag to `lightning run` as shown below:

```sh
lightning run app app.py --cloud
```

This will load the app to your account, build services, and then run the app on Lightning's platform. An `Open App` button will be shown in the Lightning Web UI when your app is ready to be launched and viewed in the browser.

The name of the app loaded to Lightning can be changed in the [`.lightningai/.lightning`](https://github.com/JustinGoheen/lightning-pod/tree/main/.lightningai/framework/.lightning) file or with

```sh
lightning run app app.py --cloud --name="what ever name you choose"
```

## Skills

_New to ML and software engineering students ..._

Do not be overwhelmed by the amount of files contained in the repo. The directories other than lightning_pod are a collection of "Hello, World!" like examples meant to help you begin to understand basic CI-CD, testing, documentation etc.

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

The ML tooling i.e. the dependencies, or stack, was selected by referring to the Lightning ecosystem repos: PyTorch Lightning, Lightning Flash, torchmetrics etc.

Non-ML tooling (CI/CD, code quality, and PR automation) includes:

- DeepSource
- CircleCI
- GitHub Actions
- Azure Pipelines
- PyTest + coverage + CodeCov
- mypy
- flake8 + Black
- pre-commit git hooks
- mergify for PRs

### Extras

This repo uses a GitHub Action for GitHub CodeQL security analysis; this action is the default action set by GitHub when enabling [code scanning](https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/about-code-scanning) for any repo.

## Cloud Development

Lightning Pod enables collaborative development with Gitpod and GitHub CodeSpaces. Please note that these tools have only been tested on creating and training a custom LightningModule i.e. it is necessary to debug Lightning and Dash apps locally. Lastly, GitHub CodeSpaces is still in beta for individual pro accounts. Gitpod offers 50 free hours per month.

<div align="center">

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/JustinGoheen/lightning-pod)

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

Gitpod and CodeSpaces use pyenv. User do not need to create an environment, as it will be created for them on launch.

Once the workspace image has finished building, do the following to teardown the example and run a trainer of your own from the provided example LightningModule:

```sh
pod teardown
pod trainer run
```

## Getting Help

Please join the [Lightning Community Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-19m2xnz2o-hC80K2vGCoGCpP4vTh6T1g) for questions about the Lightning ecosystem. Feel free to @ me in Slack if you have a question specific to this repo.

## Contributing

There is no need to submit an issue or PR to this repo. This template is exactly that – a template for others to fork or clone and improve on, and share with the community. My hopes in sharing this template is that new to ML students or PhD researchers in any domain can quickly form a project from trustworthy boilerplate.
