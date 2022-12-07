<div align="center">

# Lightning Pod

<!--[![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod) -->

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)


</div>

## Overview

Lightning Pod is a template Python environment, tooling, and system architecture for artificial intelligence and machine learning projects. The project culminates with an [app](https://01gcfpsrmb3cb4x9bc6sqvhazs.litng-ai-03.litng.ai/view/home) deployed to the Lightning Cloud platform.

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
  <summary>Notable Extras</summary>

  These frameworks and libraries are installed when creating an environment from the provided requirements utilities.

  - torchserve
  - fastapi
  - pydantic
  - gunicorn
  - uvicorn
  - click
  - rich
  - pyarrow
  - numpy

</details>

<details>
  <summary>Testing and Code Quality</summary>

  - PyTest
  - coverage
  - MyPy
  - Bandit
  - Black
  - isort
  - pre-commit

</details>

<details>
  <summary>Packaging</summary>

  - setuptools
  - build
  - twine

</details>

<details>
  <summary>CI/CD</summary>

  - GitHub Actions

</details>

### Core Code

`lightning_pod.core` contains code for LightningModule and and the Trainer.

`lightning_pod.pipeline` contains code for data preprocessing, building a Torch Dataset, and LightningDataModule.

If you only need to process data and implement an algorithm from a paper or pseudcode, you can focus on `lightning_pod.core` and `lightning_pod.pipeline` and ignore the rest of the code, so long as you follow the basic class and function naming conventions I've provided.

> Altering the naming conventions will cause the flow to break. Be sure to refactor correctly.

### Using the Template

The intent is that users [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repo, set that fork as a [template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository), then [create a new repo from their template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template), and lastly [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) their newly created repo created from the template.

> it is recommended to keep your fork of lightning-pod free of changes and synced with the lightning-pod source repo, as this ensures new features become available immediately after release

#### Creating an Environment

Base dependencies can be viewed in [setup.cfg](https://github.com/JustinGoheen/lightning-pod/blob/main/pyproject.toml).

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

## Learning Deep Learning

Lightning AI' Sebastian Raschka has created a [free series on Deep Learning](https://lightning.ai/pages/courses/deep-learning-fundamentals/).

Grant Sanderson, also known as 3blue1brown on YouTube, has provided a very useful, high level [introduction to neural networks](https://www.3blue1brown.com/topics/neural-networks). Grant's [other videos](https://www.3blue1brown.com/#lessons) are also useful for computer and data science, and mathematics in general.

NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning. Additionally, Professor Canziani was kind enough to make his course materials public [on GitHub](https://github.com/Atcold/NYU-DLSP21).

The book [Dive into Deep Learning](http://d2l.ai/#), created by a team of Amazon engineers, is availlable for free.

DeepMind has shared several lectures series created for UCL [on YouTube](https://www.youtube.com/c/DeepMind/playlists?view=50&sort=dd&shelf_id=9).

OpenAI has created [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), an introductory series in deep reinforcement learning.

## Cloud Development

Lightning Pod enables development with GitHub CodeSpaces. Please note that lightning-pod has only been tested with regard to creating and training a custom LightningModule i.e. it is necessary to debug Lightning and Dash apps locally.

<div align="center">

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

Once the workspace image has finished building, do the following to teardown the example and run a trainer of your own from the provided example LightningModule:

```sh
pod teardown
pod trainer run
```
