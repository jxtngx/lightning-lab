<!-- # Copyright Justin R. Goheen.
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
# limitations under the License. -->

<div align="center">

# Lightning Pod

![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
<a href="https://lightning.ai" ><img src ="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" height="28"/> </a>

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

## Overview

Lightning Pod is a public template for deep learning research projects using the [Lightning.ai](https://lightning.ai) ecosystem. It is meant to be minimal and high-level in nature so that template is helpful, but not bloated.

Lightning Pod is inspired by ReactJS utilities such as CRA and CRACOS in that each of those utilities provides opionated boilerplate that has now become convention.

The intent is that users create new repos from the [use this template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) button.

An example project can be found at [lightning-pod-example](https://github.com/JustinGoheen/lightning-pod-example)..

### Project Requirements and Extras

Lightning Pod provides opinionated project requirements to enable easy creation of new virtual environments. The following is a list of core requirements and extras.

<details>
  <summary>Core AI/ML Ecosystem</summary>

These are the base frameworks. Many other tools (numpy, pyarrow etc) are installed as dependencies when installing the core dependencies.

- pytorch-lightning
- lightning-app
- lightning-trainging-studio (HPO)
- torchmetrics
- weights and biases
- aim
- optuna
- hydra
- plotly
- dash
- pynecone

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

### Source Code

The source module,`lightning_pod/`, provides a structure for users to begin from. That structure is as follows:

`lightning_pod.cli` contains code for the command line interface.

`lightning_pod.core` contains code for Lightning Module and Trainer.

`lightning_pod.fabric` contains MixIns, Hooks, and utilities.

`lightning_pod.pipeline` contains code for data acquistion preprocessing, building a TorchDataset, and LightningDataModule.

`lightning_pod.components` contains Lightning Flows and Works grouped by purpose.

`lightning_pod.pages` contains code for data apps. `pages` is borrowed from React project concepts.

### Using the Template

#### Creating an Environment

Base dependencies can be viewed in [setup.cfg](https://github.com/JustinGoheen/lightning-pod/blob/main/setup.cfg), under `install_requires`.

Instructions for creating a new environment are shown below.

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
python3 -m venv .venv/
# to activate on windows
.venv\Scripts\activate.bat
# to activate on macos and Unix
source .venv/bin/activate
# install lightning-pod
pip install -e .
# if desired, install extras
pip install -r requirements/extras.txt
{{ set interpreter in IDE }}
```

</details>

## GitHub CodeSpaces

Lightning Pod enables development with GitHub CodeSpaces. Please note that lightning-pod has only been tested with regard to creating and training a custom LightningModule in the CodeSpace i.e. it is necessary to debug Lightning and Dash apps locally.

<div align="center">

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

## Learning Resources

### Reviewing Source Code

The following three videos were created by Lightning's Thomas Chaton; the videos are extremely helpful in learning how to use code search features in VS Code to navigate a project's source code, enabling a deeper understanding of what is going on under the hood of someone else's code.

> these videos were created before PyTorch Lightning was moved into the Lightning Framework mono repo

[Lightning Codebase Deep Dive 1](https://youtu.be/aEeh9ucKUkU) <br>
[Lightning Codebase Deep Dive 2](https://youtu.be/NEpRYqdsm54) <br>
[Lightning Codebase Deep Dive 3](https://youtu.be/x4d4RDNJaZk)

### General Engineering and Tools

Lightning's founder, and their lead educator have created a series of short videos called [Lightning Bits](https://lightning.ai/pages/ai-education/#bits) for beginners who need guides for using IDEs, git, and terminal.

A long standing Python community resource has been [The Hitchhiker's Guide to Python](https://docs.python-guide.org). The "guide exists to provide both novice and expert Python developers a best practice handbook for the installation, configuration, and usage of Python on a daily basis".

[VS Code](https://code.visualstudio.com/docs) and [PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html) IDEs have each provided great docs for their users. My preference is VS Code - though PyCharm does have its benefits and is absolutely a suitable alternative to VS Code. I especially like VS Code's [integrations for PyTorch and tensorboard](https://code.visualstudio.com/docs/datascience/pytorch-support). I pair [Gitkraken](https://www.gitkraken.com) and [GitLens](https://www.gitkraken.com/gitlens) with VS Code to manage my version control and contributions.

### Data Analysis

Wes McKinney, creator of Pandas and founder of Voltron Data (responsible for Ibis, Apache Arrow etc) has released his third edition of [Python for Data Analysis](https://wesmckinney.com/book/) in an open access format.

### Intro to Artificial Intelligence and Mathematics for Machine Learning

Harvard University has developed an [Introduction to Artificial Intelligence with Python](https://www.edx.org/course/cs50s-introduction-to-artificial-intelligence-with-python) course that can be audited for free.

[Artificial Intelligence: A Modern Approach](https://www.google.com/books/edition/_/koFptAEACAAJ?hl=en&sa=X&ved=2ahUKEwj3rILozs78AhV1gIQIHbMWCtsQ8fIDegQIAxBB) is the most widely used text on Artificial Intelligence in college courses.

[Mathematics for Machine Learning](https://mml-book.github.io) provides "the necessary mathematical skills to read" books that cover advanced maching learning techniques.

Grant Sanderson, also known as 3blue1brown on YouTube, has provided a very useful, high level [introduction to neural networks](https://www.3blue1brown.com/topics/neural-networks). Grant's [other videos](https://www.3blue1brown.com/#lessons) are also useful for computer and data science, and mathematics in general.

### Deep Learning

Lightning AI's Sebastian Raschka has created a [free series on Deep Learning](https://lightning.ai/pages/courses/deep-learning-fundamentals/).

NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning. Additionally, Professor Canziani was kind enough to make his course materials public [on GitHub](https://github.com/Atcold/NYU-DLSP21).

The book [Dive into Deep Learning](http://d2l.ai/#), created by a team of Amazon engineers, is availlable for free.

DeepMind has shared several lectures series created for UCL [on YouTube](https://www.youtube.com/c/DeepMind/playlists?view=50&sort=dd&shelf_id=9).

OpenAI has created [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), an introductory series in deep reinforcement learning.

### ML Ops

Weights and Biases has created a free [ML Ops](https://www.wandb.courses/courses/effective-mlops-model-development) course.
