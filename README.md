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
[![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg)](https://wandb.ai/justingoheen/lightningpod-train-wandb?workspace=user-justingoheen)

[![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-pod.svg?style=shield)

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

## Overview

Lightning Pod is a template Python environment, tooling, and architecture for deep learning research projects using the [Lightning.ai](https://lightning.ai) ecosystem. It is meant to be minimal and high-level in nature so that the project remains easy to understand across the breadth of topics.

The project culminates with a Dash UI (shown below) to display training results. The UI is implemented as a Lightning App that can be shared via Lightning Cloud.

![](assets/dash_ui.png)

The intent is that users [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repo, set that fork as a [template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository), then [create a new repo from their template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template), and lastly [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) their newly created repo.

> it is recommended to keep your fork of lightning-pod free of changes and synced with the lightning-pod source repo, as this ensures new features become available immediately

### Using Lightning Pod as an Instructional Resource

Lightning Pod does provide an example Lightning Module trained on MNIST; however, there are purposeful gaps in the data pipeline, network design, and UI that will allow instructors to use this project as a template to provide to students.

The inclusion of a GitHub CodeSpace devcontainer allows instructors to provide a uniform development environment to students. Pairing this with GitHub version control means instructors can also use the git tree to track student progress, and use development branches as a way to help students troubleshoot their way through issues.

> Additional how-to content will be created during 2023 to assist Professors and TAs in using lighting-pod for their courses.

### Covered Concepts

The following concepts are introduced at a high level in lightning-pod. While not all are discussed in this README or associated blog posts, the code base can serve as examples of each.

<details>
  <summary>Concepts</summary>

- _CI/CD_: `.github` and `.circleci` directories contain basic configs for running CircleCI and GitHub Action jobs.
- _Python Programming_: the project forces students into using interfaces (classes) and inheritance. The hope here is that early exposure to such principles will allow students to quickly upskill their programming abilities.
- _Code Quality_: when installed correctly, the project will use pre-commit to format a user's code with [black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/). MyPy and PyTest are also available. Coverage will run pytest for commits to the default branch. Tests are located in `tests/` and are based on minimal examples found in Lightning's code base.
- _Packaging_: The current (Feb 18 2023) version uses methods provided by `setuptools` in `setup.py`, `setup.cfg`, and `pyproject.toml`. Some of these will be consolidated to reflect changes in Lightning to include [`ruff`](https://github.com/charliermarsh/ruff) as a tool. Basic use of `build` and `twine` for distribution via pypi can be found in the following [quickstart](https://setuptools.pypa.io/en/latest/userguide/quickstart.html).
- _AI/ML Research Project Architecture_: while opinionated, this repo does provide a basic structure for research projects, and is certainly suitable for algorithm design research that will rely on datasets provided in torchvision, torchaudio, and torchtext. By architecture, I mean the location of directories such as `data`, `logs`, `models`, and `notebooks`. The source code is contained in `lightning_pod`, it's structure is formed from concepts found in Lightning's core projects, and in React projects given the inclusion of the `pages` module in the source directory and `assets` at root.
- _Experiment Management_: the project provides examples for [`wandb`](https://wandb.ai/site), [`tensorboard`](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html), and [`aim`](https://aimstack.io) to log hyperparameter optimization sweeps and training runs. The project also provides an example of using [Optuna](https://optuna.readthedocs.io/en/stable/) as a replacement for wandb's built-in Sweeps.
- _UIs and DataViz_: Any React app will be in a directory of its own at the project's root, and will be structured according to Next or Vite's setup. React apps may be created with a javascript utility or Pynecone.
- _CLIs_: a command line interface entrypoint `pod` serves as an example CLI. It is built with [click](https://click.palletsprojects.com/en/8.1.x/). The code for `pod` is found in `lightning_pod.cli.pod` and it is created in `setup.cfg`'s entry point option.

To some degree, the concepts would be covered in the following courses:

- Intro and Intermediate Python programming
- Intro to data acquisition and pre-processing
- Intro to user interfaces and data visualization
- Intro to AI, ML, and DL (types of agents, styles of learning, and algorithms and data structures)
- Intro to research i.e. applied machine learning or designing learning agents
- Intro to distributed systems
- Intro to hardware for AI/ML systems (HPU, IPU, TPU, GPU)

</details>

### Project Requirements and Extras

<details>
  <summary>Core AI/ML Ecosystem</summary>

These are the base frameworks. Many other tools (numpy, pyarrow etc) are installed as dependencies when installing the core dependencies.

- pytorch-lightning
- lightning-app
- lightning-trainging-studio (HPO)
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
- CircleCI

</details>

### Source Code

`lightning_pod.cli` contains code for the CLI `pod`.

`lightning_pod.core` contains code for Lightning Module and Trainer.

`lightning_pod.pipeline` contains code for data preprocessing, building a Torch Dataset, and LightningDataModule.

`lightning_pod.components` contains Lightning Flows and Works grouped by purpose.

`lightning_pod.pages` contains UIs built with Dash or PyneCone.

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

#### Command Line Interface

A CLI `pod` is provided to assist with certain project tasks and to interact with Trainer. The commands for `pod` and their effects are shown below.

<details>
  <summary>pod</summary>

`pod teardown` will destroy any existing data splits, saved predictions, logs, profilers, checkpoints, and ONNX. <br>

`pod trainer run-example` runs the Trainer in a fast-dev-run. <br>

`pod bug-report` creates a bug report to [submit issues on GitHub](https://github.com/Lightning-AI/lightning/issues) for Lightning. the report is printed to screen in terminal, and generated as a markdown file for easy submission.

`pod init` will remove boilerplate to allow users to begin their own projects.

Files removed by `pod init`:

- cached MNIST data found in `data/cache/PodDataset`
- training splits found in `data/training_split`
- saved predictions found in `data/predictions`
- PyTorch Profiler logs found in `logs/profiler`
- TensorBoard logs found in `logs/logger`
- model checkpoints found in `models/checkpoints`
- persisted ONNX model found in `models/onnx`

The flow for creating new checkpoints and an ONNX model from the provided encoder-decoder looks like:

```sh
pod teardown
pod trainer -em=wandb --persist-model --persist-predictions --persist-splits
```

Once the new Trainer has finished, the app can be viewed by running the following in terminal:

```sh
lightning run app app.py
```

</details>

> the CLI is built with [Click](https://click.palletsprojects.com/en/8.1.x/) and [Rich](https://github.com/Textualize/rich)

#### Datasets

<details>
  <summary>Scikit and PyTorch Provided Datasets</summary>

If using built-in datasets from [torchvision](https://pytorch.org/vision/stable/datasets.html), [torchaudio](https://pytorch.org/audio/stable/datasets.html), or [Lightning Bolts integration of scikit-learn datasets](https://lightning-bolts.readthedocs.io/en/latest/datamodules/sklearn.html), then creating LightningDataModules should be relatively straight forward, with little to no change necessary for the provided `lightning_pod.pipeline.datamodule`. However, be sure to pay attention to which methods and hooks are available to the respective datasets, and be ready to debug errors in `lightning_pod.pipeline.datamodule` attributed to `lightning_pod.pipeline.dataset`'s differences in hooks after using a dataset other than torchvision's MNIST.

</details>

<details>
  <summary>Custom Torch Datasets</summary>

Depending on scale and complexity, creating your own custom torch dataset can be relatively straight forward. Keep in mind that in doing so, none of the hooks available to the MNIST torch dataset used in the example will be availble to your custom dataset; you must create your own hooks and methods. You can view the source code of PyTorch and Lightning Bolts as examples of how to develop a custom dataset that will be piped to a LightningDatamodule.

A basic custom torch dataset is shown below:

```python
import pandas as pd
import torch
from torch.utils.data import Dataset


class PodDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = pd.read_csv(features_path)
        self.labels = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x, y = self.features.iloc[idx], self.labels.iloc[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

You read more on PyTorch datasets and LightningDatamodules by following the links below:

- PyTorch [Datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
- Lightning [Datamodules](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html?highlight=datamodule#what-is-a-datamodule)

> LightningDataModules handle DataLoaders; you do not need to follow the DataLoaders portion of the PyTorch tutorial

</details>

## Deploying to Lightning Cloud

Once a user has created a Lightning account, the app can be deployed to Lightning Cloud with the following command in terminal:

```bash
lightning run app app.py --cloud
```

On command to load to cloud, Lightning will look for two files in the root directory `.lightningignore` and `.lightning`.

`.lightning` is the config file.

`.lightningignore` is a more granular version of gitignore that allows users to be specific about which project files should be loaded to Lightning Cloud.

## GitHub CodeSpaces

Lightning Pod enables development with GitHub CodeSpaces. Please note that lightning-pod has only been tested with regard to creating and training a custom LightningModule in the CodeSpace i.e. it is necessary to debug Lightning and Dash apps locally.

<div align="center">

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

Once the workspace image has finished building, do the following to teardown the example and run a trainer of your own from the provided example LightningModule:

```sh
pod teardown
pod trainer run-example
```

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
