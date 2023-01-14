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

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" height=25/>   <img src ="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" height=25/>
<br>

<!-- [![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod) -->

[![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-pod.svg?style=shield)


</div>

## Overview

Lightning Pod is a template Python environment, tooling, and architecture for deep learning research projects using the [Lightning.ai](https://lightning.ai) ecosystem. The project culminates with a Dash UI (shown below) to display training results.

![](assets/dash_ui.png)


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

### Core Code

`lightning_pod.core` contains code for LightningModule and the Trainer.

`lightning_pod.pipeline` contains code for data preprocessing, building a Torch Dataset, and LightningDataModule.

If you only need to process data and implement an algorithm from a paper or pseudcode, you can focus on `lightning_pod.core` and `lightning_pod.pipeline` and ignore the rest of the code, so long as you follow the basic class and function naming conventions I've provided.

> Altering the naming conventions will cause the flow to break. Be sure to refactor correctly.

### Using the Template

The intent is that users [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) this repo, set that fork as a [template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository), then [create a new repo from their template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template), and lastly [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) their newly created repo created from the template.

> it is recommended to keep your fork of lightning-pod free of changes and synced with the lightning-pod source repo, as this ensures new features become available immediately after release

Examples can be found in [lighting-pod/examples/](examples/).

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

A [CLI](https://github.com/JustinGoheen/lightning-pod/blob/main/lightning_pod/cli/console.py) `pod` is provided to assist with certain project tasks and to interact with Trainer. The commands for `pod` and their effects are shown below.

<details>
  <summary>pod</summary>

`pod teardown` will destroy any existing data splits, saved predictions, logs, profilers, checkpoints, and ONNX. <br>

`pod trainer run-hydra` runs the example hydra Trainer. <br>

`pod trainer run-wandb --project-name=your-project-name` runs the example wandb Trainer. <br>

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


class LitDataset(Dataset):
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
lightning run app your_app_file_name.py --cloud
```

On command to load to cloud, Lightning will look for two files in the root directory `.lightningignore` and `.lightning`.

`.lightning` is the config file.

`.lightningignore` is a more granular version of gitignore that allows users to be specific about which project files should be loaded to Lightning Cloud.

## GitHub CodeSpaces

Lightning Pod enables development with GitHub CodeSpaces. Please note that lightning-pod has only been tested with regard to creating and training a custom LightningModule i.e. it is necessary to debug Lightning and Dash apps locally.

<div align="center">

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new?repo=JustinGoheen/lightning-pod)

</div>

Once the workspace image has finished building, do the following to teardown the example and run a trainer of your own from the provided example LightningModule:

```sh
pod teardown
pod trainer run-hydra
```

## Learning Resources

### Deep Learning

Lightning AI's Sebastian Raschka has created a [free series on Deep Learning](https://lightning.ai/pages/courses/deep-learning-fundamentals/).

NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning. Additionally, Professor Canziani was kind enough to make his course materials public [on GitHub](https://github.com/Atcold/NYU-DLSP21).

Grant Sanderson, also known as 3blue1brown on YouTube, has provided a very useful, high level [introduction to neural networks](https://www.3blue1brown.com/topics/neural-networks). Grant's [other videos](https://www.3blue1brown.com/#lessons) are also useful for computer and data science, and mathematics in general.

The book [Dive into Deep Learning](http://d2l.ai/#), created by a team of Amazon engineers, is availlable for free.

DeepMind has shared several lectures series created for UCL [on YouTube](https://www.youtube.com/c/DeepMind/playlists?view=50&sort=dd&shelf_id=9).

OpenAI has created [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), an introductory series in deep reinforcement learning.

### Reviewing Source Code

The following three videos were created by Lightning's Thomas Chaton; the videos are extremely helpful in learning how to use code search features in VS Code to navigate a project's source code, enabling a deeper understanding of what is going on under the hood.

> these videos were created before PyTorch Lightning was moved into the Lightning Framework mono repo

[Lightning Codebase Deep Dive 1](https://youtu.be/aEeh9ucKUkU) <br>
[Lightning Codebase Deep Dive 2](https://youtu.be/NEpRYqdsm54) <br>
[Lightning Codebase Deep Dive 3](https://youtu.be/x4d4RDNJaZk)
