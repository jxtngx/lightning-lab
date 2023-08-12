# Lightning Pod

## Overview

Lightning Pod is a public template for artificial intelligence and machine learning research projects using the [Lightning.ai](https://lightning.ai) ecosystem.

The recommended way for Lightning Pod users to create new repos is with the [use this template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) button.

An adaptation can be found at [lightning-vision](https://github.com/JustinGoheen/lightning-pod-vision).

## The Structure

### Source Module

`lightning_pod.core` should contain code for the Lightning Module and Trainer.

`lightning_pod.backend` should contain MixIns, Hooks, and utilities.

`lightning_pod.components` should contain Lightning Flows and Works for use with LightningApps, grouped by purpose for cohesion.

`lightning_pod.pipeline` should contain code for data acquistion and preprocessing, and building a TorchDataset and LightningDataModule.

`lightning_pod.api` should contain code for model serving APIs built with [FastAPI](https://fastapi.tiangolo.com/project-generation/#machine-learning-models-with-spacy-and-fastapi).

`lightning_pod.cli` should contain code for the command line interface built with [Click](https://click.palletsprojects.com/en/8.1.x/) and [Rich](https://rich.readthedocs.io/en/stable/).

`lightning_pod.pages` should contain code for data apps built with streamlit, dash, or reflex. the `pages` module naming convention is borrowed from React concepts.

`lightning_pod.conf.py` and `lightning_pod.conf.yaml` can assist with project, trainer, and sweep configurations. The .yaml file is provided for users who may want to use hydra.cc, or omegaconf with click; I prefer omegaconf and click.

### Project Root

<details>
    <summary>Root Directories and Files</summary>
    <br>

`app.py` is the Lightning App.

`assets` directory contains CSS and images for pages.

`data` directory should be used to cache the TorchDataset and training splits locally if the size of the dataset allows for local storage. additionally, this directory should be used to cache predictions during HPO sweeps.

`docs` directory should be used to store technical documentation.

`logs` directory will store logs generated from experiment managers and profilers.

`models` directory will store training checkpoints and the pre-trained ONNX model.

`notebooks` directory can be used to present exploratory data analysis, explain math concepts, and create a presentation notebook to accompany a conference style paper.

`requirements` directory should mirror base requirements and extras found in setup.cfg. the requirements directory and _requirements.txt_ at root are required by the basic CircleCI GitHub Action.

`tests` module contains unit and integration tests targeted by pytest.

`.lightning` and `.lightningignore` are used by Lightning as config files.

`setup.py` `setup.cfg` `pyproject.toml` and `MANIFEST.ini` assist with packaging the Python project.

`.pre-commit-config.yaml` is required by pre-commit to install its git-hooks.

</details>

## Base Requirements and Extras

Lightning Pod installs minimal requirements out of the box, and provides extras to make creating robust virtual environments easier. To view the requirements, in [setup.cfg](setup.cfg), see `install_requires` for the base requirements and `options.extras_require` for the available extras.

> popular alternatives are listed in the extras, and commented out to avoid installation. to use the alternatives, uncomment the line and then comment out or delete the libraries you do not want to install

The recommended install is as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full, { domain extra(s) }]"
```

where { domain extra(s) } is one of, or some combination of (vision, text, audio, rl, forecast) e.g.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full, vision]"
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

Lightning AI's Sebastian Raschka has created a [free series on Deep Learning](https://lightning.ai/pages/courses/deep-learning-fundamentals/) and has shared his [university lectures](https://sebastianraschka.com/teaching/).

NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning and has also made his his course materials public [on GitHub](https://github.com/Atcold/NYU-DLSP21).

The book [Dive into Deep Learning](http://d2l.ai/#), created by a team of Amazon engineers, is availlable for free.

DeepMind has shared several lectures series created for UCL [on YouTube](https://www.youtube.com/c/DeepMind/playlists?view=50&sort=dd&shelf_id=9).

OpenAI has created [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), an introductory series in deep reinforcement learning.

### MLOps

Weights and Biases has created a free course for MLOps titled [Effective MLOps: Model Development](https://www.wandb.courses/courses/effective-mlops-model-development)
