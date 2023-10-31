# Lightning Lab

## Overview

Lightning Lab is a public template for artificial intelligence and machine learning research projects using Lightning AI's [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/).

The recommended way for Lightning Lab users to create new repos is with the [use this template](https://github.com/new?template_name=lightning-lab&template_owner=JustinGoheen) button.

You can find domain specific variations of Lightning Lab on [my wiki](https://justingoheen.github.io/pages/lightninglabs/).

## The Structure

### Source Module

`lightninglab.cli` contains code for the command line interface built with [Typer](https://typer.tiangolo.com/).

`lightninglab.components` contains experiment utilities grouped by purpose for cohesion.

`lightninglab.core` contains code for the Lightning Module and Trainer.

`lightninglab.pipeline` contains code for data acquistion and preprocessing, and building a TorchDataset and LightningDataModule.

`lightninglab.serve` contains code for model serving APIs built with [FastAPI](https://fastapi.tiangolo.com/project-generation/#machine-learning-models-with-spacy-and-fastapi).

`lightninglab.config` assists with project, trainer, and sweep configurations.

### Project Root

`data` directory should be used to cache the TorchDataset and training splits locally if the size of the dataset allows for local storage. additionally, this directory should be used to cache predictions during HPO sweeps.

`docs` directory should be used for technical documentation.

`logs` directory contains logs generated from experiment managers and profilers.

`models` directory contains training checkpoints and the pre-trained production model.

`notebooks` directory can be used to present exploratory data analysis, explain math concepts, and create a presentation notebook to accompany a conference style paper.

`requirements` directory should mirror base requirements and extras found in setup.cfg. the requirements directory and _requirements.txt_ at root are required by the basic `Coverage` GitHub Action.

`tests` module contains unit and integration tests targeted by pytest.

`streamlit` contains the Streamlit UI.

`setup.py` `setup.cfg` `pyproject.toml` and `MANIFEST.ini` assist with packaging the Python project.

`.pre-commit-config.yaml` is required by pre-commit to install its git-hooks.

## Installation

Lightning Lab installs minimal requirements out of the box, and provides extras to make creating robust virtual environments easier. To view the requirements, in [setup.cfg](setup.cfg), see `install_requires` for the base requirements and `options.extras_require` for the available extras.

The recommended install is as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all, { domain extra(s) }]"
```

where { domain extra(s) } is one of, or some combination of (vision, text, audio, rl, forecast) e.g.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all, vision]"
```

!!! warning

   Do not install multiple variations of Lightning Lab into a single virtual environment. As this will override the `lab` CLI for each new variation that is installed.

## Refactoring the Template

Lightning Lab is a great template for deep learning projects. Using the template will require some refactoring if you intend to rename `src/lightninglab` to something like `src/textlab`. You can refactor in a few simple steps in VS Code:

1. Start by renaming the `src/lightninglab` to something like `src/textlab` or `src/imagenetlab`. Doing so will allow VS Code to refactor all instance of `lightninglab` that exists in any `.py` file.
2. Open the search pane in VS Code and search for `lightniglab` in `tests/` and replace those occurences with whatever you have renamed the source module to.
3. Next, search for `lightninglab` and replace those occurences in all `.toml` `.md` `cfg` files and string occurences in `.py` files.
4. Next, search for Lightning Lab and change that to your repo name.
5. Next, search for my name – `Justin Goheen` and replace that with either your name or GitHub username.
6. Next, search once again for my name as `justingoheen` and do the following:
   - replace the occurences in `mkdocs.yml` with your GitHub username.
   - replace the occurences in `authors.yml` with your choice of author name for your docs and blog.

## Tools and Concepts

- Hyperparameter Sweeps and experiment management with [Weights and Biases](https://wandb.ai/site)
- Command Line Interfaces with [Typer](https://typer.tiangolo.com)
- Model serving with [FastAPI](https://fastapi.tiangolo.com)
- UIs with [Streamlit](https://streamlit.io) and [Plotly](https://plotly.com/python/)
- Using [python-dotenv](https://github.com/theskumar/python-dotenv)
- Documenting code with [mkdocstrings](https://mkdocstrings.github.io) and [material-for-mkdocs](https://squidfunk.github.io/mkdocs-material/)
- Supabase's [Python client](https://supabase.com/docs/reference/python/initializing)
- [PyTest](https://docs.pytest.org/en/stable/)
- [Ruff](https://docs.astral.sh/ruff/)
- [MyPY](https://mypy.readthedocs.io/en/stable/)
- [Black](https://black.readthedocs.io/en/stable/)
- [GitHub Actions](https://github.com/features/actions)
- Version control as experiment management
