# Examples

The provided examples are lite introductions to [hydra](https://hydra.cc) and [wandb](https://wandb.ai/site). Examples of hyperparameter optimization with [lightning-training-studio](https://github.com/Lightning-AI/lightning-hpo) and [Optuna](https://optuna.readthedocs.io/en/stable/) will be added soon.

To use the examples, lightning-pod must be installed to your virtual environment. If you've not created a venv, in terminal do:

```bash
python3 -m .venv/
```

then activate with

```bash
source .venv/bin/activate
```

and then install lighting-pod with

```bash
pip install -e .
```

## Hydra

Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs

[Docs](https://hydra.cc/docs/intro/)

In order to run the hydra example, in terminal do:

```bash
pod trainer run-hydra
```

A training run will start in your terminal and lightning will output information to the terminal.

## Weights and Biases (wandb)

wandb can be used to track and visualize experiments in real time, compare baselines, and iterate quickly on ML projects.

[Docs](https://docs.wandb.ai/)

You must have a wandb account to use this example.

In order to run the wandb example, in terminal do:

```bash
pod trainer run-wandb
```

A training run will start in your terminal and lightning will output information to the terminal. Results will be synced to the project [`lightning-pod-examples`](https://wandb.ai/justingoheen/lightning-pod-examples) in your wandb account.


## Optuna

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.

[Docs](https://optuna.readthedocs.io/en/stable/reference/index.html)
