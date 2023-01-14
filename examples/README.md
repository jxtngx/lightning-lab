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

In order to run the hydra example, in terminal do:

```bash
pod trainer run-hydra
```

A training run will start in your terminal and lightning will output information to the terminal.

## Weights and Biases (wandb)

You must have a wandb account to use this example.

In order to run the wandb example, in terminal do:

```bash
pod trainer run-wandb --project-name=LP-Example
```

A training run will start in your terminal and lightning will output information to the terminal. Results will be synced to the project `LP-Example` in your wandb account.
