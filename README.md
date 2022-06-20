<div align="center">

# Hello, Lightning!


![](https://img.shields.io/badge/PyTorch_Lightning-Ecosystem-informational?style=flat&logo=pytorchlightning&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Grid.ai-Cloud_Compute-informational?style=flat&logo=grid.ai&logoColor=white&color=2bbc8a)


</div>

# Overview

This project is a template Python environment, tooling, and system architecture for [Lightning](https://www.pytorchlightning.ai/) OS that culminates with a Plotly Dash [UI](https://01g5ym5thgarst5xzx1vd6mqfa.litng-ai-03.litng.ai/view/home) deployed to the Lightning platform.

## Using the Template

The intent is that users [create a new repo from the template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) in GitHub's web interface and then clone the newly created repo in their personal account to their local machine.

## Viewing the App Locally

Once the repo has been cloned, the app can be viewed locally by running the following in terminal (assumes conda environment manager):

```sh
cd {{ path to clone }}
conda env create --file environment.yml
conda activate lightning-os
pip install lightning
pip install -e .
lightning run app app.py
```

> miniconda is not installing lightning from pip when added to the enviroment.yml; hence the `pip install lightning` that follows the environment activation 

> `pip install -e .` will install an editable version of the `lightning-pod` module to your Python environment and must be ran only once.

## Deploying to Lightning Cloud

Deploying finished applications to Lightning is simple; one needs only to add an additional flag to `lightning run` as shown below:

```sh
lightning run app app.py --cloud
```

# Software Engineering and Machine Learning

Many ML courses are taught via notebooks, leading to a possible gap in software engineering skills or outright bad practices for researchers and data science students. The resources shown below can help to build the requisite skills to be considered industry ready.

## Software Engineering

The Lightning team has created a series of [Engineering for Researchers](https://www.pytorchlightning.ai/edu/engineering-class) videos to help upskill individuals for work beyond of notebooks.

## Deep Learning

For software engineers in need of deep learning know-how, NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning. Additionally, Professor Canziani was kind enough to make public his course notes [on GitHub](https://github.com/Atcold/NYU-DLSP21).

## Additional Resources

Aside from the above, I've written a [wiki](https://github.com/JustinGoheen/lightning-pod/wiki) to help guide individuals through some of the concepts and tooling discussed below.

# Tooling

The tooling i.e. the dependencies, or stack, was selected by referring to the Lightning ecosystem repos: PyTorch Lightning, Lightning Flash, torchmetrics etc. Tooling not used by the Lightning team is also used, and is described below briefly, and in the wiki in greater detail.

## Lightning Stack

The lightning team typically uses DeepSource, CircleCI, GitHub Actions, and Azure Pipelines for top level CI/CD management. At a deeper level, the team uses PyTest + coverage + CodeCov for unit testing, mypy for type checking, flake8 + Black for linting and formatting, pre-commit to for git commit QA, and mergify for automating PR merges which pass all CI/CD checks.

Azure Pipelines, pre-commit, and mergify are not used in this project repo.

## Extras

This repo uses a GitHub Action for GitHub CodeQL security analysis; this action is the default action set by GitHub when enabling [code scanning](https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/about-code-scanning) for any repo.

Additionally, this repo includes boilerplate for reproducible environments with Gitpod or GitHub CodeSpaces.

# Getting Help

Please join the [Lightning Community Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-19m2xnz2o-hC80K2vGCoGCpP4vTh6T1g) for questions about the Lightning ecosystem. After introducing yourself in #introduce-yourself, it is best to post to the #questions channel of the Lightning Slack to receive quick support from the community. Feel free to @ me in Slack if you have a question specific to this repo.

# Contributing

There is no need to submit an issue or PR to this repo. This template is exactly that – a template for others to fork or clone and improve on, and share with the community. My hopes in sharing this template is that new to ML students or PhD researchers in any domain can quickly form a project from trustworthy boilerplate.