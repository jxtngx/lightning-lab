from setuptools import setup
from setuptools import find_packages
from pathlib import Path

console_scripts = """
[console_scripts]
pod=lightning_pod.cli.pod:main
"""

rootdir = Path(__file__).parent
long_description = (rootdir / "README.md").read_text()

setup(
    name="lightning-pod",
    version="0.0.4.1",
    description="A Lightning.ai application seed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JustinGoheen/lightning-pod",
    author="Justin Goheen",
    license="Apache 2.0",
    # install_requires=[
    #     "lightning",
    #     "lightning-grid",
    #     "pytorch_lightning",
    #     "torchvision",
    #     "dearpygui",
    #     "click",
    #     "textual",
    #     "rich-cli",
    #     "colorama",
    #     "tabulate",
    #     "plotly",
    #     "dash",
    #     "dash-bootstrap-components",
    #     "optuna",
    #     "hydra-core",
    #     "torch-tb-profilers",
    # ],
    author_email="",
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Environment :: Console",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points=console_scripts,
)
