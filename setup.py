from setuptools import setup
from setuptools import find_packages

console_scripts = """
[console_scripts]
pod=lightning_pod.cli.pod:main
"""

setup(
    name="lightning-pod",
    version="0.0.1",
    description="for local, editable installs of the lightning_pod module",
    url="https://github.com/JustinGoheen/hello-lightning",
    author="Justin Goheen",
    license="Apache 2.0",
    install_requires=[],
    author_email="",
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    entry_points=console_scripts,
)
