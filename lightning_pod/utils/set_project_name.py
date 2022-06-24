import os
from pathlib import Path


def main():
    filepath = Path(__file__)
    projectpath = filepath.parents[2]
    projectpath = str(projectpath)
    projectpath = projectpath.split(os.sep)
    return projectpath[-1]
