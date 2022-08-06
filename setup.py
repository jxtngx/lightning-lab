# Copyright Justin R. Goheen.
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
# limitations under the License.


from pathlib import Path

from setuptools import find_packages, setup

console_scripts = """
[console_scripts]
pod=lightning_pod.cli.console:main
"""

rootdir = Path(__file__).parent
long_description = (rootdir / "README.md").read_text()

setup(
    name="lightning-pod",
    version="0.0.5",
    description="A Lightning.ai application seed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JustinGoheen/lightning-pod",
    author="Justin Goheen",
    license="Apache 2.0",
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
