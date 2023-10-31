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

from setuptools import setup
from setuptools import find_packages

rootdir = Path(__file__).parent
long_description = (rootdir / "README.md").read_text()

package_name = "lightninglab"

setup(
    name=package_name,
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=[package_name, f"{package_name}.*"]),
    include_package_data=True,
    setup_requires=["wheel"],
    zip_safe=False,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    license="Apache 2.0",
    author_email="",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)