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

import os
from pathlib import Path

# SET PATHS
filepath = Path(__file__)
PROJECTPATH = filepath.parents[1]
LOGSPATH = os.path.join(PROJECTPATH, "logs")
TORCHPROFILERPATH = os.path.join(LOGSPATH, "torch_profiler")
SIMPLEPROFILERPATH = os.path.join(LOGSPATH, "simple_profiler")
CHKPTSPATH = os.path.join(PROJECTPATH, "models", "checkpoints")
MODELPATH = os.path.join(PROJECTPATH, "models", "onnx", "model.onnx")
PREDSPATH = os.path.join(PROJECTPATH, "data", "predictions", "predictions.pt")
SPLITSPATH = os.path.join(PROJECTPATH, "data", "training_split")
WANDBPATH = os.path.join(PROJECTPATH, "logs", "wandb_logs")
OPTUNAPATH = os.path.join(PROJECTPATH, "logs", "optuna")

# GLOBAL SEED
GLOBALSEED = 42
