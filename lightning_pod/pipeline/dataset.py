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

from typing import Any

from torchvision.datasets import MNIST


class PodDataset(MNIST):
    """
    Note:
        see below for a basic example of a custom torch dataset
    ```python
    import pandas as pd
    import torch
    from torch.utils.data import Dataset


    class PodDataset(Dataset):
        def __init__(self, features_path, labels_path):
            self.features = pd.read_csv(features_path)
            self.labels = pd.read_csv(labels_path)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            x, y = self.features.iloc[idx], self.labels.iloc[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    ```
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
