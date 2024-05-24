class LabDataset:
    """
    Note:
        see below for a basic example of a custom torch dataset
    ```python
    import pandas as pd
    import torch
    from torch.utils.data import Dataset


    class LabDataset(Dataset):
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
