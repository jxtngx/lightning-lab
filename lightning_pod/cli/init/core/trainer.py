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
from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.profilers import Profiler, PyTorchProfiler

from lightning_pod import conf


class PodTrainer(L.Trainer):
    def __init__(
        self,
        logger: Optional[Logger] = None,
        profiler: Optional[Profiler] = None,
        callbacks: Optional[List] = [],
        plugins: Optional[List] = [],
        set_seed: bool = True,
        **trainer_init_kwargs: Dict[str, Any]
    ) -> None:
        # SET SEED
        if set_seed:
            seed_everything(conf.GLOBALSEED, workers=True)
        super().__init__(
            logger=logger or TensorBoardLogger(conf.LOGSPATH, name="tensorboard"),
            profiler=profiler or PyTorchProfiler(dirpath=conf.TORCHPROFILERPATH, filename="profiler"),
            callbacks=callbacks + [ModelCheckpoint(dirpath=conf.CHKPTSPATH, filename="model")],
            plugins=plugins,
            **trainer_init_kwargs
        )

    def persist_predictions(self, predictions_dir: Optional[Union[str, Path]] = conf.PREDSPATH) -> None:
        self.test(ckpt_path="best", datamodule=self.datamodule)
        predictions = self.predict(self.model, self.datamodule.val_dataloader())
        torch.save(predictions, predictions_dir)
