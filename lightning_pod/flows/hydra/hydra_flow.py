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

import hydra
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf.dictconfig import DictConfig

from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule

FILEPATH = Path(__file__)


@hydra.main(
    config_path=str(FILEPATH.parent),
    config_name="trainer",
    version_base=hydra.__version__,
)
def main(cfg: DictConfig) -> None:
    # SET MODEL, DATAMODULE TRAINER
    model = PodModule()
    datamodule = PodDataModule()
    trainer = PodTrainer(callbacks=[EarlyStopping(monitor="training_loss", mode="min")], **cfg.trainer)
    # TRAIN MODEL
    trainer.fit(model=model, datamodule=datamodule)
    # IF NOT FAST DEV RUN: TEST, PREDICT, PERSIST
    if not cfg.trainer.fast_dev_run:
        # TEST MODEL
        trainer.test(ckpt_path="best", datamodule=trainer.datamodule)


if __name__ == "__main__":
    main()
