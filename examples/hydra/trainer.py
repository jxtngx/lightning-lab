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

from lightning_pod import conf
from lightning_pod.core.trainer import LitTrainer

FILEPATH = Path(__file__)


@hydra.main(
    config_path=str(FILEPATH.parent),
    config_name="trainer",
    version_base=hydra.__version__,
)
def main(cfg: DictConfig) -> None:
    # SET MODEL, DATAMODULE TRAINER
    trainer = LitTrainer(callbacks=[EarlyStopping(monitor="loss", mode="min")], **cfg.trainer)
    # TRAIN MODEL
    trainer.fit(model=trainer.model, datamodule=trainer.datamodule)
    # IF NOT FAST DEV RUN: TEST, PREDICT, PERSIST
    if not cfg.trainer.fast_dev_run:
        # TEST MODEL
        trainer.test(ckpt_path="best", datamodule=trainer.datamodule)
        # PERSIST MODEL
        input_sample = trainer.datamodule.train_data.dataset[0][0]
        trainer.model.to_onnx(conf.MODELPATH, input_sample=input_sample, export_params=True)
        # PREDICT
        predictions = trainer.predict(trainer.model, trainer.datamodule.val_dataloader())
        # PERSIST PREDICTIONS
        trainer.persist_predictions(predictions)
        # PERSIST DATA SPLITS FOR REPRODUCIBILITY
        trainer.datamodule.persist_splits()


if __name__ == "__main__":
    main()
