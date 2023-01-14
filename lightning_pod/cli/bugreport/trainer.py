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

import hydra
import lightning as L
from lightning.pytorch.demos.boring_classes import BoringModel
from omegaconf.dictconfig import DictConfig

# SET PATHS
filepath = Path(__file__)
PROJECTPATH = os.getcwd()


@hydra.main(
    config_path=str(filepath.parent),
    config_name="trainer",
    version_base=hydra.__version__,
)
def main(cfg: DictConfig) -> None:
    model = BoringModel()
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_predict_batches=cfg.trainer.limit_predict_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        deterministic=cfg.trainer.deterministic,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        enable_model_summary=cfg.trainer.enable_model_summary,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        # logger=logger,
        # profiler=profiler,
        # callbacks=callbacks,
        # plugins=plugins,
        default_root_dir=cfg.trainer.default_root_dir,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        num_nodes=cfg.trainer.num_nodes,
        num_processes=cfg.trainer.num_processes,
        gpus=cfg.trainer.gpus,
        auto_select_gpus=cfg.trainer.auto_select_gpus,
        tpu_cores=cfg.trainer.tpu_cores,
        ipus=cfg.trainer.ipus,
        overfit_batches=cfg.trainer.overfit_batches,
        track_grad_norm=cfg.trainer.track_grad_norm,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        fast_dev_run=cfg.trainer.fast_dev_run,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        min_epochs=cfg.trainer.min_epochs,
        max_steps=cfg.trainer.max_steps,
        min_steps=cfg.trainer.min_steps,
        max_time=cfg.trainer.max_time,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        weights_save_path=cfg.trainer.weights_save_path,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        resume_from_checkpoint=cfg.trainer.resume_from_checkpoint,
        benchmark=cfg.trainer.benchmark,
        reload_dataloaders_every_n_epochs=cfg.trainer.reload_dataloaders_every_n_epochs,
        auto_lr_find=cfg.trainer.auto_lr_find,
        replace_sampler_ddp=cfg.trainer.replace_sampler_ddp,
        detect_anomaly=cfg.trainer.detect_anomaly,
        auto_scale_batch_size=cfg.trainer.auto_scale_batch_size,
        amp_backend=cfg.trainer.amp_backend,
        amp_level=cfg.trainer.amp_level,
        move_metrics_to_cpu=cfg.trainer.move_metrics_to_cpu,
        multiple_trainloader_mode=cfg.trainer.multiple_trainloader_mode,
    )

    trainer.fit(model=model)


if __name__ == "__main__":
    main()
