import os
from pathlib import Path

import hydra
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from torch.utils.data import TensorDataset

from lightning_pod.core.module import LitModel
from lightning_pod.pipeline.datamodule import LitDataModule

# SET PATHS
filepath = Path(__file__)
PROJECTPATH = os.getcwd()


@hydra.main(
    config_path=str(filepath.parent),
    config_name="trainer",
    version_base=hydra.__version__,
)
def main(cfg: DictConfig) -> None:
    # SET LOGGER
    logs_dir = os.path.join(PROJECTPATH, "logs")
    logger = TensorBoardLogger(logs_dir, name="logger")
    # SET PROFILER
    profile_dir = os.path.join(logs_dir, "profiler")
    profiler = PyTorchProfiler(dirpath=profile_dir, filename="profiler")
    # SET CHECKPOINT CALLBACK
    chkpt_dir = os.path.join(PROJECTPATH, "models", "checkpoints")
    checkpoint_callback = ModelCheckpoint(dirpath=chkpt_dir, filename="model")
    # SET EARLYSTOPPING CALLBACK
    early_stopping = EarlyStopping(monitor="loss", mode="min")
    # SET CALLBACKS
    callbacks = [checkpoint_callback, early_stopping]
    # SET PLUGINS
    plugins = None
    # SET SEED
    seed_everything(42, workers=True)
    #  GET DATALOADER
    datamodule = LitDataModule()
    #  SET MODEL
    model = LitModel()
    # SET TRAINER
    trainer = Trainer(
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
        logger=logger,
        profiler=profiler,
        callbacks=callbacks,
        plugins=plugins,
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
        flush_logs_every_n_steps=cfg.trainer.flush_logs_every_n_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        weights_save_path=cfg.trainer.weights_save_path,
        weights_summary=cfg.trainer.weights_summary,
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
    # TRAIN MODEL
    trainer.fit(model=model, datamodule=datamodule)
    # IF NOT FAST DEV RUN: TEST, PREDICT, PERSIST
    if not cfg.trainer.fast_dev_run:
        # TEST MODEL
        trainer.test(ckpt_path="best", datamodule=datamodule)
        # PERSIST MODEL
        # TODO write util to search models dir and append version number
        pretrained_dir = os.path.join(PROJECTPATH, "models", "onnx")
        modelpath = os.path.join(pretrained_dir, "model.onnx")
        input_sample = datamodule.train_data.dataset[0][0]
        model.to_onnx(modelpath, input_sample=input_sample, export_params=True)
        # PREDICT
        predictions = trainer.predict(model, datamodule.val_dataloader())
        # EXPORT PREDICTIONS
        predictions = torch.vstack(predictions)  # type: ignore[arg-type]
        predictions = TensorDataset(predictions)
        predictions_dir = os.path.join(PROJECTPATH, "data", "predictions")
        prediction_fname = os.path.join(predictions_dir, "predictions.pt")
        torch.save(predictions, prediction_fname)
        # EXPORT ALL DATA SPLITS FOR REPRODUCIBILITY
        split_dir = os.path.join(PROJECTPATH, "data", "training_split")
        train_split_fname = os.path.join(split_dir, "train.pt")
        test_split_fname = os.path.join(split_dir, "test.pt")
        val_split_fname = os.path.join(split_dir, "val.pt")
        torch.save(datamodule.train_data, train_split_fname)
        torch.save(datamodule.test_data, test_split_fname)
        torch.save(datamodule.val_data, val_split_fname)


if __name__ == "__main__":
    main()
