import os
import hydra
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_pod.network.module import LitModel
from lightning_pod.pipeline.datamodule import LitDataModule

NETWORKPATH = Path(__file__).parent
PODPATH = NETWORKPATH.parents[0]
PROJECTPATH = NETWORKPATH.parents[1]


@hydra.main(config_path=NETWORKPATH, config_name="hydra_config.yaml")
def main(cfg):
    # SET LOGGER
    logs_dir = os.path.join(PROJECTPATH, "logs")
    logger = TensorBoardLogger(logs_dir, name="lightning_logs")
    # SET PROFILER
    profile_dir = os.path.join(logs_dir, "profiler")
    profiler = SimpleProfiler(dirpath=profile_dir, filename="profiler", extended=True)
    # SET CHECKPOINT CALLBACK
    chkpt_dir = os.path.join(PROJECTPATH, "models", "checkpoints")
    checkpoint_callback = ModelCheckpoint(dirpath=chkpt_dir, filename="model")
    # SET EARLYSTOPPING CALLBACK
    early_stopping = EarlyStopping(monitor="loss", mode="min")
    # SET CALLBACKS
    callbacks = [checkpoint_callback, early_stopping]
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
        limit_predict_batches=None,
        limit_test_batches=None,
        limit_val_batches=None,
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
        plugins=None,
        default_root_dir=None,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
        num_nodes=1,
        num_processes=None,
        gpus=None,
        auto_select_gpus=False,
        tpu_cores=None,
        ipus=None,
        overfit_batches=0.0,
        track_grad_norm=-1,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=None,
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        max_time=None,
        val_check_interval=None,
        flush_logs_every_n_steps=None,
        log_every_n_steps=50,
        sync_batchnorm=False,
        weights_save_path=None,
        weights_summary="top",
        num_sanity_val_steps=2,
        resume_from_checkpoint=None,
        benchmark=None,
        reload_dataloaders_every_n_epochs=0,
        auto_lr_find=False,
        replace_sampler_ddp=True,
        detect_anomaly=False,
        auto_scale_batch_size=False,
        amp_backend="negative",
        amp_level=None,
        move_metrics_to_cpu=False,
        multiple_trainloader_mode="max_size_cycle",
    )
    # TRAIN MODEL
    trainer.fit(model=model, datamodule=datamodule)
    # TEST MODEL
    trainer.test(ckpt_path="best", datamodule=datamodule)
    # PERSIST MODEL
    pretrained_dir = os.path.join(PROJECTPATH, "models", "production")
    modelpath = os.path.join(pretrained_dir, "model.onnx")
    input_sample = datamodule.train_data.dataset[0][0]
    model.to_onnx(modelpath, input_sample=input_sample, export_params=True)


if __name__ == "__main__":
    main()
