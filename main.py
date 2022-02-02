from gan import GAN
from src.dataloader import CIFAR10DataModule
import os
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = max(0, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
AVAIL_GPUS

dm = CIFAR10DataModule()
model = GAN(*dm.size())
tb_logger = pl_loggers.TensorBoardLogger("/content/drive/MyDrive/cifar_gan/")
resume_from_checkpoint = "/content/drive/MyDrive/cifar_gan/default/version_99/checkpoints/epoch=3193-step=2472155.ckpt"
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=5000,
    progress_bar_refresh_rate=20,
    logger=tb_logger,
    resume_from_checkpoint=resume_from_checkpoint,
)
trainer.fit(model, dm)
