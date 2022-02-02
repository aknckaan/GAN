import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from gan import GAN
from src.dataloader.CIFAR10_loader import CIFAR10DataModule

AVAIL_GPUS = max(0, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
AVAIL_GPUS

dm = CIFAR10DataModule()
model = GAN(*dm.size())
tb_logger = pl_loggers.TensorBoardLogger("../gan_checkpoints/")
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=5000,
    progress_bar_refresh_rate=20,
    logger=tb_logger,
)
trainer.fit(model, dm)
