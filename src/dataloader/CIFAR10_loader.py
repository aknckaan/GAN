from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "cifar10",
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Resize(32),
            ]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            CIFAR10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.CIFAR10_train, self.CIFAR10t_val = random_split(
                CIFAR10_full, [49500, 500]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.CIFAR10_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.CIFAR10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.CIFAR10_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.CIFAR10_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
