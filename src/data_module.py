from typing import Any
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from dataset import FakeDataset, custom_collate_fn
from constants import INPUT_DIMS, BATCH_SIZE, NUM_BATCHES

class FakeDataModule(LightningDataModule):

    def __init__(
        self, 
        sampler: Any
    ):
        super().__init__()

        self.train_dataset = FakeDataset(nrows = BATCH_SIZE, ncols = INPUT_DIMS, nsamples = NUM_BATCHES)
        self.val_dataset = FakeDataset(nrows = BATCH_SIZE, ncols = INPUT_DIMS, nsamples = NUM_BATCHES)
        self.sampler = sampler

    def train_dataloader(self) -> DataLoader:
        
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            sampler=self.sampler(self.train_dataset),
            collate_fn=custom_collate_fn
        )
