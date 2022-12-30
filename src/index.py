import pytorch_lightning as pl
import torch
from architecture import LinearNet
from model import DummyModel
from data_module import FakeDataModule
from torch.utils.data import RandomSampler
from constants import (
    INPUT_DIMS, 
    OUTPUT_DIMS,
    WIDTH,
    DEPTH,
    MIN_EPOCHS,
    MAX_EPOCHS
)

def main():
    
    architecture = LinearNet(input_dims=INPUT_DIMS, output_dims=OUTPUT_DIMS, widths = [WIDTH] * DEPTH, activation=torch.nn.ReLU())
    model = DummyModel(architecture=architecture)
    trainer = pl.Trainer(
        gpus=1, 
        callbacks=[],
        logger=False,
        log_every_n_steps=10,
        min_epochs=MIN_EPOCHS,
        max_epochs=MAX_EPOCHS,
    )

    trainer.fit(model, FakeDataModule(sampler=RandomSampler))


if __name__ == "__main__":
    main()
