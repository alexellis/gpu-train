from typing import Any
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn.functional as F

mse_loss = torch.nn.MSELoss(reduction="none")

class DummyModel(pl.LightningModule):
    """Lightning Module for working with RectNet

    @param input_dims
    @param output_dims
    @param depth
    @param width
    @param activation
    """

    def __init__(
        self,
        architecture: Any
    ):
        super().__init__()
        self.model = architecture

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        batch_id, (x, y) = batch[0]
        y_hat = self.forward(x)
        mse_loss = F.mse_loss(y, y_hat)

        # self.log("Train Unscaled Loss", mse_loss, sync_dist=True) # TODO: Look into sync_dist
        
        return mse_loss

    def configure_optimizers(self):
        # TODO: Figure out what is happening when we load from a checkpoint
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
