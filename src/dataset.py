import os
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import (
    Dataset, 
)

class RiskfuelBatch:
    def __init__(self, batch_id : int, x: torch.Tensor, y: torch.Tensor):
        self.batch_id = batch_id
        self.x = x
        self.y = y

    def __iter__(self):
        return iter((self.batch_id, (self.x, self.y)))

    def pin_memory(self):
        """
        Ensures pinning on custom batch type
        """

        # NOTE: Not pinning to pinned memory since this is a toy example for actuated
        #       generally you would transfer to pinned memory w/ several workers then 
        #       let torch move things to GPU  
        # self.x = self.x.pin_memory()
        # self.y = self.y.pin_memory()
        return self

class FakeDataset(Dataset):

    _cursor : int = 0

    data : Dict[str, Any]

    def __init__(self, nrows : int, ncols : int, nsamples : int):
        self.data = [ self._gen_fake_sample(nrows, ncols) for _ in range(nsamples)]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i : int):

        x = torch.tensor(self.data[i][0], device="cuda:0", dtype=torch.float32)
        y = torch.tensor(self.data[i][1], device="cuda:0", dtype=torch.float32)

        return i, (x, y)

    def _gen_fake_sample(self, nrows : int, ncols : int):
        X = np.random.rand(nrows, ncols)
        Y = np.random.rand(nrows, ncols)

        return (X, Y)
        

def custom_collate_fn(batches : List[Tuple[torch.Tensor, torch.Tensor]]):
    return [RiskfuelBatch(batch_id, x, y) for batch_id, (x, y) in batches]
