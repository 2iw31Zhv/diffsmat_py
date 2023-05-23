import numpy as np
import torch

class BlockToeplitz2D:
    def __init__(self, mat2D):
        self.mat2D = mat2D
        self.nx, self.ny = mat2D.shape
        self.nx  = (self.nx + 1) // 2
        self.ny  = (self.ny + 1) // 2
        self.ndim = self.nx * self.ny
        self.dtype = mat2D.dtype
        self.device = mat2D.device

    def to_dense(self):
        rows = torch.arange(self.nx * self.ny, dtype = torch.int64, device = self.device)
        cols = torch.arange(self.nx * self.ny, dtype = torch.int64, device = self.device)
        rows, cols = torch.meshgrid(rows, cols, indexing = "ij")
        ms = torch.div(rows, self.nx, rounding_mode = 'floor').to(torch.int64)
        ns = rows - ms * self.nx
        js = torch.div(cols, self.nx, rounding_mode = 'floor').to(torch.int64)
        ls = cols - js * self.nx
        diffi = ms - js
        diffj = ns - ls
        return self.mat2D[self.nx + diffj - 1, self.ny + diffi - 1]