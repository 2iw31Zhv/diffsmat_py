import numpy as np
import torch

class Toeplitz:
    def __init__(self, vec):
        self.vec = vec
        self.n = vec.shape[0]
        self.nbatch = 1
        if len(vec.shape) == 2:
            self.nbatch = vec.shape[1]
        self.ndim = (self.n + 1) // 2
        self.dtype = vec.dtype
        self.device = vec.device
    def to_dense(self):
        rows = torch.arange(self.ndim, dtype = torch.int64, device = self.device)
        cols = torch.arange(self.ndim, dtype = torch.int64, device = self.device)
        rows, cols = torch.meshgrid(rows, cols, indexing = "ij")
        if self.nbatch == 1:
            return self.vec[rows - cols + self.ndim - 1]
        else:
            out = torch.stack([self.vec[rows - cols + self.ndim - 1, i] for i in range(self.nbatch)], dim = 2)
            return out

import unittest
class TestToeplitzMatrix(unittest.TestCase):
    def test1(self):
        vec = torch.arange(5, device = "cpu")
        mat = Toeplitz(vec)
        gt = torch.tensor([[2, 1, 0], [3, 2, 1], [4, 3, 2]], dtype = vec.dtype, device = vec.device)
        self.assertTrue(torch.allclose(mat.to_dense(), gt))
    
    def test2(self):
        n = 3
        vec = torch.arange(5, device = "cpu")
        vecs = vec.repeat(n, 1).T
        mats = Toeplitz(vecs).to_dense()
        gt = torch.tensor([[2, 1, 0], [3, 2, 1], [4, 3, 2]], dtype = vec.dtype, device = vec.device)
        for i in range(n):
            self.assertTrue(torch.allclose(mats[:, :, i].to_dense(), gt))

if __name__ == '__main__':
    unittest.main()
