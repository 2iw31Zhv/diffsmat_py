# This file defines differentiable matrix functions used in DiffSMat.
# For dense matrix, plase refer to https://www.cs.columbia.edu/~cxz/publications/diff_smat.pdf
# Author: Ziwei Zhu

import numpy as np
import torch
from torch.autograd import Function

class sqrtm_ops_func_core(Function):
    @staticmethod
    def forward(ctx, ops, valsqrt):
        ctx.save_for_backward(valsqrt)
        return torch.diag(valsqrt) # m x m 
    @staticmethod
    def backward(ctx, grad_output):
        valsqrt = ctx.saved_tensors[0]
        Y = grad_output / torch.conj(valsqrt[:, None] + valsqrt[None, :])
        return Y, None
sqrtm_ops_core = sqrtm_ops_func_core.apply