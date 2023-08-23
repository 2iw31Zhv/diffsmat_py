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

# this function is from https://github.com/kch3782/torcwa
class Eig(torch.autograd.Function):
    broadening_parameter = 1e-10

    @staticmethod
    def forward(ctx,x):
        ctx.input = x
        eigval, eigvec = torch.linalg.eig(x)
        ctx.eigval = eigval.cpu()
        ctx.eigvec = eigvec.cpu()
        return eigval, eigvec

    @staticmethod
    def backward(ctx,grad_eigval,grad_eigvec):
        eigval = ctx.eigval.to(grad_eigval)
        eigvec = ctx.eigvec.to(grad_eigvec)

        grad_eigval = torch.diag(grad_eigval)
        s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)

        # Lorentzian broadening: get small error but stabilizing the gradient calculation
        if Eig.broadening_parameter is not None:
            F = torch.conj(s)/(torch.abs(s)**2 + Eig.broadening_parameter)
        elif s.dtype == torch.complex64:
            F = torch.conj(s)/(torch.abs(s)**2 + 1.4e-45)
        elif s.dtype == torch.complex128:
            F = torch.conj(s)/(torch.abs(s)**2 + 4.9e-324)

        diag_indices = torch.linspace(0,F.shape[-1]-1,F.shape[-1],dtype=torch.int64)
        F[diag_indices,diag_indices] = 0.
        XH = torch.transpose(torch.conj(eigvec),-2,-1)
        tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)

        grad = torch.matmul(torch.matmul(torch.inverse(XH), grad_eigval + tmp), XH)
        if not torch.is_complex(ctx.input):
            grad = torch.real(grad)
        return grad 
eig_diff = Eig.apply