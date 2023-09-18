import unittest
import matplotlib.pyplot as plt
import numpy as np
import torch
from .toeplitz import Toeplitz
from .blocktoeplitz_2d import BlockToeplitz2D

def analytical_fourier_2d(e, Lx, Ly, M, N, type = "continuous"):
    """
    Compute Analytical Fourier transform from a grid-like distribution

    Parameters
    ----------
    e : torch.Tensor Size (nx, ny)
        Grid-like distribution on the range (Lx, Ly)
        the step size of the grid is then dx = Lx / nx, dy = Ly / ny
        the grid is defined such as e[i, j] = e((i+0.5)*dx, (j+0.5)*dy) located at a cell with size (dx, dy)
        each grid is uniform
    Lx : float
    Ly : float
    M : int
    N : int

    Returns
    -------
    torch.Tensor
        Analytical Fourier transform with the size as (2*nx_harmonics-1, 2*ny_harmonics-1)
        fe2D(m, n) = \frac{1}{Lx*Ly} \int^Lx_0 \int^Ly_0 e(x, y) exp(-1j * kx[m] * x - 1j * ky[n] * y) dx dy
    """
    device = e.device
    ny, nx = e.shape
    dx = Lx / nx
    dy = Ly / ny
    ms = torch.arange(-M+1, M, device = device)
    ns = torch.arange(-N+1, N, device = device)
    kx = 2 * np.pi / Lx * ms
    ky = 2 * np.pi / Ly * ns
    xs = torch.linspace(0.5*dx, Lx-0.5*dx, nx, requires_grad=False, device=e.device, dtype = torch.float64)
    ys = torch.linspace(0.5*dy, Ly-0.5*dy, ny, requires_grad=False, device=e.device, dtype = torch.float64)
    if type == "ex":
        basis_kx_conj = torch.exp(-1j * kx[:, None] * xs[None, :]) * torch.special.sinc(ms * dx / Lx)[:, None] # (2M-1) x nx
        finvex = dx * torch.sum(basis_kx_conj[:, None, :] / e[None, :, :], dim = 2) / Lx # (2M-1) x ny
        finvex = Toeplitz(finvex)
        finvex = finvex.to_dense() # M x M x ny
        fex = torch.linalg.inv(torch.permute(finvex, (2, 0, 1))) # ny x M x M
        fey = dy * torch.exp(-1j * ky[:, None] * ys[None, :]) * torch.special.sinc(ns * dy / Ly)[:, None] / Ly # (2N-1) x ny
        fey = Toeplitz(fey)
        fey = fey.to_dense() # N x N x ny
        fe2D = torch.einsum("ijm, mkl->ikjl", fey, fex).reshape((M*N, M*N))
        return fe2D
    elif type == "ey":
        basis_ky_conj = torch.exp(-1j * ky[:, None] * ys[None, :]) * torch.special.sinc(ns * dy / Ly)[:, None] # (2N-1) x ny
        finvey = dy * torch.sum(basis_ky_conj[:, :, None] / e[None, :, :], dim = 1) / Ly # (2N-1) x nx
        finvey = Toeplitz(finvey)
        finvey = finvey.to_dense() # N x N x nx
        fey = torch.linalg.inv(torch.permute(finvey, (2, 0, 1))) # nx x N x N
        fex = dx * torch.exp(-1j * kx[:, None] * xs[None, :]) * torch.special.sinc(ms * dx / Lx)[:, None] / Lx # (2M-1) x nx
        fex = Toeplitz(fex)
        fex = fex.to_dense() # M x M x nx
        fe2D = torch.einsum("mij, klm->ikjl", fey, fex).reshape((M*N, M*N))
        return fe2D
    else:
        basis_kx_conj = torch.exp(-1j * kx[:, None] * xs[None, :]) * torch.special.sinc(ms * dx / Lx)[:, None]
        basis_ky_conj = torch.exp(-1j * ky[None, :] * ys[:, None]) * torch.special.sinc(ns * dy / Ly)[None, :]
        fe2D =  dx * dy * basis_kx_conj @ e.T @ basis_ky_conj / (Lx * Ly)
        return BlockToeplitz2D(fe2D).to_dense()


class TestFourier2D(unittest.TestCase):
    def test_continuous(self):
        e = torch.ones((3, 3), dtype = torch.complex128, device = "cpu")
        e[:, 1] = 1.445 * 1.445
        fe2D = analytical_fourier_2d(e, 1, 1, 5, 5, type = "continuous")
        fe2D_x = analytical_fourier_2d(e, 1, 1, 5, 5, type = "ex")
        fe2D_y = analytical_fourier_2d(e, 1, 1, 5, 5, type = "ey")

        if 0:
            plt.subplots(1, 3, figsize = (12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(fe2D.real)
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(fe2D_x.real)
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(fe2D_y.real)
            plt.colorbar()
            plt.savefig("test_fourier_2d.png")
            plt.close()
        self.assertTrue(fe2D.shape == fe2D_x.shape)
        self.assertTrue(fe2D.shape == fe2D_y.shape)

if __name__ == '__main__':
    unittest.main()