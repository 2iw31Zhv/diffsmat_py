import numpy as np
import torch


def analytical_fourier_2d(e, Lx, Ly, M, N, device = "cpu"):
    """
    Compute Analytical Fourier transform from a grid-like distribution

    Parameters
    ----------
    e : torch.Tensor Size (nx, ny)
        Grid-like distribution on the range (Lx, Ly)
        the step size of the grid is then dx = Lx / nx, dy = Ly / ny
        the grid is defined such as e[i, j] = e((i+0.5)*dx, (j+0.5)*dy) located at a cell with size (dx, dy)
        each grid is uniform
    nx_harmonics : int
        Number of harmonics along the x axis
        kx = 2 * pi / Lx * torch.arange(-nx_harmonics+1, nx_harmonics, device = device)
    ny_harmonics : int
        Number of harmonics along the y axis
        ky = 2 * pi / Ly * torch.arange(-ny_harmonics+1, ny_harmonics, device = device)

    Returns
    -------
    torch.Tensor
        Analytical Fourier transform with the size as (2*nx_harmonics-1, 2*ny_harmonics-1)
        fe2D(m, n) = \frac{1}{Lx*Ly} \int^Lx_0 \int^Ly_0 e(x, y) exp(-1j * kx[m] * x - 1j * ky[n] * y) dx dy
    """
    nx, ny = e.shape
    dx = Lx / nx
    dy = Ly / ny
    ms = torch.arange(-M+1, M, device = device)
    ns = torch.arange(-N+1, N, device = device)
    kx = 2 * np.pi / Lx * ms
    ky = 2 * np.pi / Ly * ns
    xs = torch.linspace(0.5*dx, Lx-0.5*dx, nx, requires_grad=False, device=e.device, dtype = torch.float64)
    ys = torch.linspace(0.5*dy, Ly-0.5*dy, ny, requires_grad=False, device=e.device, dtype = torch.float64)
    basis_kx_conj = torch.exp(-1j * kx[:, None] * xs[None, :]) * torch.special.sinc(ms * dx / Lx)[:, None]
    basis_ky_conj = torch.exp(-1j * ky[None, :] * ys[:, None]) * torch.special.sinc(ns * dy / Ly)[None, :]
    fe2D =  dx * dy * basis_kx_conj @ e @ basis_ky_conj
    return fe2D / (Lx * Ly)
