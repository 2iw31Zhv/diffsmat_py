import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from maxwell_coeff import *
from maxwell_mode import *
from scattering_matrix import *
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

nx = ny = 3 # half of the harmonics along x and y directions
nx_grid = 4 # grid number along x direction, we use analytical Fourier transform, so nx_grid can be very small
n_opt = 2 # number of optimization grid along one direction
wavelength = 1.55
Lx = 1. # period
Ly = 1. # period
n_mode = 2 # number of modes to be optimized
ny_grid = int(nx_grid * Ly / Lx) # grid number along y direction
k0 = 2 * np.pi / wavelength # free space wavevector

eps_in = torch.tensor(3.48*3.48+0j, device = device, requires_grad = False, dtype = torch.complex128)
eps_out = torch.tensor(1.+0j, device = device, requires_grad = False, dtype = torch.complex128)

de = 0.5 * torch.ones(2*n_opt//2, 2*n_opt//2, device = device, dtype = torch.float64)
de.requires_grad_(True)

ex = eps_out * torch.ones(nx_grid, ny_grid, device = device, dtype = torch.float64)
ex[nx_grid//2 - n_opt//2 : nx_grid//2 + n_opt//2, ny_grid//2 - n_opt//2 : ny_grid//2 + n_opt//2] += (eps_in - eps_out) * de

coeff = MaxwellCoeff(nx, ny, Lx, Ly, device = device)
# coeff.compute(wavelength, ex, device = device)
port_mode = MaxwellMode()
port_mode.compute_in_vacuum(wavelength, coeff.kx, coeff.ky, device = device)
# port_mode.compute(coeff, device = device)
neff_port = port_mode.valsqrt.real / k0 / k0
neff_port, port_indices = torch.sort(neff_port, descending = True)
print(neff_port[:n_mode])

plt.subplots(2, 4, figsize = (10, 5))
for i in range(2):
    plt.subplot(2, 4, i*4+1)
    plt.imshow(port_mode.Ex(port_indices[i]).detach().cpu().numpy().real)
    plt.subplot(2, 4, i*4+2)
    plt.imshow(port_mode.Ey(port_indices[i]).detach().cpu().numpy().real)
    plt.subplot(2, 4, i*4+3)
    plt.imshow(port_mode.Hx(port_indices[i]).detach().cpu().numpy().real)
    plt.subplot(2, 4, i*4+4)
    plt.imshow(port_mode.Hy(port_indices[i]).detach().cpu().numpy().real)
plt.savefig("all_modes_port.png")
plt.close()


plt.imshow(ex.detach().cpu().numpy().real)
plt.savefig("ex.png")

coeff.compute(wavelength, ex, device = device)

mode = MaxwellMode()
mode.compute(coeff, device = device)
neff = mode.valsqrt.real / k0 / k0
neff, indices = torch.sort(neff, descending = True)
print(neff[:n_mode])

plt.subplots(n_mode, 4, figsize = (10, 15))
for i in range(n_mode):
    plt.subplot(n_mode, 4, i*4+1)
    plt.imshow(mode.Ex(indices[i]).detach().cpu().numpy().real)
    plt.subplot(n_mode, 4, i*4+2)
    plt.imshow(mode.Ey(indices[i]).detach().cpu().numpy().real)
    plt.subplot(n_mode, 4, i*4+3)
    plt.imshow(mode.Hx(indices[i]).detach().cpu().numpy().real)
    plt.subplot(n_mode, 4, i*4+4)
    plt.imshow(mode.Hy(indices[i]).detach().cpu().numpy().real)

plt.savefig("all_modes.png")
plt.close()

smat = ScatteringMatrix()
smat.compute(mode, 1.)
smat.port_project(port_mode, coeff)
smatsqr = torch.abs(smat.smat)**2
plt.imshow(smatsqr[torch.diag(port_indices[:n_mode]), torch.diag(port_indices[:n_mode])].detach().cpu().numpy())
plt.colorbar()
plt.savefig("smat.png")
plt.close()



