import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from rcwa.maxwell_coeff import *
from rcwa.maxwell_mode import *
from rcwa.scattering_matrix import *
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

nx = 20 # half of the harmonics along x and y directions
ny = 20
nx_grid = 100 # grid number along x direction, we use analytical Fourier transform, so nx_grid can be very small
n_opt = 20 # number of optimization grid along one direction
wavelength = 1.55
Lx = 1. # period
Ly = 1. # period
n_mode = 2 # number of modes to be optimized
ny_grid = int(nx_grid * Ly / Lx) # grid number along y direction
k0 = 2 * np.pi / wavelength # free space wavevector

# the foreground permittivity, silicon
eps_in = torch.tensor(3.48*3.48+0j, device = device, requires_grad = False, dtype = torch.complex128)
# the background permittivity, air
eps_out = torch.tensor(1.+0j, device = device, requires_grad = False, dtype = torch.complex128)

# set the initial value of the optimized permittivity to be 0.5 * (eps_in + eps_out)
de = 0.5 * torch.ones(2*n_opt//2, 2*n_opt//2, device = device, dtype = torch.float64)
de.requires_grad_(True)
ex = eps_out * torch.ones(nx_grid, ny_grid, device = device, dtype = torch.float64)
ex[nx_grid//2 - n_opt//2 : nx_grid//2 + n_opt//2, ny_grid//2 - n_opt//2 : ny_grid//2 + n_opt//2] += (eps_in - eps_out) * de

# the code calculate the modes in vacuum using analytical way
# in vacuum, the modes are just plane waves
coeff = MaxwellCoeff(nx, ny, Lx, Ly, device = device)
port_mode = MaxwellMode()
port_mode.compute_in_vacuum(wavelength, coeff, device = device)

# search for the modes with the highest effective index
# these two modes are the perpendicularly incident plane waves with different polarization
# we use them as the input and output ports
neff_port = port_mode.valsqrt.real / k0 / k0
neff_port, port_indices = torch.sort(neff_port, descending = True)

print(port_mode.valsqrt[port_indices[:n_mode]] / k0 / k0)

# I visualized the port modes here
# It is in fourier space, because they are all zeroth-order plane waves,
# you will just find the center value to be 1 and all other values to be 0
plt.subplots(2, 4, figsize = (10, 5))
for i in range(2):
    plt.subplot(2, 4, i*4+1)
    plt.imshow(port_mode.Ex_fourier(port_indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)
    plt.subplot(2, 4, i*4+2)
    plt.imshow(port_mode.Ey_fourier(port_indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)
    plt.subplot(2, 4, i*4+3)
    plt.imshow(port_mode.Hx_fourier(port_indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)
    plt.subplot(2, 4, i*4+4)
    plt.imshow(port_mode.Hy_fourier(port_indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)
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

# the modes in the structure
# this time, I transformed them back to real space
plt.subplots(n_mode, 4, figsize = (10, 5))
for i in range(n_mode):
    plt.subplot(n_mode, 4, i*4+1)
    plt.imshow(mode.Ex(indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)
    plt.subplot(n_mode, 4, i*4+2)
    plt.imshow(mode.Ey(indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)
    plt.subplot(n_mode, 4, i*4+3)
    plt.imshow(mode.Hx(indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)
    plt.subplot(n_mode, 4, i*4+4)
    plt.imshow(mode.Hy(indices[i]).detach().cpu().numpy().real)
    plt.clim(-1, 1)

plt.savefig("all_modes.png")
plt.close()

smat = ScatteringMatrix()
smat.compute(mode, 1.)
smat.port_project(port_mode, coeff)
Tuu_2 = torch.abs(smat.Tuu())**2

select_indices = port_indices[:n_mode]
inds_i, inds_j= torch.meshgrid(select_indices, select_indices, indexing = "ij")

# now you have the scattering matrix with input and output are perpendicular plane waves
# one with x-pol and the other with y-pol, so it is a 2 x 2 matrix
plt.imshow(Tuu_2[inds_i, inds_j].detach().cpu().numpy())
plt.colorbar()
plt.clim(0, 1)
plt.axis("off")
plt.savefig("smat.png")
plt.close()



